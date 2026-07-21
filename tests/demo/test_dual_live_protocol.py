from __future__ import annotations

import base64
import io
import json
import struct
import zlib

import numpy as np
from PIL import Image
import pytest

from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MAX_PIXELS,
    MODEL_ID,
    PROTOCOL_ID,
    ProtocolError,
    decode_request,
    decode_response,
    encode_request,
    mask_sha256,
    rgb_sha256,
)


def _png_base64(array: np.ndarray, *, mode: str) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _png_header(width: int, height: int) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\0" + b"\0" * (width * 3) for _ in range(height))
    payload = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", data)
    payload += chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b"")
    return base64.b64encode(payload).decode("ascii")


def _valid_response_fixture() -> tuple[dict[str, object], dict[str, object]]:
    image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    request_id = "a" * 32
    mask = np.zeros((7, 11), dtype=np.uint8)
    mask[2:5, 3:8] = 1
    expected = {
        "protocol": PROTOCOL_ID,
        "request_id": request_id,
        "input_sha256": rgb_sha256(image),
        "model_id": MODEL_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "input_shape": (7, 11),
    }
    payload: dict[str, object] = {
        **{key: value for key, value in expected.items() if key != "input_shape"},
        "mask_sha256": mask_sha256(mask),
        "mask_png_base64": _png_base64(mask * 255, mode="L"),
        "latency_ms": 12.5,
        "execution": "live",
    }
    return payload, expected


def test_response_round_trip_preserves_bound_binary_mask() -> None:
    payload, expected = _valid_response_fixture()

    result = decode_response(payload, expected)

    assert result.request_id == "a" * 32
    assert result.input_sha256 == expected["input_sha256"]
    assert result.model_id == MODEL_ID
    assert result.checkpoint_sha256 == CHECKPOINT_SHA256
    assert result.execution == "live"
    assert result.mask.shape == (7, 11)
    assert result.mask.dtype == np.uint8
    assert result.mask_sha256 == mask_sha256(result.mask)


def test_request_round_trip_preserves_exact_rgb_and_digest() -> None:
    image = np.arange(12 * 9 * 3, dtype=np.uint8).reshape(12, 9, 3)

    payload = encode_request("a" * 32, image)
    request_id, decoded, digest = decode_request(payload)

    assert request_id == "a" * 32
    np.testing.assert_array_equal(decoded, image)
    assert digest == rgb_sha256(image)
    encoded = base64.b64decode(str(payload["image_png_base64"]), validate=True)
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    "field",
    ["protocol", "request_id", "input_sha256", "mask_sha256", "checkpoint_sha256"],
)
def test_response_rejects_every_binding_drift(field: str) -> None:
    payload, expected = _valid_response_fixture()
    payload[field] = "bad"

    with pytest.raises(ProtocolError):
        decode_response(payload, expected)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"request_id": "a" * 32}, None),
        ({"request_id": "g" * 32}, None),
        ({"request_id": "a" * 32, "protocol": PROTOCOL_ID}, None),
    ],
)
def test_request_rejects_incomplete_or_non_hex_schema(
    payload: dict[str, object], expected: object
) -> None:
    del expected
    with pytest.raises(ProtocolError):
        decode_request(payload)


def test_request_rejects_unknown_field_and_digest_or_image_drift() -> None:
    image = np.zeros((5, 8, 3), dtype=np.uint8)
    payload = encode_request("a" * 32, image)

    payload["unexpected"] = "value"
    with pytest.raises(ProtocolError):
        decode_request(payload)

    payload = encode_request("a" * 32, image)
    payload["input_sha256"] = "0" * 64
    with pytest.raises(ProtocolError):
        decode_request(payload)


def test_response_requires_exact_schema_live_execution_and_valid_latency() -> None:
    payload, expected = _valid_response_fixture()
    payload["extra"] = True
    with pytest.raises(ProtocolError):
        decode_response(payload, expected)

    for latency in (float("nan"), float("inf"), -0.1):
        payload, expected = _valid_response_fixture()
        payload["latency_ms"] = latency
        with pytest.raises(ProtocolError):
            decode_response(payload, expected)

    payload, expected = _valid_response_fixture()
    payload["execution"] = "cached"
    with pytest.raises(ProtocolError):
        decode_response(payload, expected)


def test_response_rejects_mask_geometry_and_binary_or_digest_drift() -> None:
    payload, expected = _valid_response_fixture()
    payload["mask_png_base64"] = _png_base64(np.ones((6, 11), dtype=np.uint8) * 255, mode="L")
    with pytest.raises(ProtocolError):
        decode_response(payload, expected)

    payload, expected = _valid_response_fixture()
    payload["mask_png_base64"] = _png_base64(
        np.full((7, 11), 127, dtype=np.uint8), mode="L"
    )
    with pytest.raises(ProtocolError):
        decode_response(payload, expected)

    payload, expected = _valid_response_fixture()
    payload["mask_sha256"] = "0" * 64
    with pytest.raises(ProtocolError):
        decode_response(payload, expected)


def test_payload_limit_and_pixel_limit_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload, expected = _valid_response_fixture()
    payload["mask_png_base64"] = "A" * (20 * 1024 * 1024 + 1)
    with pytest.raises(ProtocolError):
        decode_response(payload, expected)

    image = np.zeros((2, 3, 3), dtype=np.uint8)
    request = encode_request("a" * 32, image)
    monkeypatch.setattr("lesion_robustness.demo.dual_live_protocol.MAX_PIXELS", 4)
    with pytest.raises(ProtocolError):
        decode_request(request)

    assert MAX_PIXELS == 16_000_000


def test_request_maps_pillow_decompression_bombs_to_protocol_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = encode_request("a" * 32, np.zeros((2, 2, 3), dtype=np.uint8))
    payload["image_png_base64"] = _png_header(3, 2)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4)

    with pytest.raises(ProtocolError):
        decode_request(payload)


def test_protocol_errors_do_not_reflect_untrusted_payload() -> None:
    private_path = "C:\\private\\checkpoint_final.pth"
    with pytest.raises(ProtocolError) as caught:
        decode_request({"request_id": private_path})
    assert private_path.lower() not in str(caught.value).lower()
    assert private_path.lower() not in json.dumps(str(caught.value)).lower()
