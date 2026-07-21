"""Strict, hash-bound JSON payloads for the localhost nnU-Net sidecar."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import math
import re
import warnings

import numpy as np
from PIL import Image


PROTOCOL_ID = "imp.nnunet.sidecar.v1"
MODEL_ID = "L192-nnUNet-v2-raw-100ep"
CHECKPOINT_SHA256 = "3814716033afd464dacc573f92a5a44ff20eb7f2163d99b4f16ecff8aa278ea2"
MAX_PIXELS = 16_000_000
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 20 * 1024 * 1024

_REQUEST_FIELDS = frozenset({"protocol", "request_id", "input_sha256", "image_png_base64"})
_RESPONSE_FIELDS = frozenset(
    {
        "protocol",
        "request_id",
        "input_sha256",
        "model_id",
        "checkpoint_sha256",
        "mask_sha256",
        "mask_png_base64",
        "latency_ms",
        "execution",
    }
)
_EXPECTED_FIELDS = frozenset(
    {
        "protocol",
        "request_id",
        "input_sha256",
        "model_id",
        "checkpoint_sha256",
        "input_shape",
    }
)
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")


class ProtocolError(ValueError):
    """Raised when an untrusted sidecar payload fails protocol validation."""


@dataclass(frozen=True)
class SidecarResult:
    request_id: str
    input_sha256: str
    mask: np.ndarray
    mask_sha256: str
    model_id: str
    checkpoint_sha256: str
    latency_ms: float
    execution: str
    protocol: str = PROTOCOL_ID


def _invalid(kind: str) -> ProtocolError:
    return ProtocolError(f"invalid sidecar {kind}")


def _validate_geometry(height: int, width: int, *, kind: str) -> None:
    if height < 1 or width < 1 or height * width > MAX_PIXELS:
        raise _invalid(kind)


def validate_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise _invalid("image")
    _validate_geometry(int(value.shape[0]), int(value.shape[1]), kind="image")
    return np.ascontiguousarray(value)


def validate_binary_mask(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype != np.uint8:
        raise _invalid("mask")
    _validate_geometry(int(value.shape[0]), int(value.shape[1]), kind="mask")
    if not np.isin(value, (0, 1)).all():
        raise _invalid("mask")
    return np.ascontiguousarray(value)


def rgb_sha256(image: np.ndarray) -> str:
    rgb = validate_rgb(image)
    prefix = f"{rgb.shape[0]}x{rgb.shape[1]}x3|".encode("ascii")
    return hashlib.sha256(prefix + rgb.tobytes(order="C")).hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    binary = validate_binary_mask(mask)
    prefix = f"{binary.shape[0]}x{binary.shape[1]}|".encode("ascii")
    return hashlib.sha256(prefix + binary.tobytes(order="C")).hexdigest()


def _png_base64(array: np.ndarray, *, mode: str, kind: str) -> str:
    buffer = io.BytesIO()
    try:
        Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    except (OSError, ValueError) as exc:
        raise _invalid(kind) from exc
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _encoded_size(payload: Mapping[str, object], *, maximum: int, kind: str) -> None:
    try:
        size = len(json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise _invalid(kind) from exc
    if size > maximum:
        raise _invalid(kind)


def _validate_schema(
    payload: object, *, fields: frozenset[str], maximum: int, kind: str
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise _invalid(kind)
    _encoded_size(payload, maximum=maximum, kind=kind)
    return payload


def _validate_request_id(value: object, *, kind: str) -> str:
    if not isinstance(value, str) or _REQUEST_ID.fullmatch(value) is None:
        raise _invalid(kind)
    return value


def _validate_sha256(value: object, *, kind: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise _invalid(kind)
    return value


def _decode_png(value: object, *, mode: str, kind: str) -> np.ndarray:
    if not isinstance(value, str):
        raise _invalid(kind)
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as decoded:
                if decoded.format != "PNG" or decoded.mode != mode:
                    raise _invalid(kind)
                _validate_geometry(int(decoded.height), int(decoded.width), kind=kind)
                decoded.load()
                return np.array(decoded, dtype=np.uint8, copy=True)
    except (
        OSError,
        ValueError,
        UnicodeEncodeError,
        base64.binascii.Error,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise _invalid(kind) from exc


def encode_request(request_id: str, image: np.ndarray) -> dict[str, str]:
    rgb = validate_rgb(image)
    payload = {
        "protocol": PROTOCOL_ID,
        "request_id": _validate_request_id(request_id, kind="request"),
        "input_sha256": rgb_sha256(rgb),
        "image_png_base64": _png_base64(rgb, mode="RGB", kind="request"),
    }
    _encoded_size(payload, maximum=MAX_REQUEST_BYTES, kind="request")
    return payload


def decode_request(payload: object) -> tuple[str, np.ndarray, str]:
    request = _validate_schema(
        payload, fields=_REQUEST_FIELDS, maximum=MAX_REQUEST_BYTES, kind="request"
    )
    if request["protocol"] != PROTOCOL_ID:
        raise _invalid("request")
    request_id = _validate_request_id(request["request_id"], kind="request")
    digest = _validate_sha256(request["input_sha256"], kind="request")
    image = validate_rgb(_decode_png(request["image_png_base64"], mode="RGB", kind="request"))
    if not hmac.compare_digest(digest, rgb_sha256(image)):
        raise _invalid("request")
    return request_id, image, digest


def expected_bindings(request_id: str, image: np.ndarray) -> dict[str, object]:
    """Return the response bindings for one already-validated request image."""
    rgb = validate_rgb(image)
    return {
        "protocol": PROTOCOL_ID,
        "request_id": _validate_request_id(request_id, kind="expected bindings"),
        "input_sha256": rgb_sha256(rgb),
        "model_id": MODEL_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "input_shape": (int(rgb.shape[0]), int(rgb.shape[1])),
    }


def _validate_expected(expected: object) -> dict[str, object]:
    bindings = _validate_schema(
        expected, fields=_EXPECTED_FIELDS, maximum=2048, kind="expected bindings"
    )
    protocol = bindings["protocol"]
    model_id = bindings["model_id"]
    checkpoint = bindings["checkpoint_sha256"]
    if protocol != PROTOCOL_ID or model_id != MODEL_ID or checkpoint != CHECKPOINT_SHA256:
        raise _invalid("expected bindings")
    shape = bindings["input_shape"]
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
    ):
        raise _invalid("expected bindings")
    _validate_geometry(shape[0], shape[1], kind="expected bindings")
    return {
        "protocol": PROTOCOL_ID,
        "request_id": _validate_request_id(bindings["request_id"], kind="expected bindings"),
        "input_sha256": _validate_sha256(bindings["input_sha256"], kind="expected bindings"),
        "model_id": MODEL_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "input_shape": shape,
    }


def decode_response(payload: object, expected: object) -> SidecarResult:
    response = _validate_schema(
        payload, fields=_RESPONSE_FIELDS, maximum=MAX_RESPONSE_BYTES, kind="response"
    )
    bindings = _validate_expected(expected)
    for field in _EXPECTED_FIELDS - {"input_shape"}:
        value = bindings[field]
        if response[field] != value:
            raise _invalid("response")
    observed_mask_hash = _validate_sha256(response["mask_sha256"], kind="response")
    latency = response["latency_ms"]
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise _invalid("response")
    latency_ms = float(latency)
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        raise _invalid("response")
    if response["execution"] != "live":
        raise _invalid("response")
    encoded_mask = _decode_png(response["mask_png_base64"], mode="L", kind="response")
    if not np.isin(encoded_mask, (0, 255)).all():
        raise _invalid("response")
    mask = validate_binary_mask((encoded_mask // 255).astype(np.uint8))
    if mask.shape != bindings["input_shape"]:
        raise _invalid("response")
    if not hmac.compare_digest(observed_mask_hash, mask_sha256(mask)):
        raise _invalid("response")
    return SidecarResult(
        request_id=bindings["request_id"],
        input_sha256=bindings["input_sha256"],
        mask=mask,
        mask_sha256=observed_mask_hash,
        model_id=MODEL_ID,
        checkpoint_sha256=CHECKPOINT_SHA256,
        latency_ms=latency_ms,
        execution="live",
    )
