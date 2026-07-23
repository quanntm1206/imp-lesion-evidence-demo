from __future__ import annotations

import http.client
import json
import threading

import numpy as np
import pytest

from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MAX_RESPONSE_BYTES,
    MODEL_ID,
    PROTOCOL_ID,
    encode_response,
    rgb_sha256,
)
from lesion_robustness.demo.nnunet_client import (
    NnUNetClient,
    SidecarHealth,
    SidecarUnavailable,
)
from sidecar.nnunet.server import make_server


def rgb(height: int, width: int) -> np.ndarray:
    return np.full((height, width, 3), 127, dtype=np.uint8)


class FakeResponse:
    def __init__(
        self, status: int, payload: bytes, content_type: str = "application/json"
    ) -> None:
        self.status = status
        self._payload = payload
        self._content_type = content_type

    def read(self, max_bytes: int) -> bytes:
        return self._payload[:max_bytes]

    def getheader(self, name: str, default: str | None = None) -> str | None:
        if name.lower() == "content-type":
            return self._content_type
        return default


class FakeConnection:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[tuple[object, ...]] = []

    def request(self, *args: object, **_kwargs: object) -> None:
        self.requests.append(args + (_kwargs,))

    def getresponse(self) -> FakeResponse:
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response  # type: ignore[return-value]

    def close(self) -> None:
        return None


def response_with_wrong_mask_hash() -> dict[str, object]:
    image = rgb(8, 8)
    payload = encode_response(
        "a" * 32,
        rgb_sha256(image),
        np.zeros((8, 8), dtype=np.uint8),
        1.0,
    )
    payload["mask_sha256"] = "0" * 64
    return payload


def health_payload() -> dict[str, object]:
    return {
        "protocol": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "device": "cuda:0",
        "ready": True,
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0:7862",
        "http://localhost:7862",
        "https://127.0.0.1:7862",
        "http://127.0.0.1:9999",
    ],
)
def test_client_accepts_only_exact_sidecar_origin(url: str) -> None:
    with pytest.raises(ValueError, match="exact loopback sidecar"):
        NnUNetClient(url)


@pytest.mark.parametrize(
    ("response", "failure"),
    [
        (TimeoutError("C:" + "/private/model"), "timeout"),
        (FakeResponse(200, b"not-json"), "malformed_response"),
        (
            FakeResponse(200, json.dumps(response_with_wrong_mask_hash()).encode()),
            "binding_mismatch",
        ),
    ],
)
def test_client_rejects_timeout_malformed_json_and_hash_drift(
    monkeypatch: pytest.MonkeyPatch, response: object, failure: str
) -> None:
    connection = FakeConnection(response)
    monkeypatch.setattr(http.client, "HTTPConnection", lambda *_args, **_kwargs: connection)
    client = NnUNetClient()

    with pytest.raises(SidecarUnavailable, match=failure) as raised:
        client.predict("a" * 32, rgb(8, 8))

    assert "private" not in str(raised.value).lower()


def test_client_health_returns_only_pinned_ready_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection(FakeResponse(200, json.dumps(health_payload()).encode()))
    monkeypatch.setattr(http.client, "HTTPConnection", lambda *_args, **_kwargs: connection)

    assert NnUNetClient().health() == SidecarHealth(
        protocol=PROTOCOL_ID,
        model_id=MODEL_ID,
        checkpoint_sha256=CHECKPOINT_SHA256,
        device="cuda:0",
        ready=True,
    )
    assert connection.requests[0][0:2] == ("GET", "/health")


@pytest.mark.parametrize(
    "mutation",
    ["extra", "missing", "protocol", "model_id", "checkpoint_sha256", "device", "ready"],
)
def test_client_health_rejects_non_exact_identity(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    payload = health_payload()
    if mutation == "extra":
        payload["extra"] = "forged"
    elif mutation == "missing":
        payload.pop("ready")
    elif mutation == "ready":
        payload["ready"] = False
    else:
        payload[mutation] = "forged"
    connection = FakeConnection(FakeResponse(200, json.dumps(payload).encode()))
    monkeypatch.setattr(http.client, "HTTPConnection", lambda *_args, **_kwargs: connection)

    with pytest.raises(SidecarUnavailable, match="binding_mismatch"):
        NnUNetClient().health()


@pytest.mark.parametrize(
    ("response", "failure"),
    [
        (FakeResponse(503, ("C:" + "/private").encode()), "http_status"),
        (FakeResponse(200, b"{}", "text/plain"), "malformed_response"),
        (FakeResponse(200, b"x" * (MAX_RESPONSE_BYTES + 1)), "malformed_response"),
    ],
)
def test_client_rejects_http_content_type_and_oversized_responses(
    monkeypatch: pytest.MonkeyPatch, response: FakeResponse, failure: str
) -> None:
    connection = FakeConnection(response)
    monkeypatch.setattr(http.client, "HTTPConnection", lambda *_args, **_kwargs: connection)

    with pytest.raises(SidecarUnavailable, match=failure) as raised:
        NnUNetClient().predict("a" * 32, rgb(8, 8))

    assert "private" not in str(raised.value).lower()


def test_predict_posts_to_versioned_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    image = rgb(8, 8)
    payload = encode_response(
        "a" * 32,
        rgb_sha256(image),
        np.zeros(image.shape[:2], dtype=np.uint8),
        1.0,
    )
    connection = FakeConnection(FakeResponse(200, json.dumps(payload).encode()))
    monkeypatch.setattr(http.client, "HTTPConnection", lambda *_args, **_kwargs: connection)

    NnUNetClient().predict("a" * 32, image)

    assert connection.requests[0][0:2] == ("POST", "/v1/predict")


def test_client_matches_actual_sidecar_handler_without_fixed_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        model_id = MODEL_ID
        checkpoint_sha256 = CHECKPOINT_SHA256
        device = "cuda:0"
        ready = True

        def predict(self, image: np.ndarray) -> tuple[np.ndarray, float]:
            return np.zeros(image.shape[:2], dtype=np.uint8), 1.0

    server = make_server(FakeBackend(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    original_connection = http.client.HTTPConnection
    monkeypatch.setattr(
        http.client,
        "HTTPConnection",
        lambda _host, _port, timeout: original_connection(
            "127.0.0.1", server.server_port, timeout=timeout
        ),
    )
    thread.start()
    try:
        client = NnUNetClient()
        assert client.health().ready is True
        result = client.predict("a" * 32, rgb(8, 8))
        assert result.request_id == "a" * 32
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
