"""Fail-closed client for the pinned localhost nnU-Net sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import socket
from typing import Any

import numpy as np

from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MAX_RESPONSE_BYTES,
    MODEL_ID,
    PROTOCOL_ID,
    ProtocolError,
    SidecarResult,
    decode_response,
    encode_request,
    expected_bindings,
)


_SIDECAR_ORIGIN = "http://127.0.0.1:7862"
_SIDECAR_HOST = "127.0.0.1"
_SIDECAR_PORT = 7862


class SidecarUnavailable(RuntimeError):
    """A sidecar failure safe to surface to an untrusted UI."""

    def __init__(self, public_code: str) -> None:
        self.public_code = public_code
        super().__init__(public_code)


@dataclass(frozen=True)
class SidecarHealth:
    protocol: str
    model_id: str
    checkpoint_sha256: str
    device: str
    ready: bool


class NnUNetClient:
    def __init__(
        self,
        base_url: str = _SIDECAR_ORIGIN,
        timeout_seconds: float = 90.0,
    ) -> None:
        if base_url != _SIDECAR_ORIGIN:
            raise ValueError("NnUNetClient requires the exact loopback sidecar origin")
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise ValueError("sidecar timeout must be positive")
        if not 0.0 < float(timeout_seconds) < float("inf"):
            raise ValueError("sidecar timeout must be positive")
        self._timeout_seconds = float(timeout_seconds)

    def health(self) -> SidecarHealth:
        payload = self._request("GET", "/health", None)
        if set(payload) != {
            "protocol",
            "model_id",
            "checkpoint_sha256",
            "device",
            "ready",
        }:
            raise SidecarUnavailable("binding_mismatch")
        if (
            payload["protocol"] != PROTOCOL_ID
            or payload["model_id"] != MODEL_ID
            or payload["checkpoint_sha256"] != CHECKPOINT_SHA256
            or payload["device"] != "cuda:0"
            or payload["ready"] is not True
        ):
            raise SidecarUnavailable("binding_mismatch")
        return SidecarHealth(
            protocol=PROTOCOL_ID,
            model_id=MODEL_ID,
            checkpoint_sha256=CHECKPOINT_SHA256,
            device="cuda:0",
            ready=True,
        )

    def predict(self, request_id: str, image: np.ndarray) -> SidecarResult:
        try:
            request = encode_request(request_id, image)
        except ProtocolError:
            raise SidecarUnavailable("binding_mismatch") from None
        payload = self._request("POST", "/v1/predict", request)
        try:
            return decode_response(payload, expected_bindings(request_id, image))
        except ProtocolError:
            raise SidecarUnavailable("binding_mismatch") from None

    def _request(
        self, method: str, path: str, payload: dict[str, str] | None
    ) -> dict[str, Any]:
        body = (
            None
            if payload is None
            else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )
        connection: http.client.HTTPConnection | None = None
        try:
            connection = http.client.HTTPConnection(
                _SIDECAR_HOST,
                _SIDECAR_PORT,
                timeout=self._timeout_seconds,
            )
            headers = {"Accept": "application/json"}
            if body is not None:
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                raise SidecarUnavailable("http_status")
            content_type = response.getheader("Content-Type", "")
            if (
                not isinstance(content_type, str)
                or content_type.split(";", 1)[0].strip().lower() != "application/json"
            ):
                raise SidecarUnavailable("malformed_response")
            content_length = response.getheader("Content-Length")
            if content_length is not None and (
                not content_length.isdecimal() or int(content_length) > MAX_RESPONSE_BYTES
            ):
                raise SidecarUnavailable("malformed_response")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise SidecarUnavailable("malformed_response")
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise SidecarUnavailable("malformed_response")
            return decoded
        except SidecarUnavailable:
            raise
        except (TimeoutError, socket.timeout):
            raise SidecarUnavailable("timeout") from None
        except (http.client.HTTPException, OSError):
            raise SidecarUnavailable("unavailable") from None
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise SidecarUnavailable("malformed_response") from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except (http.client.HTTPException, OSError):
                    pass
