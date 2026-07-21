"""Strict loopback HTTP boundary for the persistent Loop192 predictor."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Protocol, cast

import numpy as np

from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MAX_REQUEST_BYTES,
    MODEL_ID,
    PROTOCOL_ID,
    ProtocolError,
    decode_request,
    encode_response,
)
from sidecar.nnunet.predictor import (
    ArtifactDriftError,
    InferenceOOMError,
    Loop192Predictor,
)


class PredictorBackend(Protocol):
    model_id: str
    checkpoint_sha256: str
    device: str
    ready: bool

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, float]: ...


def _is_container_namespace() -> bool:
    if os.name != "posix":
        return False
    marker = Path("/.dockerenv")
    try:
        if not stat.S_ISREG(os.lstat(marker).st_mode):
            return False
        init_namespace = os.stat("/proc/1/ns/mnt")
        process_namespace = os.stat("/proc/self/ns/mnt")
    except OSError:
        return False
    return (init_namespace.st_dev, init_namespace.st_ino) == (
        process_namespace.st_dev,
        process_namespace.st_ino,
    )


class SidecarHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], backend: PredictorBackend) -> None:
        super().__init__(address, SidecarHandler)
        self.backend = backend
        self.prediction_lock = threading.Lock()


class SidecarHandler(BaseHTTPRequestHandler):
    server: SidecarHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _send_json(self, status: int, payload: object) -> None:
        try:
            body = json.dumps(
                payload, allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        except (TypeError, ValueError):
            status = 503
            body = b'{"error":{"code":"inference_failed"}}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str) -> None:
        self._send_json(status, {"error": {"code": code}})

    def do_GET(self) -> None:
        if self.path != "/health":
            self._error(404, "invalid_request")
            return
        backend = self.server.backend
        ready = bool(backend.ready)
        self._send_json(
            200 if ready else 503,
            {
                "protocol": PROTOCOL_ID,
                "model_id": MODEL_ID,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "device": "cuda:0",
                "ready": ready,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/v1/predict":
            self._error(404, "invalid_request")
            return
        if self.headers.get("Content-Type") != "application/json":
            self._error(400, "invalid_request")
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(400, "invalid_request")
            return
        if content_length < 1 or content_length > MAX_REQUEST_BYTES:
            self._error(413 if content_length > MAX_REQUEST_BYTES else 400, "invalid_request")
            return
        try:
            body = self.rfile.read(content_length)
            if len(body) != content_length:
                raise ValueError
            payload: Any = json.loads(body)
            request_id, image, input_sha256 = decode_request(payload)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError, ProtocolError):
            self._error(400, "invalid_request")
            return

        if not self.server.prediction_lock.acquire(blocking=False):
            self._error(503, "busy")
            return
        try:
            mask, latency_ms = self.server.backend.predict(image)
            response = encode_response(request_id, input_sha256, mask, latency_ms)
        except ArtifactDriftError:
            self._error(503, "artifact_drift")
            return
        except InferenceOOMError:
            self._error(503, "oom")
            return
        except BaseException:
            self._error(503, "inference_failed")
            return
        finally:
            self.server.prediction_lock.release()
        self._send_json(200, response)


def make_server(
    backend: PredictorBackend,
    *,
    port: int = 7862,
    host: str = "127.0.0.1",
) -> SidecarHTTPServer:
    if host != "127.0.0.1" and not (
        host == "0.0.0.0" and _is_container_namespace()
    ):
        raise ValueError("invalid_bind_host")
    if (
        backend.model_id != MODEL_ID
        or backend.checkpoint_sha256 != CHECKPOINT_SHA256
        or backend.device != "cuda:0"
    ):
        raise ValueError("artifact_drift")
    return SidecarHTTPServer((host, port), backend)


def main() -> int:
    model_root = Path(os.environ.get("NNUNET_MODEL_ROOT", "/models/loop192"))
    manifest = Path(
        os.environ.get(
            "NNUNET_MODEL_MANIFEST", "/app/sidecar/nnunet/model_manifest.example.json"
        )
    )
    bind_host = os.environ.get("NNUNET_BIND_HOST", "127.0.0.1")
    try:
        backend = Loop192Predictor(model_root, manifest)
        server = make_server(cast(PredictorBackend, backend), host=bind_host)
    except BaseException as exc:
        public_code = getattr(exc, "public_code", "runtime_unavailable")
        raise SystemExit(public_code) from None
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
