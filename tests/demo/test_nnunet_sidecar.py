from __future__ import annotations

from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import threading
from typing import Iterator

import numpy as np
import pytest

from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MAX_REQUEST_BYTES,
    MODEL_ID,
    PROTOCOL_ID,
    decode_response,
    encode_request,
    expected_bindings,
)
from sidecar.nnunet.predictor import ArtifactDriftError, Loop192Predictor
from sidecar.nnunet.server import make_server


def rgb(height: int, width: int) -> np.ndarray:
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


class FakePredictor:
    model_id = MODEL_ID
    checkpoint_sha256 = CHECKPOINT_SHA256
    device = "cuda:0"
    ready = True

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.calls = 0

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        self.calls += 1
        assert image.dtype == np.uint8
        return self.mask.copy(), 4.25


class BrokenPredictor(FakePredictor):
    def __init__(self, private_path: str) -> None:
        super().__init__(np.zeros((8, 8), dtype=np.uint8))
        self.private_path = private_path

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        del image
        raise RuntimeError(self.private_path)


class BlockingPredictor(FakePredictor):
    def __init__(self, mask: np.ndarray) -> None:
        super().__init__(mask)
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        self.entered.set()
        assert self.release.wait(timeout=5)
        return super().predict(image)


@contextmanager
def running_sidecar(backend: FakePredictor) -> Iterator[tuple[str, int]]:
    server = make_server(backend, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request_json(
    address: tuple[str, int],
    method: str,
    path: str,
    payload: object | None = None,
) -> tuple[int, dict[str, object]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection = http.client.HTTPConnection(*address, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        assert response.getheader("Content-Type") == "application/json"
        return response.status, json.loads(raw)
    finally:
        connection.close()


def test_predict_endpoint_runs_backend_once_and_returns_bound_mask() -> None:
    image = rgb(13, 17)
    backend = FakePredictor(mask=np.ones((13, 17), dtype=np.uint8))

    with running_sidecar(backend) as address:
        status, response = request_json(
            address, "POST", "/v1/predict", encode_request("a" * 32, image)
        )

    assert status == 200
    assert backend.calls == 1
    decoded = decode_response(response, expected_bindings("a" * 32, image))
    assert decoded.execution == "live"
    assert decoded.mask.shape == (13, 17)


def test_sidecar_never_returns_path_or_trace_on_backend_error() -> None:
    with running_sidecar(BrokenPredictor("/private/checkpoint_final.pth")) as address:
        status, response = request_json(
            address,
            "POST",
            "/v1/predict",
            encode_request("b" * 32, rgb(8, 8)),
        )

    serialized = json.dumps(response).lower()
    assert status == 503
    assert response == {"error": {"code": "inference_failed"}}
    assert "private" not in serialized
    assert "traceback" not in serialized


def test_health_is_path_free_and_bound_to_loopback_identity() -> None:
    with running_sidecar(FakePredictor(np.zeros((3, 4), dtype=np.uint8))) as address:
        status, response = request_json(address, "GET", "/health")

    assert address[0] == "127.0.0.1"
    assert status == 200
    assert response == {
        "protocol": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "device": "cuda:0",
        "ready": True,
    }
    serialized = json.dumps(response).lower()
    assert "/models" not in serialized
    assert "checkpoint_final.pth" not in serialized


def test_container_listener_requires_an_explicit_allowlisted_host() -> None:
    backend = FakePredictor(np.zeros((3, 4), dtype=np.uint8))

    container_server = make_server(backend, port=0, host="0.0.0.0")
    try:
        assert container_server.server_address[0] == "0.0.0.0"
    finally:
        container_server.server_close()

    with pytest.raises(ValueError, match="invalid_bind_host"):
        make_server(backend, port=0, host="192.168.1.20")


def test_invalid_json_and_oversized_content_length_fail_before_inference() -> None:
    backend = FakePredictor(np.zeros((2, 2), dtype=np.uint8))
    with running_sidecar(backend) as address:
        connection = http.client.HTTPConnection(*address, timeout=5)
        connection.request(
            "POST",
            "/v1/predict",
            body=b"{",
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        invalid = json.loads(response.read())
        assert response.status == 400
        connection.close()

        connection = http.client.HTTPConnection(*address, timeout=5)
        connection.putrequest("POST", "/v1/predict")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(MAX_REQUEST_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        oversized = json.loads(response.read())
        assert response.status == 413
        connection.close()

    assert invalid == {"error": {"code": "invalid_request"}}
    assert oversized == {"error": {"code": "invalid_request"}}
    assert backend.calls == 0


def test_prediction_lock_rejects_concurrent_request_as_busy() -> None:
    image = rgb(4, 5)
    backend = BlockingPredictor(np.zeros((4, 5), dtype=np.uint8))
    first_result: list[tuple[int, dict[str, object]]] = []

    with running_sidecar(backend) as address:
        first = threading.Thread(
            target=lambda: first_result.append(
                request_json(
                    address,
                    "POST",
                    "/v1/predict",
                    encode_request("c" * 32, image),
                )
            )
        )
        first.start()
        assert backend.entered.wait(timeout=5)
        status, response = request_json(
            address,
            "POST",
            "/v1/predict",
            encode_request("d" * 32, image),
        )
        backend.release.set()
        first.join(timeout=5)

    assert status == 503
    assert response == {"error": {"code": "busy"}}
    assert first_result[0][0] == 200
    assert backend.calls == 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle_and_manifest(root: Path) -> tuple[Path, Path]:
    bundle = root / "bundle"
    bundle.mkdir()
    contents = {
        "checkpoint_final.pth": b"checkpoint",
        "plans.json": b'{"plans_name":"nnUNetPlans"}',
        "dataset.json": b'{"channel_names":{"0":"red","1":"green","2":"blue"}}',
        "dataset_fingerprint.json": b'{"spacings":[[999.0,1.0,1.0]]}',
    }
    for name, content in contents.items():
        (bundle / name).write_bytes(content)
    manifest = {
        "schema_version": "imp.nnunet.model-manifest.v1",
        "model_id": MODEL_ID,
        "runtime": {
            "distribution": "nnunetv2",
            "version": "2.8.1",
            "recovered_git_commit": "3e9fdc5fec7c8164f8fc2c6263af8be73278130e",
            "environment_status": "reconstructed",
        },
        "input": {
            "layout": "CZYX",
            "channels": 3,
            "spacing": [999.0, 1.0, 1.0],
        },
        "artifacts": {
            name: {"sha256": _sha256(bundle / name), "size": len(content)}
            for name, content in contents.items()
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return bundle, manifest_path


class FakeCuda:
    def __init__(self) -> None:
        self.synchronizations = 0

    @staticmethod
    def is_available() -> bool:
        return True

    def synchronize(self) -> None:
        self.synchronizations += 1

    @staticmethod
    def empty_cache() -> None:
        return None


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()

    @staticmethod
    def device(kind: str, index: int) -> str:
        return f"{kind}:{index}"


class FakeNnUNetPredictor:
    instances: list["FakeNnUNetPredictor"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.initializations: list[tuple[str, tuple[str, ...], str]] = []
        self.inputs: list[tuple[np.ndarray, dict[str, object]]] = []
        self.__class__.instances.append(self)

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: tuple[str, ...] | None,
        checkpoint_name: str = "checkpoint_final.pth",
    ) -> None:
        assert use_folds is not None
        self.initializations.append(
            (model_training_output_dir, use_folds, checkpoint_name)
        )

    def predict_single_npy_array(
        self,
        input_image: np.ndarray,
        image_properties: dict[str, object],
        segmentation_previous_stage: np.ndarray | None = None,
        output_file_truncated: str | None = None,
        save_or_return_probabilities: bool = False,
    ) -> np.ndarray:
        assert segmentation_previous_stage is None
        assert output_file_truncated is None
        assert not save_or_return_probabilities
        self.inputs.append((input_image.copy(), dict(image_properties)))
        return np.ones((1, 2, 3), dtype=np.uint8)


def test_predictor_verifies_artifacts_before_loading_runtime(tmp_path: Path) -> None:
    bundle, manifest = _write_bundle_and_manifest(tmp_path)
    (bundle / "plans.json").write_text("drift", encoding="utf-8")
    runtime_loaded = False

    def load_runtime() -> tuple[FakeTorch, type[FakeNnUNetPredictor], str]:
        nonlocal runtime_loaded
        runtime_loaded = True
        return FakeTorch(), FakeNnUNetPredictor, "2.8.1"

    with pytest.raises(ArtifactDriftError, match="artifact_drift"):
        Loop192Predictor(bundle, manifest, runtime_loader=load_runtime)

    assert not runtime_loaded


def test_predictor_initializes_once_and_uses_natural_image_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest = _write_bundle_and_manifest(tmp_path)
    fake_torch = FakeTorch()
    FakeNnUNetPredictor.instances.clear()
    monkeypatch.delenv("nnUNet_results", raising=False)

    def load_runtime() -> tuple[FakeTorch, type[FakeNnUNetPredictor], str]:
        assert os.environ["nnUNet_results"] == str(bundle.resolve())
        return fake_torch, FakeNnUNetPredictor, "2.8.1"

    predictor = Loop192Predictor(
        bundle,
        manifest,
        runtime_loader=load_runtime,
    )
    image = rgb(5, 7)

    mask, latency = predictor.predict(image)
    predictor.predict(image)

    assert len(FakeNnUNetPredictor.instances) == 1
    runtime = FakeNnUNetPredictor.instances[0]
    assert len(runtime.initializations) == 1
    assert runtime.initializations[0][1:] == (("all",), "checkpoint_final.pth")
    assert len(runtime.inputs) == 2
    nnunet_input, properties = runtime.inputs[0]
    assert nnunet_input.shape == (3, 1, 5, 7)
    assert nnunet_input.dtype == np.float32
    np.testing.assert_array_equal(nnunet_input[:, 0], image.transpose(2, 0, 1))
    assert properties == {"spacing": (999.0, 1.0, 1.0)}
    assert mask.shape == (5, 7)
    assert mask.dtype == np.uint8
    assert np.all(mask == 1)
    assert latency >= 0.0
    assert fake_torch.cuda.synchronizations == 4


def test_public_manifest_pins_recovered_bundle_without_private_paths() -> None:
    sidecar_root = Path(__file__).parents[2] / "sidecar" / "nnunet"
    manifest = json.loads(
        (sidecar_root / "model_manifest.example.json").read_text(encoding="utf-8")
    )

    assert manifest["runtime"] == {
        "distribution": "nnunetv2",
        "version": "2.8.1",
        "recovered_git_commit": "3e9fdc5fec7c8164f8fc2c6263af8be73278130e",
        "environment_status": "reconstructed",
    }
    assert manifest["artifacts"] == {
        "checkpoint_final.pth": {
            "sha256": CHECKPOINT_SHA256,
            "size": 267947879,
        },
        "plans.json": {
            "sha256": "b60e4defd229b03f7064dc5b66123545c91cdaa44c09d990b86690a94e1e08a7",
            "size": 6379,
        },
        "dataset.json": {
            "sha256": "eb33bcbad9d8d5c96168b3c12171392ffabf63ba4cbff4f2bf4badc98bf6487a",
            "size": 183,
        },
        "dataset_fingerprint.json": {
            "sha256": "931da8aae52ffecd726d5928009ebdcae7002e24b035fad89177e0bc81dba85c",
            "size": 274020,
        },
    }
    assert "private" not in json.dumps(manifest).lower()
    assert ":\\" not in json.dumps(manifest)


def test_container_uses_digest_pinned_cuda_base() -> None:
    sidecar_root = Path(__file__).parents[2] / "sidecar" / "nnunet"
    dockerfile = (sidecar_root / "Dockerfile").read_text(encoding="utf-8")
    plan = (
        Path(__file__).parents[2]
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-21-dual-live-demo.md"
    ).read_text(encoding="utf-8")

    assert (
        "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@"
        "sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be"
    ) in dockerfile
    assert "--break-system-packages" in dockerfile
    assert "--no-deps" not in dockerfile
    assert "python -m pip check" in dockerfile
    assert "useradd --uid 65532" in dockerfile
    assert "USER sidecar:sidecar" in dockerfile
    assert "NNUNET_BIND_HOST=0.0.0.0" in dockerfile
    assert "127.0.0.1:7862:7862" in plan


def test_reconstructed_lock_is_full_and_exactly_pinned() -> None:
    lock_path = Path(__file__).parents[2] / "sidecar" / "nnunet" / "requirements.lock"
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    pins = [line for line in lines if line and not line.startswith("#")]

    assert any("reconstructed" in line.lower() for line in lines if line.startswith("#"))
    assert any("not the original environment" in line.lower() for line in lines if line.startswith("#"))
    assert "nnunetv2==2.8.1" in pins
    assert len(pins) >= 50
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s;]+", line) for line in pins)
    assert not any(
        token in line.lower()
        for line in pins
        for token in ("git+", "http://", "https://", "file:", " @ ", "-e ")
    )
