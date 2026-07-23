from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import builtins
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lesion_robustness.demo.model_service import (
    CandidateUnavailableError,
    ControlOnlyResult,
    Loop206ComparisonService,
    ModelEndpoint,
    load_model_registry,
    load_receipt_authorized_prior,
)


class FakeModel:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.checkpoint_sha256 = hashlib.sha256(model_id.encode("ascii")).hexdigest()
        self.device = "cpu"
        self.last_input: np.ndarray | None = None
        self.call_count = 0

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        self.last_input = batch.copy()
        self.call_count += 1
        return (batch[:, 3:4] * 8.0) - 4.0

    def synchronize(self) -> None:
        return None


class FakePrior:
    def predict(self, image: np.ndarray) -> np.ndarray:
        assert image.shape == (384, 384, 3)
        return np.full((384, 384), 255, dtype=np.uint8)


def _replace_after_first_binary_read(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    replacement: bytes,
) -> None:
    original_open = Path.open
    replaced = False

    class ReplacingHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            nonlocal replaced
            result = self._handle.__exit__(exc_type, exc, traceback)
            if not replaced:
                replaced = True
                with original_open(target, "wb") as output:
                    output.write(replacement)
            return result

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def guarded_open(path: Path, *args, **kwargs):
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        handle = original_open(path, *args, **kwargs)
        if Path(path).resolve() == target.resolve() and mode == "rb" and not replaced:
            return ReplacingHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", guarded_open)


def _snapshot_bytes(source) -> bytes:
    if hasattr(source, "open"):
        with source.open() as handle:
            return handle.read()
    if hasattr(source, "read"):
        return source.read()
    return Path(source).read_bytes()


@pytest.fixture
def endpoints() -> tuple[FakeModel, FakeModel]:
    return FakeModel("control-s206"), FakeModel("candidate-s206")


def _authorized_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import lesion_robustness.demo.model_service as module

    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"status":"passed"}', encoding="ascii")
    monkeypatch.setattr(
        module,
        "load_deployment_prior_with_receipt_hash",
        lambda *_args, **_kwargs: (
            FakePrior(),
            hashlib.sha256(receipt.read_bytes()).hexdigest(),
        ),
    )
    return load_receipt_authorized_prior(
        tmp_path / "prior.joblib",
        receipt,
        expected_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
    )


def test_arbitrary_comparison_fails_closed_before_inference_without_receipt(
    endpoints: tuple[FakeModel, FakeModel],
) -> None:
    control, candidate = endpoints
    service = Loop206ComparisonService(control, candidate)

    with pytest.raises(CandidateUnavailableError, match="receipt-authorized"):
        service.compare(np.full((240, 320, 3), 128, dtype=np.uint8))

    assert control.call_count == 0
    assert candidate.call_count == 0


def test_constructor_rejects_duck_typed_prior(
    endpoints: tuple[FakeModel, FakeModel],
) -> None:
    control, candidate = endpoints
    forged = SimpleNamespace(prior=FakePrior(), receipt_sha256="a" * 64)

    with pytest.raises(TypeError, match="ReceiptAuthorizedPrior"):
        Loop206ComparisonService(control, candidate, forged)

    assert control.call_count == 0
    assert candidate.call_count == 0


def test_service_builds_zero_and_receipt_authorized_candidate_channels(
    endpoints: tuple[FakeModel, FakeModel],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, candidate = endpoints
    prior = _authorized_prior(tmp_path, monkeypatch)
    service = Loop206ComparisonService(control, candidate, prior)

    result = service.compare(np.full((240, 320, 3), 128, dtype=np.uint8))

    assert control.last_input is not None
    assert candidate.last_input is not None
    np.testing.assert_array_equal(control.last_input[0, 3], 0.0)
    assert candidate.last_input[0, 3].max() == 1.0
    assert result.control_mask.shape == (240, 320)
    assert result.candidate_mask.shape == (240, 320)
    assert result.prior_receipt_sha256 == hashlib.sha256(
        b'{"status":"passed"}'
    ).hexdigest()


def test_public_fixed_comparison_accepts_only_allowlisted_identifier_and_corruption(
    endpoints: tuple[FakeModel, FakeModel],
) -> None:
    control, candidate = endpoints
    service = Loop206ComparisonService(control, candidate)
    parameters = inspect.signature(service.compare_fixed).parameters

    assert list(parameters) == ["identifier", "corruption"]
    with pytest.raises(CandidateUnavailableError, match="production fixed provider"):
        service.compare_fixed("allowlisted-sample", corruption="clean")
    with pytest.raises(TypeError):
        service.compare_fixed(
            np.full((240, 320, 3), 90, dtype=np.uint8),
            preprocessed_rgb=np.full((384, 384, 3), 100, dtype=np.uint8),
            fixed=object(),
        )
    assert control.call_count == 0
    assert candidate.call_count == 0


def test_constructor_rejects_fixture_fixed_provider(
    endpoints: tuple[FakeModel, FakeModel],
) -> None:
    control, candidate = endpoints

    with pytest.raises(TypeError, match="production-authorized fixed provider"):
        Loop206ComparisonService(
            control,
            candidate,
            fixed_provider=SimpleNamespace(authorize=lambda *_args: object()),
        )


def test_fixed_result_metadata_uses_an_explicit_allowlist() -> None:
    import lesion_robustness.demo.model_service as module

    authorized = SimpleNamespace(
        group_key="group-fixed",
        sample_id="sample-fixed",
        corruption="clean",
        candidate_manifest_sha256="1" * 64,
        candidate_data_sha256="2" * 64,
        zero_manifest_sha256="3" * 64,
        zero_data_sha256="4" * 64,
        mask_sha256_raw="5" * 64,
        mask_sha256_binary="6" * 64,
        mask_sha256_runtime="7" * 64,
        historical_cache_provenance_drift=True,
        metadata={"local_path": "C:" + "/private", "comparison_source": "forged"},
    )

    metadata = module._fixed_public_metadata(authorized)

    assert metadata == {
        "comparison_source": "exact_fixed_cache",
        "group_key": "group-fixed",
        "sample_id": "sample-fixed",
        "corruption": "clean",
        "candidate_manifest_sha256": "1" * 64,
        "candidate_data_sha256": "2" * 64,
        "zero_manifest_sha256": "3" * 64,
        "zero_data_sha256": "4" * 64,
        "mask_sha256_raw": "5" * 64,
        "mask_sha256_binary": "6" * 64,
        "mask_sha256_runtime": "7" * 64,
        "historical_cache_provenance_drift": True,
    }


def test_control_only_result_cannot_be_mistaken_for_comparison(
    endpoints: tuple[FakeModel, FakeModel],
) -> None:
    control, candidate = endpoints
    service = Loop206ComparisonService(control, candidate)

    result = service.preview_control(np.full((120, 160, 3), 75, dtype=np.uint8))

    assert isinstance(result, ControlOnlyResult)
    assert result.mode == "control_only"
    assert result.control_mask.shape == (120, 160)
    assert candidate.call_count == 0
    assert not any("candidate" in field.name or "comparison" in field.name for field in fields(result))


def test_service_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    control = tmp_path / "control.pt"
    candidate = tmp_path / "candidate.pt"
    control.write_bytes(b"control")
    candidate.write_bytes(b"candidate")
    registry = tmp_path / "registry.json"
    payload = json.loads(
        (Path(__file__).resolve().parents[2] / "demo/model_registry.example.json").read_text(
            encoding="ascii"
        )
    )
    registry.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ValueError, match="control checkpoint hash"):
        load_model_registry(
            registry,
            environ={
                "IMP_LOOP206_CONTROL_CHECKPOINT": str(control),
                "IMP_LOOP206_CANDIDATE_CHECKPOINT": str(candidate),
            },
        )


def test_checkpoint_hash_and_torch_load_use_the_same_captured_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lesion_robustness.demo.model_service as module

    control_bytes = b"verified-control-checkpoint"
    candidate_bytes = b"verified-candidate-checkpoint"
    control = tmp_path / "control.pt"
    candidate = tmp_path / "candidate.pt"
    control.write_bytes(control_bytes)
    candidate.write_bytes(candidate_bytes)
    payload = deepcopy(module.PINNED_REGISTRY)
    payload["control"]["checkpoint_sha256"] = hashlib.sha256(control_bytes).hexdigest()
    payload["candidate"]["checkpoint_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(payload), encoding="ascii")
    monkeypatch.setattr(module, "PINNED_REGISTRY", payload)
    _replace_after_first_binary_read(monkeypatch, control, b"replacement-checkpoint")
    loaded_bytes: dict[str, bytes] = {}

    def fake_load(source, *, model_id, checkpoint_sha256, device, seed):
        loaded_bytes[model_id] = _snapshot_bytes(source)
        endpoint = FakeModel(model_id)
        endpoint.checkpoint_sha256 = checkpoint_sha256
        role = "control" if model_id == payload["control"]["model_id"] else "candidate"
        preprocessing = {
            "extra_channel": {
                "enabled": True,
                "type": "loop206_contour_cache",
                "require_input_sha256": True,
                "cache_manifest": f"pilot_cache_v2_{'zero_control' if role == 'control' else 'candidate'}/manifest.json",
            }
        }
        return endpoint, preprocessing, {"low_contrast": {"factor": 0.5}}

    monkeypatch.setattr(module, "_load_torch_endpoint", fake_load)
    original_import = builtins.__import__

    def reject_eager_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch is intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_eager_torch)

    loaded = module.load_model_registry(
        registry,
        environ={
            "IMP_LOOP206_CONTROL_CHECKPOINT": str(control),
            "IMP_LOOP206_CANDIDATE_CHECKPOINT": str(candidate),
        },
        device="cpu",
    )

    assert loaded_bytes[payload["control"]["model_id"]] == control_bytes
    assert loaded_bytes[payload["candidate"]["model_id"]] == candidate_bytes
    assert loaded.control.checkpoint_sha256 == hashlib.sha256(control_bytes).hexdigest()
    assert loaded.prior is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version="loop206.demo.models.v2"),
        lambda value: value["control"].update(model_id="forged-control"),
        lambda value: value["candidate"].update(checkpoint_sha256="0" * 64),
        lambda value: value["control"].update(checkpoint_env="FORGED_CONTROL"),
        lambda value: value.update(prior_env="FORGED_PRIOR"),
        lambda value: value.update(prior_receipt_sha256="a" * 64),
    ],
)
def test_registry_rejects_mutable_authority_before_environment_resolution(
    tmp_path: Path, mutation
) -> None:
    payload = json.loads(
        (Path(__file__).resolve().parents[2] / "demo/model_registry.example.json").read_text(
            encoding="ascii"
        )
    )
    mutation(payload)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ValueError, match="pinned model registry"):
        load_model_registry(registry, environ={})


def test_model_endpoint_contract_is_runtime_checkable(endpoints: tuple[FakeModel, FakeModel]) -> None:
    assert isinstance(endpoints[0], ModelEndpoint)


def test_release_without_approved_receipt_pin_rejects_configured_prior_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lesion_robustness.demo.model_service as module

    control = tmp_path / "control.pt"
    candidate = tmp_path / "candidate.pt"
    control.write_bytes(b"control")
    candidate.write_bytes(b"candidate")
    payload = deepcopy(module.PINNED_REGISTRY)
    payload["control"]["checkpoint_sha256"] = hashlib.sha256(b"control").hexdigest()
    payload["candidate"]["checkpoint_sha256"] = hashlib.sha256(b"candidate").hexdigest()
    payload["prior_receipt_sha256"] = None
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(payload), encoding="ascii")
    monkeypatch.setattr(module, "PINNED_REGISTRY", payload)

    def fake_endpoint(_snapshot, *, model_id, checkpoint_sha256, device, seed):
        endpoint = FakeModel(model_id)
        endpoint.checkpoint_sha256 = checkpoint_sha256
        role = "control" if model_id == payload["control"]["model_id"] else "candidate"
        preprocessing = {
            "extra_channel": {
                "enabled": True,
                "type": "loop206_contour_cache",
                "require_input_sha256": True,
                "cache_manifest": f"pilot_cache_v2_{'zero_control' if role == 'control' else 'candidate'}/manifest.json",
            }
        }
        return endpoint, preprocessing, {"low_contrast": {"factor": 0.5}}

    monkeypatch.setattr(module, "_load_torch_endpoint", fake_endpoint)
    effects: list[str] = []
    monkeypatch.setattr(
        module,
        "load_receipt_authorized_prior",
        lambda *_args, **_kwargs: effects.append("loaded"),
    )

    with pytest.raises(ValueError, match="disabled by the pinned registry"):
        module.load_model_registry(
            registry,
            environ={
                payload["control"]["checkpoint_env"]: str(control),
                payload["candidate"]["checkpoint_env"]: str(candidate),
                payload["prior_env"]: str(tmp_path / "forged.joblib"),
                payload["prior_receipt_env"]: str(tmp_path / "forged.json"),
            },
            device="cpu",
        )

    assert effects == []


def test_registry_passes_exact_approved_receipt_pin_to_prior_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lesion_robustness.demo.model_service as module

    control = tmp_path / "control.pt"
    candidate = tmp_path / "candidate.pt"
    control.write_bytes(b"control")
    candidate.write_bytes(b"candidate")
    payload = deepcopy(module.PINNED_REGISTRY)
    payload["control"]["checkpoint_sha256"] = hashlib.sha256(b"control").hexdigest()
    payload["candidate"]["checkpoint_sha256"] = hashlib.sha256(b"candidate").hexdigest()
    payload["prior_receipt_sha256"] = "9" * 64
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(payload), encoding="ascii")
    monkeypatch.setattr(module, "PINNED_REGISTRY", payload)

    def fake_endpoint(_snapshot, *, model_id, checkpoint_sha256, device, seed):
        endpoint = FakeModel(model_id)
        endpoint.checkpoint_sha256 = checkpoint_sha256
        role = "control" if model_id == payload["control"]["model_id"] else "candidate"
        preprocessing = {
            "extra_channel": {
                "enabled": True,
                "type": "loop206_contour_cache",
                "require_input_sha256": True,
                "cache_manifest": f"pilot_cache_v2_{'zero_control' if role == 'control' else 'candidate'}/manifest.json",
            }
        }
        return endpoint, preprocessing, {"low_contrast": {"factor": 0.5}}

    monkeypatch.setattr(module, "_load_torch_endpoint", fake_endpoint)
    observed: list[str] = []
    authorized = SimpleNamespace(receipt_sha256="9" * 64)

    def fake_prior(_artifact, _receipt, *, expected_receipt_sha256):
        observed.append(expected_receipt_sha256)
        return authorized

    monkeypatch.setattr(module, "load_receipt_authorized_prior", fake_prior)

    loaded = module.load_model_registry(
        registry,
        environ={
            payload["control"]["checkpoint_env"]: str(control),
            payload["candidate"]["checkpoint_env"]: str(candidate),
            payload["prior_env"]: str(tmp_path / "prior.joblib"),
            payload["prior_receipt_env"]: str(tmp_path / "receipt.json"),
        },
        device="cpu",
    )

    assert loaded.prior is authorized
    assert observed == ["9" * 64]


def test_registry_accepts_only_the_expected_arm_specific_cache_manifest() -> None:
    import lesion_robustness.demo.model_service as module

    common = {
        "contrast_stretch": {"enabled": True},
        "extra_channel": {
            "enabled": True,
            "type": "loop206_contour_cache",
            "require_input_sha256": True,
        },
    }
    control = json.loads(json.dumps(common))
    candidate = json.loads(json.dumps(common))
    control["extra_channel"]["cache_manifest"] = (
        ".artifacts/preprocessing_search/loop206_leac_drlse/"
        "pilot_cache_v2_zero_control/manifest.json"
    )
    candidate["extra_channel"]["cache_manifest"] = (
        ".artifacts/preprocessing_search/loop206_leac_drlse/"
        "pilot_cache_v2_candidate/manifest.json"
    )

    selected = module._validate_preprocessing_pair(control, candidate)

    assert selected == control
    candidate["contrast_stretch"]["enabled"] = False
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        module._validate_preprocessing_pair(control, candidate)
