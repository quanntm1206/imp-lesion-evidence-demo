from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lesion_robustness.demo.fixed_cache import FixedCacheRecord, sha256_rgb_array
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


@pytest.fixture
def endpoints() -> tuple[FakeModel, FakeModel]:
    return FakeModel("control-s206"), FakeModel("candidate-s206")


def _authorized_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import lesion_robustness.demo.model_service as module

    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"status":"passed"}', encoding="ascii")
    monkeypatch.setattr(module, "load_deployment_prior", lambda *_args: FakePrior())
    return load_receipt_authorized_prior(tmp_path / "prior.joblib", receipt)


def test_arbitrary_comparison_fails_closed_before_inference_without_receipt(
    endpoints: tuple[FakeModel, FakeModel],
) -> None:
    control, candidate = endpoints
    service = Loop206ComparisonService(control, candidate)

    with pytest.raises(CandidateUnavailableError, match="receipt-authorized"):
        service.compare(np.full((240, 320, 3), 128, dtype=np.uint8))

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


def test_fixed_comparison_returns_both_arms_at_original_geometry(
    endpoints: tuple[FakeModel, FakeModel],
) -> None:
    control, candidate = endpoints
    service = Loop206ComparisonService(control, candidate)
    original = np.full((240, 320, 3), 90, dtype=np.uint8)
    preprocessed = np.full((384, 384, 3), 100, dtype=np.uint8)
    fixed = FixedCacheRecord(
        group_key="fixed-group",
        corruption="clean",
        input_rgb_sha256=sha256_rgb_array(preprocessed),
        control_channel=np.zeros((384, 384), dtype=np.uint8),
        candidate_channel=np.full((384, 384), 255, dtype=np.uint8),
        metadata={"historical_cache_provenance_drift": True},
    )

    result = service.compare_fixed(original, preprocessed_rgb=preprocessed, fixed=fixed)

    assert control.call_count == 1
    assert candidate.call_count == 1
    assert result.control_probability.shape == (240, 320)
    assert result.candidate_probability.shape == (240, 320)
    assert result.control_mask.shape == (240, 320)
    assert result.candidate_mask.shape == (240, 320)
    assert result.prior_receipt_sha256 is None
    assert result.metadata["comparison_source"] == "exact_fixed_cache"


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
    registry.write_text(
        json.dumps(
            {
                "schema_version": "loop206.demo.models.v1",
                "control": {
                    "model_id": "L206-control-s206",
                    "checkpoint_env": "CONTROL_PT",
                    "checkpoint_sha256": "0" * 64,
                },
                "candidate": {
                    "model_id": "L206-contour-channel-s206",
                    "checkpoint_env": "CANDIDATE_PT",
                    "checkpoint_sha256": hashlib.sha256(b"candidate").hexdigest(),
                },
                "prior_env": "PRIOR",
                "prior_receipt_env": "PRIOR_RECEIPT",
            }
        ),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="control checkpoint hash"):
        load_model_registry(
            registry,
            environ={"CONTROL_PT": str(control), "CANDIDATE_PT": str(candidate)},
        )


def test_model_endpoint_contract_is_runtime_checkable(endpoints: tuple[FakeModel, FakeModel]) -> None:
    assert isinstance(endpoints[0], ModelEndpoint)


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
