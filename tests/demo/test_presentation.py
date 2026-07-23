from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lesion_robustness.demo.immutable_io import ImmutableSnapshot
from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MODEL_ID,
    PROTOCOL_ID,
    SidecarResult,
    mask_sha256,
    rgb_sha256,
)
import lesion_robustness.demo.presentation as presentation
from lesion_robustness.demo.dual_live_service import DualLiveService
from lesion_robustness.demo.nnunet_client import SidecarUnavailable
from lesion_robustness.demo.live_inputs import (
    LiveInputEvidence,
    synthetic_evidence,
    upload_evidence,
)
from lesion_robustness.release_manifest import load_release_manifest, runtime_projection

from lesion_robustness.demo.presentation import (
    NO_GT_MESSAGE,
    build_control_receipt,
    build_dual_live_receipt,
    build_fixed_receipt,
    render_clean_evidence,
    render_dual_live_ledger,
    render_legacy_table,
    render_metrics,
)


ROOT = Path(__file__).resolve().parents[2]


def _manifest():
    return load_release_manifest()


def _registry() -> dict:
    return json.loads(
        (ROOT / "demo/data/evidence_registry.json").read_text(encoding="ascii")
    )


def _fixed_result() -> SimpleNamespace:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    return SimpleNamespace(
        original_rgb=image,
        control_mask=mask,
        candidate_mask=mask,
        control_overlay=image,
        candidate_overlay=image,
        control_latency_ms=12.5,
        candidate_latency_ms=14.5,
        total_latency_ms=27.0,
        device="cpu",
        control_model_id="control",
        candidate_model_id="candidate",
        control_checkpoint_sha256="a" * 64,
        candidate_checkpoint_sha256="b" * 64,
        prior_receipt_sha256=None,
        metadata={
            "comparison_source": "exact_fixed_cache",
            "group_key": "component:verified",
            "sample_id": "ISIC_verified",
            "corruption": "clean",
            "candidate_manifest_sha256": "c" * 64,
            "candidate_data_sha256": "d" * 64,
            "zero_manifest_sha256": "e" * 64,
            "zero_data_sha256": "f" * 64,
            "mask_sha256_raw": "1" * 64,
            "mask_sha256_binary": "2" * 64,
            "mask_sha256_runtime": ImmutableSnapshot.decoded_binary_mask_sha256(mask),
            "historical_cache_provenance_drift": True,
            "path": "E:" + "/private/data.jpg",
        },
    )


def complete_dual_result() -> SimpleNamespace:
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    imp_mask = np.zeros((16, 24), dtype=np.uint8)
    nnunet_mask = np.ones((16, 24), dtype=np.uint8)
    imp_identity = runtime_projection()["imp"]
    return SimpleNamespace(
        request_id="a" * 32,
        input_sha256=rgb_sha256(image),
        original_rgb=image,
        receipt_eligible=True,
        total_latency_ms=18.5,
        total_latency_scope="dual_service_run_end_to_end_wall",
        imp=SimpleNamespace(
            status="completed",
            mask=imp_mask,
            model_id=imp_identity["model_id"],
            checkpoint_sha256=imp_identity["checkpoint_sha256"],
            preprocessing="imp_runtime_control",
            reported_latency_ms=7.0,
            coordinator_latency_ms=8.0,
            reported_latency_scope="imp_model_forward_cuda_sync",
            coordinator_latency_scope="imp_preview_control_end_to_end_wall",
            device="cuda:0",
        ),
        nnunet=SimpleNamespace(
            status="completed",
            mask=nnunet_mask,
            model_id="L192-nnUNet-v2-raw-100ep",
            checkpoint_sha256=CHECKPOINT_SHA256,
            preprocessing="raw_rgb_256",
            reported_latency_ms=11.5,
            coordinator_latency_ms=17.5,
            reported_latency_scope="nnunet_predict_single_npy_array_cuda_sync",
            coordinator_latency_scope="nnunet_localhost_client_end_to_end_wall",
            device="cuda:0",
            protocol=PROTOCOL_ID,
        ),
    )


def incomplete_dual_result() -> SimpleNamespace:
    result = complete_dual_result()
    result.receipt_eligible = False
    return result


def _build_dual_receipt(result: object) -> dict[str, object]:
    return build_dual_live_receipt(
        result,
        _manifest(),
        upload_evidence(result.original_rgb),
    )


class ReceiptImp:
    def preview_control(self, image: np.ndarray) -> SimpleNamespace:
        imp_identity = runtime_projection()["imp"]
        return SimpleNamespace(
            control_mask=np.zeros(image.shape[:2], dtype=np.uint8),
            control_overlay=image.copy(),
            control_latency_ms=2.0,
            device="cuda:0",
            control_model_id=imp_identity["model_id"],
            control_checkpoint_sha256=imp_identity["checkpoint_sha256"],
        )


class CpuReceiptImp(ReceiptImp):
    def preview_control(self, image: np.ndarray) -> SimpleNamespace:
        result = super().preview_control(image)
        result.device = "cpu"
        return result


class ReceiptNnUNet:
    def predict(self, request_id: str, image: np.ndarray) -> SidecarResult:
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        return SidecarResult(
            request_id=request_id,
            input_sha256=rgb_sha256(image),
            mask=mask,
            mask_sha256=mask_sha256(mask),
            model_id=MODEL_ID,
            checkpoint_sha256=CHECKPOINT_SHA256,
            latency_ms=3.0,
            execution="live",
        )


class UnavailableReceiptNnUNet:
    def predict(self, _request_id: str, _image: np.ndarray) -> SidecarResult:
        raise SidecarUnavailable("timeout")


def test_clean_evidence_keeps_validation_and_train_screen_scopes_separate() -> None:
    html = render_clean_evidence(_registry())

    assert "protected_validation" in html
    assert "train_screen" in html
    assert "0.8959" in html and "0.9019" in html
    assert "-0.0313" in html
    normalized = html.lower()
    assert "adaptive development and checkpoint-selection validation" in normalized
    assert "selection-optimistic" in normalized
    assert "after averaging three selected seeds and three views" in normalized
    assert "76 groups as whole split-group clusters" in normalized
    assert "conditional on those selected seeds" in normalized
    assert "SOTA" not in html


def test_legacy_rows_always_render_warning() -> None:
    html = render_legacy_table(_registry())

    assert "legacy_patient_contaminated" in html
    assert "13" in html and "3" in html


def test_no_ground_truth_uses_exact_unavailable_message() -> None:
    assert render_metrics(None) == NO_GT_MESSAGE


def test_metric_table_formats_both_authorized_arms() -> None:
    text = render_metrics(
        {
            "control": {"dice": 1.0, "iou": 1.0, "boundary_f1": 1.0, "hd95": 0.0, "assd": 0.0},
            "candidate": {"dice": 0.75, "iou": 0.6, "boundary_f1": 0.5, "hd95": 2.0, "assd": 1.0},
        }
    )

    assert "Control" in text and "Candidate" in text
    assert "Dice" in text and "HD95" in text


def test_fixed_receipt_is_allowlisted_and_path_free() -> None:
    receipt = build_fixed_receipt(_fixed_result(), metrics=None, registry=_registry())
    encoded = json.dumps(receipt, sort_keys=True)

    assert receipt["evidence_badge"] == [
        "train_screen",
        "exact_fixed_cache",
        "historical_cache_provenance_drift",
    ]
    assert receipt["models"]["control"]["checkpoint_sha256"] == "a" * 64
    assert receipt["models"]["candidate"]["checkpoint_sha256"] == "b" * 64
    assert receipt["ground_truth_binding"] is None
    for forbidden in ("E:/", "private", "path", "url", "filename", "environment"):
        assert forbidden not in encoded.lower()


def test_fixed_metric_receipt_binds_verified_ground_truth_hashes() -> None:
    result = _fixed_result()
    metrics = {
        arm: {metric: 1.0 for metric in ("dice", "iou", "boundary_f1", "hd95", "assd")}
        for arm in ("control", "candidate")
    }

    receipt = build_fixed_receipt(result, metrics=metrics, registry=_registry())

    assert receipt["ground_truth_binding"] == {
        "mask_sha256_raw": "1" * 64,
        "mask_sha256_binary": "2" * 64,
        "mask_sha256_runtime": result.metadata["mask_sha256_runtime"],
    }


def test_control_receipt_contains_no_candidate_payload() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    result = SimpleNamespace(
        mode="control_only",
        original_rgb=image,
        control_mask=np.zeros((32, 32), dtype=np.uint8),
        control_overlay=image,
        control_latency_ms=9.0,
        device="cpu",
        control_model_id="control",
        control_checkpoint_sha256="a" * 64,
        metadata={"result_type": "arbitrary_image_control_only"},
    )

    receipt = build_control_receipt(result, registry=_registry())

    assert receipt["mode"] == "control_only"
    assert "candidate" not in json.dumps(receipt).lower()


def test_dual_receipt_binds_input_outputs_models_protocol_and_latency() -> None:
    receipt = _build_dual_receipt(complete_dual_result())

    assert receipt["schema_version"] == "imp.dual_live.receipt.v2"
    assert receipt["execution"] == "live_sequential"
    assert receipt["evidence_class"] == "illustrative_arbitrary_upload_no_ground_truth"
    assert receipt["request_id"] == "a" * 32
    assert receipt["input"] == {
        "rgb_sha256": complete_dual_result().input_sha256,
        "dimensions": {"height": 16, "width": 24, "channels": 3},
        "evidence_kind": "arbitrary_upload",
    }
    assert set(receipt["models"]) == {"imp", "nnunet"}
    assert receipt["models"]["nnunet"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert receipt["models"]["nnunet"]["segmentation_sha256"] == mask_sha256(
        complete_dual_result().nnunet.mask
    )
    assert receipt["protocol_id"] == PROTOCOL_ID
    assert receipt["models"]["nnunet"]["runtime_status"] == "reconstructed_not_original_equivalent"
    assert receipt["loop192_validation_status"] == "val_gate_failed_no_test"
    assert receipt["release_manifest_sha256"] == _manifest().digest
    assert receipt["total_latency"] == {
        "coordinator_ms": 18.5,
        "coordinator_scope": "dual_service_run_end_to_end_wall",
    }
    serialized = json.dumps(receipt).lower()
    for forbidden in ("dice", "iou", "bf1", "hd95", "assd", "path", "username", "diagnosis"):
        assert forbidden not in serialized


def test_dual_receipt_requires_explicit_valid_input_evidence() -> None:
    with pytest.raises(ValueError, match="input evidence"):
        build_dual_live_receipt(complete_dual_result(), _manifest(), None)


def test_dual_receipt_rejects_mismatched_kind_and_evidence_class() -> None:
    result = complete_dual_result()
    forged = SimpleNamespace(
        kind="arbitrary_upload",
        evidence_class="illustrative_public_sample_no_ground_truth",
        rgb_sha256=result.input_sha256,
        sample_id=None,
        source_dataset=None,
        source_page=None,
        image_license=None,
        training_exposure={},
        ground_truth_used=False,
        ground_truth_not_loaded=True,
    )

    with pytest.raises(ValueError, match="input evidence"):
        build_dual_live_receipt(result, _manifest(), forged)


def test_dual_receipt_rejects_evidence_hash_before_building_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = complete_dual_result()
    evidence = upload_evidence(result.original_rgb)
    object.__setattr__(evidence, "rgb_sha256", "b" * 64)

    def unexpected_payload(*_args: object) -> dict[str, object]:
        raise AssertionError("receipt payload must not be constructed")

    monkeypatch.setattr(presentation, "_dual_live_payload", unexpected_payload)

    with pytest.raises(ValueError, match="input evidence binding mismatch"):
        build_dual_live_receipt(result, _manifest(), evidence)


def test_dual_receipt_upload_schema_has_no_public_metadata() -> None:
    result = complete_dual_result()
    receipt = build_dual_live_receipt(result, _manifest(), upload_evidence(result.original_rgb))

    assert receipt["input"] == {
        "rgb_sha256": result.input_sha256,
        "dimensions": {"height": 16, "width": 24, "channels": 3},
        "evidence_kind": "arbitrary_upload",
    }


def test_incomplete_dual_result_cannot_build_receipt() -> None:
    with pytest.raises(ValueError, match="complete live result"):
        _build_dual_receipt(incomplete_dual_result())


def test_dual_receipt_rejects_nonfinite_latency() -> None:
    result = complete_dual_result()
    result.nnunet.reported_latency_ms = float("nan")

    with pytest.raises(ValueError, match="finite"):
        _build_dual_receipt(result)


def test_dual_receipt_rejects_total_latency_below_coordinator_arm() -> None:
    result = complete_dual_result()
    result.total_latency_ms = 17.4

    with pytest.raises(ValueError, match="total_latency"):
        _build_dual_receipt(result)


@pytest.mark.parametrize(
    ("arm", "field", "value"),
    [
        ("imp", "model_id", "SOTA-model"),
        ("nnunet", "preprocessing", "http://example.test/input"),
        ("imp", "device", "Clinical-device"),
    ],
)
def test_dual_receipt_rejects_prohibited_public_metadata(
    arm: str, field: str, value: str
) -> None:
    result = complete_dual_result()
    setattr(getattr(result, arm), field, value)

    with pytest.raises(ValueError, match="unsafe"):
        _build_dual_receipt(result)


def test_dual_service_complete_result_builds_receipt() -> None:
    result = DualLiveService(ReceiptImp(), ReceiptNnUNet()).run(
        np.zeros((9, 13, 3), dtype=np.uint8)
    )

    receipt = _build_dual_receipt(result)

    assert result.receipt_eligible
    assert receipt["models"]["imp"]["preprocessing"] == "imp_runtime_control"
    assert receipt["models"]["nnunet"]["preprocessing"] == "nnunet_natural_image_2d_czyx"
    assert receipt["total_latency"]["coordinator_ms"] >= 0.0


def test_dual_service_cpu_imp_result_builds_receipt() -> None:
    result = DualLiveService(CpuReceiptImp(), ReceiptNnUNet()).run(
        np.zeros((9, 13, 3), dtype=np.uint8)
    )

    receipt = _build_dual_receipt(result)

    assert result.receipt_eligible
    assert receipt["models"]["imp"]["device"] == "cpu"


def test_dual_service_incomplete_result_cannot_build_receipt() -> None:
    result = DualLiveService(ReceiptImp(), UnavailableReceiptNnUNet()).run(
        np.zeros((9, 13, 3), dtype=np.uint8)
    )

    with pytest.raises(ValueError, match="complete live result"):
        _build_dual_receipt(result)


def test_dual_ledger_is_path_free_and_scope_limited() -> None:
    ledger = render_dual_live_ledger(complete_dual_result())

    assert "Live sequential result" in ledger
    assert "Ground truth not supplied" in ledger
    assert "reconstructed runtime" in ledger
    assert "coordinator wall" in ledger
    assert "not a controlled efficiency benchmark" in ledger
    assert "Dice" not in ledger and "path" not in ledger.lower()


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (lambda result: synthetic_evidence(result.original_rgb), "illustrative_synthetic_no_ground_truth"),
        (lambda result: upload_evidence(result.original_rgb), "illustrative_arbitrary_upload_no_ground_truth"),
    ],
)
def test_receipt_distinguishes_nonpublic_input_evidence(evidence, expected: str) -> None:
    result = complete_dual_result()
    receipt = build_dual_live_receipt(result, _manifest(), evidence(result))

    assert receipt["evidence_class"] == expected
    assert "metrics" not in receipt


@pytest.mark.parametrize("value", [None, object()])
def test_missing_or_unknown_input_evidence_fails_closed(value: object) -> None:
    with pytest.raises(ValueError, match="input evidence"):
        build_dual_live_receipt(complete_dual_result(), _manifest(), value)


def test_receipt_rejects_unknown_evidence_kind() -> None:
    result = complete_dual_result()
    evidence = LiveInputEvidence(
        "arbitrary_upload", "illustrative_arbitrary_upload_no_ground_truth",
        result.input_sha256, None, None, None, None, {}, False, True,
    )
    object.__setattr__(evidence, "kind", "unknown")

    with pytest.raises(ValueError, match="input evidence"):
        build_dual_live_receipt(result, _manifest(), evidence)


def test_public_receipt_has_provenance_and_segmentation_hashes_without_private_fields() -> None:
    result = complete_dual_result()
    evidence = LiveInputEvidence(
        "public_sample",
        "illustrative_public_sample_no_ground_truth",
        result.input_sha256,
        "ISIC_0000050",
        "isic2018",
        "https://challenge.isic-archive.com/data/",
        "CC-0",
        {
            "L206-control-s206": "excluded_from_308_fit_in_76_group_train_screen_holdout",
            "L192-nnUNet-v2-raw-100ep": "included_in_clean_v3_2008_training_rows",
        },
        False,
        True,
    )

    receipt = build_dual_live_receipt(result, _manifest(), evidence)

    assert receipt["input"]["sample_id"] == "ISIC_0000050"
    assert receipt["input"]["license"] == "CC-0"
    assert receipt["models"]["imp"]["segmentation_sha256"] == mask_sha256(result.imp.mask)
    assert receipt["models"]["nnunet"]["segmentation_sha256"] == mask_sha256(result.nnunet.mask)
    assert set(receipt["input"]) <= {
        "rgb_sha256", "dimensions", "evidence_kind", "sample_id", "source_dataset",
        "source_page", "license", "training_exposure", "ground_truth_used",
        "ground_truth_not_loaded",
    }
    serialized = json.dumps(receipt).lower()
    for forbidden in ("path", "filename", "username", "diagnosis", "traceback", "dice", "iou", "bf1", "hd95", "assd"):
        assert forbidden not in serialized
