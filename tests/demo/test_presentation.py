from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lesion_robustness.demo.presentation import (
    NO_GT_MESSAGE,
    build_control_receipt,
    build_fixed_receipt,
    render_clean_evidence,
    render_legacy_table,
    render_metrics,
)


ROOT = Path(__file__).resolve().parents[2]


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
            "historical_cache_provenance_drift": True,
            "path": "E:/private/data.jpg",
        },
    )


def test_clean_evidence_keeps_validation_and_train_screen_scopes_separate() -> None:
    html = render_clean_evidence(_registry())

    assert "protected_validation" in html
    assert "train_screen" in html
    assert "0.8959" in html and "0.9019" in html
    assert "-0.0313" in html
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
    for forbidden in ("E:/", "private", "path", "url", "filename", "environment"):
        assert forbidden not in encoded.lower()


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
