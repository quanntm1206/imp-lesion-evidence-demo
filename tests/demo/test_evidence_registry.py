from __future__ import annotations

import json
from pathlib import Path

import pytest

from lesion_robustness.evidence_registry import (
    EvidenceSources,
    build_registry,
    validate_registry,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="ascii")
    return path


@pytest.fixture
def frozen_reports(tmp_path: Path) -> EvidenceSources:
    loop191 = _write_json(
        tmp_path / "loop191.json",
        {
            "loop": 191,
            "status": "val_gate_failed_no_test",
            "test_opened": False,
            "pre_pilot_integrity": {
                "clean_v3": {
                    "status": "complete",
                    "rows": 2869,
                    "split_counts": {"train": 2008, "val": 431, "test": 430},
                    "identity_overlap_violations": 0,
                }
            },
            "candidates": [
                {
                    "id": "L191-C0-clean-v3-IMP-control",
                    "role": "control",
                    "metrics": {
                        "robust_mean": {
                            "dice": 0.895870479294128,
                            "iou": 0.8276836720163411,
                            "precision": 0.9087624468161243,
                            "recall": 0.9127938055574955,
                            "boundary_f1": 0.4145296468027299,
                        }
                    },
                }
            ],
        },
    )
    loop192 = _write_json(
        tmp_path / "loop192.json",
        {
            "loop": 192,
            "status": "val_gate_failed_no_test",
            "test_opened": False,
            "candidate_id": "L192-nnUNet-v2-raw-100ep",
            "evaluation_protocol": {
                "model_image_size": [256, 256],
                "metric_image_size": [384, 384],
                "boundary_tolerance": 2,
                "distance_units": "pixels_on_metric_canvas",
            },
            "candidate": {
                "robust_mean": {
                    "dice": 0.9019177076063616,
                    "iou": 0.8353954025733779,
                    "precision": 0.9055585401108206,
                    "recall": 0.9246376021289167,
                    "boundary_f1": 0.43691577678397314,
                }
            },
        },
    )
    loop206 = _write_json(
        tmp_path / "loop206.json",
        {
            "loop": 206,
            "status": "closed_no_go_pause_requested",
            "decision": "stop_loop206_decision_only_no_validation_test_ph2",
            "protected_panels_sealed": True,
            "evidence_validation": {"passed": True},
            "robust_deltas": {
                "dice": -0.031296243954732295,
                "boundary_f1": -0.014658313347547247,
            },
            "bootstrap": {
                "dice": {
                    "point_delta": -0.03129624395473221,
                    "ci95_lower": -0.049121296024302145,
                    "ci95_upper": -0.015627817085354864,
                    "paired_seed_count": 3,
                    "group_count": 76,
                    "corruption_count": 3,
                    "resamples": 10000,
                },
                "boundary_f1": {
                    "point_delta": -0.01465831334754726,
                    "ci95_lower": -0.030758654691150956,
                    "ci95_upper": 0.0010438469457382654,
                    "paired_seed_count": 3,
                    "group_count": 76,
                    "corruption_count": 3,
                    "resamples": 10000,
                },
            },
        },
    )
    panel = tmp_path / "locked_panel_results.tex"
    panel.write_text(
        "TEST & IMP-SegFormer-B3 & 0.8913 & 0.8962 & 0.8224 & 0.4240 \\\\\n"
        "TEST & Vanilla SegFormer-B3 & 0.8897 & 0.8968 & 0.8179 & 0.3969 \\\\\n"
        "TEST & EGE-UNet & 0.8616 & 0.8617 & 0.7809 & 0.0817 \\\\\n"
        "TEST & nnU-Net v2 & 0.8911 & 0.8923 & 0.8246 & 0.5381 \\\\\n",
        encoding="ascii",
    )
    bootstrap = tmp_path / "bootstrap_deltas.tex"
    bootstrap.write_text(
        "Dice IMP--nnU-Net & 0.0002 & [-0.0030, 0.0035] \\\\\n",
        encoding="ascii",
    )
    return EvidenceSources(loop191, loop192, loop206, panel, bootstrap)


def test_registry_separates_validation_screen_and_legacy(
    frozen_reports: EvidenceSources, tmp_path: Path
) -> None:
    registry = build_registry(frozen_reports, project_root=tmp_path)
    validate_registry(registry)
    rows = {row["model_id"]: row for row in registry["observations"]}
    assert rows["L191-C0-clean-v3-IMP-control"]["evidence_class"] == "protected_validation"
    assert rows["L192-nnUNet-v2-raw-100ep"]["robust_dice"] == 0.9019177076063616
    assert rows["L206-contour-vs-control"]["evidence_class"] == "train_screen"
    assert rows["Loop170-IMP"]["evidence_class"] == "legacy_patient_contaminated"
    assert registry["scientific_sota_status"] == "not_established"


def test_registry_rejects_legacy_as_comparable(
    frozen_reports: EvidenceSources, tmp_path: Path
) -> None:
    registry = build_registry(frozen_reports, project_root=tmp_path)
    legacy = next(row for row in registry["observations"] if row["model_id"] == "Loop170-IMP")
    legacy["scientific_comparable"] = True
    with pytest.raises(ValueError, match="legacy"):
        validate_registry(registry)


def test_registry_rejects_source_hash_drift(
    frozen_reports: EvidenceSources, tmp_path: Path
) -> None:
    registry = build_registry(frozen_reports, project_root=tmp_path)
    registry["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registry hash"):
        validate_registry(registry)


def test_registry_rejects_forged_release_digest_with_current_registry_hash(
    frozen_reports: EvidenceSources, tmp_path: Path
) -> None:
    registry = build_registry(frozen_reports, project_root=tmp_path)
    registry["release_manifest_sha256"] = "a" * 64

    with pytest.raises(ValueError, match="release manifest"):
        validate_registry(registry)
