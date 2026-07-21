from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.demo.verify_nnunet_bundle import verify_bundle


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fake_loop192_bundle(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    bundle = tmp_path / "loop192"
    bundle.mkdir()
    artifacts = {
        "checkpoint_final.pth": b"checkpoint",
        "nnUNetPlans.json": b'{"plans": 192}',
        "dataset_fingerprint.json": b'{"fingerprint": 192}',
        "dataset.json": b'{"dataset": 192}',
        "plans.json": b'{"inference_plans": 192}',
        "runtime_identity.json": b'{"trainer": "nnUNetTrainer"}',
        "requirements.lock": b"nnunetv2==2.5.1\n",
    }
    for filename, content in artifacts.items():
        (bundle / filename).write_bytes(content)
    report: dict[str, object] = {
        "candidate_id": "loop192-nnunet-clean-v3",
        "provenance": {
            "checkpoint_sha256": _sha256(artifacts["checkpoint_final.pth"]),
            "plans_sha256": _sha256(artifacts["nnUNetPlans.json"]),
            "fingerprint_sha256": _sha256(artifacts["dataset_fingerprint.json"]),
        },
    }
    return bundle, report


def test_bundle_verifier_binds_required_hashes_and_omits_paths(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)

    receipt = verify_bundle(bundle, report)

    provenance = report["provenance"]
    assert receipt["schema_version"] == "loop192.recovery.receipt.v1"
    assert receipt["checkpoint_sha256"] == provenance["checkpoint_sha256"]
    assert receipt["plans_sha256"] == provenance["plans_sha256"]
    assert receipt["fingerprint_sha256"] == provenance["fingerprint_sha256"]
    assert receipt["source_vhd_unchanged"] is True
    assert "path" not in json.dumps(receipt).lower()


def test_bundle_verifier_stops_on_hash_drift(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    (bundle / "checkpoint_final.pth").write_bytes(b"drift")

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        verify_bundle(bundle, report)


def test_bundle_verifier_requires_recovery_metadata(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    (bundle / "requirements.lock").unlink()

    with pytest.raises(FileNotFoundError, match="required Loop192 metadata missing: requirements.lock"):
        verify_bundle(bundle, report)


def test_bundle_verifier_rejects_malformed_report(tmp_path: Path) -> None:
    bundle, _ = fake_loop192_bundle(tmp_path)

    with pytest.raises(KeyError, match="provenance"):
        verify_bundle(bundle, {"candidate_id": "loop192-nnunet-clean-v3"})
