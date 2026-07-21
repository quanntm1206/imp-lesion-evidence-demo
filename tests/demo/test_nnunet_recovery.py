from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys

import pytest

from scripts.demo import verify_nnunet_bundle as verifier


EXPECTED_PINS = {
    "checkpoint_sha256": "3814716033afd464dacc573f92a5a44ff20eb7f2163d99b4f16ecff8aa278ea2",
    "plans_sha256": "b60e4defd229b03f7064dc5b66123545c91cdaa44c09d990b86690a94e1e08a7",
    "fingerprint_sha256": "931da8aae52ffecd726d5928009ebdcae7002e24b035fad89177e0bc81dba85c",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _allow_fake_pins(monkeypatch: pytest.MonkeyPatch, report: dict[str, object]) -> None:
    provenance = report["provenance"]
    assert isinstance(provenance, dict)
    monkeypatch.setattr(verifier, "PINNED_HASHES", provenance.copy(), raising=False)


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
        "candidate_id": "L192-nnUNet-v2-raw-100ep",
        "provenance": {
            "checkpoint_sha256": _sha256(artifacts["checkpoint_final.pth"]),
            "plans_sha256": _sha256(artifacts["nnUNetPlans.json"]),
            "fingerprint_sha256": _sha256(artifacts["dataset_fingerprint.json"]),
        },
        "source_vhd_proof": {
            "before": {
                "length": 552_000_000_000,
                "creation_time_utc": "2026-07-01T00:00:00Z",
                "last_write_time_utc": "2026-07-01T00:00:00Z",
            },
            "after": {
                "length": 552_000_000_000,
                "creation_time_utc": "2026-07-01T00:00:00Z",
                "last_write_time_utc": "2026-07-01T00:00:00Z",
            },
        },
    }
    return bundle, report


def test_bundle_verifier_binds_required_hashes_and_omits_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    _allow_fake_pins(monkeypatch, report)

    receipt = verifier.verify_bundle(bundle, report)

    provenance = report["provenance"]
    assert receipt["schema_version"] == "loop192.recovery.receipt.v1"
    assert receipt["checkpoint_sha256"] == provenance["checkpoint_sha256"]
    assert receipt["plans_sha256"] == provenance["plans_sha256"]
    assert receipt["fingerprint_sha256"] == provenance["fingerprint_sha256"]
    assert receipt["source_vhd_unchanged"] is True
    assert "path" not in json.dumps(receipt).lower()


def test_bundle_verifier_stops_on_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    _allow_fake_pins(monkeypatch, report)
    (bundle / "checkpoint_final.pth").write_bytes(b"drift")

    with pytest.raises(ValueError, match="checkpoint hash does not match Loop192 pin"):
        verifier.verify_bundle(bundle, report)


def test_bundle_verifier_requires_recovery_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    _allow_fake_pins(monkeypatch, report)
    (bundle / "requirements.lock").unlink()

    with pytest.raises(FileNotFoundError, match="required Loop192 metadata missing: requirements.lock"):
        verifier.verify_bundle(bundle, report)


def test_bundle_verifier_rejects_malformed_report(tmp_path: Path) -> None:
    bundle, _ = fake_loop192_bundle(tmp_path)

    with pytest.raises(KeyError, match="provenance"):
        verifier.verify_bundle(bundle, {"candidate_id": "L192-nnUNet-v2-raw-100ep"})


def test_bundle_verifier_rejects_path_bearing_candidate_id(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    report["candidate_id"] = r"C:\private\loop192"

    with pytest.raises(ValueError, match="candidate_id must not contain a local path"):
        verifier.verify_bundle(bundle, report)


def test_bundle_verifier_rejects_unpinned_candidate_id(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    report["candidate_id"] = "L191-C0-clean-v3-IMP-control"

    with pytest.raises(ValueError, match="candidate_id does not match Loop192"):
        verifier.verify_bundle(bundle, report)


def test_bundle_verifier_requires_unchanged_source_vhd_proof(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    report.pop("source_vhd_proof")

    with pytest.raises(ValueError, match="source VHD unchanged proof required"):
        verifier.verify_bundle(bundle, report)


def test_bundle_verifier_rejects_changed_source_vhd_proof(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    proof = report["source_vhd_proof"]
    assert isinstance(proof, dict)
    after = proof["after"]
    assert isinstance(after, dict)
    after["length"] = 552_000_000_001

    with pytest.raises(ValueError, match="source VHD changed after recovery"):
        verifier.verify_bundle(bundle, report)


def test_bundle_verifier_pins_exact_loop192_hashes() -> None:
    assert dict(getattr(verifier, "PINNED_HASHES", {})) == EXPECTED_PINS


def test_bundle_verifier_rejects_matching_unpinned_bundle_and_report(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)

    with pytest.raises(ValueError, match="checkpoint provenance does not match Loop192 pin"):
        verifier.verify_bundle(bundle, report)


def test_bundle_verifier_rejects_report_provenance_outside_expected_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    _allow_fake_pins(monkeypatch, report)
    provenance = report["provenance"]
    assert isinstance(provenance, dict)
    provenance["checkpoint_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="checkpoint provenance does not match Loop192 pin"):
        verifier.verify_bundle(bundle, report)


def test_bundle_verifier_cli_writes_atomic_sorted_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    report["encoding_probe"] = "caf\u00e9"
    report.update(
        recovery_backend="container-readonly-7zip",
        parser_warning="Headers Error",
        runtime_status="reconstructed_required",
    )
    _allow_fake_pins(monkeypatch, report)
    report_path = bundle / ".verification-report.json"
    receipt_path = bundle / "recovery_receipt.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    expected = verifier.verify_bundle(bundle, report)
    expected.update(
        recovery_backend="container-readonly-7zip",
        parser_warning="Headers Error",
        runtime_status="reconstructed_required",
    )
    link_calls: list[tuple[Path, Path]] = []
    real_link = os.link

    def observed_link(source, destination) -> None:
        link_calls.append((Path(source), Path(destination)))
        real_link(source, destination)

    monkeypatch.setattr(os, "link", observed_link)

    result = verifier.main(
        [
            "--bundle",
            str(bundle),
            "--report",
            str(report_path),
            "--receipt",
            str(receipt_path),
        ]
    )

    assert result == 0
    assert len(link_calls) == 1
    temporary, destination = link_calls[0]
    assert temporary.parent.resolve() == bundle.resolve()
    assert destination.resolve() == receipt_path.resolve()
    assert not temporary.exists()
    assert receipt_path.read_text(encoding="utf-8") == (
        json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    assert str(tmp_path) not in receipt_path.read_text(encoding="utf-8")


def test_bundle_verifier_cli_rejects_malformed_report(tmp_path: Path) -> None:
    bundle, _ = fake_loop192_bundle(tmp_path)
    report_path = bundle / ".verification-report.json"
    receipt_path = bundle / "recovery_receipt.json"
    report_path.write_text("{malformed", encoding="ascii")

    with pytest.raises(json.JSONDecodeError):
        verifier.main(
            [
                "--bundle",
                str(bundle),
                "--report",
                str(report_path),
                "--receipt",
                str(receipt_path),
            ]
        )

    assert not receipt_path.exists()


def test_bundle_verifier_cli_rejects_receipt_outside_bundle(tmp_path: Path) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    report_path = bundle / ".verification-report.json"
    report_path.write_text(json.dumps(report), encoding="ascii")

    with pytest.raises(ValueError, match="receipt parent must equal resolved bundle"):
        verifier.main(
            [
                "--bundle",
                str(bundle),
                "--report",
                str(report_path),
                "--receipt",
                str(tmp_path / "outside-receipt.json"),
            ]
        )


@pytest.mark.parametrize(
    "artifact_name",
    ["checkpoint_final.pth", "nnUNetPlans.json", "runtime_identity.json"],
)
def test_bundle_verifier_cli_never_overwrites_bundle_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    _allow_fake_pins(monkeypatch, report)
    report_path = bundle / ".verification-report.json"
    report_path.write_text(json.dumps(report), encoding="ascii")
    artifact_path = bundle / artifact_name
    before = artifact_path.read_bytes()

    with pytest.raises(
        ValueError, match="receipt basename must be exactly recovery_receipt.json"
    ):
        verifier.main(
            [
                "--bundle",
                str(bundle),
                "--report",
                str(report_path),
                "--receipt",
                str(artifact_path),
            ]
        )

    assert artifact_path.read_bytes() == before


def test_bundle_verifier_cli_rejects_existing_exact_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    _allow_fake_pins(monkeypatch, report)
    report_path = bundle / ".verification-report.json"
    report_path.write_text(json.dumps(report), encoding="ascii")
    receipt_path = bundle / "recovery_receipt.json"
    before = b"preserve-existing-receipt"
    receipt_path.write_bytes(before)

    with pytest.raises(ValueError, match="receipt must not already exist"):
        verifier.main(
            [
                "--bundle",
                str(bundle),
                "--report",
                str(report_path),
                "--receipt",
                str(receipt_path),
            ]
        )

    assert receipt_path.read_bytes() == before


def test_bundle_verifier_cli_reads_report_from_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    _allow_fake_pins(monkeypatch, report)
    receipt_path = bundle / "recovery_receipt.json"
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(report)))

    result = verifier.main(
        [
            "--bundle",
            str(bundle),
            "--report",
            "-",
            "--receipt",
            str(receipt_path),
        ]
    )

    assert result == 0
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == (
        verifier.verify_bundle(bundle, report)
    )


@pytest.mark.parametrize(
    ("full_flag", "abbreviated_flag"),
    [("--bundle", "--bund"), ("--report", "--repo"), ("--receipt", "--rece")],
)
def test_bundle_verifier_cli_rejects_abbreviated_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    full_flag: str,
    abbreviated_flag: str,
) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    _allow_fake_pins(monkeypatch, report)
    report_path = bundle / ".verification-report.json"
    receipt_path = bundle / "recovery_receipt.json"
    report_path.write_text(json.dumps(report), encoding="ascii")
    arguments = [
        "--bundle",
        str(bundle),
        "--report",
        str(report_path),
        "--receipt",
        str(receipt_path),
    ]
    arguments[arguments.index(full_flag)] = abbreviated_flag

    with pytest.raises(SystemExit):
        verifier.main(arguments)


@pytest.mark.parametrize("kind", ["report", "receipt"])
def test_bundle_verifier_cli_rejects_reparse_paths(
    tmp_path: Path, kind: str
) -> None:
    bundle, report = fake_loop192_bundle(tmp_path)
    report_target = bundle / "report-target.json"
    report_target.write_text(json.dumps(report), encoding="ascii")
    report_path = bundle / ".verification-report.json"
    receipt_path = bundle / "recovery_receipt.json"
    if kind == "report":
        link, target = report_path, report_target
    else:
        report_path.write_text(json.dumps(report), encoding="ascii")
        link, target = receipt_path, tmp_path / "receipt-target.json"
        target.write_text("{}", encoding="ascii")
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symlink unavailable: {error}")

    with pytest.raises(ValueError, match=f"{kind} must not be a reparse point"):
        verifier.main(
            [
                "--bundle",
                str(bundle),
                "--report",
                str(report_path),
                "--receipt",
                str(receipt_path),
            ]
        )
