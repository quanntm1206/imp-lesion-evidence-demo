from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.demo.write_fast_release_handoff import build_packet

from lesion_robustness.demo.fast_release_handoff import (
    DEFERRED_JOBS,
    validate_fast_release_handoff,
    write_fast_release_handoff,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260723T033658169Z"
NEW_HANDOFF_RUN_ID = "20260723T040700000Z"


def _sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _packet() -> dict[str, object]:
    return {
        "schema_version": "imp.fast_release.handoff.v1",
        "run_id": RUN_ID,
        "status": "blocked_missing_prerequisite",
        "release_manifest_sha256": _sha("release/imp_release_manifest.json"),
        "paper_artifact_manifest_sha256": _sha("paper/clean_v3_loop206/artifact_manifest.json"),
        "paper_pdf_sha256": _sha("paper/clean_v3_loop206/main.pdf"),
        "presentation_manifest_sha256": _sha("outputs/imp-lesion-evidence-defense-manifest.json"),
        "html_sha256": _sha("outputs/imp-lesion-evidence-defense.html"),
        "pptx_sha256": _sha("outputs/imp-lesion-evidence-defense.pptx"),
        "presentation_pdf_sha256": _sha("outputs/imp-lesion-evidence-defense.pdf"),
        "acceptance_packet_sha256": _sha(
            f"demo_runtime/acceptance/imp.dual_live.e2e.v1/{RUN_ID}/acceptance.json"
        ),
        "acceptance_status": "blocked",
        "visual_qa_status": "passed",
        "runtime_status": "blocked_missing_prerequisite",
        "determinism_status": "blocked",
        "cloudflare_status": "deferred_external_dependency",
        "p1_status": "not_promoted",
        "test_v3": False,
        "ph2": False,
        "deferred_jobs": list(DEFERRED_JOBS),
        "reason": "Missing required private runtime prerequisites.",
        "artifact_class": "private_runtime_prerequisite",
    }


def _write(path: Path, payload: object) -> None:
    path.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    )


def test_missing_required_key_rejected(tmp_path: Path) -> None:
    payload = _packet()
    del payload["runtime_status"]
    path = tmp_path / "missing.json"
    _write(path, payload)
    with pytest.raises(ValueError, match="required"):
        validate_fast_release_handoff(path)


def test_duplicate_json_key_rejected(tmp_path: Path) -> None:
    payload = _packet()
    path = tmp_path / "duplicate.json"
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    raw = raw.replace('"status":"blocked_missing_prerequisite"', '"status":"blocked","status":"blocked_missing_prerequisite"') + "\n"
    path.write_bytes(raw.encode("ascii"))
    with pytest.raises(ValueError, match="duplicate"):
        validate_fast_release_handoff(path)


def test_stale_hash_rejected(tmp_path: Path) -> None:
    payload = _packet()
    payload["release_manifest_sha256"] = "0" * 64
    path = tmp_path / "stale.json"
    _write(path, payload)
    with pytest.raises(ValueError, match="stale|hash"):
        validate_fast_release_handoff(path)


def test_noncanonical_bytes_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(_packet(), indent=2) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="canonical"):
        validate_fast_release_handoff(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("p1_status", "promoted"), ("test_v3", True), ("ph2", True), ("deferred_jobs", ["imp-206"])],
)
def test_sealed_flags_and_deferred_jobs_rejected(tmp_path: Path, field: str, value: object) -> None:
    payload = _packet()
    payload[field] = value
    path = tmp_path / f"bad-{field}.json"
    _write(path, payload)
    with pytest.raises(ValueError):
        validate_fast_release_handoff(path)


def test_blocked_packet_requires_acceptance_hash_and_reason(tmp_path: Path) -> None:
    payload = _packet()
    payload["acceptance_packet_sha256"] = None
    path = tmp_path / "blocked.json"
    _write(path, payload)
    with pytest.raises(ValueError, match="acceptance_packet_sha256"):
        validate_fast_release_handoff(path)


def test_blocked_runtime_cannot_be_labeled_valid(tmp_path: Path) -> None:
    payload = _packet()
    payload["status"] = "valid"
    path = tmp_path / "mixed-status.json"
    _write(path, payload)
    with pytest.raises(ValueError, match="status"):
        validate_fast_release_handoff(path)


def test_unknown_top_level_field_rejected(tmp_path: Path) -> None:
    payload = _packet()
    payload["unexpected"] = "drift"
    path = tmp_path / "extra.json"
    _write(path, payload)
    with pytest.raises(ValueError, match="keys"):
        validate_fast_release_handoff(path)


def test_writer_is_canonical_and_refuses_byte_drift(tmp_path: Path) -> None:
    packet = _packet()
    path = write_fast_release_handoff(tmp_path, packet)
    assert path == tmp_path / RUN_ID / "handoff.json"
    assert path.read_bytes().endswith(b"\n")
    assert validate_fast_release_handoff(path).run_id == RUN_ID
    drifted = dict(packet)
    drifted["reason"] = "different"
    with pytest.raises(ValueError, match="drift"):
        write_fast_release_handoff(tmp_path, drifted)


def test_cli_packet_preserves_blocked_runtime_with_independent_handoff_run(
    tmp_path: Path,
) -> None:
    packet = build_packet(
        run_id=NEW_HANDOFF_RUN_ID,
        release_manifest=ROOT / "release/imp_release_manifest.json",
        paper_manifest=ROOT / "paper/clean_v3_loop206/artifact_manifest.json",
        presentation_manifest=ROOT / "outputs/imp-lesion-evidence-defense-manifest.json",
        acceptance_packet=ROOT / f"demo_runtime/acceptance/imp.dual_live.e2e.v1/{RUN_ID}/acceptance.json",
        visual_review=ROOT / "outputs/visual-evidence/visual-review.json",
    )

    path = write_fast_release_handoff(tmp_path, packet)
    assert packet["status"] == "blocked_missing_prerequisite"
    assert packet["paper_pdf_status"] == "current"
    assert packet["runtime_status"] == "blocked_missing_prerequisite"
    assert packet["determinism_status"] == "blocked"
    assert packet["cloudflare_status"] == "deferred_external_dependency"
    assert packet["acceptance_status"] == "blocked"
    assert packet["test_v3"] is False and packet["ph2"] is False
    assert validate_fast_release_handoff(path).run_id == NEW_HANDOFF_RUN_ID
