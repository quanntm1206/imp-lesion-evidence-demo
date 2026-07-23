from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys

import pytest

from lesion_robustness.research.rq1_data import (
    RQ1_V2_CAPABILITY,
    audit_data,
    load_protocol,
    read_authorized_rows,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments" / "rq1_v2" / "protocol.json"
AUDIT_SCRIPT = ROOT / "scripts" / "research" / "audit_rq1_v2_data.py"


class SealedIndexSpy:
    def __fspath__(self) -> str:
        raise AssertionError("sealed index was resolved")


def test_protocol_freezes_rq1_v2_capability_and_reproducibility() -> None:
    protocol = load_protocol(PROTOCOL)

    assert protocol.schema_version == "imp.rq1_v2.protocol.v1"
    assert protocol.train_count == 2008
    assert protocol.validation_count == 431
    assert protocol.test_v3_access is False
    assert protocol.seeds == (206, 1206, 2206)
    assert protocol.dataset_index_status == "unresolved_blocked"
    assert protocol.dataset_index_sha256 is None
    assert tuple(RQ1_V2_CAPABILITY) == ("train", "validation")


def test_protocol_rejects_test_rows_before_any_index_open() -> None:
    with pytest.raises(ValueError, match="test-v3 is sealed"):
        read_authorized_rows(SealedIndexSpy(), "test")


def test_protocol_rejects_unknown_split_before_any_index_open() -> None:
    with pytest.raises(ValueError, match="not authorized"):
        read_authorized_rows(SealedIndexSpy(), "holdout")


def test_unresolved_index_status_rejects_before_any_index_open() -> None:
    protocol = load_protocol(PROTOCOL)

    with pytest.raises(ValueError, match="dataset index is unresolved_blocked"):
        read_authorized_rows(SealedIndexSpy(), "train", protocol)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "forged", "schema_version"),
        ("research_question", "forged", "research_question"),
        ("evidence_class", "forged", "evidence_class"),
        ("train_count", 1, "train_count"),
        ("validation_count", 1, "validation_count"),
        ("test_v3_access", True, "test-v3"),
        ("clean_v3_manifest_sha256", "0" * 64, "clean_v3_manifest_sha256"),
        ("seeds", (206,), "seeds"),
        ("conditions", ("clean",), "conditions"),
        ("metrics", ("dice",), "metrics"),
        ("boundary_tolerance_original_pixels", 3, "boundary tolerance"),
        ("probability_restore", "forged", "probability_restore"),
        ("threshold", 0.4, "threshold"),
        ("primary_endpoint", "forged", "primary_endpoint"),
        ("secondary_endpoint", "forged", "secondary_endpoint"),
        ("claim_limit", "forged", "claim_limit"),
    ],
)
def test_audit_revalidates_complete_frozen_protocol_before_rows(
    field: str, value: object, message: str
) -> None:
    protocol = load_protocol(PROTOCOL)
    verified = replace(
        protocol, dataset_index_status="verified", dataset_index_sha256="d" * 64
    )
    drifted = replace(verified, **{field: value})

    with pytest.raises(ValueError, match=message):
        audit_data((), drifted)


@pytest.mark.parametrize(
    ("status", "digest", "message"),
    [
        ("verified", None, "verified dataset index requires"),
        ("unresolved_blocked", "d" * 64, "unresolved_blocked dataset index forbids"),
        ("forged", "d" * 64, "dataset_index_status"),
    ],
)
def test_protocol_rejects_invalid_index_status_digest_pair(
    status: str, digest: str | None, message: str
) -> None:
    protocol = load_protocol(PROTOCOL)
    drifted = replace(
        protocol, dataset_index_status=status, dataset_index_sha256=digest
    )

    with pytest.raises(ValueError, match=message):
        audit_data((), drifted)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("test_v3_access", True, "test-v3 must remain sealed"),
        ("train_count", 2007, "train_count"),
        ("validation_count", 430, "validation_count"),
        ("seeds", [206], "seeds"),
    ],
)
def test_protocol_rejects_contract_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="ascii"))
    payload[field] = value
    candidate = tmp_path / "protocol.json"
    candidate.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ValueError, match=message):
        load_protocol(candidate)


def test_protocol_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    text = PROTOCOL.read_text(encoding="ascii")
    candidate = tmp_path / "duplicate.json"
    candidate.write_text(text.replace('"train_count": 2008,', '"train_count": 2008,\n  "train_count": 2008,'), encoding="ascii")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_protocol(candidate)


def test_protocol_admits_future_verified_index_without_code_constant(
    tmp_path: Path,
) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="ascii"))
    payload["dataset_index_status"] = "verified"
    payload["dataset_index_sha256"] = "d" * 64
    candidate = tmp_path / "verified-protocol.json"
    candidate.write_text(json.dumps(payload), encoding="ascii")

    loaded = load_protocol(candidate)

    assert loaded.dataset_index_status == "verified"
    assert loaded.dataset_index_sha256 == "d" * 64


def test_audit_entry_rejects_unresolved_before_index_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(AUDIT_SCRIPT),
            "--protocol",
            str(PROTOCOL),
            "--index",
            str(tmp_path / "must-not-open.json"),
            "--output",
            str(tmp_path / "must-not-write.json"),
        ],
    )

    with pytest.raises(ValueError, match="dataset index is unresolved_blocked"):
        runpy.run_path(str(AUDIT_SCRIPT), run_name="__main__")
