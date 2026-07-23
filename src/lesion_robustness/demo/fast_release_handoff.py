"""Canonical, evidence-bound fast-release handoff packets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA_VERSION = "imp.fast_release.handoff.v1"
DEFERRED_JOBS = (
    "imp-206",
    "nnunet-206",
    "imp-1206",
    "nnunet-1206",
    "imp-2206",
    "nnunet-2206",
)
HASH_FIELDS = {
    "release_manifest_sha256": "release/imp_release_manifest.json",
    "paper_artifact_manifest_sha256": "paper/clean_v3_loop206/artifact_manifest.json",
    "paper_pdf_sha256": "paper/clean_v3_loop206/main.pdf",
    "presentation_manifest_sha256": "outputs/imp-lesion-evidence-defense-manifest.json",
    "html_sha256": "outputs/imp-lesion-evidence-defense.html",
    "pptx_sha256": "outputs/imp-lesion-evidence-defense.pptx",
    "presentation_pdf_sha256": "outputs/imp-lesion-evidence-defense.pdf",
}
REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "status",
        *HASH_FIELDS,
        "acceptance_packet_sha256",
        "acceptance_status",
        "visual_qa_status",
        "runtime_status",
        "determinism_status",
        "cloudflare_status",
        "p1_status",
        "test_v3",
        "ph2",
        "deferred_jobs",
    }
)
OPTIONAL_KEYS = frozenset({"reason", "artifact_class", "paper_pdf_status"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")
_ABS_WIN = re.compile(r"(?i)\b[A-Z]:[\\/]")

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FastReleaseHandoff:
    schema_version: str
    run_id: str
    status: str
    release_manifest_sha256: str | None
    paper_artifact_manifest_sha256: str | None
    paper_pdf_sha256: str | None
    presentation_manifest_sha256: str | None
    html_sha256: str | None
    pptx_sha256: str | None
    presentation_pdf_sha256: str | None
    acceptance_packet_sha256: str
    acceptance_status: str
    visual_qa_status: str
    runtime_status: str
    determinism_status: str
    cloudflare_status: str
    p1_status: str
    test_v3: bool
    ph2: bool
    deferred_jobs: tuple[str, ...]
    reason: str | None = None
    artifact_class: str | None = None
    payload: Mapping[str, object] | None = None

    def __getitem__(self, key: str) -> object:
        if self.payload is None:
            raise KeyError(key)
        return self.payload[key]


def _canonical(payload: Mapping[str, object]) -> bytes:
    try:
        return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("handoff payload is not canonical JSON") from exc


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, object]:
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_pairs_no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "duplicate JSON key" in str(exc):
            raise
        raise ValueError("handoff JSON is unreadable") from exc
    if not isinstance(raw, dict):
        raise ValueError("handoff root must be an object")
    if raw_bytes != _canonical(raw):
        raise ValueError("handoff bytes are not canonical")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _acceptance_path(expected_hash: str) -> Path:
    root = ROOT / "demo_runtime" / "acceptance" / "imp.dual_live.e2e.v1"
    try:
        candidates = [
            item / "acceptance.json"
            for item in root.iterdir()
            if item.is_dir() and (item / "acceptance.json").is_file()
        ]
    except OSError as exc:
        raise ValueError("handoff acceptance packet missing") from exc
    matches = [path for path in candidates if _sha256_file(path) == expected_hash]
    if len(matches) != 1:
        raise ValueError("handoff acceptance packet missing or ambiguous")
    return matches[0]


def _reject_unsafe_fields(value: object, *, key: str = "") -> None:
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            lowered = str(child_key).lower()
            if lowered in {"path", "url", "private", "private_path", "private_url", "filename", "file_path"} or lowered.endswith("_path") or lowered.endswith("_url"):
                raise ValueError("handoff path/URL/private fields are forbidden")
            if lowered in {"handoff_sha256", "self_sha256", "packet_sha256"} or ("handoff" in lowered and lowered.endswith(("_sha256", "_digest"))):
                raise ValueError("handoff self-hash field is forbidden")
            _reject_unsafe_fields(child_value, key=lowered)
    elif isinstance(value, list):
        for item in value:
            _reject_unsafe_fields(item, key=key)
    elif isinstance(value, str):
        if value.lower().startswith(("http://", "https://")) or _ABS_WIN.search(value):
            raise ValueError("handoff path/URL/private fields are forbidden")


def _validate_hash(value: object, field: str, *, allow_null: bool) -> str | None:
    if value is None and allow_null:
        return None
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256 hash")
    return value


def validate_fast_release_handoff(path: Path) -> FastReleaseHandoff:
    """Read, canonicalize, and verify one handoff packet."""
    payload = _read(Path(path))
    _reject_unsafe_fields(payload)
    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        raise ValueError(f"handoff required keys missing: {sorted(missing)}")
    extra = set(payload) - REQUIRED_KEYS - OPTIONAL_KEYS
    if extra:
        raise ValueError(f"handoff keys mismatch: {sorted(extra)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("handoff schema mismatch")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("handoff run_id is invalid")
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("handoff status is invalid")
    blocked = status.startswith("blocked")
    for field in HASH_FIELDS:
        value = _validate_hash(payload.get(field), field, allow_null=blocked)
        if value is not None:
            expected_path = ROOT / HASH_FIELDS[field]
            if not expected_path.is_file():
                raise ValueError(f"handoff artifact missing: {HASH_FIELDS[field]}")
            if _sha256_file(expected_path) != value:
                raise ValueError(f"handoff stale hash: {field}")
    acceptance_hash = _validate_hash(payload.get("acceptance_packet_sha256"), "acceptance_packet_sha256", allow_null=False)
    assert acceptance_hash is not None
    acceptance_path = _acceptance_path(acceptance_hash)
    acceptance = _read(acceptance_path)
    if acceptance.get("run_id") != acceptance_path.parent.name:
        raise ValueError("handoff acceptance packet run_id mismatch")
    for field in ("acceptance_status", "visual_qa_status", "runtime_status", "determinism_status", "cloudflare_status"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"handoff {field} is invalid")
    if blocked != payload["runtime_status"].startswith("blocked") or blocked != (payload["acceptance_status"] == "blocked"):
        raise ValueError("handoff status does not match blocked statuses")
    if acceptance.get("status") != payload["acceptance_status"] or acceptance.get("runtime") != payload["runtime_status"]:
        raise ValueError("handoff acceptance statuses mismatch")
    for source_field, handoff_field in (("determinism", "determinism_status"), ("cloudflare", "cloudflare_status")):
        if source_field in acceptance and acceptance[source_field] != payload[handoff_field]:
            raise ValueError("handoff acceptance statuses mismatch")
    if payload.get("p1_status") != "not_promoted":
        raise ValueError("handoff p1_status must be not_promoted")
    if payload.get("test_v3") is not False:
        raise ValueError("handoff test_v3 is sealed false")
    if payload.get("ph2") is not False:
        raise ValueError("handoff ph2 is sealed false")
    jobs = payload.get("deferred_jobs")
    if jobs != list(DEFERRED_JOBS):
        raise ValueError("handoff deferred_jobs mismatch")
    if blocked:
        if payload.get("acceptance_status", "").startswith(("pass", "complete", "valid")):
            raise ValueError("blocked handoff cannot claim acceptance pass")
        if not isinstance(payload.get("reason"), str) or not isinstance(payload.get("artifact_class"), str):
            raise ValueError("blocked handoff requires reason and artifact_class")
    elif payload.get("reason") is not None or payload.get("artifact_class") is not None:
        raise ValueError("valid handoff cannot carry blocked reason")
    return FastReleaseHandoff(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        status=status,
        **{field: payload[field] for field in HASH_FIELDS},
        acceptance_packet_sha256=acceptance_hash,
        acceptance_status=payload["acceptance_status"],
        visual_qa_status=payload["visual_qa_status"],
        runtime_status=payload["runtime_status"],
        determinism_status=payload["determinism_status"],
        cloudflare_status=payload["cloudflare_status"],
        p1_status="not_promoted",
        test_v3=False,
        ph2=False,
        deferred_jobs=tuple(jobs),
        reason=payload.get("reason"),
        artifact_class=payload.get("artifact_class"),
        payload=payload,
    )


def write_fast_release_handoff(output_dir: Path, packet: Mapping[str, object]) -> Path:
    """Write one canonical append-only packet, refusing byte drift."""
    if not isinstance(packet, Mapping):
        raise ValueError("handoff packet must be a mapping")
    _reject_unsafe_fields(packet)
    run_id = packet.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("handoff run_id is invalid")
    target = Path(output_dir) / run_id / "handoff.json"
    encoded = _canonical(dict(packet))
    if target.exists():
        if target.read_bytes() != encoded:
            raise ValueError("handoff byte drift on rerun")
        return target
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_bytes(encoded)
    return target
