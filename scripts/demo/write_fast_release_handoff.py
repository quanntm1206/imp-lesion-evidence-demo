from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from lesion_robustness.demo.fast_release_handoff import (
    DEFERRED_JOBS,
    SCHEMA_VERSION,
    validate_fast_release_handoff,
    write_fast_release_handoff,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _artifact(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value or ":" in value:
        raise ValueError(f"{label} path is invalid")
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    return path


def _pinned_hash(path: Path, expected: object, label: str) -> str:
    actual = _sha256(path)
    if expected != actual:
        raise ValueError(f"{label} hash is stale")
    return actual


def build_packet(
    *,
    run_id: str,
    release_manifest: Path,
    paper_manifest: Path,
    presentation_manifest: Path,
    acceptance_packet: Path,
    visual_review: Path,
) -> dict[str, object]:
    release_hash = _sha256(release_manifest)
    paper = _json_object(paper_manifest, "paper artifact manifest")
    presentation = _json_object(presentation_manifest, "presentation manifest")
    acceptance = _json_object(acceptance_packet, "acceptance packet")
    visual = _json_object(visual_review, "visual review packet")

    acceptance_run_id = acceptance.get("run_id")
    if not isinstance(acceptance_run_id, str) or acceptance_run_id != acceptance_packet.parent.name:
        raise ValueError("acceptance packet run_id mismatch")
    if paper.get("release_manifest_sha256") != release_hash:
        raise ValueError("paper artifact manifest release hash is stale")
    if presentation.get("current_release_manifest_sha256") != release_hash:
        raise ValueError("presentation manifest release hash is stale")

    paper_pdf = paper.get("paper_pdf")
    if not isinstance(paper_pdf, dict):
        raise ValueError("paper PDF manifest is invalid")
    paper_pdf_path = _artifact(paper_manifest.parent, paper_pdf.get("path"), "paper PDF")
    paper_pdf_hash = _pinned_hash(paper_pdf_path, paper_pdf.get("sha256"), "paper PDF")
    paper_pdf_status = paper_pdf.get("status")
    if not isinstance(paper_pdf_status, str) or not paper_pdf_status:
        raise ValueError("paper PDF status is invalid")

    raw_files = presentation.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("presentation manifest files are invalid")
    files: dict[str, dict[str, object]] = {}
    for item in raw_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("presentation manifest file is invalid")
        files[Path(item["path"]).suffix.lower()] = item
    if set(files) != {".html", ".pptx", ".pdf"}:
        raise ValueError("presentation artifact set mismatch")

    presentation_hashes: dict[str, str] = {}
    for suffix, field in ((".html", "html_sha256"), (".pptx", "pptx_sha256"), (".pdf", "presentation_pdf_sha256")):
        item = files[suffix]
        artifact = _artifact(ROOT, item.get("path"), f"presentation {suffix}")
        presentation_hashes[field] = _pinned_hash(artifact, item.get("sha256"), f"presentation {suffix}")

    runtime = acceptance.get("runtime")
    acceptance_status = acceptance.get("status")
    cloudflare = acceptance.get("cloudflare")
    if not isinstance(runtime, str) or not isinstance(acceptance_status, str):
        raise ValueError("acceptance packet status is invalid")
    blocked = acceptance_status == "blocked" or runtime.startswith("blocked")
    determinism = acceptance.get("determinism")
    if determinism is None and blocked:
        determinism = "blocked"
    if not isinstance(determinism, str):
        raise ValueError("acceptance determinism status is missing")
    if not isinstance(cloudflare, str):
        raise ValueError("acceptance cloudflare status is missing")
    visual_status = visual.get("visual_qa")
    if not isinstance(visual_status, str) or not visual_status:
        raise ValueError("visual QA status is invalid")

    packet: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": runtime if blocked else "valid",
        "release_manifest_sha256": release_hash,
        "paper_artifact_manifest_sha256": _sha256(paper_manifest),
        "paper_pdf_sha256": paper_pdf_hash,
        "paper_pdf_status": paper_pdf_status,
        "presentation_manifest_sha256": _sha256(presentation_manifest),
        **presentation_hashes,
        "acceptance_packet_sha256": _sha256(acceptance_packet),
        "acceptance_status": acceptance_status,
        "visual_qa_status": visual_status,
        "runtime_status": runtime,
        "determinism_status": determinism,
        "cloudflare_status": cloudflare,
        "p1_status": "not_promoted",
        "test_v3": False,
        "ph2": False,
        "deferred_jobs": list(DEFERRED_JOBS),
    }
    if blocked:
        reason = acceptance.get("reason")
        artifact_class = acceptance.get("artifact_class")
        if not isinstance(reason, str) or not isinstance(artifact_class, str):
            raise ValueError("blocked acceptance requires reason and artifact_class")
        packet.update(reason=reason, artifact_class=artifact_class)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--paper-manifest", type=Path, required=True)
    parser.add_argument("--presentation-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-packet", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, default=Path("outputs/visual-evidence/visual-review.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    packet = build_packet(
        run_id=args.run_id,
        release_manifest=args.release_manifest,
        paper_manifest=args.paper_manifest,
        presentation_manifest=args.presentation_manifest,
        acceptance_packet=args.acceptance_packet,
        visual_review=args.visual_review,
    )
    output = write_fast_release_handoff(args.output_root, packet)
    handoff = validate_fast_release_handoff(output)
    print(f"handoff_status={handoff.status}")
    print(f"path={output.as_posix()}")
    print(f"sha256={_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
