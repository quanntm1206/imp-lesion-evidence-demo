"""Validate one release authority and write its checked-in projections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from lesion_robustness.release_manifest import (
    DEFAULT_MANIFEST,
    deck_projection,
    load_release_manifest,
    registry_projection,
    sidecar_model_manifest_projection,
    validate_projection,
)
from lesion_robustness.evidence_registry import _registry_hash
from lesion_robustness.demo.live_inputs import recompute_public_selection


ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = ROOT.parents[1]
LOOP192_REPORT = BASE_ROOT / ".artifacts/preprocessing_search/current_bdou_loop192_nnunet_clean_v3_report.json"
LOOP192_REPORT_SHA256 = "852a67544ad34d64139dbf913e740e4866e5f72dfe10a1d89d4f573f1711d064"
LOOP192_TRAIN_CASE_KEY_SHA256 = "a3a1e6c2f5e8d0800d6c78df27b280266e009ebe9b4cd270da25c21d1b37f388"
TRAINING_EXPOSURE = {
    "L206-control-s206": "excluded_from_308_fit_in_76_group_train_screen_holdout",
    "L192-nnUNet-v2-raw-100ep": "included_in_clean_v3_2008_training_rows",
}
CASE_KEY_FIELDS = ("original_id", "split_group", "image_path", "mask_path")


def _raw_csv_row_hash(path: Path, row_number: int) -> str:
    lines = path.read_bytes().splitlines()
    if row_number < 1 or row_number > len(lines):
        raise ValueError("public sample CSV row is unavailable")
    return hashlib.sha256(lines[row_number - 1]).hexdigest()


def _train_case_key(rows: list[dict[str, str]]) -> tuple[set[str], str]:
    digest = hashlib.sha256()
    sample_ids: set[str] = set()
    for row in rows:
        if row.get("split") != "train":
            continue
        sample_id = row.get("original_id", "")
        if not sample_id or sample_id in sample_ids:
            raise ValueError("public sample Loop192 training membership mismatch")
        sample_ids.add(sample_id)
        try:
            encoded = "\x1f".join(row[field] for field in CASE_KEY_FIELDS).encode("utf-8")
        except KeyError as exc:
            raise ValueError("public sample Loop192 training metadata mismatch") from exc
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return sample_ids, digest.hexdigest()


def verify_public_sample_evidence(
    manifest_path: Path,
    *,
    dataset_index: Path,
    clean_v3_manifest: Path,
    loop192_report: Path = LOOP192_REPORT,
) -> None:
    """Admit public samples only when source CSV and deterministic selection agree."""
    manifest = load_release_manifest(manifest_path)
    selection = recompute_public_selection(dataset_index, manifest)
    if selection.ordered_universe_sha256 != manifest.public_samples.selection["ordered_universe_sha256"]:
        raise ValueError("public sample ordered universe mismatch")
    csv_hash = hashlib.sha256(clean_v3_manifest.read_bytes()).hexdigest()
    with clean_v3_manifest.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_ids, train_case_key = _train_case_key(rows)
    report_bytes = loop192_report.read_bytes()
    try:
        report = json.loads(report_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("public sample Loop192 report is invalid") from exc
    provenance = report.get("provenance", {})
    if (
        hashlib.sha256(report_bytes).hexdigest() != LOOP192_REPORT_SHA256
        or report.get("candidate_id") != "L192-nnUNet-v2-raw-100ep"
        or report.get("test_opened") is not False
        or provenance.get("manifest_sha256") != csv_hash
        or provenance.get("train_case_key_sha256") != LOOP192_TRAIN_CASE_KEY_SHA256
        or train_case_key != LOOP192_TRAIN_CASE_KEY_SHA256
    ):
        raise ValueError("public sample Loop192 training metadata mismatch")
    for sample in manifest.public_samples.samples:
        evidence = sample.license_evidence
        if dict(sample.training_exposure) != TRAINING_EXPOSURE:
            raise ValueError("public sample training exposure mismatch")
        if csv_hash != evidence.clean_v3_manifest_sha256 or _raw_csv_row_hash(clean_v3_manifest, evidence.csv_row_number) != evidence.raw_csv_row_sha256:
            raise ValueError("public sample license evidence mismatch")
        row = rows[evidence.csv_row_number - 2]
        if (
            row.get("original_id") != sample.sample_id
            or row.get("source_dataset") != sample.source_dataset
            or row.get("image_license") != sample.license_id
            or row.get("sha256_raw") != sample.sha256_raw
            or row.get("sha256_rgb") != sample.sha256_rgb
            or row.get("split") != "train"
            or sample.sample_id not in train_ids
        ):
            raise ValueError("public sample CSV provenance mismatch")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
    )


def build_sidecar(manifest_path: Path = DEFAULT_MANIFEST) -> Path:
    manifest = load_release_manifest(manifest_path)
    sidecar = ROOT / "sidecar" / "nnunet" / "model_manifest.example.json"
    _write_json(sidecar, sidecar_model_manifest_projection(manifest.path))
    validate_projection(sidecar, manifest.path)
    return sidecar


def build(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    dataset_index: Path = BASE_ROOT / "demo_runtime" / "loop206_dataset_index.json",
    clean_v3_manifest: Path = BASE_ROOT / "data" / "splits" / "clean_v3_manifest.csv",
) -> tuple[Path, ...]:
    manifest = load_release_manifest(manifest_path)
    verify_public_sample_evidence(
        manifest.path, dataset_index=dataset_index, clean_v3_manifest=clean_v3_manifest
    )
    registry = ROOT / "demo" / "model_registry.example.json"
    _write_json(registry, registry_projection(manifest.path))

    evidence = ROOT / "demo" / "data" / "evidence_registry.json"
    evidence_payload = json.loads(evidence.read_text(encoding="ascii"))
    evidence_payload["release_manifest_sha256"] = manifest.digest
    evidence_payload["registry_sha256"] = _registry_hash(evidence_payload)
    _write_json(evidence, evidence_payload)

    deck = ROOT / "presentation" / "interactive" / "content.json"
    deck_payload = json.loads(deck.read_text(encoding="utf-8"))
    deck_payload["release_manifest_sha256"] = manifest.digest
    deck_payload["release_comparisons"] = deck_projection(manifest.path)["comparisons"]
    _write_json(deck, deck_payload)
    sidecar = build_sidecar(manifest.path)
    for artifact in (registry, evidence, deck, sidecar):
        validate_projection(artifact, manifest.path)
    return registry, evidence, deck, sidecar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-index", type=Path, default=BASE_ROOT / "demo_runtime" / "loop206_dataset_index.json")
    parser.add_argument("--clean-v3-manifest", type=Path, default=BASE_ROOT / "data" / "splits" / "clean_v3_manifest.csv")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sidecar-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        verify_public_sample_evidence(args.manifest, dataset_index=args.dataset_index, clean_v3_manifest=args.clean_v3_manifest)
        for artifact in (
            ROOT / "demo" / "model_registry.example.json",
            ROOT / "demo" / "data" / "evidence_registry.json",
            ROOT / "presentation" / "interactive" / "content.json",
            ROOT / "sidecar" / "nnunet" / "model_manifest.example.json",
        ):
            validate_projection(artifact, args.manifest)
        return 0
    if args.sidecar_only:
        build_sidecar(args.manifest)
        return 0
    build(args.manifest, dataset_index=args.dataset_index, clean_v3_manifest=args.clean_v3_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
