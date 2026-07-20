"""Capture deterministic, non-protected Loop206 fixed-cache examples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from lesion_robustness.demo.app import run_fixed_comparison
from lesion_robustness.demo.model_service import load_model_registry
from lesion_robustness.evidence_registry import validate_registry
from lesion_robustness.image_utils import read_mask, resize_image_and_mask


PROVENANCE_MANIFEST_SHA256 = (
    "4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-registry", type=Path, required=True)
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--zero-manifest", type=Path, required=True)
    parser.add_argument("--live-config", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def _safe_path(root: Path, relative: object) -> Path:
    path = (root / str(relative)).resolve()
    path.relative_to(root.resolve())
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_display_authorization(
    provenance_manifest: str | Path,
    selected_rows: Sequence[dict],
    *,
    expected_sha256: str = PROVENANCE_MANIFEST_SHA256,
) -> dict[str, object]:
    path = Path(provenance_manifest)
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("qualitative provenance manifest hash mismatch")
    with path.open(encoding="utf-8", newline="") as handle:
        provenance_rows = list(csv.DictReader(handle))
    if len(selected_rows) != 3:
        raise ValueError("qualitative provenance authorization requires three samples")

    for selected in selected_rows:
        sample_id = str(selected.get("sample_id", ""))
        matches = [row for row in provenance_rows if row.get("isic_image_id") == sample_id]
        if len(matches) != 1:
            raise ValueError("qualitative provenance sample is missing or duplicated")
        provenance = matches[0]
        if (
            provenance.get("original_id") != sample_id
            or provenance.get("source_dataset") != selected.get("source_dataset")
            or provenance.get("split") != selected.get("source_split")
        ):
            raise ValueError("qualitative provenance sample identity mismatch")
        if provenance.get("dataset_license") != "legacy_isic_challenge_terms":
            raise ValueError("qualitative provenance dataset license is not accepted")
        if provenance.get("image_license") != "CC-0":
            raise ValueError("qualitative provenance image license is not accepted")
        if provenance.get("mask_variant") != "challenge_ground_truth":
            raise ValueError("qualitative provenance ground truth is not authorized")
        if (
            provenance.get("sha256_raw") != selected.get("sha256_raw")
            or provenance.get("sha256_rgb") != selected.get("sha256_rgb")
        ):
            raise ValueError("qualitative provenance image hash binding mismatch")

    return {
        "schema_version": "loop206.qualitative_display_authorization.v1",
        "provenance_manifest_sha256": actual_sha256,
        "dataset_license": "legacy_isic_challenge_terms",
        "image_license": "CC-0",
        "mask_variant": "challenge_ground_truth",
        "identity_field": "isic_image_id",
        "hash_binding": "sha256_raw+sha256_rgb",
        "authorized_sample_count": 3,
    }


def _authorized_mask(raw_row: dict, roots: tuple[Path, ...]) -> np.ndarray:
    if (
        raw_row.get("role") != "holdout"
        or raw_row.get("split") != "train_screen_holdout"
        or raw_row.get("source_split") != "train"
    ):
        raise ValueError("qualitative ground truth is not train-screen authorized")
    root_index = int(raw_row["mask_root"])
    if root_index not in range(len(roots)):
        raise ValueError("qualitative mask root binding mismatch")
    mask = read_mask(_safe_path(roots[root_index], raw_row["mask_relative"]))
    blank = np.zeros((*mask.shape, 3), dtype=np.uint8)
    _, resized = resize_image_and_mask(blank, mask, (384, 384))
    if resized is None:
        raise ValueError("qualitative ground truth resize failed")
    return np.ascontiguousarray(resized, dtype=np.uint8)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    registry = json.loads(args.evidence_registry.read_text(encoding="ascii"))
    validate_registry(registry)
    index = json.loads(args.dataset_index.read_text(encoding="ascii"))
    roots = tuple(path.expanduser().resolve() for path in args.dataset_root)
    rows = sorted(
        (
            row
            for row in index["rows"]
            if row.get("role") == "holdout"
            and row.get("split") == "train_screen_holdout"
            and row.get("source_split") == "train"
        ),
        key=lambda row: (str(row["sample_id"]), str(row["group_key"])),
    )
    if len(rows) != 76:
        raise ValueError("qualitative selection requires the exact 76-row train screen")
    # Fixed positions prevent prediction- or metric-based qualitative selection.
    selected = [rows[0], rows[len(rows) // 2], rows[-1]]
    display_authorization = load_display_authorization(
        args.provenance_manifest,
        selected,
    )

    model_environment = {
        "IMP_LOOP206_CONTROL_CHECKPOINT": str(args.control_checkpoint),
        "IMP_LOOP206_CANDIDATE_CHECKPOINT": str(args.candidate_checkpoint),
    }
    loaded = load_model_registry(
        args.model_registry, environ=model_environment, device=args.device
    )
    provider = loaded.build_fixed_provider(
        dataset_index=args.dataset_index,
        dataset_roots=roots,
        candidate_manifest=args.candidate_manifest,
        zero_manifest=args.zero_manifest,
        live_config=args.live_config,
    )
    service = loaded.build_service(fixed_provider=provider)

    originals: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    ground_truths: list[np.ndarray] = []
    receipts: list[str] = []
    for row in selected:
        ground_truth = _authorized_mask(row, roots)
        response = run_fixed_comparison(
            service,
            registry,
            str(row["group_key"]),
            "clean",
            ground_truth,
        )
        if not response.ok or response.mode != "exact_fixed_cache":
            raise RuntimeError("real fixed-cache GPU comparison failed closed")
        if response.receipt is None or response.receipt.get("device") != "cuda":
            raise RuntimeError("qualitative receipt is not real CUDA evidence")
        originals.append(np.asarray(response.original_rgb, dtype=np.uint8))
        controls.append(np.asarray(response.control_mask, dtype=np.uint8))
        candidates.append(np.asarray(response.candidate_mask, dtype=np.uint8))
        ground_truths.append(ground_truth)
        receipt = dict(response.receipt)
        receipt["display_authorization"] = display_authorization
        receipts.append(json.dumps(receipt, sort_keys=True, separators=(",", ":")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        original=np.stack(originals),
        control=np.stack(controls),
        candidate=np.stack(candidates),
        ground_truth=np.stack(ground_truths),
        receipt_json=np.asarray(receipts),
        display_authorization_json=np.asarray(
            json.dumps(display_authorization, sort_keys=True, separators=(",", ":"))
        ),
        selection_rule=np.asarray(
            "first, middle, last after sorting all 76 train-screen rows by sample_id and group_key"
        ),
    )


if __name__ == "__main__":
    main()
