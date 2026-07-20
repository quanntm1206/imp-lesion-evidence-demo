"""Capture deterministic, non-protected Loop206 fixed-cache examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from lesion_robustness.demo.app import run_fixed_comparison
from lesion_robustness.demo.model_service import load_model_registry
from lesion_robustness.evidence_registry import validate_registry
from lesion_robustness.image_utils import read_mask, resize_image_and_mask


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-registry", type=Path, required=True)
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
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
        receipts.append(
            json.dumps(response.receipt, sort_keys=True, separators=(",", ":"))
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        original=np.stack(originals),
        control=np.stack(controls),
        candidate=np.stack(candidates),
        ground_truth=np.stack(ground_truths),
        receipt_json=np.asarray(receipts),
        selection_rule=np.asarray(
            "first, middle, last after sorting all 76 train-screen rows by sample_id and group_key"
        ),
    )


if __name__ == "__main__":
    main()
