"""Capture deterministic, non-protected Loop206 fixed-cache examples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from lesion_robustness.demo.app import run_fixed_comparison
from lesion_robustness.demo.immutable_io import ImmutableSnapshot
from lesion_robustness.demo.loop206_prior import load_dataset_index
from lesion_robustness.demo.model_service import load_model_registry
from lesion_robustness.evidence_registry import validate_registry


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


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_display_authorization(
    provenance_manifest: str | Path,
    selected_rows: Sequence[dict],
    *,
    expected_sha256: str = PROVENANCE_MANIFEST_SHA256,
) -> dict[str, object]:
    path = Path(provenance_manifest)
    snapshot = ImmutableSnapshot.read(path)
    actual_sha256 = snapshot.sha256
    if actual_sha256 != expected_sha256:
        raise ValueError("qualitative provenance manifest hash mismatch")
    with io.StringIO(snapshot.text("utf-8"), newline="") as handle:
        provenance_rows = list(csv.DictReader(handle))
    if len(selected_rows) != 3:
        raise ValueError("qualitative provenance authorization requires three samples")

    mask_bindings: list[dict[str, str]] = []
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
        mask_sha256_raw = str(selected.get("mask_sha256_raw", ""))
        mask_sha256_binary = str(selected.get("mask_sha256_binary", ""))
        if not _is_sha256(mask_sha256_raw) or not _is_sha256(mask_sha256_binary):
            raise ValueError("qualitative provenance mask hash binding mismatch")
        mask_bindings.append(
            {
                "group_key": str(selected.get("group_key", "")),
                "sample_id": sample_id,
                "mask_sha256_raw": mask_sha256_raw,
                "mask_sha256_binary": mask_sha256_binary,
            }
        )

    return {
        "schema_version": "loop206.qualitative_display_authorization.v1",
        "provenance_manifest_sha256": actual_sha256,
        "dataset_license": "legacy_isic_challenge_terms",
        "image_license": "CC-0",
        "mask_variant": "challenge_ground_truth",
        "identity_field": "isic_image_id",
        "hash_binding": "sha256_raw+sha256_rgb+mask_sha256_raw+mask_sha256_binary",
        "mask_bindings_sha256": _canonical_hash(
            sorted(mask_bindings, key=lambda row: (row["sample_id"], row["group_key"]))
        ),
        "authorized_sample_count": 3,
    }


def load_qualitative_selection(
    dataset_index: str | Path, roots: Sequence[Path]
) -> tuple[list[dict], list[np.ndarray]]:
    _, holdout, index = load_dataset_index(dataset_index, dataset_roots=roots)
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
    typed = {row.group_key: row for row in holdout}
    if len(typed) != 76 or set(typed) != {str(row["group_key"]) for row in rows}:
        raise ValueError("qualitative verified holdout binding mismatch")
    selected = [rows[0], rows[len(rows) // 2], rows[-1]]
    masks: list[np.ndarray] = []
    for row in selected:
        verified = typed[str(row["group_key"])]
        if verified.mask is None:
            raise ValueError("qualitative verified ground truth is missing")
        masks.append(np.ascontiguousarray(verified.mask, dtype=np.uint8).copy())
    return selected, masks


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    registry = json.loads(args.evidence_registry.read_text(encoding="ascii"))
    validate_registry(registry)
    roots = tuple(path.expanduser().resolve() for path in args.dataset_root)
    selected, verified_masks = load_qualitative_selection(args.dataset_index, roots)
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
    for row, ground_truth in zip(selected, verified_masks, strict=True):
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
