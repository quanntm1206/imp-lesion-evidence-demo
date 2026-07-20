"""Render the evidence-bounded Task 10 result figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


PANEL_CAPTION = "illustrative; not protected-test evidence"
EVIDENCE_BADGE = [
    "train_screen",
    "exact_fixed_cache",
    "historical_cache_provenance_drift",
]
PDF_METADATA = {
    "Title": "Evidence-bounded Loop206 figures",
    "Author": "IMP Project",
    "Subject": "Train-screen evidence",
    "Keywords": "train-screen, negative ablation, qualitative evidence",
    "CreationDate": None,
    "ModDate": None,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )


def _build_delta(output: Path) -> None:
    metrics = ["Robust Dice", "Boundary F1"]
    points = np.asarray([-0.0313, -0.0147])
    lower = np.asarray([-0.0491, -0.0308])
    upper = np.asarray([-0.0156, 0.0010])
    positions = np.asarray([1, 0])

    figure, axis = plt.subplots(figsize=(6.8, 2.35), constrained_layout=True)
    axis.axvline(0, color="#4b5563", linewidth=0.9, linestyle=(0, (3, 3)))
    axis.errorbar(
        points,
        positions,
        xerr=np.vstack((points - lower, upper - points)),
        fmt="o",
        color="#216b64",
        ecolor="#216b64",
        elinewidth=1.6,
        capsize=4,
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.7,
    )
    for point, low, high, y in zip(points, lower, upper, positions):
        axis.text(
            high + 0.0018,
            y,
            f"{point:+.4f}  [{low:+.4f}, {high:+.4f}]",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#1d211e",
        )
    axis.set_yticks(positions, metrics)
    axis.set_ylim(-0.55, 1.55)
    axis.set_xlim(-0.057, 0.022)
    axis.set_xlabel("Candidate minus control (95% paired cluster-bootstrap CI)")
    axis.set_title("Loop206 train-screen paired deltas", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#d8d4ca", linewidth=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    figure.savefig(output, format="pdf", metadata=PDF_METADATA)
    plt.close(figure)


def _disagreement_overlay(
    original: np.ndarray, control: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    overlay = original.astype(np.float32).copy()
    control_only = (control > 0) & (candidate == 0)
    candidate_only = (candidate > 0) & (control == 0)
    overlay[control_only] = 0.32 * overlay[control_only] + 0.68 * np.asarray(
        [159, 63, 44], dtype=np.float32
    )
    overlay[candidate_only] = 0.32 * overlay[candidate_only] + 0.68 * np.asarray(
        [33, 107, 100], dtype=np.float32
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _load_bundle(path: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {
            key: np.asarray(bundle[key])
            for key in ("original", "control", "candidate", "ground_truth")
        }
        receipts = [json.loads(str(value)) for value in bundle["receipt_json"]]
    if any(value.shape[0] != 3 for value in arrays.values()) or len(receipts) != 3:
        raise ValueError("qualitative bundle must contain exactly three examples")
    sample_ids: set[str] = set()
    for receipt in receipts:
        if (
            receipt.get("schema_version") != "loop206.demo.receipt.v1"
            or receipt.get("mode") != "exact_fixed_cache"
            or receipt.get("evidence_badge") != EVIDENCE_BADGE
            or receipt.get("device") != "cuda"
            or receipt.get("historical_cache_provenance_drift") is not True
            or receipt.get("sample", {}).get("corruption") != "clean"
        ):
            raise ValueError("qualitative receipt violates the evidence contract")
        sample_ids.add(str(receipt["sample"]["sample_id"]))
    if len(sample_ids) != 3:
        raise ValueError("qualitative examples must be unique")
    return arrays, receipts


def _build_qualitative(
    arrays: dict[str, np.ndarray], receipts: list[dict], output: Path
) -> None:
    figure, axes = plt.subplots(3, 5, figsize=(11.8, 7.35), constrained_layout=False)
    figure.subplots_adjust(left=0.025, right=0.995, top=0.875, bottom=0.045, wspace=0.035, hspace=0.28)
    titles = [
        "Original",
        "Control mask",
        "Candidate mask",
        "Disagreement overlay",
        "Ground truth (authorized)",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontweight="bold", pad=6)
    figure.suptitle(
        "Loop206 fixed-cache qualitative comparison\n"
        "train_screen / exact_fixed_cache / historical_cache_provenance_drift",
        x=0.025,
        y=0.985,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color="#1d211e",
    )
    for row in range(3):
        original = arrays["original"][row]
        control = arrays["control"][row]
        candidate = arrays["candidate"][row]
        ground_truth = arrays["ground_truth"][row]
        panels = [
            original,
            control,
            candidate,
            _disagreement_overlay(original, control, candidate),
            ground_truth,
        ]
        for column, panel in enumerate(panels):
            axis = axes[row, column]
            display = panel
            if panel.ndim == 2 and int(panel.max()) <= 1:
                display = panel * 255
            axis.imshow(display, cmap=None if panel.ndim == 3 else "gray", vmin=0, vmax=255)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#9f9480")
                spine.set_linewidth(0.55)
        sample_id = str(receipts[row]["sample"]["sample_id"])
        axes[row, 0].text(
            0.015,
            0.975,
            f"({chr(97 + row)}) {sample_id}",
            transform=axes[row, 0].transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            color="white",
            bbox={"facecolor": "#1d211e", "edgecolor": "none", "pad": 2.2, "alpha": 0.88},
        )
        axes[row, 2].text(
            0.5,
            -0.065,
            PANEL_CAPTION,
            transform=axes[row, 2].transAxes,
            va="top",
            ha="center",
            fontsize=7.2,
            color="#4b5563",
        )
    figure.text(
        0.995,
        0.012,
        "Disagreement: rust = control only; teal = candidate only.",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#4b5563",
    )
    figure.savefig(output, format="pdf", dpi=240, metadata=PDF_METADATA)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    arrays, receipts = _load_bundle(args.receipt_bundle)
    _build_delta(args.output_dir / "loop206_delta.pdf")
    _build_qualitative(arrays, receipts, args.output_dir / "qualitative_demo.pdf")
    (args.output_dir / "qualitative_demo_receipts.json").write_text(
        json.dumps(
            {
                "schema_version": "loop206.qualitative_receipts.v1",
                "selection_rule": "first, middle, last after sorting all 76 train-screen rows by sample_id and group_key",
                "panel_caption": PANEL_CAPTION,
                "receipts": receipts,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
