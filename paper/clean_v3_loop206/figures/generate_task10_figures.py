"""Render the evidence-bounded Task 10 result figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple, Sequence

import matplotlib.pyplot as plt
import numpy as np

from lesion_robustness.demo.immutable_io import ImmutableSnapshot
from lesion_robustness.evidence_registry import validate_registry


PANEL_CAPTION = "illustrative; not protected-test evidence"
EVIDENCE_BADGE = [
    "train_screen",
    "exact_fixed_cache",
    "historical_cache_provenance_drift",
]
PROVENANCE_MANIFEST_SHA256 = (
    "4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102"
)
PDF_METADATA = {
    "Title": "Evidence-bounded Loop206 figures",
    "Author": "IMP Project",
    "Subject": "Train-screen evidence",
    "Keywords": "train-screen, negative ablation, qualitative evidence",
    "CreationDate": None,
    "ModDate": None,
}


class MetricDelta(NamedTuple):
    point_delta: float
    ci95: tuple[float, float]


class Loop206DeltaEvidence(NamedTuple):
    registry_sha256: str
    dice: MetricDelta
    boundary_f1: MetricDelta


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--receipt-bundle", type=Path, required=True)
    parser.add_argument("--expected-receipt-bundle-sha256", required=True)
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


def _finite(value: object, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _ci95(value: object, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must contain two bounds")
    lower = _finite(value[0], f"{field}.lower")
    upper = _finite(value[1], f"{field}.upper")
    if lower > upper:
        raise ValueError(f"{field} bounds are reversed")
    return lower, upper


def _exact_row(rows: object, *, field: str, value: str, label: str) -> dict:
    if not isinstance(rows, list):
        raise ValueError(f"evidence registry {label} rows are missing")
    matches = [row for row in rows if isinstance(row, dict) and row.get(field) == value]
    if len(matches) != 1:
        raise ValueError(f"Loop206 {label} is missing or duplicated")
    return matches[0]


def load_loop206_delta_evidence(path: str | Path) -> Loop206DeltaEvidence:
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError("evidence registry must contain one JSON object")
    validate_registry(payload)
    observation = _exact_row(
        payload.get("observations"),
        field="model_id",
        value="L206-contour-vs-control",
        label="observation",
    )
    comparison = _exact_row(
        payload.get("comparisons"),
        field="comparison_id",
        value="L206-contour-minus-control",
        label="comparison",
    )
    expected_observation = {
        "display_name": "Loop206 contour channel minus zero-channel control",
        "partition": "train_screen",
        "evidence_class": "train_screen",
        "metric_contract": "loop206_train_screen_three_corruptions_t2",
        "seed_count": 3,
        "group_count": 76,
        "corruption_count": 3,
        "bootstrap_resamples": 10000,
        "source_ids": ["loop206_report"],
    }
    if any(observation.get(key) != value for key, value in expected_observation.items()):
        raise ValueError("Loop206 observation semantic contract mismatch")
    if (
        comparison.get("evidence_class") != "train_screen"
        or comparison.get("metric") != "robust_dice"
        or comparison.get("source_ids") != ["loop206_report"]
    ):
        raise ValueError("Loop206 comparison semantic contract mismatch")

    dice = MetricDelta(
        _finite(observation.get("robust_dice_delta"), "robust_dice_delta"),
        _ci95(observation.get("robust_dice_ci95"), "robust_dice_ci95"),
    )
    boundary_f1 = MetricDelta(
        _finite(observation.get("robust_bf1_delta"), "robust_bf1_delta"),
        _ci95(observation.get("robust_bf1_ci95"), "robust_bf1_ci95"),
    )
    comparison_delta = _finite(comparison.get("point_delta"), "comparison.point_delta")
    comparison_ci = _ci95(comparison.get("ci95"), "comparison.ci95")
    if comparison_delta != dice.point_delta or comparison_ci != dice.ci95:
        raise ValueError("Loop206 comparison does not match its observation")
    return Loop206DeltaEvidence(
        registry_sha256=str(payload["registry_sha256"]),
        dice=dice,
        boundary_f1=boundary_f1,
    )


def _build_delta(evidence: Loop206DeltaEvidence, output: Path) -> None:
    metrics = ["Robust Dice", "Boundary F1"]
    points = np.asarray(
        [evidence.dice.point_delta, evidence.boundary_f1.point_delta]
    )
    lower = np.asarray([evidence.dice.ci95[0], evidence.boundary_f1.ci95[0]])
    upper = np.asarray([evidence.dice.ci95[1], evidence.boundary_f1.ci95[1]])
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
    axis.set_xlabel("Candidate minus control (95% split-group bootstrap CI; seeds fixed)")
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


def _load_bundle(
    snapshot: ImmutableSnapshot, *, expected_registry_sha256: str
) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    with np.load(snapshot.open(), allow_pickle=False) as bundle:
        arrays = {
            key: np.asarray(bundle[key])
            for key in ("original", "control", "candidate", "ground_truth")
        }
        receipts = [json.loads(str(value)) for value in bundle["receipt_json"]]
        display_authorization = json.loads(str(bundle["display_authorization_json"]))
    if any(value.shape[0] != 3 for value in arrays.values()) or len(receipts) != 3:
        raise ValueError("qualitative bundle must contain exactly three examples")
    sample_ids: set[str] = set()
    mask_bindings: list[dict[str, str]] = []
    for index, receipt in enumerate(receipts):
        if (
            receipt.get("schema_version") != "loop206.demo.receipt.v1"
            or receipt.get("mode") != "exact_fixed_cache"
            or receipt.get("evidence_badge") != EVIDENCE_BADGE
            or receipt.get("device") != "cuda"
            or receipt.get("historical_cache_provenance_drift") is not True
            or receipt.get("sample", {}).get("corruption") != "clean"
            or receipt.get("display_authorization") != display_authorization
            or receipt.get("evidence_registry_sha256") != expected_registry_sha256
        ):
            raise ValueError("qualitative receipt violates the evidence contract")
        sample_id = str(receipt["sample"]["sample_id"])
        group_key = str(receipt["sample"]["group_key"])
        sample_ids.add(sample_id)
        ground_truth_binding = receipt.get("ground_truth_binding")
        if not isinstance(ground_truth_binding, dict):
            raise ValueError("qualitative ground truth receipt binding is missing")
        mask_sha256_raw = str(ground_truth_binding.get("mask_sha256_raw", ""))
        mask_sha256_binary = str(
            ground_truth_binding.get("mask_sha256_binary", "")
        )
        mask_sha256_runtime = str(
            ground_truth_binding.get("mask_sha256_runtime", "")
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (mask_sha256_raw, mask_sha256_binary, mask_sha256_runtime)
        ):
            raise ValueError("qualitative ground truth receipt hash is invalid")
        if (
            ImmutableSnapshot.decoded_binary_mask_sha256(arrays["ground_truth"][index])
            != mask_sha256_runtime
        ):
            raise ValueError("qualitative ground truth receipt hash mismatch")
        mask_bindings.append(
            {
                "group_key": group_key,
                "sample_id": sample_id,
                "mask_sha256_raw": mask_sha256_raw,
                "mask_sha256_binary": mask_sha256_binary,
            }
        )
    if len(sample_ids) != 3:
        raise ValueError("qualitative examples must be unique")
    if (
        display_authorization.get("schema_version")
        != "loop206.qualitative_display_authorization.v1"
        or display_authorization.get("image_license") != "CC-0"
        or display_authorization.get("mask_variant") != "challenge_ground_truth"
        or display_authorization.get("provenance_manifest_sha256")
        != PROVENANCE_MANIFEST_SHA256
        or display_authorization.get("authorized_sample_count") != 3
        or display_authorization.get("hash_binding")
        != "sha256_raw+sha256_rgb+mask_sha256_raw+mask_sha256_binary"
    ):
        raise ValueError("qualitative display authorization is invalid")
    encoded_bindings = json.dumps(
        sorted(mask_bindings, key=lambda row: (row["sample_id"], row["group_key"])),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if display_authorization.get("mask_bindings_sha256") != hashlib.sha256(
        encoded_bindings
    ).hexdigest():
        raise ValueError("qualitative display mask binding mismatch")
    return arrays, receipts, display_authorization


def _build_qualitative(
    arrays: dict[str, np.ndarray], receipts: list[dict], output: Path
) -> None:
    figure, axes = plt.subplots(3, 5, figsize=(11.8, 8.1), constrained_layout=False)
    figure.subplots_adjust(left=0.025, right=0.995, top=0.875, bottom=0.055, wspace=0.035, hspace=0.43)
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
    caption_all_modality_panels(axes)
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


def caption_all_modality_panels(axes: np.ndarray) -> int:
    count = 0
    for axis in axes.flat:
        axis.text(
            0.5,
            -0.055,
            PANEL_CAPTION,
            transform=axis.transAxes,
            va="top",
            ha="center",
            fontsize=7.0,
            fontstretch="condensed",
            color="#4b5563",
        )
        count += 1
    if count != 15:
        raise ValueError("qualitative layout must contain exactly 15 modality panels")
    return count


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    evidence = load_loop206_delta_evidence(args.evidence_registry)
    bundle_snapshot = ImmutableSnapshot.read(args.receipt_bundle)
    bundle_sha256 = bundle_snapshot.sha256
    if bundle_sha256 != args.expected_receipt_bundle_sha256:
        raise ValueError("qualitative receipt bundle SHA-256 mismatch")
    arrays, receipts, display_authorization = _load_bundle(
        bundle_snapshot,
        expected_registry_sha256=evidence.registry_sha256,
    )
    _build_delta(evidence, args.output_dir / "loop206_delta.pdf")
    _build_qualitative(arrays, receipts, args.output_dir / "qualitative_demo.pdf")
    (args.output_dir / "qualitative_demo_receipts.json").write_text(
        json.dumps(
            {
                "schema_version": "loop206.qualitative_receipts.v1",
                "selection_rule": "first, middle, last after sorting all 76 train-screen rows by sample_id and group_key",
                "panel_caption": PANEL_CAPTION,
                "evidence_registry_sha256": evidence.registry_sha256,
                "runtime_bundle_sha256": bundle_sha256,
                "display_authorization": display_authorization,
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
