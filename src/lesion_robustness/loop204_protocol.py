"""Train-only PCDS-MR feasibility protocol for Loop204.

The module is intentionally evaluation-only: it reads immutable image/mask pairs,
builds deterministic corruptions, and never imports or invokes training code.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from lesion_robustness.image_utils import read_mask, read_rgb
from lesion_robustness.corruptions import apply_corruption
from lesion_robustness.pcds_mr import PCDSMRConfig, apply_pcds_mr
from lesion_robustness.preprocessing import preprocess_image_from_config, saliency_proxy


SEED = 204
MAX_SAMPLES = 256
CORRUPTIONS = ("clean", "low_contrast", "gaussian_noise")
BASE_PREPROCESSING = {
    "normalize": True,
    "contrast_stretch": {"enabled": True, "lower_percentile": 1.0, "upper_percentile": 99.0},
    "clahe": {"enabled": True, "color_space": "lab", "clip_limit": 2.0, "tile_grid_size": [8, 8]},
    "filter": {"type": "median", "kernel_size": 3},
    "color_constancy": {"enabled": False},
    "gamma": {"enabled": False},
    "hair_removal": {"enabled": False},
    "sharpen": {"enabled": False},
    "extra_channel": {"enabled": False},
}
PROTECTED_SPLITS = frozenset(
    {"val", "validation", "test", "test_v3", "ph2", "external", "external_audit"}
)
GROUP_FIELDS = (
    "clean_v3_component_id",
    "split_group",
    "patient_id",
    "lesion_id",
    "duplicate_group",
)
SAMPLE_ID_FIELDS = ("sample_id", "original_id", "isic_image_id", "image_id")
HARD_GATES = {
    "median_auroc_delta_min": 0.03,
    "bootstrap_ci_lower_strict_min": 0.0,
    "median_coverage_min_each_corruption": 0.97,
    "median_support_area_ratio_max": 0.65,
    "fallback_rate_max_each_corruption": 0.45,
    "fallback_improvement_vs_loop203_min": 0.15,
    "invalid_border_rate_max": 0.05,
    "median_mass_concentration_delta_min": 0.08,
    "corruption_auroc_regression_max": 0.05,
    "redundancy_correlation_max": 0.95,
}


class Loop204ProtocolError(ValueError):
    """Raised when a Loop204 safety, input, or metric invariant is violated."""


@dataclass(frozen=True)
class PCDSMRMaps:
    pcds_raw: np.ndarray
    support: np.ndarray
    pcds_mr: np.ndarray
    post_modulation: np.ndarray
    fallback_used: bool
    fallback_reason: str
    invalid_border: bool
    invalid_border_fraction: float


def _expanded_bbox_mask(mask: np.ndarray, margin_fraction: float = 0.20) -> np.ndarray:
    support = np.asarray(mask, dtype=bool)
    ys, xs = np.where(support)
    output = np.zeros(support.shape, dtype=bool)
    if ys.size == 0:
        return output
    height, width = support.shape
    box_height = int(ys.max() - ys.min() + 1)
    box_width = int(xs.max() - xs.min() + 1)
    margin_y = round(box_height * float(margin_fraction))
    margin_x = round(box_width * float(margin_fraction))
    y0, y1 = max(0, int(ys.min()) - margin_y), min(height, int(ys.max()) + 1 + margin_y)
    x0, x1 = max(0, int(xs.min()) - margin_x), min(width, int(xs.max()) + 1 + margin_x)
    output[y0:y1, x0:x1] = True
    return output


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pcds_config_sha256(config: PCDSMRConfig) -> str:
    return sha256_bytes(canonical_json_bytes(asdict(config)))


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object) -> str:
    return str(value or "").strip()


def _stable_digest(seed: int, *values: object) -> bytes:
    joined = "::".join([str(int(seed)), *(_text(value) for value in values)])
    return hashlib.sha256(joined.encode("utf-8")).digest()


def _sample_id(row: Mapping[str, object]) -> str:
    for field in SAMPLE_ID_FIELDS:
        value = _text(row.get(field))
        if value:
            return value
    image_path = _text(row.get("image_path"))
    if image_path:
        return Path(image_path).stem
    raise Loop204ProtocolError("manifest row has no stable sample identifier")


def _reject_protected_source(row: Mapping[str, object]) -> None:
    for field in ("split", "source_split", "dataset", "source_dataset", "cohort"):
        value = _text(row.get(field)).lower()
        tokens = {token for token in value.replace("-", "_").split("_") if token}
        if value in PROTECTED_SPLITS or "ph2" in tokens:
            raise Loop204ProtocolError(
                f"Loop204 train-only screen rejects protected source {field}={value!r}"
            )


def validate_train_only_rows(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise Loop204ProtocolError("Loop204 train-only screen received no rows")
    seen_ids: set[str] = set()
    for row in rows:
        split = _text(row.get("split")).lower()
        if split != "train":
            raise Loop204ProtocolError(f"Loop204 fit/screen rejects non-train split {split!r}")
        _reject_protected_source(row)
        sample_id = _sample_id(row)
        if sample_id in seen_ids:
            raise Loop204ProtocolError(f"duplicate Loop204 sample identifier: {sample_id}")
        seen_ids.add(sample_id)
        for field in ("image_path", "mask_path"):
            if not _text(row.get(field)):
                raise Loop204ProtocolError(f"Loop204 row {sample_id!r} is missing {field}")
        if not any(_text(row.get(field)) for field in GROUP_FIELDS):
            raise Loop204ProtocolError(
                f"Loop204 row {sample_id!r} has no patient/group identity"
            )


def read_clean_v3_manifest(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise Loop204ProtocolError("Clean-v3 manifest has no header")
        rows = [dict(row) for row in reader]
    if not rows:
        raise Loop204ProtocolError("Clean-v3 manifest is empty")
    if "split" not in reader.fieldnames:
        raise Loop204ProtocolError("Clean-v3 manifest is missing split")
    return rows


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _sampling_components(rows: Sequence[Mapping[str, object]]) -> list[list[int]]:
    dsu = _DisjointSet(len(rows))
    owners: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for field in GROUP_FIELDS:
            value = _text(row.get(field))
            if not value:
                continue
            key = (field, value)
            previous = owners.get(key)
            if previous is None:
                owners[key] = index
            else:
                dsu.union(index, previous)
    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(dsu.find(index), []).append(index)
    return list(components.values())


def select_train_samples(
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    seed: int = SEED,
    max_samples: int = MAX_SAMPLES,
) -> list[dict[str, Any]]:
    """Select one deterministic representative per linked patient/group component."""

    if int(max_samples) < 1 or int(max_samples) > MAX_SAMPLES:
        raise Loop204ProtocolError(f"max_samples must be in [1, {MAX_SAMPLES}]")
    train_rows = [dict(row) for row in manifest_rows if _text(row.get("split")).lower() == "train"]
    if not train_rows:
        raise Loop204ProtocolError("Clean-v3 manifest has no train rows")
    validate_train_only_rows(train_rows)

    selected: list[dict[str, Any]] = []
    for component in _sampling_components(train_rows):
        ranked = sorted(
            component,
            key=lambda index: (
                _stable_digest(seed, "row", _sample_id(train_rows[index])),
                _sample_id(train_rows[index]),
            ),
        )
        row = dict(train_rows[ranked[0]])
        member_ids = sorted(_sample_id(train_rows[index]) for index in component)
        component_hash = sha256_bytes(canonical_json_bytes(member_ids))
        row["loop204_sampling_group"] = component_hash
        row["loop204_component_size"] = len(component)
        selected.append(row)

    selected.sort(
        key=lambda row: (
            _stable_digest(seed, "group", row["loop204_sampling_group"]),
            _sample_id(row),
        )
    )
    selected = selected[: int(max_samples)]
    validate_train_only_rows(selected)
    return selected


def resolve_manifest_path(manifest_path: str | Path, value: object) -> Path:
    raw = Path(_text(value)).expanduser()
    if not raw.is_absolute():
        raw = Path(manifest_path).resolve().parent / raw
    return raw.resolve()


def build_corruptions(
    image: np.ndarray,
    sample_id: str,
    *,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise Loop204ProtocolError("Loop204 images must be RGB uint8")
    noise_seed = int.from_bytes(_stable_digest(seed, "noise", sample_id)[:8], "big")
    return {
        "clean": value.copy(),
        "low_contrast": apply_corruption(value, "low_contrast", factor=0.5),
        "gaussian_noise": apply_corruption(value, "gaussian_noise", sigma=0.05, seed=noise_seed),
    }


def _normalize_map(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise Loop204ProtocolError("saliency maps must be finite HxW arrays")
    low, high = float(value.min()), float(value.max())
    if high - low <= 1e-8:
        return np.zeros_like(value, dtype=np.float32)
    return ((value - low) / (high - low)).astype(np.float32, copy=False)


def compute_pcds_mr(
    image: np.ndarray, *, config: PCDSMRConfig = PCDSMRConfig()
) -> PCDSMRMaps:
    """Return diagnostic maps from the exact production PCDS-MR transform."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise Loop204ProtocolError("PCDS-MR requires RGB uint8 input")
    result = apply_pcds_mr(rgb, config=config)
    raw = _normalize_map(result.score)
    support = result.support_mask.astype(bool, copy=False)
    mr_map = _normalize_map(raw * support.astype(np.float32))
    post = _normalize_map(saliency_proxy(result.image))
    invalid_border = result.metadata.mode == "local_only" and not result.metadata.applied
    return PCDSMRMaps(
        pcds_raw=raw.astype(np.float32, copy=False),
        support=support.astype(bool, copy=False),
        pcds_mr=mr_map.astype(np.float32, copy=False),
        post_modulation=post.astype(np.float32, copy=False),
        fallback_used=not bool(result.metadata.applied),
        fallback_reason=result.metadata.fallback_reason or "none",
        invalid_border=bool(invalid_border),
        invalid_border_fraction=1.0 - float(result.metadata.border_valid_fraction),
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    sorted_values = flat[order]
    ranks = np.empty(flat.size, dtype=np.float64)
    start = 0
    while start < flat.size:
        end = start + 1
        while end < flat.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pixel_auroc(target: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(target, dtype=bool).reshape(-1)
    scores = np.asarray(score, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or not np.isfinite(scores).all():
        raise Loop204ProtocolError("AUROC target/score mismatch or non-finite score")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise Loop204ProtocolError("pixel AUROC requires lesion and background pixels")
    positive_rank_sum = float(_rankdata(scores)[labels].sum())
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks, right_ranks = _rankdata(left), _rankdata(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return 1.0 if np.array_equal(left_ranks, right_ranks) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def saliency_mass_concentration(target: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(target, dtype=bool)
    values = np.clip(np.asarray(score, dtype=np.float64), 0.0, None)
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    return float(values[labels].sum() / total)


def downsample_metric_inputs(
    mask: np.ndarray, maps: Mapping[str, np.ndarray], *, max_side: int = 128
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target = np.asarray(mask) > 0
    if target.ndim != 2:
        raise Loop204ProtocolError("metric mask must be HxW")
    height, width = target.shape
    scale = min(1.0, float(max_side) / max(height, width))
    output_shape = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized_mask = cv2.resize(target.astype(np.uint8), output_shape, interpolation=cv2.INTER_NEAREST) > 0
    resized_maps = {}
    for name, value in maps.items():
        interpolation = cv2.INTER_NEAREST if name == "pcds_mr_support" else cv2.INTER_AREA
        resized_maps[name] = cv2.resize(
            np.asarray(value, dtype=np.float32), output_shape, interpolation=interpolation
        )
    return resized_mask, resized_maps


def evaluate_maps(
    mask: np.ndarray,
    old_proxy: np.ndarray,
    maps: PCDSMRMaps,
    *,
    max_side: int = 128,
) -> dict[str, Any]:
    target, values = downsample_metric_inputs(
        mask,
        {
            "saliency_proxy": _normalize_map(old_proxy),
            "pcds_raw": maps.pcds_raw,
            "pcds_mr": maps.pcds_mr,
            "post_modulation": maps.post_modulation,
            "pcds_mr_support": maps.support.astype(np.float32),
        },
        max_side=max_side,
    )
    if not np.any(target) or np.all(target):
        raise Loop204ProtocolError("downsampled diagnostic mask must contain lesion and background")
    map_names = (
        "saliency_proxy",
        "pcds_raw",
        "pcds_mr_support",
        "pcds_mr",
        "post_modulation",
    )
    map_metrics = {}
    for name in map_names:
        map_metrics[name] = {
            "pixel_auroc": pixel_auroc(target, values[name]),
            "saliency_mass_concentration": saliency_mass_concentration(target, values[name]),
            "spearman_vs_old_proxy": 1.0
            if name == "saliency_proxy"
            else spearman_correlation(values["saliency_proxy"], values[name]),
        }
    crop_support = _expanded_bbox_mask(values["pcds_mr_support"] >= 0.5)
    coverage = float(crop_support[target].mean())
    support_ratio = float(crop_support.mean())
    correlation = spearman_correlation(values["saliency_proxy"], values["pcds_mr"])
    crop_fallback = bool(
        not crop_support.any()
        or coverage < HARD_GATES["median_coverage_min_each_corruption"]
        or support_ratio > HARD_GATES["median_support_area_ratio_max"]
    )
    return {
        "metric_shape": [int(target.shape[0]), int(target.shape[1])],
        "maps": map_metrics,
        "lesion_coverage": coverage,
        "support_area_ratio": support_ratio,
        "spearman_vs_old_proxy": correlation,
        "auroc_delta": map_metrics["pcds_mr"]["pixel_auroc"]
        - map_metrics["saliency_proxy"]["pixel_auroc"],
        "mass_concentration_delta": map_metrics["pcds_mr"][
            "saliency_mass_concentration"
        ]
        - map_metrics["saliency_proxy"]["saliency_mass_concentration"],
        "fallback_used": crop_fallback,
        "transform_fallback_used": maps.fallback_used,
        "fallback_reason": maps.fallback_reason,
        "invalid_border": maps.invalid_border,
        "invalid_border_fraction": maps.invalid_border_fraction,
    }


def bootstrap_paired_auroc_delta(
    old_auroc: Sequence[float],
    candidate_auroc: Sequence[float],
    *,
    seed: int = SEED,
    iterations: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    old = np.asarray(old_auroc, dtype=np.float64)
    candidate = np.asarray(candidate_auroc, dtype=np.float64)
    if old.ndim != 1 or candidate.shape != old.shape or old.size < 2:
        raise Loop204ProtocolError("paired AUROC bootstrap requires >=2 matched observations")
    if not np.isfinite(old).all() or not np.isfinite(candidate).all():
        raise Loop204ProtocolError("paired AUROC bootstrap requires finite values")
    if int(iterations) < 100:
        raise Loop204ProtocolError("paired AUROC bootstrap requires at least 100 iterations")
    deltas = candidate - old
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        sampled = rng.integers(0, deltas.size, size=deltas.size)
        estimates[index] = float(np.median(deltas[sampled]))
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "seed": int(seed),
        "iterations": int(iterations),
        "confidence": float(confidence),
        "estimate": float(np.median(deltas)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
    }


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise Loop204ProtocolError(f"cannot summarize empty/non-finite metric {key}")
    return float(np.median(values))


def summarize_screen(
    sample_metrics: Sequence[Mapping[str, Any]],
    *,
    loop203_fallback_baseline: float,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    if not 0.0 <= float(loop203_fallback_baseline) <= 1.0:
        raise Loop204ProtocolError("Loop203 fallback baseline must be in [0, 1]")
    expected = set(CORRUPTIONS)
    by_sample: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in sample_metrics:
        sample_id, corruption = _text(row.get("sample_id")), _text(row.get("corruption"))
        if not sample_id or corruption not in expected:
            raise Loop204ProtocolError("invalid sample metric identity/corruption")
        if corruption in by_sample.setdefault(sample_id, {}):
            raise Loop204ProtocolError(f"duplicate metric for {sample_id}/{corruption}")
        by_sample[sample_id][corruption] = row
    if len(by_sample) < 2:
        raise Loop204ProtocolError("Loop204 screen requires at least two sampled groups")
    for sample_id, views in by_sample.items():
        if set(views) != expected:
            raise Loop204ProtocolError(f"sample {sample_id!r} lacks matched corruptions")

    corruption_summary: dict[str, Any] = {}
    for corruption in CORRUPTIONS:
        rows = [views[corruption] for views in by_sample.values()]
        corruption_summary[corruption] = {
            "count": len(rows),
            "median_auroc": _median(
                [
                    {"value": row["maps"]["pcds_mr"]["pixel_auroc"]}
                    for row in rows
                ],
                "value",
            ),
            "median_old_proxy_auroc": _median(
                [{"value": row["maps"]["saliency_proxy"]["pixel_auroc"]} for row in rows],
                "value",
            ),
            "median_auroc_delta": _median(rows, "auroc_delta"),
            "median_coverage": _median(rows, "lesion_coverage"),
            "median_support_area_ratio": _median(rows, "support_area_ratio"),
            "median_mass_concentration_delta": _median(rows, "mass_concentration_delta"),
            "median_spearman_vs_old_proxy": _median(rows, "spearman_vs_old_proxy"),
            "fallback_rate": float(np.mean([bool(row["fallback_used"]) for row in rows])),
            "invalid_border_rate": float(np.mean([bool(row["invalid_border"]) for row in rows])),
            "median_invalid_border_fraction": _median(rows, "invalid_border_fraction"),
        }

    all_rows = [views[corruption] for views in by_sample.values() for corruption in CORRUPTIONS]
    paired_old, paired_candidate = [], []
    for views in by_sample.values():
        paired_old.append(
            float(np.mean([views[name]["maps"]["saliency_proxy"]["pixel_auroc"] for name in CORRUPTIONS]))
        )
        paired_candidate.append(
            float(np.mean([views[name]["maps"]["pcds_mr"]["pixel_auroc"] for name in CORRUPTIONS]))
        )
    bootstrap = bootstrap_paired_auroc_delta(
        paired_old, paired_candidate, seed=SEED, iterations=bootstrap_iterations
    )
    overall = {
        "sample_count": len(by_sample),
        "metric_row_count": len(all_rows),
        "median_auroc_delta": _median(all_rows, "auroc_delta"),
        "median_support_area_ratio": _median(all_rows, "support_area_ratio"),
        "median_mass_concentration_delta": _median(all_rows, "mass_concentration_delta"),
        "median_spearman_vs_old_proxy": _median(all_rows, "spearman_vs_old_proxy"),
        "fallback_rate": float(np.mean([bool(row["fallback_used"]) for row in all_rows])),
        "invalid_border_rate": float(np.mean([bool(row["invalid_border"]) for row in all_rows])),
    }
    return {
        "overall": overall,
        "corruptions": corruption_summary,
        "paired_auroc_delta_bootstrap": bootstrap,
        "loop203_fallback_baseline": float(loop203_fallback_baseline),
    }


def evaluate_hard_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    overall = summary["overall"]
    corruptions = summary["corruptions"]
    bootstrap = summary["paired_auroc_delta_bootstrap"]
    baseline = float(summary["loop203_fallback_baseline"])
    fallback_limit = min(
        HARD_GATES["fallback_rate_max_each_corruption"],
        baseline - HARD_GATES["fallback_improvement_vs_loop203_min"],
    )
    clean_auroc = float(corruptions["clean"]["median_auroc"])
    regressions = {
        name: clean_auroc - float(corruptions[name]["median_auroc"])
        for name in CORRUPTIONS
        if name != "clean"
    }
    no_gain = (
        float(overall["median_auroc_delta"]) <= 0.0
        and float(overall["median_mass_concentration_delta"]) <= 0.0
    )
    redundant = (
        float(overall["median_spearman_vs_old_proxy"])
        > HARD_GATES["redundancy_correlation_max"]
        and no_gain
    )

    results = {
        "median_auroc_delta": {
            "passed": float(overall["median_auroc_delta"])
            >= HARD_GATES["median_auroc_delta_min"],
            "observed": float(overall["median_auroc_delta"]),
            "required": ">= 0.03",
        },
        "paired_auroc_delta_ci_lower": {
            "passed": float(bootstrap["lower"])
            > HARD_GATES["bootstrap_ci_lower_strict_min"],
            "observed": float(bootstrap["lower"]),
            "required": "> 0",
        },
        "coverage_each_corruption": {
            "passed": all(
                float(corruptions[name]["median_coverage"])
                >= HARD_GATES["median_coverage_min_each_corruption"]
                for name in CORRUPTIONS
            ),
            "observed": {
                name: float(corruptions[name]["median_coverage"]) for name in CORRUPTIONS
            },
            "required": ">= 0.97 each corruption",
        },
        "support_area_ratio": {
            "passed": float(overall["median_support_area_ratio"])
            <= HARD_GATES["median_support_area_ratio_max"],
            "observed": float(overall["median_support_area_ratio"]),
            "required": "<= 0.65",
        },
        "fallback_each_corruption": {
            "passed": fallback_limit >= 0.0
            and all(
                float(corruptions[name]["fallback_rate"]) <= fallback_limit
                for name in CORRUPTIONS
            ),
            "observed": {
                name: float(corruptions[name]["fallback_rate"]) for name in CORRUPTIONS
            },
            "required": f"<= 0.45 and <= Loop203 baseline - 0.15 ({fallback_limit:.6f})",
        },
        "invalid_border": {
            "passed": float(overall["invalid_border_rate"])
            <= HARD_GATES["invalid_border_rate_max"],
            "observed": float(overall["invalid_border_rate"]),
            "required": "<= 0.05",
        },
        "mass_concentration_delta": {
            "passed": float(overall["median_mass_concentration_delta"])
            >= HARD_GATES["median_mass_concentration_delta_min"],
            "observed": float(overall["median_mass_concentration_delta"]),
            "required": ">= 0.08",
        },
        "corruption_auroc_regression": {
            "passed": all(
                value <= HARD_GATES["corruption_auroc_regression_max"]
                for value in regressions.values()
            ),
            "observed": regressions,
            "required": "<= 0.05 versus clean",
        },
        "old_proxy_redundancy": {
            "passed": not redundant,
            "observed": {
                "median_spearman": float(overall["median_spearman_vs_old_proxy"]),
                "no_auroc_or_mass_gain": no_gain,
            },
            "required": "reject only when correlation > 0.95 and neither AUROC nor mass improves",
        },
    }
    return {
        "passed": all(bool(result["passed"]) for result in results.values()),
        "results": results,
        "thresholds": dict(HARD_GATES),
    }


def evaluate_sample(
    row: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    config: PCDSMRConfig = PCDSMRConfig(),
    max_side: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_train_only_rows([row])
    sample_id = _sample_id(row)
    image_path = resolve_manifest_path(manifest_path, row["image_path"])
    mask_path = resolve_manifest_path(manifest_path, row["mask_path"])
    image, mask = read_rgb(image_path), read_mask(mask_path)
    if image.shape[:2] != mask.shape:
        raise Loop204ProtocolError(f"image/mask shape mismatch for {sample_id}")
    if not np.any(mask) or np.all(mask):
        raise Loop204ProtocolError(f"diagnostic mask must contain lesion/background for {sample_id}")
    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST).astype(bool)

    metrics: list[dict[str, Any]] = []
    views = build_corruptions(image, sample_id)
    if tuple(views) != CORRUPTIONS:
        raise Loop204ProtocolError("corruption order/set changed")
    for corruption in CORRUPTIONS:
        view = preprocess_image_from_config(views[corruption], BASE_PREPROCESSING)
        old = saliency_proxy(view)
        maps = compute_pcds_mr(view, config=config)
        metric = evaluate_maps(mask, old, maps, max_side=max_side)
        metric.update(
            {
                "sample_id": sample_id,
                "sampling_group": _text(row["loop204_sampling_group"]),
                "split": "train",
                "corruption": corruption,
            }
        )
        metrics.append(metric)
    provenance = {
        "sample_id": sample_id,
        "sampling_group": _text(row["loop204_sampling_group"]),
        "component_size": int(row.get("loop204_component_size", 1)),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "image_sha256": sha256_file(image_path),
        "mask_sha256": sha256_file(mask_path),
    }
    return metrics, provenance


def sampled_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = [
        {
            "sample_id": _sample_id(row),
            "sampling_group": _text(row.get("loop204_sampling_group")),
            "image_path": _text(row.get("image_path")),
            "mask_path": _text(row.get("mask_path")),
            "split": _text(row.get("split")).lower(),
        }
        for row in rows
    ]
    return sha256_bytes(canonical_json_bytes(payload))
