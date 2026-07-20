"""Train-only saliency-constrained local-edge active contour for Loop206."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from lesion_robustness.metrics import segmentation_metrics


SEED = 206
CORRUPTIONS = ("clean", "low_contrast", "gaussian_noise")
HARD_GATES = {
    "bf1_delta_overall_min": 0.040,
    "bf1_delta_each_corruption_min": 0.020,
    "bootstrap_bf1_lower_strict_min": 0.0,
    "dice_delta_min": -0.010,
    "precision_delta_min": 0.020,
    "recall_delta_min": -0.015,
    "auroc_regression_max": 0.010,
    "auprc_regression_max": 0.010,
    "fallback_rate_max_each_corruption": 0.25,
    "invalid_border_rate_max_each_corruption": 0.05,
    "assd_relative_improvement_min": 0.10,
    "hd95_relative_regression_max": 0.02,
    "runtime_median_seconds_max": 1.0,
    "runtime_p95_seconds_max": 2.0,
}


class Loop206ProtocolError(ValueError):
    """Raised when a Loop206 scientific or numerical invariant is violated."""


@dataclass(frozen=True)
class ActiveContourConfig:
    name: str
    iterations: int
    smoothing: int
    balloon: int
    init_threshold_offset: float
    gradient_sigma: float = 1.5
    gradient_alpha: float = 12.0
    saliency_edge_weight: float = 0.35
    low_probability: float = 0.01
    high_probability: float = 0.75
    border_margin: int = 1
    min_area_scale: float = 0.20
    max_area_scale: float = 2.00


CANDIDATE_CONFIGS = (
    ActiveContourConfig("shrink_low_20_s1", 20, 1, -1, 0.00),
    ActiveContourConfig("shrink_low_30_s2", 30, 2, -1, 0.00),
    ActiveContourConfig("shrink_mid_25_s1", 25, 1, -1, 0.025),
    ActiveContourConfig("neutral_mid_20_s1", 20, 1, 0, 0.025),
    ActiveContourConfig("neutral_mid_30_s2", 30, 2, 0, 0.050),
    ActiveContourConfig("expand_core_25_s1", 25, 1, 1, 0.100),
)


def _validate_inputs(image: np.ndarray, probability: np.ndarray, threshold: float) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise Loop206ProtocolError("Loop206 image must be RGB uint8")
    if probability.shape != image.shape[:2] or not np.isfinite(probability).all():
        raise Loop206ProtocolError("Loop206 probability map must be finite and shape-matched")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise Loop206ProtocolError("Loop206 probability map must be in [0, 1]")
    if not 0.0 < float(threshold) < 1.0:
        raise Loop206ProtocolError("Loop206 initialization threshold must be in (0, 1)")


def _validate_config(config: ActiveContourConfig) -> None:
    if not config.name or config.iterations < 1 or config.smoothing < 0:
        raise Loop206ProtocolError("active-contour iteration/smoothing config is invalid")
    if config.balloon not in {-1, 0, 1}:
        raise Loop206ProtocolError("active-contour balloon must be -1, 0, or 1")
    if config.gradient_sigma <= 0.0 or config.gradient_alpha <= 0.0:
        raise Loop206ProtocolError("active-contour gradient config is invalid")
    if not 0.0 <= config.saliency_edge_weight <= 1.0:
        raise Loop206ProtocolError("saliency edge weight must be in [0, 1]")
    if not 0.0 <= config.low_probability < config.high_probability <= 1.0:
        raise Loop206ProtocolError("saliency barrier probabilities are invalid")
    if config.border_margin < 0 or config.min_area_scale <= 0.0:
        raise Loop206ProtocolError("active-contour topology config is invalid")
    if config.max_area_scale <= config.min_area_scale:
        raise Loop206ProtocolError("active-contour max area scale must exceed min area scale")


def local_edge_indicator(
    image: np.ndarray,
    probability: np.ndarray,
    *,
    config: ActiveContourConfig,
) -> np.ndarray:
    """Build a Lab edge indicator modulated by the deployable saliency map."""

    _validate_config(config)
    _validate_inputs(image, probability, 0.5)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    sigma = float(config.gradient_sigma)
    lab = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gradients = []
    for channel in range(3):
        gx = cv2.Sobel(lab[..., channel], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[..., channel], cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(gx * gx + gy * gy)
    magnitude = np.sqrt(np.sum(gradients, axis=0)).astype(np.float32)
    scale = float(np.quantile(magnitude, 0.95))
    if scale > 1e-8:
        magnitude /= scale
    edge = 1.0 / np.sqrt(1.0 + float(config.gradient_alpha) * magnitude * magnitude)
    barrier = np.clip(
        (probability - float(config.low_probability))
        / max(float(config.high_probability - config.low_probability), 1e-6),
        0.0,
        1.0,
    )
    weight = float(config.saliency_edge_weight)
    indicator = edge * ((1.0 - weight) + weight * barrier)
    return np.clip(indicator, 1e-4, 1.0).astype(np.float32)


def _max_saliency_component(mask: np.ndarray, probability: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask.astype(bool, copy=False)
    scores = [float(probability[labels == index].sum()) for index in range(1, count)]
    selected = int(np.argmax(scores)) + 1
    return labels == selected


def refine_active_contour(
    image: np.ndarray,
    probability: np.ndarray,
    base_threshold: float,
    *,
    config: ActiveContourConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine an automatic saliency mask without using a ground-truth prompt."""

    _validate_config(config)
    probability = np.asarray(probability, dtype=np.float32)
    _validate_inputs(image, probability, base_threshold)
    initial_threshold = float(
        np.clip(base_threshold + config.init_threshold_offset, 0.001, 0.999)
    )
    baseline = probability >= float(base_threshold)
    initial = probability >= initial_threshold
    if not initial.any() or not baseline.any():
        return baseline.astype(np.uint8), {
            "fallback_used": True,
            "fallback_reason": "empty_initial_or_baseline",
            "initial_threshold": initial_threshold,
        }
    indicator = local_edge_indicator(image, probability, config=config)
    try:
        from skimage.segmentation import morphological_geodesic_active_contour

        refined = morphological_geodesic_active_contour(
            indicator,
            int(config.iterations),
            init_level_set=initial.astype(np.int8),
            smoothing=int(config.smoothing),
            threshold="auto",
            balloon=int(config.balloon),
        ).astype(bool)
    except Exception as exc:
        return baseline.astype(np.uint8), {
            "fallback_used": True,
            "fallback_reason": f"active_contour_error:{type(exc).__name__}",
            "initial_threshold": initial_threshold,
        }

    refined &= probability >= float(config.low_probability)
    refined |= probability >= float(config.high_probability)
    margin = int(config.border_margin)
    if margin > 0:
        border = np.zeros(refined.shape, dtype=bool)
        border[:margin, :] = True
        border[-margin:, :] = True
        border[:, :margin] = True
        border[:, -margin:] = True
        refined[border & (probability < config.high_probability)] = False
    refined = binary_fill_holes(refined).astype(bool)
    refined = _max_saliency_component(refined, probability)
    baseline_area = int(baseline.sum())
    area = int(refined.sum())
    area_scale = area / max(1, baseline_area)
    fallback = (
        area == 0
        or area_scale < float(config.min_area_scale)
        or area_scale > float(config.max_area_scale)
    )
    if fallback:
        refined = baseline
    return refined.astype(np.uint8), {
        "fallback_used": bool(fallback),
        "fallback_reason": "area_guard" if fallback else "none",
        "initial_threshold": initial_threshold,
        "area_scale": float(area_scale),
    }


def _border_fraction(mask: np.ndarray) -> float:
    border = np.zeros(mask.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    return float(np.asarray(mask, dtype=bool)[border].mean())


def evaluate_refinement(
    image: np.ndarray,
    probability: np.ndarray,
    target: np.ndarray,
    base_threshold: float,
    *,
    config: ActiveContourConfig,
    runtime_seconds: float = 0.0,
) -> dict[str, Any]:
    """Compare the refined binary mask against the unrefined Loop205 mask."""

    target_b = np.asarray(target) > 0
    if target_b.shape != probability.shape or not target_b.any() or target_b.all():
        raise Loop206ProtocolError("Loop206 target must contain lesion/background")
    refined, metadata = refine_active_contour(
        image, probability, base_threshold, config=config
    )
    baseline = np.asarray(probability) >= float(base_threshold)
    base_metrics = segmentation_metrics(
        baseline,
        target_b,
        include_boundary=True,
        boundary_tolerance=2,
        empty_boundary_distance_policy="image_diagonal",
    )
    refined_metrics = segmentation_metrics(
        refined,
        target_b,
        include_boundary=True,
        boundary_tolerance=2,
        empty_boundary_distance_policy="image_diagonal",
    )
    base_assd = float(base_metrics["assd"])
    base_hd95 = float(base_metrics["hd95"])
    assd_relative_improvement = (base_assd - float(refined_metrics["assd"])) / max(
        base_assd, 1e-6
    )
    hd95_relative_regression = (float(refined_metrics["hd95"]) - base_hd95) / max(
        base_hd95, 1e-6
    )
    return {
        "config": config.name,
        "baseline": base_metrics,
        "refined": refined_metrics,
        "dice_delta": float(refined_metrics["dice"] - base_metrics["dice"]),
        "precision_delta": float(
            refined_metrics["precision"] - base_metrics["precision"]
        ),
        "recall_delta": float(refined_metrics["recall"] - base_metrics["recall"]),
        "boundary_f1_delta": float(
            refined_metrics["boundary_f1"] - base_metrics["boundary_f1"]
        ),
        "assd_relative_improvement": float(assd_relative_improvement),
        "hd95_relative_regression": float(hd95_relative_regression),
        "auroc_regression": 0.0,
        "auprc_regression": 0.0,
        "fallback_used": bool(metadata["fallback_used"]),
        "fallback_reason": metadata["fallback_reason"],
        "invalid_border": _border_fraction(refined) > 0.5,
        "invalid_border_fraction": _border_fraction(refined),
        "runtime_seconds": float(runtime_seconds),
        "metadata": metadata,
    }


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise Loop206ProtocolError(f"cannot summarize {key}")
    return float(np.median(values))


def summarize_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise Loop206ProtocolError("cannot summarize empty Loop206 metrics")
    per_corruption: dict[str, Any] = {}
    for corruption in CORRUPTIONS:
        subset = [row for row in rows if row.get("corruption") == corruption]
        if not subset:
            raise Loop206ProtocolError(f"missing Loop206 corruption {corruption}")
        per_corruption[corruption] = {
            "count": len(subset),
            "median_boundary_f1_delta": _median(subset, "boundary_f1_delta"),
            "median_dice_delta": _median(subset, "dice_delta"),
            "median_precision_delta": _median(subset, "precision_delta"),
            "median_recall_delta": _median(subset, "recall_delta"),
            "median_assd_relative_improvement": _median(
                subset, "assd_relative_improvement"
            ),
            "median_hd95_relative_regression": _median(
                subset, "hd95_relative_regression"
            ),
            "fallback_rate": float(np.mean([bool(row["fallback_used"]) for row in subset])),
            "invalid_border_rate": float(
                np.mean([bool(row["invalid_border"]) for row in subset])
            ),
            "runtime_median_seconds": _median(subset, "runtime_seconds"),
            "runtime_p95_seconds": float(
                np.quantile([float(row["runtime_seconds"]) for row in subset], 0.95)
            ),
        }
    return {
        "overall": {
            "count": len(rows),
            "median_boundary_f1_delta": _median(rows, "boundary_f1_delta"),
            "median_dice_delta": _median(rows, "dice_delta"),
            "median_precision_delta": _median(rows, "precision_delta"),
            "median_recall_delta": _median(rows, "recall_delta"),
            "median_assd_relative_improvement": _median(
                rows, "assd_relative_improvement"
            ),
            "median_hd95_relative_regression": _median(
                rows, "hd95_relative_regression"
            ),
            "max_auroc_regression": max(float(row["auroc_regression"]) for row in rows),
            "max_auprc_regression": max(float(row["auprc_regression"]) for row in rows),
            "fallback_rate": float(np.mean([bool(row["fallback_used"]) for row in rows])),
            "invalid_border_rate": float(
                np.mean([bool(row["invalid_border"]) for row in rows])
            ),
            "runtime_median_seconds": _median(rows, "runtime_seconds"),
            "runtime_p95_seconds": float(
                np.quantile([float(row["runtime_seconds"]) for row in rows], 0.95)
            ),
        },
        "corruptions": per_corruption,
    }


def select_config(
    cases: Sequence[Mapping[str, Any]],
    *,
    configs: Sequence[ActiveContourConfig] = CANDIDATE_CONFIGS,
) -> dict[str, Any]:
    """Select one contour policy from OOF maps outside the held-out fold."""

    results: list[dict[str, Any]] = []
    for config in configs:
        rows = []
        for case in cases:
            metric = evaluate_refinement(
                case["image"],
                case["probability"],
                case["mask"],
                float(case["base_threshold"]),
                config=config,
            )
            metric["corruption"] = case["corruption"]
            rows.append(metric)
        summary = summarize_metrics(rows)
        overall = summary["overall"]
        feasible = (
            overall["median_boundary_f1_delta"] >= 0.020
            and overall["median_dice_delta"] >= -0.010
            and overall["median_precision_delta"] >= 0.010
            and overall["median_recall_delta"] >= -0.015
            and all(
                summary["corruptions"][name]["fallback_rate"] <= 0.25
                and summary["corruptions"][name]["invalid_border_rate"] <= 0.05
                for name in CORRUPTIONS
            )
        )
        score = (
            overall["median_boundary_f1_delta"]
            + 0.50 * overall["median_precision_delta"]
            + 0.25 * overall["median_dice_delta"]
            + 0.15 * overall["median_assd_relative_improvement"]
            - 0.20 * overall["fallback_rate"]
            - 0.20 * overall["invalid_border_rate"]
        )
        results.append(
            {
                "config": config,
                "summary": summary,
                "feasible": bool(feasible),
                "score": float(score),
            }
        )
    feasible_results = [item for item in results if item["feasible"]]
    selected = max(feasible_results or results, key=lambda item: (item["score"], item["config"].name))
    return {
        "selected_config": selected["config"],
        "selected_summary": selected["summary"],
        "selected_score": selected["score"],
        "feasible_config_found": bool(feasible_results),
        "ranking": [
            {
                "config": item["config"].name,
                "score": item["score"],
                "feasible": item["feasible"],
                "overall": item["summary"]["overall"],
            }
            for item in sorted(results, key=lambda item: item["score"], reverse=True)
        ],
    }


def bootstrap_bf1_delta(
    rows: Sequence[Mapping[str, Any]], *, iterations: int = 2000, seed: int = SEED
) -> dict[str, float | int]:
    by_sample: dict[str, list[float]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample_id"]), []).append(
            float(row["boundary_f1_delta"])
        )
    values = np.asarray(
        [float(np.mean(by_sample[key])) for key in sorted(by_sample)], dtype=np.float64
    )
    if values.size < 2 or iterations < 100:
        raise Loop206ProtocolError("BF1 bootstrap requires >=2 samples and >=100 iterations")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        estimates[index] = float(np.median(values[rng.integers(0, values.size, values.size)]))
    return {
        "seed": int(seed),
        "iterations": int(iterations),
        "estimate": float(np.median(values)),
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
    }


def evaluate_hard_gates(
    summary: Mapping[str, Any], bootstrap: Mapping[str, Any]
) -> dict[str, Any]:
    overall = summary["overall"]
    corruptions = summary["corruptions"]
    results = {
        "bf1_delta_overall": {
            "passed": float(overall["median_boundary_f1_delta"])
            >= HARD_GATES["bf1_delta_overall_min"],
            "observed": float(overall["median_boundary_f1_delta"]),
            "required": ">= 0.040",
        },
        "bf1_delta_each_corruption": {
            "passed": all(
                float(corruptions[name]["median_boundary_f1_delta"])
                >= HARD_GATES["bf1_delta_each_corruption_min"]
                for name in CORRUPTIONS
            ),
            "observed": {
                name: float(corruptions[name]["median_boundary_f1_delta"])
                for name in CORRUPTIONS
            },
            "required": ">= 0.020 each corruption",
        },
        "bootstrap_bf1_lower": {
            "passed": float(bootstrap["lower"])
            > HARD_GATES["bootstrap_bf1_lower_strict_min"],
            "observed": float(bootstrap["lower"]),
            "required": "> 0",
        },
        "dice_delta": {
            "passed": float(overall["median_dice_delta"])
            >= HARD_GATES["dice_delta_min"],
            "observed": float(overall["median_dice_delta"]),
            "required": ">= -0.010",
        },
        "precision_delta": {
            "passed": float(overall["median_precision_delta"])
            >= HARD_GATES["precision_delta_min"],
            "observed": float(overall["median_precision_delta"]),
            "required": ">= 0.020",
        },
        "recall_delta": {
            "passed": float(overall["median_recall_delta"])
            >= HARD_GATES["recall_delta_min"],
            "observed": float(overall["median_recall_delta"]),
            "required": ">= -0.015",
        },
        "probability_ranking": {
            "passed": float(overall["max_auroc_regression"])
            <= HARD_GATES["auroc_regression_max"]
            and float(overall["max_auprc_regression"])
            <= HARD_GATES["auprc_regression_max"],
            "observed": {
                "auroc_regression": float(overall["max_auroc_regression"]),
                "auprc_regression": float(overall["max_auprc_regression"]),
            },
            "required": "AUROC/AUPRC regression <= 0.010",
        },
        "fallback_each_corruption": {
            "passed": all(
                float(corruptions[name]["fallback_rate"])
                <= HARD_GATES["fallback_rate_max_each_corruption"]
                for name in CORRUPTIONS
            ),
            "observed": {
                name: float(corruptions[name]["fallback_rate"]) for name in CORRUPTIONS
            },
            "required": "<= 0.25 each corruption",
        },
        "invalid_border_each_corruption": {
            "passed": all(
                float(corruptions[name]["invalid_border_rate"])
                <= HARD_GATES["invalid_border_rate_max_each_corruption"]
                for name in CORRUPTIONS
            ),
            "observed": {
                name: float(corruptions[name]["invalid_border_rate"])
                for name in CORRUPTIONS
            },
            "required": "<= 0.05 each corruption",
        },
        "assd_relative_improvement": {
            "passed": float(overall["median_assd_relative_improvement"])
            >= HARD_GATES["assd_relative_improvement_min"],
            "observed": float(overall["median_assd_relative_improvement"]),
            "required": ">= 0.10",
        },
        "hd95_relative_regression": {
            "passed": float(overall["median_hd95_relative_regression"])
            <= HARD_GATES["hd95_relative_regression_max"],
            "observed": float(overall["median_hd95_relative_regression"]),
            "required": "<= 0.02",
        },
        "runtime_median": {
            "passed": float(overall["runtime_median_seconds"])
            <= HARD_GATES["runtime_median_seconds_max"],
            "observed": float(overall["runtime_median_seconds"]),
            "required": "<= 1.0 seconds/image",
        },
        "runtime_p95": {
            "passed": float(overall["runtime_p95_seconds"])
            <= HARD_GATES["runtime_p95_seconds_max"],
            "observed": float(overall["runtime_p95_seconds"]),
            "required": "<= 2.0 seconds/image",
        },
    }
    return {
        "passed": all(bool(item["passed"]) for item in results.values()),
        "results": results,
        "thresholds": dict(HARD_GATES),
    }
