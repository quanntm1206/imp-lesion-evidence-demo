"""Original-geometry probability restoration and finite RQ1 metrics."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class MetricRow:
    dice: float
    iou: float
    precision: float
    recall: float
    boundary_f1: float
    hd95: float
    assd: float
    hd95_normalized: float
    assd_normalized: float
    empty_policy: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def restore_probability(probability: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly restore float probabilities before thresholding."""
    height, width = original_hw
    if height <= 0 or width <= 0:
        raise ValueError("original geometry must be positive")
    source = np.asarray(probability, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("probability must be a two-dimensional float array")
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.ascontiguousarray(source))[None, None]
    restored = functional.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )[0, 0]
    return np.ascontiguousarray(restored.detach().cpu().numpy().astype(np.float32))


def _binary(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    return value >= np.float32(0.5)


def _boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(
        mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    return mask & ~eroded.astype(bool)


def _distance(boundary: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 5)


def score(mask: np.ndarray, ground_truth: np.ndarray) -> MetricRow:
    """Score binary masks in original geometry with finite empty-mask penalties."""
    prediction = _binary(mask)
    target = _binary(ground_truth)
    if prediction.shape != target.shape:
        raise ValueError("prediction and ground truth geometry mismatch")
    height, width = prediction.shape
    diagonal = float(math.hypot(height, width))
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    if pred_count == 0 and target_count == 0:
        return MetricRow(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, "both_empty")
    if pred_count == 0 or target_count == 0:
        return MetricRow(0.0, 0.0, 0.0, 0.0, 0.0, diagonal, diagonal, 1.0, 1.0, "one_empty_diagonal_penalty")

    true_positive = float(np.logical_and(prediction, target).sum())
    union = float(np.logical_or(prediction, target).sum())
    dice = 2.0 * true_positive / float(pred_count + target_count)
    iou = true_positive / union
    precision = true_positive / float(pred_count)
    recall = true_positive / float(target_count)
    pred_boundary = _boundary(prediction)
    target_boundary = _boundary(target)
    pred_distance = _distance(target_boundary)
    target_distance = _distance(pred_boundary)
    distances = np.concatenate((pred_distance[pred_boundary], target_distance[target_boundary])).astype(np.float64)
    tolerance = np.float32(2.0)
    boundary_precision = float((pred_distance[pred_boundary] <= tolerance).mean()) if pred_boundary.any() else 0.0
    boundary_recall = float((target_distance[target_boundary] <= tolerance).mean()) if target_boundary.any() else 0.0
    boundary_f1 = 0.0 if boundary_precision + boundary_recall == 0.0 else 2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
    hd95 = float(np.percentile(distances, 95))
    assd = float(np.mean(distances))
    return MetricRow(
        float(dice), float(iou), float(precision), float(recall), float(boundary_f1),
        hd95, assd, hd95 / diagonal, assd / diagonal, None
    )
