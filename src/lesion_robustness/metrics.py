from __future__ import annotations

import numpy as np


def _binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(mask) > threshold).astype(bool)


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask_b = _binary(mask)
    if not mask_b.any():
        return np.zeros_like(mask_b, dtype=bool)
    import cv2

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(
        mask_b.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return np.logical_and(mask_b, ~eroded)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    import cv2

    diameter = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def boundary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: int = 2,
    empty_distance_policy: str = "infinite",
) -> dict[str, float]:
    """Compute boundary-sensitive metrics for binary segmentation masks.

    Metrics:
    - boundary_f1: F-score of predicted/target boundary pixels matched within tolerance.
    - boundary_iou: IoU over tolerance-dilated boundary bands.
    - hd95: symmetric 95th-percentile boundary distance in pixels.
    - assd: average symmetric boundary distance in pixels.
    """
    if tolerance < 0:
        raise ValueError("boundary tolerance must be non-negative")
    if empty_distance_policy not in {"infinite", "image_diagonal"}:
        raise ValueError(
            "empty distance policy must be 'infinite' or 'image_diagonal'"
        )
    pred_boundary = _boundary(pred)
    target_boundary = _boundary(target)

    pred_count = pred_boundary.sum(dtype=np.float64)
    target_count = target_boundary.sum(dtype=np.float64)
    if pred_count == 0 and target_count == 0:
        return {
            "boundary_f1": 1.0,
            "boundary_iou": 1.0,
            "hd95": 0.0,
            "assd": 0.0,
        }
    if pred_count == 0 or target_count == 0:
        empty_distance = (
            float(np.hypot(max(pred_boundary.shape[0] - 1, 0), max(pred_boundary.shape[1] - 1, 0)))
            if empty_distance_policy == "image_diagonal"
            else float("inf")
        )
        return {
            "boundary_f1": 0.0,
            "boundary_iou": 0.0,
            "hd95": empty_distance,
            "assd": empty_distance,
        }

    pred_band = _dilate(pred_boundary, tolerance)
    target_band = _dilate(target_boundary, tolerance)
    boundary_precision = np.logical_and(pred_boundary, target_band).sum(dtype=np.float64) / pred_count
    boundary_recall = np.logical_and(target_boundary, pred_band).sum(dtype=np.float64) / target_count
    boundary_den = boundary_precision + boundary_recall
    boundary_f1 = 0.0 if boundary_den == 0 else 2.0 * boundary_precision * boundary_recall / boundary_den

    band_union = np.logical_or(pred_band, target_band).sum(dtype=np.float64)
    boundary_iou = 1.0 if band_union == 0 else np.logical_and(pred_band, target_band).sum(dtype=np.float64) / band_union

    import cv2

    pred_distance = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    target_distance = cv2.distanceTransform((~target_boundary).astype(np.uint8), cv2.DIST_L2, 3)
    distances = np.concatenate(
        [
            target_distance[pred_boundary].astype(np.float64),
            pred_distance[target_boundary].astype(np.float64),
        ]
    )
    return {
        "boundary_f1": float(boundary_f1),
        "boundary_iou": float(boundary_iou),
        "hd95": float(np.percentile(distances, 95)),
        "assd": float(np.mean(distances)),
    }


def segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    include_boundary: bool = False,
    boundary_tolerance: int = 2,
    empty_boundary_distance_policy: str = "infinite",
) -> dict[str, float]:
    """Compute Dice, IoU, precision and recall for binary segmentation masks."""
    pred_b = _binary(pred)
    target_b = _binary(target)

    tp = np.logical_and(pred_b, target_b).sum(dtype=np.float64)
    fp = np.logical_and(pred_b, ~target_b).sum(dtype=np.float64)
    fn = np.logical_and(~pred_b, target_b).sum(dtype=np.float64)
    pred_sum = tp + fp
    target_sum = tp + fn

    dice_den = pred_sum + target_sum
    union = tp + fp + fn

    dice = 1.0 if dice_den == 0 else (2.0 * tp) / dice_den
    iou = 1.0 if union == 0 else tp / union
    precision = 1.0 if pred_sum == 0 else tp / pred_sum
    recall = 1.0 if target_sum == 0 else tp / target_sum

    metrics = {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }
    if include_boundary:
        metrics.update(
            boundary_metrics(
                pred,
                target,
                tolerance=boundary_tolerance,
                empty_distance_policy=empty_boundary_distance_policy,
            )
        )
    return metrics


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot average an empty metric list")
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
