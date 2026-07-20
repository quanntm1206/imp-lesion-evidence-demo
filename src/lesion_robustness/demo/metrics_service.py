"""Optional ground-truth metric evaluation for the demo."""

from __future__ import annotations

import numpy as np

from lesion_robustness.metrics import segmentation_metrics


MAX_MASK_PIXELS = 16_000_000


def _decode_mask(mask: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim == 2:
        height, width = value.shape
        channels = None
    elif value.ndim == 3 and value.shape[2] == 3:
        height, width = value.shape[:2]
        channels = 3
    else:
        raise ValueError(f"{label} must be a grayscale or RGB mask")
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} must have positive geometry")
    if height * width > MAX_MASK_PIXELS:
        raise ValueError(f"{label} exceeds the 16 megapixels limit")

    if value.dtype == np.bool_:
        numeric = value.astype(np.uint8)
    elif value.dtype == np.uint8:
        numeric = value
    elif np.issubdtype(value.dtype, np.floating):
        if not np.isfinite(value).all():
            raise ValueError(f"{label} must contain finite values")
        if not np.logical_or(value == 0.0, value == 1.0).all():
            raise ValueError(f"{label} binary float mask contains ambiguous soft values")
        numeric = (value * 255.0).astype(np.uint8)
    else:
        raise ValueError(f"{label} must use uint8 or binary float pixels")

    if channels is None and np.logical_or(numeric == 0, numeric == 1).all():
        return numeric.astype(bool, copy=False)
    if channels is None:
        decoded = numeric
    else:
        decoded = np.rint(
            numeric[..., 0].astype(np.float32) * 0.299
            + numeric[..., 1].astype(np.float32) * 0.587
            + numeric[..., 2].astype(np.float32) * 0.114
        )
    return decoded >= 127


def evaluate_optional_ground_truth(
    control: np.ndarray,
    candidate: np.ndarray,
    gt: np.ndarray | None,
) -> dict[str, dict[str, float]] | None:
    """Return per-arm metrics only when an uploaded ground-truth mask exists."""
    if gt is None:
        return None

    control_mask = _decode_mask(control, label="control mask")
    candidate_mask = _decode_mask(candidate, label="candidate mask")
    ground_truth_mask = _decode_mask(gt, label="ground-truth mask")
    if (
        control_mask.shape != ground_truth_mask.shape
        or candidate_mask.shape != ground_truth_mask.shape
    ):
        raise ValueError("metric mask geometry mismatch")

    return {
        "control": segmentation_metrics(
            control_mask,
            ground_truth_mask,
            include_boundary=True,
            boundary_tolerance=2,
            empty_boundary_distance_policy="image_diagonal",
        ),
        "candidate": segmentation_metrics(
            candidate_mask,
            ground_truth_mask,
            include_boundary=True,
            boundary_tolerance=2,
            empty_boundary_distance_policy="image_diagonal",
        ),
    }
