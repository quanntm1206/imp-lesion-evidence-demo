from __future__ import annotations

import cv2
import numpy as np


def luminance_stats(image: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(gray.mean()), float(gray.std())


def _luminance(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def laplacian_variance(image: np.ndarray) -> float:
    gray = _luminance(image)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def entropy(image: np.ndarray) -> float:
    gray = _luminance(image).astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = hist / max(float(hist.sum()), 1.0)
    probabilities = probabilities[probabilities > 0.0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def jpeg_blockiness_score(image: np.ndarray, *, block_size: int = 8) -> float:
    """Estimate 8x8 blocking artifacts from boundary vs non-boundary luminance jumps."""
    gray = _luminance(image)
    if gray.shape[0] <= block_size or gray.shape[1] <= block_size:
        return 0.0

    vertical_diffs = np.abs(np.diff(gray, axis=1))
    horizontal_diffs = np.abs(np.diff(gray, axis=0))

    vertical_boundary = vertical_diffs[:, block_size - 1 :: block_size]
    horizontal_boundary = horizontal_diffs[block_size - 1 :: block_size, :]

    vertical_mask = np.ones(vertical_diffs.shape[1], dtype=bool)
    vertical_mask[block_size - 1 :: block_size] = False
    horizontal_mask = np.ones(horizontal_diffs.shape[0], dtype=bool)
    horizontal_mask[block_size - 1 :: block_size] = False

    boundary_mean = float(np.concatenate([vertical_boundary.ravel(), horizontal_boundary.ravel()]).mean())
    non_boundary_values = np.concatenate(
        [vertical_diffs[:, vertical_mask].ravel(), horizontal_diffs[horizontal_mask, :].ravel()]
    )
    non_boundary_mean = float(non_boundary_values.mean()) if non_boundary_values.size else 0.0
    return max(0.0, boundary_mean - non_boundary_mean) / 255.0


def quality_features(image: np.ndarray) -> dict[str, float]:
    mean, std = luminance_stats(image)
    return {
        "luminance_mean": mean,
        "luminance_std": std,
        "laplacian_var": laplacian_variance(image),
        "entropy": entropy(image),
        "jpeg_blockiness": jpeg_blockiness_score(image),
    }


def classify_quality(
    image: np.ndarray,
    *,
    low_brightness_mean: float = 122.0,
    low_contrast_std: float = 13.0,
) -> str:
    mean, std = luminance_stats(image)
    if mean < low_brightness_mean:
        return "low_brightness"
    if std < low_contrast_std:
        return "low_contrast"
    return "clean"


def classify_domain_quality(
    image: np.ndarray,
    *,
    low_brightness_mean: float = 122.0,
    low_contrast_std: float = 13.0,
    jpeg_blockiness_min: float = 0.02,
) -> str:
    features = quality_features(image)
    if features["luminance_mean"] < low_brightness_mean:
        return "low_brightness"
    if features["jpeg_blockiness"] >= jpeg_blockiness_min:
        return "jpeg_compression"
    if features["luminance_std"] < low_contrast_std:
        return "low_contrast"
    return "clean"


def parse_quality_thresholds(spec: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError("quality thresholds must use NAME:THRESHOLD format.")
        name, value = item.split(":", 1)
        thresholds[name.strip()] = float(value)
    if "clean" not in thresholds:
        raise ValueError("quality thresholds must include clean.")
    return thresholds
