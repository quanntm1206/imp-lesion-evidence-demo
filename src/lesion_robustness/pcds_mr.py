from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class PCDSMRConfig:
    border_ring_fraction: float = 0.10
    local_kernel_sizes: tuple[int, ...] = (7, 15, 31)
    global_weight: float = 0.70
    local_weight: float = 0.30
    low_threshold: float = 0.20
    high_threshold: float = 0.45
    close_kernel_size: int = 7
    min_support_fraction: float = 0.005
    max_support_fraction: float = 0.85
    min_valid_border_fraction: float = 0.25
    modulation_alpha: float = 0.35
    modulation_blur_sigma: float = 15.0
    max_delta_l: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 < self.border_ring_fraction <= 0.5:
            raise ValueError("border_ring_fraction must be in (0, 0.5]")
        if not self.local_kernel_sizes or any(size < 3 or size % 2 == 0 for size in self.local_kernel_sizes):
            raise ValueError("local_kernel_sizes must contain positive odd values >= 3")
        if self.global_weight < 0.0 or self.local_weight < 0.0:
            raise ValueError("PCDS-MR score weights must be non-negative")
        if self.global_weight + self.local_weight <= 0.0:
            raise ValueError("at least one PCDS-MR score weight must be positive")
        if not 0.0 <= self.low_threshold <= self.high_threshold <= 1.0:
            raise ValueError("PCDS-MR thresholds must satisfy 0 <= low <= high <= 1")
        if self.close_kernel_size < 1 or self.close_kernel_size % 2 == 0:
            raise ValueError("close_kernel_size must be a positive odd value")
        if not 0.0 <= self.min_support_fraction < self.max_support_fraction <= 1.0:
            raise ValueError("support fractions must satisfy 0 <= min < max <= 1")
        if not 0.0 <= self.min_valid_border_fraction <= 1.0:
            raise ValueError("min_valid_border_fraction must be in [0, 1]")
        if self.modulation_alpha < 0.0:
            raise ValueError("modulation_alpha must be non-negative")
        if self.modulation_blur_sigma <= 0.0:
            raise ValueError("modulation_blur_sigma must be positive")
        if self.max_delta_l < 0.0:
            raise ValueError("max_delta_l must be non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> PCDSMRConfig:
        if not values:
            return cls()
        aliases = {
            "gaussian_kernel_sizes": "local_kernel_sizes",
            "hysteresis_low": "low_threshold",
            "hysteresis_high": "high_threshold",
            "closing_kernel_size": "close_kernel_size",
            "modulation_sigma": "modulation_blur_sigma",
            "max_l_delta": "max_delta_l",
        }
        allowed = set(cls.__dataclass_fields__)
        kwargs: dict[str, Any] = {}
        for raw_key, value in values.items():
            if raw_key == "enabled":
                continue
            key = aliases.get(raw_key, raw_key)
            if key not in allowed:
                raise ValueError(f"Unsupported preprocessing.pcds_mr parameter: {raw_key}")
            kwargs[key] = tuple(value) if key == "local_kernel_sizes" else value
        return cls(**kwargs)


@dataclass(frozen=True)
class PCDSMRMetadata:
    mode: str
    applied: bool
    fallback_reason: str | None
    border_valid_fraction: float
    excluded_fraction: float
    support_fraction: float
    global_reference_lab: tuple[float, float, float] | None
    delta_l_min: float
    delta_l_max: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PCDSMRResult:
    image: np.ndarray
    support_mask: np.ndarray
    score: np.ndarray
    exclusion_mask: np.ndarray
    metadata: PCDSMRMetadata


def _require_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PCDS-MR expects an RGB image with shape HxWx3")
    if image.dtype != np.uint8:
        raise ValueError("PCDS-MR expects uint8 RGB input")
    return image


def _ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 color difference for standard CIELAB arrays."""
    first = np.asarray(lab1, dtype=np.float64)
    second = np.asarray(lab2, dtype=np.float64)
    l1, a1, b1 = np.moveaxis(first, -1, 0)
    l2, a2, b2 = np.moveaxis(second, -1, 0)

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = np.hypot(a1_prime, b1)
    c2_prime = np.hypot(a2_prime, b2)
    h1_prime = np.mod(np.arctan2(b1, a1_prime), 2.0 * np.pi)
    h2_prime = np.mod(np.arctan2(b2, a2_prime), 2.0 * np.pi)

    delta_l = l2 - l1
    delta_c = c2_prime - c1_prime
    delta_h_angle = h2_prime - h1_prime
    delta_h_angle = np.where(delta_h_angle > np.pi, delta_h_angle - 2.0 * np.pi, delta_h_angle)
    delta_h_angle = np.where(delta_h_angle < -np.pi, delta_h_angle + 2.0 * np.pi, delta_h_angle)
    delta_h_angle = np.where(c1_prime * c2_prime == 0.0, 0.0, delta_h_angle)
    delta_h = 2.0 * np.sqrt(c1_prime * c2_prime) * np.sin(0.5 * delta_h_angle)

    l_bar = 0.5 * (l1 + l2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)
    hue_sum = h1_prime + h2_prime
    hue_diff = np.abs(h1_prime - h2_prime)
    h_bar_prime = np.where(
        c1_prime * c2_prime == 0.0,
        hue_sum,
        np.where(
            hue_diff <= np.pi,
            0.5 * hue_sum,
            np.where(hue_sum < 2.0 * np.pi, 0.5 * (hue_sum + 2.0 * np.pi), 0.5 * (hue_sum - 2.0 * np.pi)),
        ),
    )

    t = (
        1.0
        - 0.17 * np.cos(h_bar_prime - np.deg2rad(30.0))
        + 0.24 * np.cos(2.0 * h_bar_prime)
        + 0.32 * np.cos(3.0 * h_bar_prime + np.deg2rad(6.0))
        - 0.20 * np.cos(4.0 * h_bar_prime - np.deg2rad(63.0))
    )
    delta_theta = np.deg2rad(30.0) * np.exp(-((np.rad2deg(h_bar_prime) - 275.0) / 25.0) ** 2)
    c_bar_prime7 = c_bar_prime**7
    r_c = 2.0 * np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25.0**7))
    l_term = (l_bar - 50.0) ** 2
    s_l = 1.0 + 0.015 * l_term / np.sqrt(20.0 + l_term)
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -np.sin(2.0 * delta_theta) * r_c

    l_scaled = delta_l / s_l
    c_scaled = delta_c / s_c
    h_scaled = delta_h / s_h
    delta_e = np.sqrt(np.maximum(0.0, l_scaled**2 + c_scaled**2 + h_scaled**2 + r_t * c_scaled * h_scaled))
    return delta_e.astype(np.float32)


def _robust_unit_scale(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_values = values[valid_mask & np.isfinite(values)]
    if valid_values.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    lo = float(np.percentile(valid_values, 1.0))
    hi = float(np.percentile(valid_values, 99.5))
    if hi <= lo + 1e-6:
        hi = float(valid_values.max())
    if hi <= lo + 1e-6:
        return np.zeros(values.shape, dtype=np.float32)
    scaled = np.clip((values.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    scaled[~valid_mask] = 0.0
    return scaled


def _border_ring(shape: tuple[int, int], fraction: float) -> np.ndarray:
    height, width = shape
    thickness = max(1, int(np.ceil(min(height, width) * fraction)))
    ring = np.zeros((height, width), dtype=bool)
    ring[:thickness, :] = True
    ring[-thickness:, :] = True
    ring[:, :thickness] = True
    ring[:, -thickness:] = True
    return ring


def _artifact_exclusion_mask(image: np.ndarray) -> np.ndarray:
    # Import lazily so the standalone algorithm can reuse preprocessing detectors without a module cycle.
    from lesion_robustness.preprocessing import (
        detect_black_frame_mask,
        detect_dark_artifact_mask,
        specular_highlight_mask,
    )

    return (
        detect_black_frame_mask(image)
        | detect_dark_artifact_mask(image)
        | specular_highlight_mask(image)
    )


def _local_score(lab: np.ndarray, valid_mask: np.ndarray, kernel_sizes: tuple[int, ...]) -> np.ndarray:
    distances: list[np.ndarray] = []
    for kernel_size in kernel_sizes:
        local_reference = cv2.GaussianBlur(
            lab,
            (int(kernel_size), int(kernel_size)),
            sigmaX=0.0,
            sigmaY=0.0,
            borderType=cv2.BORDER_REFLECT101,
        )
        distances.append(_ciede2000(lab, local_reference))
    return _robust_unit_scale(np.maximum.reduce(distances), valid_mask)


def _robust_background_reference(lab: np.ndarray, valid_border: np.ndarray) -> np.ndarray:
    """Estimate border skin color after rejecting chromatic outliers by median/MAD."""
    values = lab[valid_border]
    reference = np.median(values, axis=0).astype(np.float32)
    distances = _ciede2000(values, reference)
    center = float(np.median(distances))
    mad = float(np.median(np.abs(distances - center)))
    cutoff = center + 3.0 * max(1.4826 * mad, 1e-3)
    inliers = values[distances <= cutoff]
    if inliers.shape[0] >= max(16, round(values.shape[0] * 0.25)):
        reference = np.median(inliers, axis=0).astype(np.float32)
    return reference


def _hysteresis_reconstruction(
    score: np.ndarray,
    exclusion_mask: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    low = (score >= low_threshold) & ~exclusion_mask
    high = (score >= high_threshold) & low
    if not high.any():
        return np.zeros(score.shape, dtype=bool)
    count, labels = cv2.connectedComponents(low.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(labels[high])] = True
    keep[0] = False
    return keep[labels]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    inverse = ((1 - padded) * 255).astype(np.uint8)
    flood_mask = np.zeros((inverse.shape[0] + 2, inverse.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(inverse, flood_mask, (0, 0), 128)
    holes = inverse == 255
    return mask | holes[1:-1, 1:-1]


def _postprocess_support(mask: np.ndarray, exclusion_mask: np.ndarray, close_kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return _fill_holes(closed) & ~exclusion_mask


def apply_pcds_mr(
    image: np.ndarray,
    *,
    config: PCDSMRConfig | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> PCDSMRResult:
    """Apply deterministic mask-free PCDS-MR LAB-L modulation."""
    image_u8 = _require_rgb_uint8(image)
    cfg = config or PCDSMRConfig()
    artifact_mask = _artifact_exclusion_mask(image_u8)
    if exclusion_mask is not None:
        supplied_mask = np.asarray(exclusion_mask, dtype=bool)
        if supplied_mask.shape != image_u8.shape[:2]:
            raise ValueError("PCDS-MR exclusion_mask must match the image height and width")
        artifact_mask |= supplied_mask

    valid_mask = ~artifact_mask
    lab = cv2.cvtColor(image_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    local_score = _local_score(lab, valid_mask, cfg.local_kernel_sizes)
    ring = _border_ring(image_u8.shape[:2], cfg.border_ring_fraction)
    ring_size = int(ring.sum())
    valid_border = ring & valid_mask
    border_valid_fraction = float(valid_border.sum() / max(ring_size, 1))

    reference: tuple[float, float, float] | None = None
    if border_valid_fraction < cfg.min_valid_border_fraction:
        mode = "local_only"
        score = local_score
    else:
        reference_values = _robust_background_reference(lab, valid_border)
        reference = tuple(float(value) for value in reference_values)
        global_distance = _ciede2000(lab, reference_values)
        global_score = _robust_unit_scale(global_distance, valid_mask)
        total_weight = cfg.global_weight + cfg.local_weight
        score = (cfg.global_weight * global_score + cfg.local_weight * local_score) / total_weight
        score = score.astype(np.float32)
        score[artifact_mask] = 0.0
        mode = "global_local"

    support = _hysteresis_reconstruction(
        score,
        artifact_mask,
        cfg.low_threshold,
        cfg.high_threshold,
    )
    support = _postprocess_support(support, artifact_mask, cfg.close_kernel_size)
    support_fraction = float(support.mean())

    fallback_reason: str | None = None
    if support_fraction < cfg.min_support_fraction:
        fallback_reason = "support_below_minimum"
    elif support_fraction > cfg.max_support_fraction:
        fallback_reason = "support_above_maximum"

    if fallback_reason is not None or cfg.modulation_alpha == 0.0 or cfg.max_delta_l == 0.0:
        metadata = PCDSMRMetadata(
            mode=mode,
            applied=False,
            fallback_reason=fallback_reason,
            border_valid_fraction=border_valid_fraction,
            excluded_fraction=float(artifact_mask.mean()),
            support_fraction=support_fraction,
            global_reference_lab=reference,
            delta_l_min=0.0,
            delta_l_max=0.0,
        )
        return PCDSMRResult(image_u8.copy(), support, score, artifact_mask, metadata)

    l_channel = lab[:, :, 0].astype(np.float32)
    smooth_l = cv2.GaussianBlur(
        l_channel,
        (0, 0),
        sigmaX=cfg.modulation_blur_sigma,
        sigmaY=cfg.modulation_blur_sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    delta_l = cfg.modulation_alpha * (l_channel - smooth_l)
    delta_l = np.clip(delta_l, -cfg.max_delta_l, cfg.max_delta_l)
    delta_l *= support.astype(np.float32)

    modulated_lab = lab.copy()
    modulated_lab[:, :, 0] = np.clip(l_channel + delta_l, 0.0, 100.0)
    modulated_rgb = cv2.cvtColor(modulated_lab, cv2.COLOR_LAB2RGB)
    output = np.clip(np.rint(modulated_rgb * 255.0), 0, 255).astype(np.uint8)
    metadata = PCDSMRMetadata(
        mode=mode,
        applied=True,
        fallback_reason=None,
        border_valid_fraction=border_valid_fraction,
        excluded_fraction=float(artifact_mask.mean()),
        support_fraction=support_fraction,
        global_reference_lab=reference,
        delta_l_min=float(delta_l.min()),
        delta_l_max=float(delta_l.max()),
    )
    return PCDSMRResult(output, support, score, artifact_mask, metadata)
