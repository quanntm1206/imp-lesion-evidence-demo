from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from lesion_robustness.image_quality import entropy, jpeg_blockiness_score, laplacian_variance


GLCM_FEATURE_NAMES = (
    "noise_sigma",
    "jpeg_blockiness",
    "luminance_mean",
    "luminance_std",
    "laplacian_log1p",
    "entropy",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
)


def _require_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("restoration preprocessing expects an RGB HxWx3 image")
    return image.astype(np.uint8, copy=False)


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Estimate additive high-frequency noise using a robust Laplacian response."""
    image_u8 = _require_rgb_uint8(image)
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if min(gray.shape) < 5:
        return 0.0
    kernel = np.asarray(
        [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
        dtype=np.float32,
    )
    response = cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
    interior = np.abs(response[1:-1, 1:-1])
    estimate = np.sqrt(np.pi / 2.0) * float(interior.mean()) / 6.0
    return max(estimate * 255.0, 0.0)


def adaptive_wiener_filter(
    image: np.ndarray,
    *,
    kernel_size: int = 5,
    blend: float = 0.45,
    noise_sigma: float | None = None,
) -> np.ndarray:
    """Apply a conservative local Wiener estimate and blend it with the source."""
    image_u8 = _require_rgb_uint8(image)
    kernel = max(3, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    if not 0.0 <= float(blend) <= 1.0:
        raise ValueError("Wiener blend must be in [0, 1]")
    sigma = estimate_noise_sigma(image_u8) if noise_sigma is None else float(noise_sigma)
    if sigma < 0.0 or not np.isfinite(sigma):
        raise ValueError("Wiener noise sigma must be finite and non-negative")

    source = image_u8.astype(np.float32) / 255.0
    noise_variance = (sigma / 255.0) ** 2
    restored = np.empty_like(source)
    for channel in range(3):
        plane = source[:, :, channel]
        local_mean = cv2.boxFilter(
            plane, cv2.CV_32F, (kernel, kernel), normalize=True, borderType=cv2.BORDER_REFLECT_101
        )
        local_second = cv2.boxFilter(
            plane * plane,
            cv2.CV_32F,
            (kernel, kernel),
            normalize=True,
            borderType=cv2.BORDER_REFLECT_101,
        )
        local_variance = np.maximum(local_second - local_mean * local_mean, 0.0)
        gain = np.maximum(local_variance - noise_variance, 0.0) / np.maximum(local_variance, 1e-8)
        restored[:, :, channel] = local_mean + gain * (plane - local_mean)

    mixed = (1.0 - float(blend)) * source + float(blend) * restored
    return np.clip(np.rint(mixed * 255.0), 0, 255).astype(np.uint8)


def jpeg_deblock_filter(
    image: np.ndarray,
    *,
    block_size: int = 8,
    strength: float = 0.25,
    min_discontinuity: float = 4.0,
    max_local_activity: float = 24.0,
) -> np.ndarray:
    """Smooth likely JPEG block boundaries while preserving active image edges."""
    image_u8 = _require_rgb_uint8(image)
    if int(block_size) < 2:
        raise ValueError("JPEG deblock block_size must be >= 2")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("JPEG deblock strength must be in [0, 1]")
    if min_discontinuity < 0.0 or max_local_activity < 0.0:
        raise ValueError("JPEG deblock thresholds must be non-negative")

    out = image_u8.astype(np.float32).copy()
    height, width = out.shape[:2]

    for x in range(int(block_size), width, int(block_size)):
        if x < 2 or x + 1 >= width:
            continue
        p1, p0 = out[:, x - 2, :].copy(), out[:, x - 1, :].copy()
        q0, q1 = out[:, x, :].copy(), out[:, x + 1, :].copy()
        boundary = np.abs(q0 - p0)
        local = 0.5 * (np.abs(p0 - p1) + np.abs(q1 - q0))
        mask = (boundary >= local + float(min_discontinuity)) & (local <= float(max_local_activity))
        delta = 0.5 * float(strength) * (q0 - p0) * mask
        out[:, x - 1, :] = p0 + delta
        out[:, x, :] = q0 - delta
        out[:, x - 2, :] = p1 + 0.25 * delta
        out[:, x + 1, :] = q1 - 0.25 * delta

    for y in range(int(block_size), height, int(block_size)):
        if y < 2 or y + 1 >= height:
            continue
        p1, p0 = out[y - 2, :, :].copy(), out[y - 1, :, :].copy()
        q0, q1 = out[y, :, :].copy(), out[y + 1, :, :].copy()
        boundary = np.abs(q0 - p0)
        local = 0.5 * (np.abs(p0 - p1) + np.abs(q1 - q0))
        mask = (boundary >= local + float(min_discontinuity)) & (local <= float(max_local_activity))
        delta = 0.5 * float(strength) * (q0 - p0) * mask
        out[y - 1, :, :] = p0 + delta
        out[y, :, :] = q0 - delta
        out[y - 2, :, :] = p1 + 0.25 * delta
        out[y + 1, :, :] = q1 - 0.25 * delta

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _glcm_statistics(gray: np.ndarray, *, levels: int = 16) -> dict[str, float]:
    if levels < 4 or levels > 64:
        raise ValueError("GLCM levels must be in [4, 64]")
    quantized = np.minimum((gray.astype(np.int32) * levels) // 256, levels - 1)
    matrices = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if dy == 0:
            left = quantized[:, :-1]
            right = quantized[:, 1:]
        elif dx == 1:
            left = quantized[:-1, :-1]
            right = quantized[1:, 1:]
        elif dx == -1:
            left = quantized[:-1, 1:]
            right = quantized[1:, :-1]
        else:
            left = quantized[:-1, :]
            right = quantized[1:, :]
        counts = np.bincount((left * levels + right).ravel(), minlength=levels * levels)
        matrix = counts.reshape(levels, levels).astype(np.float64)
        matrix += matrix.T
        total = float(matrix.sum())
        matrices.append(matrix / total if total > 0.0 else matrix)

    glcm = np.mean(matrices, axis=0)
    indices = np.arange(levels, dtype=np.float64)
    ii, jj = np.meshgrid(indices, indices, indexing="ij")
    contrast = float(np.sum((ii - jj) ** 2 * glcm))
    homogeneity = float(np.sum(glcm / (1.0 + (ii - jj) ** 2)))
    energy = float(np.sum(glcm * glcm))
    px = glcm.sum(axis=1)
    py = glcm.sum(axis=0)
    mean_x = float(np.sum(indices * px))
    mean_y = float(np.sum(indices * py))
    std_x = float(np.sqrt(np.sum((indices - mean_x) ** 2 * px)))
    std_y = float(np.sqrt(np.sum((indices - mean_y) ** 2 * py)))
    if std_x <= 1e-12 or std_y <= 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.sum((ii - mean_x) * (jj - mean_y) * glcm) / (std_x * std_y))
    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
    }


def restoration_quality_features(image: np.ndarray, *, glcm_levels: int = 16) -> dict[str, float]:
    image_u8 = _require_rgb_uint8(image)
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    features = {
        "noise_sigma": estimate_noise_sigma(image_u8),
        "jpeg_blockiness": jpeg_blockiness_score(image_u8),
        "luminance_mean": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "laplacian_log1p": float(np.log1p(max(laplacian_variance(image_u8), 0.0))),
        "entropy": entropy(image_u8),
    }
    features.update(_glcm_statistics(gray, levels=int(glcm_levels)))
    return features


@lru_cache(maxsize=8)
def _load_router_manifest(path: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError("restoration router manifest status must be passed")
    if payload.get("fit_split") != "train":
        raise ValueError("restoration router must be fit on split=train only")
    forbidden = {str(item).lower() for item in payload.get("observed_splits", [])} - {"train"}
    if forbidden:
        raise ValueError(f"restoration router manifest contains forbidden splits: {sorted(forbidden)}")
    if tuple(payload.get("feature_names", [])) != GLCM_FEATURE_NAMES:
        raise ValueError("restoration router feature schema mismatch")
    return payload


def load_router_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"restoration router manifest missing: {resolved}")
    return _load_router_manifest(str(resolved), resolved.stat().st_mtime_ns)


def classify_restoration_route(
    image: np.ndarray,
    manifest: Mapping[str, Any],
) -> tuple[str, float, dict[str, float]]:
    features = restoration_quality_features(image, glcm_levels=int(manifest.get("glcm_levels", 16)))
    vector = np.asarray([features[name] for name in GLCM_FEATURE_NAMES], dtype=np.float64)
    mean = np.asarray(manifest["feature_mean"], dtype=np.float64)
    std = np.asarray(manifest["feature_std"], dtype=np.float64)
    if vector.shape != mean.shape or vector.shape != std.shape:
        raise ValueError("restoration router normalization shape mismatch")
    standardized = (vector - mean) / np.maximum(std, 1e-8)
    class_names = list(manifest["class_names"])
    centroids = np.asarray([manifest["centroids"][name] for name in class_names], dtype=np.float64)
    distances = np.mean((centroids - standardized[None, :]) ** 2, axis=1)
    order = np.argsort(distances)
    predicted = class_names[int(order[0])]
    best = float(distances[order[0]])
    second = float(distances[order[1]]) if len(order) > 1 else best
    margin = max(second - best, 0.0) / max(second, 1e-8)
    threshold = float(manifest.get("class_margin_thresholds", {}).get(predicted, 1.0))
    if predicted == "gaussian_noise" and margin >= threshold:
        return "wiener", margin, features
    if predicted == "jpeg_compression" and margin >= threshold:
        return "deblock", margin, features
    return "identity", margin, features


def apply_restoration_preprocessing(image: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    image_u8 = _require_rgb_uint8(image)
    mode = str(config.get("mode", "identity")).strip().lower()
    wiener_cfg = config.get("wiener", {}) if isinstance(config.get("wiener", {}), Mapping) else {}
    deblock_cfg = config.get("deblock", {}) if isinstance(config.get("deblock", {}), Mapping) else {}

    def apply_wiener(value: np.ndarray) -> np.ndarray:
        return adaptive_wiener_filter(
            value,
            kernel_size=int(wiener_cfg.get("kernel_size", 5)),
            blend=float(wiener_cfg.get("blend", 0.45)),
        )

    def apply_deblock(value: np.ndarray) -> np.ndarray:
        return jpeg_deblock_filter(
            value,
            block_size=int(deblock_cfg.get("block_size", 8)),
            strength=float(deblock_cfg.get("strength", 0.25)),
            min_discontinuity=float(deblock_cfg.get("min_discontinuity", 4.0)),
            max_local_activity=float(deblock_cfg.get("max_local_activity", 24.0)),
        )

    if mode in {"identity", "none", "false", ""}:
        return image_u8.copy()
    if mode == "wiener":
        return apply_wiener(image_u8)
    if mode == "deblock":
        return apply_deblock(image_u8)
    if mode == "combined":
        return apply_wiener(apply_deblock(image_u8))
    if mode == "glcm_router":
        manifest_path = config.get("router_manifest")
        if not manifest_path:
            raise ValueError("glcm_router restoration requires router_manifest")
        route, _, _ = classify_restoration_route(image_u8, load_router_manifest(str(manifest_path)))
        if route == "wiener":
            return apply_wiener(image_u8)
        if route == "deblock":
            return apply_deblock(image_u8)
        return image_u8.copy()
    raise ValueError(f"Unsupported restoration preprocessing mode: {mode}")
