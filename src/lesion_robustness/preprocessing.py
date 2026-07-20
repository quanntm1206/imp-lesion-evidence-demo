from __future__ import annotations

from copy import deepcopy
from typing import Any

import cv2
import numpy as np

from lesion_robustness.pcds_mr import PCDSMRConfig, apply_pcds_mr
from lesion_robustness.packed_extra_channel import PACKED_TYPES
from lesion_robustness.restoration_preprocessing import apply_restoration_preprocessing


def _require_rgb_uint8(image: np.ndarray, name: str) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} expects an RGB image with shape HxWx3")
    return image.astype(np.uint8)


def apply_clahe_lab(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB color space."""
    image_u8 = _require_rgb_uint8(image, "CLAHE LAB preprocessing")
    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def apply_filter(image: np.ndarray, filter_type: str = "none", kernel_size: int = 3) -> np.ndarray:
    kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    if filter_type in {"none", None}:
        return image
    if filter_type == "median":
        return cv2.medianBlur(image, kernel)
    if filter_type == "gaussian":
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0)
    if filter_type == "bilateral":
        return cv2.bilateralFilter(image, d=kernel, sigmaColor=50, sigmaSpace=50)
    if filter_type == "guided":
        return guided_filter_rgb(image, radius=max(1, kernel // 2), epsilon=1e-3)
    if filter_type == "edge_preserving":
        return cv2.edgePreservingFilter(image, flags=1, sigma_s=30, sigma_r=0.2)
    raise ValueError(f"Unsupported filter type: {filter_type}")


def gray_world_color_constancy(image: np.ndarray) -> np.ndarray:
    """Normalize RGB channel means with the Gray World assumption."""
    image_u8 = _require_rgb_uint8(image, "Gray World color constancy")
    image_f = image_u8.astype(np.float32)
    channel_means = image_f.mean(axis=(0, 1))
    target_mean = float(channel_means.mean())
    scales = np.divide(
        target_mean,
        channel_means,
        out=np.ones_like(channel_means, dtype=np.float32),
        where=channel_means > 1e-6,
    )
    return np.clip(image_f * scales[None, None, :], 0, 255).astype(np.uint8)


def shades_of_gray_color_constancy(
    image: np.ndarray,
    *,
    minkowski_norm: float = 6.0,
    gain_clip: tuple[float, float] | None = (0.85, 1.15),
) -> np.ndarray:
    """Apply Shades-of-Gray color constancy with optional conservative gain clipping."""
    if minkowski_norm <= 0:
        raise ValueError("Shades-of-Gray minkowski_norm must be positive")
    image_u8 = _require_rgb_uint8(image, "Shades-of-Gray color constancy")
    image_f = image_u8.astype(np.float32)
    channel_norms = np.power(
        np.mean(np.power(image_f + 1e-6, minkowski_norm), axis=(0, 1)),
        1.0 / minkowski_norm,
    )
    target_norm = float(channel_norms.mean())
    gains = np.divide(
        target_norm,
        channel_norms,
        out=np.ones_like(channel_norms, dtype=np.float32),
        where=channel_norms > 1e-6,
    )
    if gain_clip is not None:
        lo, hi = gain_clip
        if hi < lo:
            raise ValueError("Shades-of-Gray gain_clip upper bound must be >= lower bound")
        gains = np.clip(gains, float(lo), float(hi))
    return np.clip(image_f * gains[None, None, :], 0, 255).astype(np.uint8)


def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply power-law gamma correction; gamma below 1 brightens mid-tones."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    image_u8 = _require_rgb_uint8(image, "gamma correction")
    table = np.array([((value / 255.0) ** gamma) * 255.0 for value in range(256)])
    return cv2.LUT(image_u8, np.clip(table, 0, 255).astype(np.uint8))


def apply_single_scale_retinex(image: np.ndarray, *, sigma: float = 30.0) -> np.ndarray:
    """Apply a conservative single-scale Retinex illumination correction."""
    if sigma <= 0:
        raise ValueError("retinex sigma must be positive")
    image_u8 = _require_rgb_uint8(image, "single-scale Retinex")
    image_f = image_u8.astype(np.float32) + 1.0
    surround = cv2.GaussianBlur(image_f, (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
    retinex = np.log(image_f) - np.log(surround)
    out = np.empty_like(retinex)
    for channel in range(retinex.shape[2]):
        channel_values = retinex[:, :, channel]
        lo, hi = np.percentile(channel_values, (1.0, 99.0))
        if hi <= lo:
            out[:, :, channel] = image_u8[:, :, channel]
        else:
            out[:, :, channel] = np.clip((channel_values - lo) * (255.0 / (hi - lo)), 0, 255)
    return out.astype(np.uint8)


def _normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    values_f = values.astype(np.float32)
    finite = np.isfinite(values_f)
    if not finite.any():
        return np.zeros(values_f.shape, dtype=np.uint8)
    lo, hi = np.percentile(values_f[finite], (1.0, 99.0))
    if hi <= lo:
        return np.zeros(values_f.shape, dtype=np.uint8)
    return np.clip((values_f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def illumination_invariant_gray(image: np.ndarray) -> np.ndarray:
    """Return a log-chromaticity illumination-invariant grayscale proxy."""
    image_u8 = _require_rgb_uint8(image, "illumination invariant gray")
    image_f = image_u8.astype(np.float32) + 1.0
    log_rgb = np.log(image_f)
    log_mean = log_rgb.mean(axis=2)
    chroma = log_rgb[:, :, 1] - 0.5 * (log_rgb[:, :, 0] + log_rgb[:, :, 2])
    return _normalize_to_uint8(chroma - 0.25 * log_mean)


def shade_attenuated_gray(image: np.ndarray, *, sigma: float = 15.0) -> np.ndarray:
    """Estimate a luminance channel with large-scale shading attenuated."""
    if sigma <= 0:
        raise ValueError("shade attenuation sigma must be positive")
    image_u8 = _require_rgb_uint8(image, "shade attenuated gray")
    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32) + 1.0
    background = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
    ratio = l_channel / background
    return np.clip(ratio * 127.5, 0, 255).astype(np.uint8)


def chromaticity_red_channel(image: np.ndarray) -> np.ndarray:
    """Return normalized red chromaticity R/(R+G+B)."""
    image_u8 = _require_rgb_uint8(image, "red chromaticity")
    image_f = image_u8.astype(np.float32)
    denom = image_f.sum(axis=2) + 1e-6
    return np.clip((image_f[:, :, 0] / denom) * 255.0, 0, 255).astype(np.uint8)


def optical_density_pc1_channel(image: np.ndarray) -> np.ndarray:
    """Return the first principal component in optical-density skin-pigment space."""
    image_u8 = _require_rgb_uint8(image, "optical density PC1")
    rgb = np.clip(image_u8.astype(np.float32), 1.0, 255.0) / 255.0
    od = -np.log(rgb)
    flat = od.reshape(-1, 3)
    centered = flat - flat.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return np.zeros(image_u8.shape[:2], dtype=np.uint8)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    pc1 = (centered @ vh[0]).reshape(image_u8.shape[:2])
    return _normalize_to_uint8(pc1)


def guided_filter_rgb(image: np.ndarray, *, radius: int = 2, epsilon: float = 1e-3) -> np.ndarray:
    """Small self-guided filter implementation for boundary-preserving smoothing."""
    if radius < 1:
        raise ValueError("guided filter radius must be >= 1")
    if epsilon <= 0:
        raise ValueError("guided filter epsilon must be positive")
    image_u8 = _require_rgb_uint8(image, "guided filter")
    out = np.empty_like(image_u8, dtype=np.float32)
    kernel = (2 * int(radius) + 1, 2 * int(radius) + 1)
    for channel in range(3):
        guide = image_u8[:, :, channel].astype(np.float32) / 255.0
        src = guide
        mean_i = cv2.boxFilter(guide, ddepth=-1, ksize=kernel, normalize=True)
        mean_p = cv2.boxFilter(src, ddepth=-1, ksize=kernel, normalize=True)
        corr_i = cv2.boxFilter(guide * guide, ddepth=-1, ksize=kernel, normalize=True)
        corr_ip = cv2.boxFilter(guide * src, ddepth=-1, ksize=kernel, normalize=True)
        var_i = corr_i - mean_i * mean_i
        cov_ip = corr_ip - mean_i * mean_p
        a = cov_ip / (var_i + float(epsilon))
        b = mean_p - a * mean_i
        mean_a = cv2.boxFilter(a, ddepth=-1, ksize=kernel, normalize=True)
        mean_b = cv2.boxFilter(b, ddepth=-1, ksize=kernel, normalize=True)
        out[:, :, channel] = (mean_a * guide + mean_b) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_shading_correction(image: np.ndarray, *, sigma: float = 15.0) -> np.ndarray:
    """Correct broad illumination falloff by dividing each channel by a smooth field."""
    if sigma <= 0:
        raise ValueError("shading correction sigma must be positive")
    image_u8 = _require_rgb_uint8(image, "shading correction")
    image_f = image_u8.astype(np.float32) + 1.0
    corrected = np.empty_like(image_f)
    for channel in range(3):
        bg = cv2.GaussianBlur(image_f[:, :, channel], (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
        ratio = image_f[:, :, channel] / bg
        corrected[:, :, channel] = ratio * float(np.mean(bg))
    return np.clip(corrected, 0, 255).astype(np.uint8)


def detect_black_frame_mask(
    image: np.ndarray,
    *,
    threshold: int = 8,
    min_border_fraction: float = 0.35,
) -> np.ndarray:
    """Detect dark connected components touching image borders."""
    image_u8 = _require_rgb_uint8(image, "black-frame detection")
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    dark = gray <= int(threshold)
    if not dark.any():
        return np.zeros(gray.shape, dtype=bool)
    border = np.zeros_like(dark, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    side_fractions = (
        float(dark[0, :].mean()),
        float(dark[-1, :].mean()),
        float(dark[:, 0].mean()),
        float(dark[:, -1].mean()),
    )
    if max(side_fractions) < float(min_border_fraction):
        return np.zeros(gray.shape, dtype=bool)
    n_labels, labels = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    keep = np.zeros(n_labels, dtype=bool)
    keep[np.unique(labels[border & dark])] = True
    keep[0] = False
    return keep[labels]


def vignette_mask(image: np.ndarray) -> np.ndarray:
    """Return a radial edge-falloff prior, bright at corners and dark at center."""
    image_u8 = _require_rgb_uint8(image, "vignette mask")
    height, width = image_u8.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    dist = np.sqrt(((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2)
    return np.clip(dist / max(float(dist.max()), 1e-6) * 255.0, 0, 255).astype(np.uint8)


def specular_highlight_mask(
    image: np.ndarray,
    *,
    value_threshold: int = 235,
    saturation_threshold: int = 40,
) -> np.ndarray:
    """Detect small bright low-saturation specular highlights."""
    image_u8 = _require_rgb_uint8(image, "specular highlight detection")
    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 2] >= int(value_threshold)) & (hsv[:, :, 1] <= int(saturation_threshold))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    return mask_u8.astype(bool)


def inpaint_specular_highlights(
    image: np.ndarray,
    *,
    value_threshold: int = 235,
    saturation_threshold: int = 40,
    inpaint_radius: int = 3,
    max_mask_ratio: float | None = 0.05,
) -> np.ndarray:
    """Inpaint only small detected specular highlights."""
    image_u8 = _require_rgb_uint8(image, "specular highlight inpainting")
    mask = specular_highlight_mask(
        image_u8,
        value_threshold=value_threshold,
        saturation_threshold=saturation_threshold,
    )
    if not mask.any():
        return image_u8.copy()
    if max_mask_ratio is not None and float(mask.mean()) > max_mask_ratio:
        return image_u8.copy()
    return cv2.inpaint(image_u8, mask.astype(np.uint8) * 255, int(inpaint_radius), cv2.INPAINT_TELEA)


def superpixel_boundary_proxy(image: np.ndarray) -> np.ndarray:
    """Return a lightweight region-boundary proxy using mean-shift smoothing plus Canny."""
    image_u8 = _require_rgb_uint8(image, "superpixel boundary proxy")
    smoothed = cv2.pyrMeanShiftFiltering(image_u8, sp=8, sr=18)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return _normalize_to_uint8(np.sqrt(gx * gx + gy * gy))


def saliency_proxy(image: np.ndarray) -> np.ndarray:
    """Return color-distance saliency from the median border skin color."""
    image_u8 = _require_rgb_uint8(image, "saliency proxy")
    border = np.concatenate(
        [
            image_u8[0, :, :],
            image_u8[-1, :, :],
            image_u8[:, 0, :],
            image_u8[:, -1, :],
        ],
        axis=0,
    )
    skin = np.median(border.astype(np.float32), axis=0)
    dist = np.linalg.norm(image_u8.astype(np.float32) - skin[None, None, :], axis=2)
    return _normalize_to_uint8(dist)


def lbp_texture_proxy(image: np.ndarray) -> np.ndarray:
    """Return a local texture proxy inspired by LBP/edge-energy responses."""
    image_u8 = _require_rgb_uint8(image, "LBP texture proxy")
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    center = gray[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        neigh = gray[1 + dy : 1 + dy + center.shape[0], 1 + dx : 1 + dx + center.shape[1]]
        code |= ((neigh >= center).astype(np.uint8) << bit)
    padded = np.pad(code, 1, mode="edge")
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    return _normalize_to_uint8(0.5 * padded.astype(np.float32) + lap)


def gabor_texture_proxy(image: np.ndarray) -> np.ndarray:
    """Return a Gabor-response texture map."""
    image_u8 = _require_rgb_uint8(image, "Gabor texture proxy")
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    responses = []
    for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
        kernel = cv2.getGaborKernel((9, 9), 3.0, theta, 6.0, 0.5, 0, ktype=cv2.CV_32F)
        responses.append(np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel)))
    return _normalize_to_uint8(np.maximum.reduce(responses))


def wavelet_energy_proxy(image: np.ndarray) -> np.ndarray:
    """Return a simple high-frequency energy map approximating wavelet detail energy."""
    image_u8 = _require_rgb_uint8(image, "wavelet energy proxy")
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    low = cv2.pyrUp(cv2.pyrDown(gray), dstsize=(gray.shape[1], gray.shape[0]))
    return _normalize_to_uint8(np.abs(gray - low))


def detect_dark_artifact_mask(
    image: np.ndarray,
    *,
    blackhat_kernel_size: int = 9,
    threshold: int = 10,
) -> np.ndarray:
    """Detect dark thin artifacts such as hairs using black-hat morphology."""
    image_u8 = _require_rgb_uint8(image, "dark artifact detection")
    kernel_size = max(3, blackhat_kernel_size if blackhat_kernel_size % 2 == 1 else blackhat_kernel_size + 1)
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask.astype(bool)


def remove_dark_artifacts(
    image: np.ndarray,
    *,
    blackhat_kernel_size: int = 9,
    threshold: int = 10,
    inpaint_radius: int = 3,
    max_mask_ratio: float | None = None,
) -> np.ndarray:
    """Inpaint detected dark hair/artifact pixels with Telea inpainting."""
    image_u8 = _require_rgb_uint8(image, "dark artifact removal")
    mask = detect_dark_artifact_mask(
        image_u8,
        blackhat_kernel_size=blackhat_kernel_size,
        threshold=threshold,
    )
    if not mask.any():
        return image_u8.copy()
    if max_mask_ratio is not None and float(mask.mean()) > max_mask_ratio:
        return image_u8.copy()
    return cv2.inpaint(image_u8, mask.astype(np.uint8) * 255, inpaint_radius, cv2.INPAINT_TELEA)


def unsharp_mask(
    image: np.ndarray,
    *,
    amount: float = 0.5,
    kernel_size: int = 3,
) -> np.ndarray:
    """Sharpen local edges with a conservative unsharp mask."""
    image_u8 = _require_rgb_uint8(image, "unsharp mask")
    kernel = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    blurred = cv2.GaussianBlur(image_u8, (kernel, kernel), sigmaX=0)
    return cv2.addWeighted(image_u8, 1.0 + amount, blurred, -amount, 0)


def contrast_stretch_lab_l(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Stretch the LAB-L channel to improve low-contrast dermoscopy images."""
    image_u8 = _require_rgb_uint8(image, "LAB-L contrast stretch")
    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    lo = float(np.percentile(l_channel, lower_percentile))
    hi = float(np.percentile(l_channel, upper_percentile))
    if hi <= lo:
        return image_u8.copy()
    stretched_l = (l_channel.astype(np.float32) - lo) * (255.0 / (hi - lo))
    stretched_l = np.clip(stretched_l, 0, 255).astype(np.uint8)
    enhanced = cv2.merge((stretched_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def _lab_l_channel(image: np.ndarray) -> np.ndarray:
    image_u8 = _require_rgb_uint8(image, "Luminance channel extraction")
    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)
    return lab[:, :, 0]


def append_luminance_channel(
    image: np.ndarray,
    *,
    channel_type: str = "lab_l",
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Append a fourth luminance/illumination channel to an RGB image."""
    image_u8 = _require_rgb_uint8(image, "extra luminance channel")
    normalized = str(channel_type).lower()
    if normalized in {"none", "", "false"}:
        return image_u8.copy()
    if normalized == "lab_l":
        extra = _lab_l_channel(image_u8)
    elif normalized in {"zero_mask", "zeros", "empty_mask"}:
        extra = np.zeros(image_u8.shape[:2], dtype=np.uint8)
    elif normalized == "clahe_l":
        l_channel = _lab_l_channel(image_u8)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        extra = clahe.apply(l_channel)
    elif normalized in {"hsv_v", "v_star"}:
        hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
        extra = hsv[:, :, 2]
    elif normalized in {"illumination_gray", "illum_gray", "illumination_invariant_gray"}:
        extra = illumination_invariant_gray(image_u8)
    elif normalized in {"shade_attenuated", "shading_attenuated", "shade_attenuated_gray"}:
        extra = shade_attenuated_gray(image_u8)
    elif normalized in {"chromaticity_red", "red_chromaticity", "rgb_norm_band"}:
        extra = chromaticity_red_channel(image_u8)
    elif normalized in {"optical_density_pc1", "od_pc1", "pigment_pc1", "melanin_proxy"}:
        extra = optical_density_pc1_channel(image_u8)
    elif normalized in {"dark_artifact_mask", "hair_mask", "artifact_mask"}:
        extra = detect_dark_artifact_mask(image_u8).astype(np.uint8) * 255
    elif normalized in {"black_frame_mask", "frame_mask"}:
        extra = detect_black_frame_mask(image_u8).astype(np.uint8) * 255
    elif normalized in {"vignette_mask", "falloff_mask"}:
        extra = vignette_mask(image_u8)
    elif normalized in {"specular_mask", "highlight_mask"}:
        extra = specular_highlight_mask(image_u8).astype(np.uint8) * 255
    elif normalized in {"superpixel_boundary", "slic_boundary", "region_boundary"}:
        extra = superpixel_boundary_proxy(image_u8)
    elif normalized in {"saliency", "saliency_proxy"}:
        extra = saliency_proxy(image_u8)
    elif normalized in {"lbp_texture", "texture_lbp"}:
        extra = lbp_texture_proxy(image_u8)
    elif normalized in {"gabor_texture", "texture_gabor"}:
        extra = gabor_texture_proxy(image_u8)
    elif normalized in {"wavelet_energy", "frequency_energy", "high_frequency_energy"}:
        extra = wavelet_energy_proxy(image_u8)
    else:
        raise ValueError(f"Unsupported extra channel type: {channel_type}")
    return np.concatenate([image_u8, extra[:, :, None].astype(np.uint8)], axis=2)


def preprocess_image(
    image: np.ndarray,
    *,
    restoration_config: dict[str, Any] | None = None,
    pcds_mr: bool = False,
    pcds_mr_config: PCDSMRConfig | None = None,
    use_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
    filter_type: str = "none",
    filter_kernel_size: int = 3,
    use_contrast_stretch: bool = False,
    contrast_stretch_lower: float = 1.0,
    contrast_stretch_upper: float = 99.0,
    color_constancy: str | None = None,
    gamma: float | None = None,
    remove_hair: bool = False,
    hair_blackhat_kernel_size: int = 9,
    hair_threshold: int = 10,
    hair_inpaint_radius: int = 3,
    hair_max_mask_ratio: float | None = None,
    sharpen: bool = False,
    sharpen_amount: float = 0.5,
    sharpen_kernel_size: int = 3,
    retinex: bool = False,
    retinex_sigma: float = 30.0,
    shading_correction: bool = False,
    shading_correction_sigma: float = 15.0,
    specular_inpaint: bool = False,
    specular_value_threshold: int = 235,
    specular_saturation_threshold: int = 40,
    specular_inpaint_radius: int = 3,
    specular_max_mask_ratio: float | None = 0.05,
    extra_channel_type: str | None = None,
) -> np.ndarray:
    out = image.copy()
    if restoration_config is not None:
        out = apply_restoration_preprocessing(out, restoration_config)
    if remove_hair:
        out = remove_dark_artifacts(
            out,
            blackhat_kernel_size=hair_blackhat_kernel_size,
            threshold=hair_threshold,
            inpaint_radius=hair_inpaint_radius,
            max_mask_ratio=hair_max_mask_ratio,
        )
    normalized_color_constancy = str(color_constancy).lower() if color_constancy is not None else "none"
    if normalized_color_constancy not in {"none", "", "false"}:
        if normalized_color_constancy != "gray_world":
            raise ValueError(f"Unsupported color constancy type: {color_constancy}")
        out = gray_world_color_constancy(out)
    if gamma is not None:
        out = apply_gamma_correction(out, gamma=gamma)
    if retinex:
        out = apply_single_scale_retinex(out, sigma=retinex_sigma)
    if shading_correction:
        out = apply_shading_correction(out, sigma=shading_correction_sigma)
    if specular_inpaint:
        out = inpaint_specular_highlights(
            out,
            value_threshold=specular_value_threshold,
            saturation_threshold=specular_saturation_threshold,
            inpaint_radius=specular_inpaint_radius,
            max_mask_ratio=specular_max_mask_ratio,
        )
    if use_clahe:
        out = apply_clahe_lab(out, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
    if use_contrast_stretch:
        out = contrast_stretch_lab_l(
            out,
            lower_percentile=contrast_stretch_lower,
            upper_percentile=contrast_stretch_upper,
        )
    out = apply_filter(out, filter_type=filter_type, kernel_size=filter_kernel_size)
    if sharpen:
        out = unsharp_mask(out, amount=sharpen_amount, kernel_size=sharpen_kernel_size)
    # Apply the candidate only after the established Loop20 representation.
    if pcds_mr:
        out = apply_pcds_mr(out, config=pcds_mr_config).image
    if (
        extra_channel_type not in {None, "", "none", "false"}
        and str(extra_channel_type).lower() not in PACKED_TYPES
    ):
        out = append_luminance_channel(
            out,
            channel_type=str(extra_channel_type),
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
        )
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _select_quality_branch(image: np.ndarray, quality_aware_cfg: dict[str, Any]) -> str:
    hair_branch = quality_aware_cfg.get("hair_artifact")
    if hair_branch is not None:
        hair_cfg = hair_branch.get("hair_removal", {}) if isinstance(hair_branch, dict) else {}
        mask = detect_dark_artifact_mask(
            image,
            blackhat_kernel_size=int(hair_cfg.get("blackhat_kernel_size", 9)),
            threshold=int(hair_cfg.get("threshold", 10)),
        )
        min_ratio = float(quality_aware_cfg.get("hair_artifact_min_ratio", 0.002))
        if float(mask.mean()) >= min_ratio:
            return "hair_artifact"

    from lesion_robustness.image_quality import classify_domain_quality

    return classify_domain_quality(
        image,
        low_brightness_mean=float(quality_aware_cfg.get("low_brightness_mean", 122.0)),
        low_contrast_std=float(quality_aware_cfg.get("low_contrast_std", 13.0)),
        jpeg_blockiness_min=float(quality_aware_cfg.get("jpeg_blockiness_min", 0.02)),
    )


def _resolve_quality_aware_config(image: np.ndarray, preprocessing: dict[str, Any]) -> dict[str, Any]:
    quality_aware_cfg = preprocessing.get("quality_aware", {})
    if not isinstance(quality_aware_cfg, dict) or not bool(quality_aware_cfg.get("enabled", True)):
        return preprocessing
    base_cfg = {key: value for key, value in preprocessing.items() if key != "quality_aware"}
    branch = _select_quality_branch(image, quality_aware_cfg)
    branch_cfg = quality_aware_cfg.get(branch, quality_aware_cfg.get("clean", {}))
    if not isinstance(branch_cfg, dict):
        branch_cfg = {}
    return _deep_merge(base_cfg, branch_cfg)


def _resolve_quality_policy_config(image: np.ndarray, preprocessing: dict[str, Any]) -> dict[str, Any]:
    policy_cfg = preprocessing.get("quality_policy", {})
    if not isinstance(policy_cfg, dict) or not bool(policy_cfg.get("enabled", False)):
        return preprocessing
    base_cfg = {key: value for key, value in preprocessing.items() if key != "quality_policy"}
    specular_ratio = float(
        specular_highlight_mask(
            image,
            value_threshold=int(policy_cfg.get("specular_value_threshold", 235)),
            saturation_threshold=int(policy_cfg.get("specular_saturation_threshold", 40)),
        ).mean()
    )
    gray = cv2.cvtColor(_require_rgb_uint8(image, "quality policy"), cv2.COLOR_RGB2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if specular_ratio >= float(policy_cfg.get("specular_min_ratio", 0.005)) or blur_var <= float(
        policy_cfg.get("blur_laplacian_var_max", -1.0)
    ):
        branch = "specular_or_blur"
    else:
        branch = "clean"
    branch_cfg = policy_cfg.get(branch, {})
    if not isinstance(branch_cfg, dict):
        branch_cfg = {}
    return _deep_merge(base_cfg, branch_cfg)


def preprocess_image_from_config(image: np.ndarray, preprocessing: dict[str, Any] | None) -> np.ndarray:
    """Apply preprocessing from a repo config block, including quality-aware branches."""
    cfg = _resolve_quality_policy_config(image, preprocessing or {})
    cfg = _resolve_quality_aware_config(image, cfg)
    restoration_cfg = cfg.get("restoration", {})
    pcds_mr_cfg = cfg.get("pcds_mr", {})
    clahe = cfg.get("clahe", {})
    filter_cfg = cfg.get("filter", {})
    contrast_cfg = cfg.get("contrast_stretch", {})
    color_constancy_cfg = cfg.get("color_constancy", {})
    gamma_cfg = cfg.get("gamma", {})
    hair_cfg = cfg.get("hair_removal", {})
    sharpen_cfg = cfg.get("sharpen", {})
    retinex_cfg = cfg.get("retinex", {})
    shading_cfg = cfg.get("shading_correction", {})
    specular_cfg = cfg.get("specular_inpaint", {})
    extra_channel_cfg = cfg.get("extra_channel", {})
    return preprocess_image(
        image,
        restoration_config=(
            dict(restoration_cfg) if bool(restoration_cfg.get("enabled", False)) else None
        ),
        pcds_mr=bool(pcds_mr_cfg.get("enabled", False)),
        pcds_mr_config=(
            PCDSMRConfig.from_mapping(pcds_mr_cfg)
            if bool(pcds_mr_cfg.get("enabled", False))
            else None
        ),
        use_clahe=bool(clahe.get("enabled", False)),
        clahe_clip_limit=float(clahe.get("clip_limit", 2.0)),
        clahe_tile_grid_size=tuple(clahe.get("tile_grid_size", (8, 8))),
        filter_type=filter_cfg.get("type", "none"),
        filter_kernel_size=int(filter_cfg.get("kernel_size", 3)),
        use_contrast_stretch=bool(contrast_cfg.get("enabled", False)),
        contrast_stretch_lower=float(contrast_cfg.get("lower_percentile", 1.0)),
        contrast_stretch_upper=float(contrast_cfg.get("upper_percentile", 99.0)),
        color_constancy=(
            color_constancy_cfg.get("type", "gray_world")
            if bool(color_constancy_cfg.get("enabled", False))
            else None
        ),
        gamma=float(gamma_cfg.get("value", 1.0)) if bool(gamma_cfg.get("enabled", False)) else None,
        remove_hair=bool(hair_cfg.get("enabled", False)),
        hair_blackhat_kernel_size=int(hair_cfg.get("blackhat_kernel_size", 9)),
        hair_threshold=int(hair_cfg.get("threshold", 10)),
        hair_inpaint_radius=int(hair_cfg.get("inpaint_radius", 3)),
        hair_max_mask_ratio=(
            float(hair_cfg["max_mask_ratio"]) if "max_mask_ratio" in hair_cfg else None
        ),
        sharpen=bool(sharpen_cfg.get("enabled", False)),
        sharpen_amount=float(sharpen_cfg.get("amount", 0.5)),
        sharpen_kernel_size=int(sharpen_cfg.get("kernel_size", 3)),
        retinex=bool(retinex_cfg.get("enabled", False)),
        retinex_sigma=float(retinex_cfg.get("sigma", 30.0)),
        shading_correction=bool(shading_cfg.get("enabled", False)),
        shading_correction_sigma=float(shading_cfg.get("sigma", 15.0)),
        specular_inpaint=bool(specular_cfg.get("enabled", False)),
        specular_value_threshold=int(specular_cfg.get("value_threshold", 235)),
        specular_saturation_threshold=int(specular_cfg.get("saturation_threshold", 40)),
        specular_inpaint_radius=int(specular_cfg.get("inpaint_radius", 3)),
        specular_max_mask_ratio=(
            float(specular_cfg["max_mask_ratio"]) if "max_mask_ratio" in specular_cfg else 0.05
        ),
        extra_channel_type=(
            extra_channel_cfg.get("type", "lab_l")
            if bool(extra_channel_cfg.get("enabled", False))
            else None
        ),
    )
