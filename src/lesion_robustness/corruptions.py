from __future__ import annotations

from io import BytesIO
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from lesion_robustness.image_utils import as_float01, restore_dtype


def low_brightness(image: np.ndarray, factor: float = 0.6) -> np.ndarray:
    degraded = as_float01(image) * factor
    return restore_dtype(degraded, image)


def low_contrast(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    degraded = (as_float01(image) - 0.5) * factor + 0.5
    return restore_dtype(degraded, image)


def gaussian_noise(image: np.ndarray, sigma: float = 0.05, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = as_float01(image) + rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return restore_dtype(noisy, image)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma)


def jpeg_compression(image: np.ndarray, quality: int = 40) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    pil_image = Image.fromarray(image.astype(np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)


def uneven_illumination(image: np.ndarray, strength: float = 0.35) -> np.ndarray:
    height, width = image.shape[:2]
    gradient = np.linspace(1.0 - strength, 1.0 + strength, width, dtype=np.float32)
    field = np.tile(gradient, (height, 1))
    if image.ndim == 3:
        field = field[..., None]
    return restore_dtype(as_float01(image) * field, image)


def hair_artifact(
    image: np.ndarray,
    line_count: int = 8,
    thickness: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.copy()
    height, width = out.shape[:2]
    color = (15, 15, 15) if out.ndim == 3 else 15
    for _ in range(line_count):
        x1, x2 = rng.integers(0, width, size=2)
        y1, y2 = rng.integers(0, height, size=2)
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)
    return out


CORRUPTIONS: dict[str, Callable[..., np.ndarray]] = {
    "low_brightness": low_brightness,
    "low_contrast": low_contrast,
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "jpeg_compression": jpeg_compression,
    "uneven_illumination": uneven_illumination,
    "hair_artifact": hair_artifact,
}


SEEDED_CORRUPTIONS = {"gaussian_noise", "hair_artifact"}


def deterministic_corruption_kwargs(
    name: str,
    kwargs: dict,
    *,
    base_seed: int,
    index: int,
) -> dict:
    out = dict(kwargs)
    if name in SEEDED_CORRUPTIONS and "seed" not in out:
        out["seed"] = int(base_seed) + int(index)
    return out


def apply_corruption(image: np.ndarray, name: str, **kwargs) -> np.ndarray:
    try:
        corruption = CORRUPTIONS[name]
    except KeyError as exc:
        names = ", ".join(sorted(CORRUPTIONS))
        raise ValueError(f"Unknown corruption '{name}'. Available: {names}") from exc
    return corruption(image, **kwargs)
