"""Geometry and presentation helpers for the Loop206 demo."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MODEL_SHAPE = (384, 384)
MAX_PIXELS = 16_000_000
MIN_SIDE = 32


@dataclass(frozen=True)
class PreparedImage:
    original_rgb: np.ndarray
    model_rgb: np.ndarray
    original_shape: tuple[int, int]


def _validate_rgb_uint8(image: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{label} must be an RGB HxWx3 array")
    if value.dtype != np.uint8:
        raise ValueError(f"{label} must use uint8 pixels")
    return value


def prepare_image(image: np.ndarray) -> PreparedImage:
    source = _validate_rgb_uint8(image, label="image")
    height, width = map(int, source.shape[:2])
    if min(height, width) < MIN_SIDE:
        raise ValueError(f"image minimum side is {MIN_SIDE} pixels")
    if height * width > MAX_PIXELS:
        raise ValueError("image exceeds the 16 megapixels limit")
    original = np.ascontiguousarray(source).copy()
    resized = cv2.resize(
        original,
        (MODEL_SHAPE[1], MODEL_SHAPE[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return PreparedImage(
        original_rgb=original,
        model_rgb=np.ascontiguousarray(resized, dtype=np.uint8),
        original_shape=(height, width),
    )


def restore_probability(
    probability: np.ndarray, original_shape: tuple[int, int]
) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("probability must be a finite 2D array")
    height, width = map(int, original_shape)
    if min(height, width) <= 0:
        raise ValueError("original geometry must be positive")
    restored = cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(restored, 0.0, 1.0).astype(np.float32, copy=False)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.35,
    color: tuple[int, int, int] = (230, 68, 46),
) -> np.ndarray:
    source = _validate_rgb_uint8(image, label="overlay image")
    binary = np.asarray(mask)
    if binary.shape != source.shape[:2]:
        raise ValueError("overlay mask geometry mismatch")
    opacity = float(alpha)
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("overlay alpha must be within [0, 1]")
    tint = np.asarray(color, dtype=np.int16)
    if tint.shape != (3,) or np.any(tint < 0) or np.any(tint > 255):
        raise ValueError("overlay color must contain three uint8 values")
    output = source.copy()
    selected = binary.astype(bool)
    if selected.any():
        blended = (
            output[selected].astype(np.float32) * (1.0 - opacity)
            + tint.astype(np.float32) * opacity
        )
        output[selected] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return output
