from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def as_float01(image: np.ndarray) -> np.ndarray:
    """Return image as float32 in [0, 1]."""
    arr = image.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def restore_dtype(float_image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] float image back to the dtype/range of reference."""
    clipped = np.clip(float_image, 0.0, 1.0)
    if np.issubdtype(reference.dtype, np.integer):
        return np.rint(clipped * 255.0).astype(reference.dtype)
    return clipped.astype(reference.dtype, copy=False)


def drop_os_page_cache_hint(path: str | Path) -> None:
    """Best-effort Linux hint to evict a just-read file from OS page cache."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        return
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def read_rgb(path: str | Path, *, drop_os_page_cache_after_read: bool = False) -> np.ndarray:
    """Read an RGB image as uint8."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if drop_os_page_cache_after_read:
        drop_os_page_cache_hint(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_image_hw(path: str | Path, *, drop_os_page_cache_after_read: bool = False) -> tuple[int, int]:
    """Read image height/width from metadata without decoding the pixel array."""
    with Image.open(path) as image:
        width, height = image.size
    if drop_os_page_cache_after_read:
        drop_os_page_cache_hint(path)
    return int(height), int(width)


def read_mask(path: str | Path, *, drop_os_page_cache_after_read: bool = False) -> np.ndarray:
    """Read a binary mask as uint8 values {0, 1}."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if drop_os_page_cache_after_read:
        drop_os_page_cache_hint(path)
    return (mask > 127).astype(np.uint8)


def read_soft_mask(path: str | Path, *, drop_os_page_cache_after_read: bool = False) -> np.ndarray:
    """Read a grayscale mask as float32 probabilities in [0, 1]."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if drop_os_page_cache_after_read:
        drop_os_page_cache_hint(path)
    return np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)


def read_multi_mask(
    paths: list[str | Path] | tuple[str | Path, ...],
    *,
    mode: str,
    drop_os_page_cache_after_read: bool = False,
) -> np.ndarray:
    """Combine multiple annotator masks into a soft mean or binary union target."""
    if not paths:
        raise ValueError("read_multi_mask requires at least one mask path")
    masks = [
        read_soft_mask(path, drop_os_page_cache_after_read=drop_os_page_cache_after_read)
        for path in paths
    ]
    shapes = {mask.shape for mask in masks}
    if len(shapes) != 1:
        raise ValueError(f"Multi-annotator masks must have identical shapes, got {sorted(shapes)}")
    stacked = np.stack(masks, axis=0)
    if mode == "soft_mean":
        return stacked.mean(axis=0).astype(np.float32, copy=False)
    if mode == "union":
        return (stacked.max(axis=0) > 0.5).astype(np.float32)
    raise ValueError(f"Unsupported multi-mask mode: {mode}")


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray | None,
    image_size: tuple[int, int],
    *,
    mask_mode: str = "binary",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resize image and optional mask to (height, width)."""
    height, width = image_size
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if mask is None:
        return resized_image, None
    if mask_mode == "binary":
        resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return resized_image, (resized_mask > 0).astype(np.uint8)
    if mask_mode == "soft":
        resized_mask = cv2.resize(mask.astype(np.float32, copy=False), (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image, np.clip(resized_mask, 0.0, 1.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported mask_mode: {mask_mode}")
