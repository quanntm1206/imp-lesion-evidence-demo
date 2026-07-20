from __future__ import annotations

import numpy as np
import pytest

from lesion_robustness.demo.geometry import (
    overlay_mask,
    prepare_image,
    restore_probability,
)


def test_prepare_image_preserves_original_and_resizes_to_model_geometry() -> None:
    image = np.full((240, 320, 3), 127, dtype=np.uint8)

    prepared = prepare_image(image)

    assert prepared.original_rgb.shape == (240, 320, 3)
    assert prepared.model_rgb.shape == (384, 384, 3)
    assert prepared.original_shape == (240, 320)
    assert not np.shares_memory(prepared.original_rgb, image)


@pytest.mark.parametrize(
    "image, message",
    [
        (np.zeros((31, 64, 3), dtype=np.uint8), "minimum side"),
        (np.zeros((4001, 4001, 3), dtype=np.uint8), "16 megapixels"),
        (np.zeros((64, 64), dtype=np.uint8), "RGB"),
        (np.zeros((64, 64, 3), dtype=np.float32), "uint8"),
    ],
)
def test_prepare_image_rejects_unsafe_inputs(image: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_image(image)


def test_restore_probability_uses_bilinear_geometry_before_thresholding() -> None:
    probability = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    restored = restore_probability(probability, (4, 6))

    assert restored.shape == (4, 6)
    assert restored.dtype == np.float32
    assert 0.0 < restored[1, 2] < 1.0
    np.testing.assert_array_equal(restored >= 0.5, np.tile([0, 0, 0, 1, 1, 1], (4, 1)))


def test_overlay_mask_is_bounded_and_does_not_mutate_source() -> None:
    image = np.full((48, 64, 3), 240, dtype=np.uint8)
    source = image.copy()
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[8:40, 12:52] = 1

    overlay = overlay_mask(image, mask, alpha=0.35, color=(255, 32, 16))

    np.testing.assert_array_equal(image, source)
    assert overlay.dtype == np.uint8
    assert overlay.min() >= 0
    assert overlay.max() <= 255
    assert np.any(overlay != source)
    with pytest.raises(ValueError, match="alpha"):
        overlay_mask(image, mask, alpha=1.1)
