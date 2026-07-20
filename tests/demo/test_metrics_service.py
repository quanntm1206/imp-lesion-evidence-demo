from __future__ import annotations

import numpy as np
import pytest

from lesion_robustness.demo.metrics_service import evaluate_optional_ground_truth


def test_no_ground_truth_returns_none_without_decoding_predictions() -> None:
    invalid = np.full((16, 16), np.nan, dtype=np.float32)

    assert evaluate_optional_ground_truth(invalid, invalid, None) is None


def test_perfect_masks_return_unit_overlap_and_zero_distance() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1

    metrics = evaluate_optional_ground_truth(mask, mask, mask)

    assert metrics is not None
    for arm in ("control", "candidate"):
        assert metrics[arm]["dice"] == 1.0
        assert metrics[arm]["iou"] == 1.0
        assert metrics[arm]["boundary_f1"] == 1.0
        assert metrics[arm]["hd95"] == 0.0
        assert metrics[arm]["assd"] == 0.0


def test_decodes_grayscale_and_rgb_uint8_masks_at_127() -> None:
    grayscale = np.array([[126, 127], [255, 0]], dtype=np.uint8)
    rgb = np.repeat(grayscale[..., None], 3, axis=2)

    metrics = evaluate_optional_ground_truth(grayscale, rgb, grayscale)

    assert metrics is not None
    assert metrics["control"]["dice"] == 1.0
    assert metrics["candidate"]["dice"] == 1.0


def test_accepts_binary_float_masks() -> None:
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1.0

    metrics = evaluate_optional_ground_truth(mask, mask, mask)

    assert metrics is not None
    assert metrics["control"]["dice"] == 1.0


@pytest.mark.parametrize(
    "invalid, message",
    [
        (np.zeros((16, 16, 4), dtype=np.uint8), "grayscale or RGB"),
        (np.full((16, 16), np.nan, dtype=np.float32), "finite"),
        (np.full((16, 16), np.inf, dtype=np.float32), "finite"),
        (np.full((16, 16), 0.25, dtype=np.float32), "binary"),
        (np.zeros((4000, 4001), dtype=np.uint8), "16 megapixels"),
        (np.zeros((16, 16), dtype=np.int16), "uint8 or binary float"),
    ],
)
def test_rejects_unsafe_or_ambiguous_masks(
    invalid: np.ndarray, message: str
) -> None:
    valid = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        evaluate_optional_ground_truth(invalid, valid, valid)


def test_rejects_geometry_mismatch_for_each_metric_input() -> None:
    valid = np.zeros((16, 16), dtype=np.uint8)
    mismatch = np.zeros((16, 15), dtype=np.uint8)

    with pytest.raises(ValueError, match="geometry mismatch"):
        evaluate_optional_ground_truth(valid, mismatch, valid)
    with pytest.raises(ValueError, match="geometry mismatch"):
        evaluate_optional_ground_truth(valid, valid, mismatch)
