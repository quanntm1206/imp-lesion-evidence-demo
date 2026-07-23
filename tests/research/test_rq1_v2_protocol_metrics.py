from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import cv2
import PIL
from scripts.research.freeze_rq1_v2_conditions import freeze

from lesion_robustness.research.rq1_metrics import restore_probability, score
from lesion_robustness.research.rq1_protocol import (
    CONDITIONS,
    apply_condition,
    build_condition_panel,
    condition_seed,
    load_condition_golden,
    imp_input_hashes,
    nnunet_input_hashes,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "rq1_v2_condition_input.rgb"
GOLDEN = ROOT / "tests" / "fixtures" / "rq1_v2_condition_golden.json"
PROTOCOL = ROOT / "experiments" / "rq1_v2" / "protocol.json"

EXPECTED_SEEDS = {
    "clean": 11014148199122613057,
    "low_brightness": 4216101521240603699,
    "low_contrast": 6026483641637571443,
    "gaussian_noise": 2654481043629223872,
    "gaussian_blur": 11325026900250914310,
    "jpeg_compression": 15707255423797903273,
}
EXPECTED_HASHES = {
    "clean": "12adc9dff80688800f2f591f0da6ab2f8109d61d910697801f57669ec0d719d3",
    "low_brightness": "b43f19fa01e921406ad6053bff6f73cedcc243de60293171f23ea6aa8edea44f",
    "low_contrast": "2c93c55179f11c23fb1fae891945001f367158d85517da42b6534bed6581d778",
    "gaussian_noise": "4996d88cc057841a22251ce945ecb84a8e02a4b4296f7a4365f3cdffed372f73",
    "gaussian_blur": "098fd884dbff5e22e2c146ea002d94e87c51dde54ecc74a79fb4a73decceb411",
    "jpeg_compression": "896c0575202304eccbf8a69fa8302ac66db3d09d60368fe0ffa01e481c5eaa56",
}


def _fixture_rgb() -> np.ndarray:
    raw = FIXTURE.read_bytes()
    assert len(raw) == 32 * 32 * 3
    return np.frombuffer(raw, dtype=np.uint8).reshape(32, 32, 3)


def test_synthetic_fixture_is_exact_contiguous_rgb_arange():
    rgb = _fixture_rgb()
    expected = (np.arange(32 * 32 * 3, dtype=np.uint32) % 256).astype(np.uint8)
    assert rgb.flags.c_contiguous
    assert rgb.dtype == np.uint8
    assert np.array_equal(rgb.reshape(-1), expected)


def test_condition_seed_is_first_eight_sha256_bytes_big_endian():
    value = "protocol|group|sample|gaussian_noise"
    expected = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
    assert condition_seed("protocol", "group", "sample", "gaussian_noise") == expected
    assert condition_seed("protocol", "group", "sample", "gaussian_noise") == expected
    assert condition_seed("protocol", "group", "sample", "clean") != expected


def test_gaussian_noise_matches_direct_pcg64_reference_vector():
    rgb = _fixture_rgb()
    seed = condition_seed("protocol", "group", "sample", "gaussian_noise")
    source = rgb.astype(np.float32) / np.float32(255.0)
    rng = np.random.Generator(np.random.PCG64(seed))
    expected = source + rng.standard_normal(source.shape, dtype=np.float32) * np.float32(0.05)
    expected = np.floor(np.clip(expected, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)
    assert np.array_equal(apply_condition(rgb, "gaussian_noise", seed), expected)


def test_each_condition_matches_independent_frozen_reference_vector():
    rgb = _fixture_rgb()
    golden = load_condition_golden()
    protocol_sha = golden["protocol_sha256"]
    row = (golden["fixture_group_key"], golden["fixture_sample_id"])
    for name in CONDITIONS:
        seed = condition_seed(protocol_sha, row[0], row[1], name)
        output = apply_condition(rgb, name, seed)
        assert seed == EXPECTED_SEEDS[name]
        assert hashlib.sha256(output.tobytes()).hexdigest() == EXPECTED_HASHES[name]
        assert output.shape == (32, 32, 3)


def test_panel_is_ordered_cached_and_both_arms_share_bytes():
    rgb = _fixture_rgb()
    protocol = json.loads(PROTOCOL.read_text(encoding="ascii"))
    golden = load_condition_golden()
    protocol["protocol_sha256"] = golden["protocol_sha256"]
    row = {
        "group_key": golden["fixture_group_key"],
        "sample_id": golden["fixture_sample_id"],
    }
    panel = build_condition_panel(rgb, row, protocol)
    again = build_condition_panel(rgb, row, protocol)
    assert panel.names == tuple(CONDITIONS)
    assert panel.hashes == again.hashes
    assert panel.seeds == again.seeds
    assert panel.imp_inputs == panel.nnunet_inputs
    assert panel.imp_inputs is panel.nnunet_inputs
    assert imp_input_hashes(panel) == nnunet_input_hashes(panel)
    assert all(value.dtype == np.uint8 and value.flags.c_contiguous for value in panel.imp_inputs)
    assert panel.hashes == golden["condition_rgb_sha256"]
    assert panel.seeds == golden["condition_uint64_seeds"]
    assert panel.ordered_panel_sha256 == golden["ordered_panel_sha256"]
    assert golden["protocol_sha256"] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert golden["input_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert golden["dependency_versions"] == {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
    }
    with pytest.raises(ValueError):
        panel.imp_inputs[0][0, 0, 0] = 0


def test_freezer_refuses_to_overwrite_existing_golden(tmp_path: Path):
    output = tmp_path / "golden.json"
    freeze(PROTOCOL, FIXTURE, output)
    original = output.read_bytes()
    output.write_bytes(original + b"drift")
    drifted = output.read_bytes()
    with pytest.raises(ValueError, match="existing golden drift"):
        freeze(PROTOCOL, FIXTURE, output)
    assert output.read_bytes() == drifted


def test_probability_is_bilinear_restored_before_threshold():
    probability = np.array(
        [[0.0, 1.0], [1.0, 0.0]], dtype=np.float32
    )
    restored = restore_probability(probability, original_hw=(5, 7))
    assert restored.shape == (5, 7)
    assert restored.dtype == np.float32
    # Independent align_corners=False oracle, including exact 0.5 threshold semantics.
    expected = np.array(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=bool,
    )
    assert np.array_equal(restored >= np.float32(0.5), expected)


def test_empty_mask_distance_policy_is_finite_and_normalized():
    empty = np.zeros((3, 4), dtype=np.uint8)
    nonempty = empty.copy()
    nonempty[1, 2] = 1
    both = score(empty, empty)
    one = score(empty, nonempty)
    assert both.hd95_normalized == 0.0
    assert both.assd_normalized == 0.0
    assert one.hd95_normalized == 1.0
    assert one.assd_normalized == 1.0
    assert one.hd95 == pytest.approx(np.hypot(3, 4))
    assert one.assd == pytest.approx(np.hypot(3, 4))
    assert one.empty_policy == "one_empty_diagonal_penalty"
    assert np.isfinite(one.hd95_normalized)


def test_identical_full_frame_masks_have_finite_perfect_boundary_metrics():
    full = np.ones((3, 4), dtype=np.uint8)

    result = score(full, full)

    assert result.boundary_f1 == 1.0
    assert result.hd95 == 0.0
    assert result.assd == 0.0
    assert np.isfinite(result.hd95_normalized)
    assert np.isfinite(result.assd_normalized)


def test_nonidentical_full_frame_mask_comparison_has_finite_boundary_metrics():
    full = np.ones((5, 6), dtype=np.uint8)
    comparison = full.copy()
    comparison[0, 0] = 0

    result = score(full, comparison)

    assert result.boundary_f1 == 1.0
    assert 0.0 <= result.hd95 <= 1.0
    assert 0.0 <= result.assd <= 1.0
    assert np.isfinite(result.hd95_normalized)
    assert np.isfinite(result.assd_normalized)
