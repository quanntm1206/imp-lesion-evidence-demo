from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lesion_robustness.demo.fixed_cache import (
    FixedCacheExpectations,
    FixedCachePair,
    sha256_rgb_array,
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    directory: Path,
    *,
    arm: str,
    image: np.ndarray,
    channel: np.ndarray,
    group_key: str = "group-fixed",
    corruption: str = "clean",
    image_path: str = "/historical/absolute/path.jpg",
) -> Path:
    directory.mkdir()
    data_path = directory / "contours.uint8.mmap"
    data_path.write_bytes(np.ascontiguousarray(channel, dtype=np.uint8).tobytes())
    row = {
        "index": 0,
        "sample_id": "sample-fixed",
        "group_key": group_key,
        "image_path": image_path,
        "corruption": corruption,
        "source_split": "train",
        "runtime_split": "train_screen_holdout",
        "fold": 4,
        "holdout_dataset_index": 0,
        "input_rgb_sha256": sha256_rgb_array(image),
        "base_threshold": 0.07500000000000001,
        "locked_config": "neutral_mid_30_s2",
        "candidate_fallback_used": False,
        "candidate_fallback_reason": "none",
    }
    payload = {
        "schema_version": "loop206.leakage_safe_pilot_cache.v2",
        "artifact_type": "loop206_packed_binary_channel",
        "status": "passed",
        "arm": arm,
        "count": 1,
        "shape": [4, 4],
        "source_row_count": 1,
        "fit_clean_rows": 0,
        "holdout_rows_per_corruption": 1,
        "source_split_counts": {"train": 1},
        "allowed_runtime_splits": ["train", "train_screen_holdout"],
        "runtime_split_counts": {"train_screen_holdout": 1},
        "corruption_counts": {"clean": 1},
        "input_rgb_sha256_count": 1,
        "locked_active_contour_config": {"name": "neutral_mid_30_s2"},
        "provenance": {"config_sha256": "a" * 64},
        "data": {
            "dtype": "uint8",
            "file": data_path.name,
            "sha256": _sha256(data_path),
        },
        "rows": [row],
        "rows_sha256": _canonical_hash([row]),
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="ascii")
    return manifest


def _pair(tmp_path: Path, *, zero_value: int = 0) -> tuple[FixedCachePair, np.ndarray]:
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    zero = _write_manifest(
        tmp_path / "zero",
        arm="zero_control",
        image=image,
        channel=np.full((4, 4), zero_value, dtype=np.uint8),
    )
    candidate = _write_manifest(
        tmp_path / "candidate",
        arm="candidate",
        image=image,
        channel=np.full((4, 4), 255, dtype=np.uint8),
    )
    expectations = FixedCacheExpectations(
        count=1,
        shape=(4, 4),
        candidate_manifest_sha256=_sha256(candidate),
        candidate_data_sha256=_sha256(candidate.parent / "contours.uint8.mmap"),
        zero_manifest_sha256=_sha256(zero),
        zero_data_sha256=_sha256(zero.parent / "contours.uint8.mmap"),
    )
    return FixedCachePair(candidate, zero, expectations=expectations), image


def _holdout_row() -> dict[str, object]:
    return {
        "sample_id": "sample-fixed",
        "group_key": "group-fixed",
        "role": "holdout",
        "fold": 4,
        "source_split": "train",
    }


def test_fixed_lookup_uses_group_and_corruption_not_historical_path(tmp_path: Path) -> None:
    pair, image = _pair(tmp_path)

    record = pair.lookup_fixture(_holdout_row(), corruption="clean", input_rgb=image)

    np.testing.assert_array_equal(record.control_channel, 0)
    np.testing.assert_array_equal(record.candidate_channel, 255)
    assert record.group_key == "group-fixed"
    assert record.corruption == "clean"
    assert not hasattr(record, "metadata")


def test_fixed_cache_rejects_declared_hash_mismatch(tmp_path: Path) -> None:
    pair, _ = _pair(tmp_path)
    bad = deepcopy(pair.expectations)
    object.__setattr__(bad, "candidate_manifest_sha256", "0" * 64)

    with pytest.raises(ValueError, match="candidate manifest hash"):
        FixedCachePair(pair.candidate.manifest_path, pair.zero.manifest_path, expectations=bad)


@pytest.mark.parametrize(
    "row, corruption, mutate_image, message",
    [
        ({**_holdout_row(), "group_key": "wrong"}, "clean", False, "group"),
        (_holdout_row(), "gaussian_noise", False, "corruption"),
        (_holdout_row(), "clean", True, "input RGB hash"),
        ({**_holdout_row(), "role": "fit"}, "clean", False, "holdout"),
    ],
)
def test_fixed_lookup_rejects_wrong_binding(
    tmp_path: Path,
    row: dict[str, object],
    corruption: str,
    mutate_image: bool,
    message: str,
) -> None:
    pair, image = _pair(tmp_path)
    if mutate_image:
        image = image.copy()
        image[0, 0, 0] ^= 1

    with pytest.raises((KeyError, ValueError), match=message):
        pair.lookup_fixture(row, corruption=corruption, input_rgb=image)


def test_fixed_cache_rejects_nonzero_control_channel(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="all-zero"):
        _pair(tmp_path, zero_value=1)


def test_fixture_cache_cannot_accept_caller_asserted_runtime_provenance(
    tmp_path: Path,
) -> None:
    pair, _ = _pair(tmp_path)

    with pytest.raises(TypeError):
        FixedCachePair(
            pair.candidate.manifest_path,
            pair.zero.manifest_path,
            expectations=pair.expectations,
            runtime_config_sha256="a" * 64,
        )


def test_fixture_cache_exposes_no_free_form_production_lookup(tmp_path: Path) -> None:
    pair, _ = _pair(tmp_path)

    assert not hasattr(pair, "lookup")


def test_fixed_presentation_is_the_internal_runtime_image_not_source_geometry() -> None:
    import lesion_robustness.demo.fixed_cache as module

    decoded_source = np.full((96, 128, 3), 12, dtype=np.uint8)
    runtime = np.full((4, 4, 3), 34, dtype=np.uint8)

    presentation = module._fixed_presentation_rgb(decoded_source, runtime)

    np.testing.assert_array_equal(presentation, runtime)
    assert not np.shares_memory(presentation, runtime)


@pytest.mark.parametrize("arm", ["candidate", "zero"])
def test_validated_cache_does_not_observe_post_construction_file_mutation(
    tmp_path: Path, arm: str
) -> None:
    pair, image = _pair(tmp_path)
    cache = pair.candidate if arm == "candidate" else pair.zero
    replacement = 0 if arm == "candidate" else 255
    with cache.data_path.open("r+b") as handle:
        handle.write(bytes([replacement]) * (4 * 4))
        handle.flush()

    record = pair.lookup_fixture(_holdout_row(), corruption="clean", input_rgb=image)

    assert not cache.data.flags.writeable
    np.testing.assert_array_equal(record.candidate_channel, 255)
    np.testing.assert_array_equal(record.control_channel, 0)
    assert cache.data_sha256 == (
        pair.expectations.candidate_data_sha256
        if arm == "candidate"
        else pair.expectations.zero_data_sha256
    )
