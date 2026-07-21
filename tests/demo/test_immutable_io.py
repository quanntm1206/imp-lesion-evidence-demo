from __future__ import annotations

import hashlib
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import pytest

from lesion_robustness.demo.immutable_io import ImmutableSnapshot


def test_large_snapshot_streams_to_owned_file_and_survives_source_replacement(
    tmp_path: Path,
) -> None:
    expected = bytes(range(251)) * 8
    source = tmp_path / "artifact.bin"
    source.write_bytes(expected)

    snapshot = ImmutableSnapshot.read(source, chunk_size=17, spool_limit=64)
    source.write_bytes(b"replacement")

    assert snapshot.size == len(expected)
    assert snapshot.is_file_backed
    with snapshot.open() as handle:
        assert handle.read() == expected
    with snapshot.open() as handle:
        assert handle.read() == expected


def test_rgb_hash_and_decode_use_the_captured_image_bytes(tmp_path: Path) -> None:
    expected = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    source = tmp_path / "image.png"
    Image.fromarray(expected, mode="RGB").save(source)
    snapshot = ImmutableSnapshot.read(source)
    Image.fromarray(np.zeros_like(expected), mode="RGB").save(source)

    decoded = snapshot.decode_rgb()

    np.testing.assert_array_equal(decoded, expected)
    expected_hash = hashlib.sha256(b"4x5x3|" + expected.tobytes()).hexdigest()
    assert snapshot.decoded_rgb_sha256(decoded) == expected_hash


def test_binary_mask_hash_and_decode_use_the_captured_bytes(tmp_path: Path) -> None:
    expected = np.zeros((4, 5), dtype=np.uint8)
    expected[1:3, 2:4] = 1
    source = tmp_path / "mask.png"
    Image.fromarray(expected * 255, mode="L").save(source)
    snapshot = ImmutableSnapshot.read(source)
    Image.fromarray(np.zeros_like(expected), mode="L").save(source)

    decoded = snapshot.decode_binary_mask()

    np.testing.assert_array_equal(decoded, expected)
    expected_hash = hashlib.sha256(b"4x5|" + expected.tobytes()).hexdigest()
    assert snapshot.decoded_binary_mask_sha256(decoded) == expected_hash


def test_verified_source_decode_is_independent_of_public_upload_pixel_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "trusted-indexed-source.png"
    Image.new("L", (4_000, 4_250), color=127).save(source)
    snapshot = ImmutableSnapshot.read(source)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 16_000_000)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
        decoded = snapshot.decode_rgb()

    assert decoded.shape == (4_250, 4_000, 3)
