from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from lesion_robustness.data_manifest import sha256_file, sha256_rgb
from lesion_robustness.demo.data_index import resolve_loop206_rows


FIELDS = (
    "image_path",
    "mask_path",
    "split",
    "source_dataset",
    "original_id",
    "sha256_raw",
    "sha256_rgb",
    "source_split",
    "loop205_group_key",
    "loop205_fold",
    "loop206_pilot_role",
)


def _write_case(root: Path, sample_id: str) -> tuple[Path, Path]:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{sample_id}.png"
    mask_path = mask_dir / f"{sample_id}_segmentation.png"
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[..., 0] = 80
    image[3:9, 4:12, 1] = 160
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:9, 4:12] = 255
    assert cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert cv2.imwrite(str(mask_path), mask)
    return image_path, mask_path


def _write_manifest(path: Path, image: Path, mask: Path, *, raw_hash: str | None = None) -> Path:
    row = {
        "image_path": f"/home/admin_mugen/datasets/source/images/{image.name}",
        "mask_path": f"/home/admin_mugen/datasets/source/masks/{mask.name}",
        "split": "train_screen_holdout",
        "source_dataset": "isic2018",
        "original_id": image.stem,
        "sha256_raw": raw_hash or sha256_file(image),
        "sha256_rgb": sha256_rgb(image),
        "source_split": "train",
        "loop205_group_key": "component:g1",
        "loop205_fold": "4",
        "loop206_pilot_role": "holdout",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return path


def test_resolver_matches_basename_and_both_hashes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image, mask = _write_case(dataset, "ISIC_0000001")
    manifest = _write_manifest(tmp_path / "manifest.csv", image, mask)
    rows = resolve_loop206_rows(manifest, [dataset], expected_rows=1)
    assert len(rows) == 1
    assert rows[0].image_path == image.resolve()
    assert rows[0].mask_path == mask.resolve()
    assert rows[0].role == "holdout"


def test_resolver_rejects_hash_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image, mask = _write_case(dataset, "ISIC_0000002")
    manifest = _write_manifest(tmp_path / "manifest.csv", image, mask, raw_hash="0" * 64)
    with pytest.raises(ValueError, match="unique hash-verified image"):
        resolve_loop206_rows(manifest, [dataset], expected_rows=1)


def test_resolver_rejects_duplicate_basename(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    image, mask = _write_case(first_root, "ISIC_0000003")
    second_image, second_mask = _write_case(second_root, "ISIC_0000003")
    assert second_image.read_bytes() == image.read_bytes()
    assert second_mask.read_bytes() == mask.read_bytes()
    manifest = _write_manifest(tmp_path / "manifest.csv", image, mask)
    with pytest.raises(ValueError, match="unique hash-verified image"):
        resolve_loop206_rows(manifest, [first_root, second_root], expected_rows=1)
