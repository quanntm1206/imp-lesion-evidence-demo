from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from lesion_robustness.demo.source_fetch import (
    extract_selected_members,
    extract_selected_members_from_archives,
)


def _archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_extract_selected_members_uses_case_insensitive_basenames(tmp_path: Path) -> None:
    archive = _archive(
        tmp_path / "source.zip",
        {
            "nested/ISIC_0001.jpg": b"image",
            "nested/ISIC_0001_Segmentation.png": b"mask",
            "nested/unrelated.txt": b"ignore",
        },
    )
    output = tmp_path / "output"
    paths = extract_selected_members(
        archive,
        {"isic_0001.jpg", "isic_0001_segmentation.png"},
        output,
    )
    assert {path.name.lower() for path in paths} == {
        "isic_0001.jpg",
        "isic_0001_segmentation.png",
    }
    assert (output / "ISIC_0001.jpg").read_bytes() == b"image"


def test_extract_selected_members_rejects_missing_or_duplicate_names(tmp_path: Path) -> None:
    missing = _archive(tmp_path / "missing.zip", {"folder/a.jpg": b"a"})
    with pytest.raises(ValueError, match="missing archive members"):
        extract_selected_members(missing, {"a.jpg", "a_mask.png"}, tmp_path / "missing")
    duplicate = _archive(
        tmp_path / "duplicate.zip",
        {"one/a.jpg": b"a", "two/A.JPG": b"b"},
    )
    with pytest.raises(ValueError, match="duplicate archive basename"):
        extract_selected_members(duplicate, {"a.jpg"}, tmp_path / "duplicate")


def test_extract_selected_members_combines_disjoint_archives(tmp_path: Path) -> None:
    training = _archive(tmp_path / "training.zip", {"train/a.jpg": b"a"})
    test = _archive(tmp_path / "test.zip", {"test/b.jpg": b"b"})
    paths = extract_selected_members_from_archives(
        [training, test],
        {"a.jpg", "b.jpg"},
        tmp_path / "combined",
    )
    assert {path.name for path in paths} == {"a.jpg", "b.jpg"}
