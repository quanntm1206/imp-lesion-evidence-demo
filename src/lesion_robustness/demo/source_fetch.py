from __future__ import annotations

from contextlib import contextmanager
import csv
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping, Sequence
import zipfile


SOURCE_ARCHIVES: dict[str, dict[str, str | list[str]]] = {
    "isic2016": {
        "images": [
            "https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Training_Data.zip",
            "https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Test_Data.zip",
        ],
        "masks": [
            "https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip",
            "https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip",
        ],
    },
    "isic2017": {
        "images": "https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Data.zip",
        "masks": "https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part1_GroundTruth.zip",
    },
    "isic2018": {
        "images": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip",
        "masks": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip",
    },
}
SOURCE_PAGE = "https://challenge.isic-archive.com/data/"


@contextmanager
def _open_binary(source: str | Path) -> Iterator[BinaryIO]:
    value = str(source)
    if value.startswith(("https://", "http://")):
        import fsspec

        with fsspec.open(value, mode="rb", block_size=1 << 20).open() as handle:
            yield handle
    else:
        with Path(source).open("rb") as handle:
            yield handle


def extract_selected_members(
    source: str | Path,
    required_basenames: Sequence[str] | set[str],
    output_dir: str | Path,
) -> list[Path]:
    return extract_selected_members_from_archives([source], required_basenames, output_dir)


def extract_selected_members_from_archives(
    sources: Sequence[str | Path],
    required_basenames: Sequence[str] | set[str],
    output_dir: str | Path,
) -> list[Path]:
    required = {str(name).strip().lower() for name in required_basenames if str(name).strip()}
    if not required:
        return []
    destination_root = Path(output_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    outputs: list[Path] = []
    for source in sources:
        with _open_binary(source) as handle, zipfile.ZipFile(handle) as archive:
            lookup: dict[str, zipfile.ZipInfo] = {}
            duplicates: set[str] = set()
            for member in archive.infolist():
                basename = Path(member.filename).name.lower()
                if member.is_dir() or basename not in required:
                    continue
                if basename in lookup or basename in found:
                    duplicates.add(basename)
                lookup[basename] = member
            if duplicates:
                raise ValueError(f"duplicate archive basename: {sorted(duplicates)}")
            for basename in sorted(lookup):
                member = lookup[basename]
                data = archive.read(member)
                destination = destination_root / Path(member.filename).name
                if destination.exists():
                    if destination.read_bytes() != data:
                        raise FileExistsError(
                            f"refusing to overwrite different dataset file: {destination}"
                        )
                else:
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                found.add(basename)
                outputs.append(destination)
    missing = sorted(required - found)
    if missing:
        raise ValueError(f"missing archive members: {missing[:10]} count={len(missing)}")
    return outputs


def required_files_by_source(manifest: str | Path) -> dict[str, dict[str, set[str]]]:
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        source = row.get("source_dataset", "").strip().lower()
        if source not in SOURCE_ARCHIVES:
            raise ValueError(f"unsupported Loop206 source dataset: {source}")
        entry = required.setdefault(source, {"images": set(), "masks": set()})
        entry["images"].add(Path(row["image_path"]).name.lower())
        entry["masks"].add(Path(row["mask_path"]).name.lower())
    if sum(len(entry["images"]) for entry in required.values()) != len(rows):
        raise ValueError("Loop206 source manifest image names are not unique")
    return required


def fetch_manifest_sources(
    manifest: str | Path,
    output_root: str | Path,
    *,
    archives: Mapping[str, Mapping[str, str | Sequence[str]]] = SOURCE_ARCHIVES,
) -> dict[str, dict[str, int]]:
    required = required_files_by_source(manifest)
    root = Path(output_root)
    counts: dict[str, dict[str, int]] = {}
    for source in sorted(required):
        if source not in archives:
            raise ValueError(f"missing archive contract for source: {source}")
        counts[source] = {}
        for kind in ("images", "masks"):
            configured = archives[source][kind]
            source_archives = [configured] if isinstance(configured, str) else list(configured)
            outputs = extract_selected_members_from_archives(
                source_archives,
                required[source][kind],
                root / source / kind,
            )
            counts[source][kind] = len(outputs)
    return counts
