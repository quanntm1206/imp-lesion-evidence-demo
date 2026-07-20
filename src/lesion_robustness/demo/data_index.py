from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from lesion_robustness.data_manifest import sha256_file, sha256_rgb
from lesion_robustness.image_utils import read_mask, read_rgb


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ResolvedRow:
    sample_id: str
    source_dataset: str
    split: str
    source_split: str
    role: str
    group_key: str
    fold: int
    image_path: Path
    mask_path: Path
    sha256_raw: str
    sha256_rgb: str


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Loop206 manifest is empty")
    return rows


def _file_index(roots: Sequence[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in roots:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.is_dir():
            continue
        for path in resolved_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                index.setdefault(path.name.lower(), []).append(path.resolve())
    return index


def _verified_image(candidates: Iterable[Path], raw_hash: str, rgb_hash: str) -> Path:
    matched = [
        path
        for path in candidates
        if sha256_file(path) == raw_hash and sha256_rgb(path) == rgb_hash
    ]
    if len(matched) != 1:
        raise ValueError(
            "Loop206 row requires one unique hash-verified image; "
            f"found {len(matched)}"
        )
    return matched[0]


def _unique_mask(candidates: Iterable[Path]) -> Path:
    matched = list(candidates)
    if len(matched) != 1:
        raise ValueError(f"Loop206 row requires one unique mask; found {len(matched)}")
    return matched[0]


def _validate_manifest_contract(rows: list[dict[str, str]], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"Loop206 manifest requires {expected_rows} rows, found {len(rows)}")
    group_keys = [row.get("loop205_group_key", "").strip() for row in rows]
    if any(not key for key in group_keys) or len(set(group_keys)) != len(group_keys):
        raise ValueError("Loop206 manifest group keys must be non-empty and unique")
    if any(row.get("source_split", "").strip().lower() != "train" for row in rows):
        raise ValueError("Loop206 manifest must contain source_split=train only")
    roles = {row.get("loop206_pilot_role", "").strip().lower() for row in rows}
    if not roles <= {"fit", "holdout"} or not roles:
        raise ValueError("Loop206 manifest roles must be fit or holdout")
    fit_groups = {
        row["loop205_group_key"]
        for row in rows
        if row.get("loop206_pilot_role", "").strip().lower() == "fit"
    }
    holdout_groups = set(group_keys) - fit_groups
    if fit_groups & holdout_groups:
        raise ValueError("Loop206 fit/holdout group overlap")
    if expected_rows == 384 and (len(fit_groups) != 308 or len(holdout_groups) != 76):
        raise ValueError("Loop206 full manifest requires 308 fit and 76 holdout groups")


def resolve_loop206_rows(
    manifest: str | Path,
    roots: Sequence[str | Path],
    *,
    expected_rows: int = 384,
) -> list[ResolvedRow]:
    manifest_path = Path(manifest)
    rows = _read_manifest(manifest_path)
    _validate_manifest_contract(rows, int(expected_rows))
    root_paths = [Path(root).expanduser().resolve() for root in roots]
    index = _file_index(root_paths)
    resolved: list[ResolvedRow] = []
    for row in rows:
        image_name = Path(row["image_path"]).name.lower()
        mask_name = Path(row["mask_path"]).name.lower()
        image_path = _verified_image(
            index.get(image_name, []),
            row.get("sha256_raw", "").strip().lower(),
            row.get("sha256_rgb", "").strip().lower(),
        )
        mask_path = _unique_mask(index.get(mask_name, []))
        image = read_rgb(image_path)
        mask = read_mask(mask_path)
        if image.shape[:2] != mask.shape or not mask.any() or mask.all():
            raise ValueError(f"Loop206 image/mask geometry is invalid for {image_name}")
        try:
            fold = int(row["loop205_fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Loop206 row has invalid fold for {image_name}") from exc
        if fold not in range(5):
            raise ValueError(f"Loop206 row has invalid fold for {image_name}")
        resolved.append(
            ResolvedRow(
                sample_id=row.get("original_id", "").strip() or Path(image_name).stem,
                source_dataset=row.get("source_dataset", "").strip().lower(),
                split=row.get("split", "").strip().lower(),
                source_split="train",
                role=row.get("loop206_pilot_role", "").strip().lower(),
                group_key=row["loop205_group_key"].strip(),
                fold=fold,
                image_path=image_path,
                mask_path=mask_path,
                sha256_raw=row.get("sha256_raw", "").strip().lower(),
                sha256_rgb=row.get("sha256_rgb", "").strip().lower(),
            )
        )
    return resolved


def _under_root(path: Path, roots: Sequence[Path]) -> tuple[int, str]:
    for index, root in enumerate(roots):
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        return index, relative.as_posix()
    raise ValueError(f"resolved dataset path escaped configured roots: {path}")


def build_index_payload(rows: Sequence[ResolvedRow], roots: Sequence[str | Path]) -> dict:
    root_paths = [Path(root).expanduser().resolve() for root in roots]
    records = []
    for row in rows:
        image_root, image_relative = _under_root(row.image_path, root_paths)
        mask_root, mask_relative = _under_root(row.mask_path, root_paths)
        record = asdict(row)
        record.pop("image_path")
        record.pop("mask_path")
        record.update(
            {
                "image_root": image_root,
                "image_relative": image_relative,
                "mask_root": mask_root,
                "mask_relative": mask_relative,
            }
        )
        records.append(record)
    return {
        "schema_version": "loop206.demo.dataset_index.v1",
        "root_count": len(root_paths),
        "row_count": len(records),
        "fit_count": sum(row.role == "fit" for row in rows),
        "holdout_count": sum(row.role == "holdout" for row in rows),
        "rows": records,
    }


def write_index(payload: dict, output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    temporary.write_text(data, encoding="ascii")
    temporary.replace(destination)
