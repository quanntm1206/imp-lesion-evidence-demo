from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


SOURCE_PRIORITY = {"isic2016": 1, "isic2017": 2, "isic2018": 3}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rgb(path: str | Path) -> str:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    payload = f"{rgb.shape[0]}x{rgb.shape[1]}x3|".encode("ascii") + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def phash_hex(path: str | Path, *, hash_size: int = 8, high_frequency_factor: int = 4) -> str:
    import cv2

    size = int(hash_size) * int(high_frequency_factor)
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = cv2.dct(gray)
    low = dct[:hash_size, :hash_size]
    median = float(np.median(low.reshape(-1)[1:]))
    bits = low > median
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def phash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def split_group_identifier(record: dict) -> str:
    if record.get("split_group"):
        return str(record["split_group"])
    if record.get("patient_id"):
        return f"patient:{record['patient_id']}"
    if record.get("lesion_id"):
        return f"lesion:{record['lesion_id']}"
    return f"duplicate:{record.get('duplicate_group') or record.get('isic_image_id') or record.get('original_id')}"


def build_ima_staple_candidates(
    image_rows: Iterable[dict],
    segmentation_rows: Iterable[dict],
    *,
    blocked_ids: set[str],
    accepted_image_licenses: set[str],
) -> tuple[list[dict], list[dict]]:
    image_by_id = {str(row["isic_id"]): dict(row) for row in image_rows}
    candidates: list[dict] = []
    excluded: list[dict] = []
    for segmentation in segmentation_rows:
        if str(segmentation.get("annotator", "")) != "ST":
            continue
        isic_id = str(segmentation["ISIC_id"])
        image = image_by_id.get(isic_id, {})
        row = {
            **image,
            **dict(segmentation),
            "original_id": isic_id,
            "isic_image_id": isic_id,
            "source_dataset": "ima_pp",
            "sampling_group": "ima_pp_staple",
            "consensus_type": "STAPLE",
            "image_license": str(image.get("copyright_license", "")),
        }
        if isic_id in blocked_ids:
            excluded.append({**row, "exclude_reason": "blocked_existing_id"})
            continue
        if row["image_license"] not in accepted_image_licenses:
            excluded.append({**row, "exclude_reason": "unaccepted_image_license"})
            continue
        candidates.append(row)
    return candidates, excluded


def find_near_duplicate_pairs(records: Iterable[dict], *, max_distance: int = 4) -> list[dict]:
    """Find pHash-near records with a small BK-tree, excluding exact RGB duplicates."""

    class Node:
        def __init__(self, value: int, index: int) -> None:
            self.value = value
            self.indices = [index]
            self.children: dict[int, Node] = {}

    rows = list(records)
    root: Node | None = None
    pairs: list[dict] = []
    for index, row in enumerate(rows):
        value = int(str(row["phash"]), 16)
        if root is None:
            root = Node(value, index)
            continue
        stack = [root]
        while stack:
            node = stack.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= max_distance:
                for previous in node.indices:
                    if rows[previous].get("sha256_rgb") == row.get("sha256_rgb"):
                        continue
                    pairs.append(
                        {
                            "left_id": str(rows[previous].get("original_id", previous)),
                            "right_id": str(row.get("original_id", index)),
                            "distance": int(distance),
                        }
                    )
            lower = distance - max_distance
            upper = distance + max_distance
            stack.extend(child for edge, child in node.children.items() if lower <= edge <= upper)

        node = root
        while True:
            distance = (value ^ node.value).bit_count()
            if distance == 0:
                node.indices.append(index)
                break
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = Node(value, index)
                break
            node = child
    return pairs


def _record_group_key(record: dict) -> str:
    if record.get("sha256_rgb"):
        return f"exact:{record['sha256_rgb']}"
    if record.get("isic_image_id"):
        return f"isic:{record['isic_image_id']}"
    return f"record:{record.get('original_id', record.get('image_path', ''))}"


def canonicalize_records(records: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    rows_all = [dict(raw) for raw in records]
    parents = list(range(len(rows_all)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, int] = {}
    for index, row in enumerate(rows_all):
        isic_id = str(row.get("isic_image_id", ""))
        rgb_hash = str(row.get("sha256_rgb", ""))
        if isic_id:
            if isic_id in seen_ids:
                union(index, seen_ids[isic_id])
            else:
                seen_ids[isic_id] = index
        if rgb_hash:
            if rgb_hash in seen_hashes:
                union(index, seen_hashes[rgb_hash])
            else:
                seen_hashes[rgb_hash] = index

    grouped_indices: dict[int, list[int]] = {}
    for index in range(len(rows_all)):
        grouped_indices.setdefault(find(index), []).append(index)
    groups: list[list[dict]] = []
    for indices in grouped_indices.values():
        rows = [rows_all[index] for index in indices]
        hashes = {str(row.get("sha256_rgb", "")) for row in rows if row.get("sha256_rgb")}
        ids = {str(row.get("isic_image_id", "")) for row in rows if row.get("isic_image_id")}
        if len(hashes) == 1:
            group = f"exact:{next(iter(hashes))}"
        elif len(ids) == 1:
            group = f"isic:{next(iter(ids))}"
        else:
            group = f"linked:{min(indices)}"
        for row in rows:
            row["duplicate_group"] = group
        groups.append(rows)

    canonical: list[dict] = []
    excluded: list[dict] = []
    for rows in groups:
        winner = max(
            rows,
            key=lambda row: (
                SOURCE_PRIORITY.get(str(row.get("source_dataset", "")).lower(), 0),
                str(row.get("original_id", "")),
            ),
        )
        canonical.append(winner)
        for row in rows:
            if row is winner:
                continue
            excluded.append({**row, "exclude_reason": "duplicate_lower_priority"})
    return canonical, excluded


def _stable_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{int(seed)}::{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def deterministic_group_split(
    records: Iterable[dict],
    *,
    ratios: dict[str, float],
    seed: int,
    group_field: str = "duplicate_group",
) -> list[dict]:
    if not ratios or abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")
    rows = [dict(record) for record in records]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = str(row.get(group_field) or row.get("isic_image_id") or row.get("original_id") or index)
        groups[group].append(index)

    total = len(rows)
    targets = {name: float(ratio) * total for name, ratio in ratios.items()}
    counts = {name: 0 for name in ratios}
    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), _stable_fraction(seed, item[0])))
    for group, indices in ordered_groups:
        candidates = sorted(
            ratios,
            key=lambda name: (
                -((targets[name] - counts[name]) / max(targets[name], 1.0)),
                _stable_fraction(seed, f"{group}:{name}"),
            ),
        )
        chosen = candidates[0]
        for index in indices:
            rows[index]["split"] = chosen
        counts[chosen] += len(indices)
    return rows


def validate_mask_pair(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.95,
) -> dict:
    with Image.open(image_path) as image:
        image_size = image.size
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
        mask_size = mask_image.size
    if mask_size != image_size:
        return {"valid": False, "reason": "shape_mismatch", "image_size": image_size, "mask_size": mask_size}
    foreground = mask > 0
    area_ratio = float(foreground.mean())
    if area_ratio == 0.0:
        return {"valid": False, "reason": "empty_mask", "area_ratio": area_ratio}
    if area_ratio < float(min_area_ratio):
        return {"valid": False, "reason": "mask_too_small", "area_ratio": area_ratio}
    if area_ratio > float(max_area_ratio):
        return {"valid": False, "reason": "mask_too_large", "area_ratio": area_ratio}
    return {"valid": True, "reason": None, "area_ratio": area_ratio, "image_size": image_size}
