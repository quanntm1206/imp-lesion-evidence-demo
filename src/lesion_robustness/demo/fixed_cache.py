"""Strict, path-independent access to the immutable Loop206 fixed caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lesion_robustness.packed_extra_channel import sha256_rgb_array


CACHE_SCHEMA = "loop206.leakage_safe_pilot_cache.v2"
ARTIFACT_TYPE = "loop206_packed_binary_channel"


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


@dataclass(frozen=True)
class FixedCacheExpectations:
    count: int
    shape: tuple[int, int]
    candidate_manifest_sha256: str
    candidate_data_sha256: str
    zero_manifest_sha256: str
    zero_data_sha256: str

    @classmethod
    def loop206(cls) -> "FixedCacheExpectations":
        return cls(
            count=536,
            shape=(384, 384),
            candidate_manifest_sha256="48e48290507eff6e4da8357e3310db9305a920f731c5b49890851d058d892255",
            candidate_data_sha256="3f49e43524772b9eee17a146ff47cb15361cf78b2ce77f8c5b25c46b8f019ebb",
            zero_manifest_sha256="b92bd22e5425354b46bc019f3ab6d3daddc24568670717be2654c8938894c0da",
            zero_data_sha256="c8f67865341c41e506c41f9ef3221861d2c4a12f771c7eee4159886fc718fa18",
        )


@dataclass(frozen=True)
class FixedCacheRecord:
    group_key: str
    corruption: str
    input_rgb_sha256: str
    control_channel: np.ndarray
    candidate_channel: np.ndarray
    metadata: dict[str, Any]


class _ValidatedCache:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        role: str,
        expected_arm: str,
        expected_manifest_sha256: str,
        expected_data_sha256: str,
        expected_count: int,
        expected_shape: tuple[int, int],
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        actual_manifest_hash = _sha256_file(self.manifest_path)
        if actual_manifest_hash != expected_manifest_sha256:
            raise ValueError(f"Loop206 {role} manifest hash mismatch")
        payload = json.loads(self.manifest_path.read_text(encoding="ascii"))
        exact = {
            "schema_version": CACHE_SCHEMA,
            "artifact_type": ARTIFACT_TYPE,
            "status": "passed",
            "arm": expected_arm,
            "count": int(expected_count),
            "shape": [int(expected_shape[0]), int(expected_shape[1])],
            "input_rgb_sha256_count": int(expected_count),
        }
        for field, expected in exact.items():
            if payload.get(field) != expected:
                raise ValueError(f"Loop206 {role} cache {field} contract mismatch")
        if payload.get("source_split_counts") != {"train": int(expected_count)}:
            raise ValueError(f"Loop206 {role} cache source split contract mismatch")
        allowed = payload.get("allowed_runtime_splits")
        if allowed != ["train", "train_screen_holdout"]:
            raise ValueError(f"Loop206 {role} cache runtime split contract mismatch")

        data = payload.get("data")
        if not isinstance(data, dict) or data.get("dtype") != "uint8":
            raise ValueError(f"Loop206 {role} cache data contract mismatch")
        filename = str(data.get("file", ""))
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Loop206 {role} cache data path mismatch")
        data_path = (self.manifest_path.parent / filename).resolve()
        try:
            data_path.relative_to(self.manifest_path.parent)
        except ValueError as exc:
            raise ValueError(f"Loop206 {role} cache data path escape") from exc
        expected_size = int(expected_count) * int(expected_shape[0]) * int(expected_shape[1])
        if not data_path.is_file() or data_path.stat().st_size != expected_size:
            raise ValueError(f"Loop206 {role} cache data size mismatch")
        actual_data_hash = _sha256_file(data_path)
        if data.get("sha256") != expected_data_sha256 or actual_data_hash != expected_data_sha256:
            raise ValueError(f"Loop206 {role} cache data hash mismatch")

        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != int(expected_count):
            raise ValueError(f"Loop206 {role} cache row count mismatch")
        if payload.get("rows_sha256") != _canonical_hash(rows):
            raise ValueError(f"Loop206 {role} cache row hash mismatch")
        indices: list[int] = []
        lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or type(row.get("index")) is not int:
                raise ValueError(f"Loop206 {role} cache row contract mismatch")
            index = int(row["index"])
            indices.append(index)
            group_key = str(row.get("group_key", ""))
            corruption = str(row.get("corruption", "")).strip().lower()
            input_hash = str(row.get("input_rgb_sha256", ""))
            key = (group_key, corruption)
            if (
                not group_key
                or not corruption
                or not _is_sha256(input_hash)
                or row.get("source_split") != "train"
                or key in lookup
            ):
                raise ValueError(f"Loop206 {role} cache row contract mismatch")
            lookup[key] = row
        if indices != list(range(int(expected_count))):
            raise ValueError(f"Loop206 {role} cache contiguous indices mismatch")

        self.payload = payload
        self.rows = rows
        self.lookup = lookup
        self.data_path = data_path
        self.data_sha256 = actual_data_hash
        self.manifest_sha256 = actual_manifest_hash
        self.data = np.memmap(
            data_path,
            mode="r",
            dtype=np.uint8,
            shape=(int(expected_count), int(expected_shape[0]), int(expected_shape[1])),
        )


class FixedCachePair:
    def __init__(
        self,
        candidate_manifest: str | Path,
        zero_manifest: str | Path,
        *,
        expectations: FixedCacheExpectations | None = None,
        runtime_config_sha256: str | None = None,
    ) -> None:
        self.expectations = expectations or FixedCacheExpectations.loop206()
        self.candidate = _ValidatedCache(
            candidate_manifest,
            role="candidate",
            expected_arm="candidate",
            expected_manifest_sha256=self.expectations.candidate_manifest_sha256,
            expected_data_sha256=self.expectations.candidate_data_sha256,
            expected_count=self.expectations.count,
            expected_shape=self.expectations.shape,
        )
        self.zero = _ValidatedCache(
            zero_manifest,
            role="zero",
            expected_arm="zero_control",
            expected_manifest_sha256=self.expectations.zero_manifest_sha256,
            expected_data_sha256=self.expectations.zero_data_sha256,
            expected_count=self.expectations.count,
            expected_shape=self.expectations.shape,
        )
        if self.candidate.rows != self.zero.rows:
            raise ValueError("Loop206 candidate/zero row metadata mismatch")
        if np.any(self.zero.data):
            raise ValueError("Loop206 zero cache violates the all-zero invariant")
        historical_config = str(
            self.candidate.payload.get("provenance", {}).get("config_sha256", "")
        )
        if not _is_sha256(historical_config):
            raise ValueError("Loop206 candidate cache provenance contract mismatch")
        self.historical_cache_provenance_drift = (
            None
            if runtime_config_sha256 is None
            else historical_config != str(runtime_config_sha256).strip().lower()
        )

    def lookup(
        self,
        dataset_row: Mapping[str, Any],
        *,
        corruption: str,
        input_rgb: np.ndarray,
    ) -> FixedCacheRecord:
        if dataset_row.get("role") != "holdout" or int(dataset_row.get("fold", -1)) != 4:
            raise ValueError("Loop206 fixed cache accepts holdout rows only")
        group_key = str(dataset_row.get("group_key", ""))
        view = str(corruption).strip().lower()
        key = (group_key, view)
        if key not in self.candidate.lookup:
            known_groups = {group for group, _ in self.candidate.lookup}
            label = "group" if group_key not in known_groups else "corruption"
            raise KeyError(f"Loop206 fixed cache {label} binding mismatch")
        candidate_row = self.candidate.lookup[key]
        zero_row = self.zero.lookup.get(key)
        if zero_row is None or candidate_row != zero_row:
            raise ValueError("Loop206 candidate/zero row metadata mismatch")
        if str(dataset_row.get("sample_id", "")) != str(candidate_row.get("sample_id", "")):
            raise ValueError("Loop206 fixed cache dataset row sample mismatch")
        actual_input_hash = sha256_rgb_array(input_rgb)
        if actual_input_hash != str(candidate_row["input_rgb_sha256"]):
            raise ValueError("Loop206 fixed cache input RGB hash mismatch")
        candidate_index = int(candidate_row["index"])
        zero_index = int(zero_row["index"])
        control = np.asarray(self.zero.data[zero_index], dtype=np.uint8).copy()
        candidate = np.asarray(self.candidate.data[candidate_index], dtype=np.uint8).copy()
        return FixedCacheRecord(
            group_key=group_key,
            corruption=view,
            input_rgb_sha256=actual_input_hash,
            control_channel=control,
            candidate_channel=candidate,
            metadata={
                "historical_cache_provenance_drift": self.historical_cache_provenance_drift,
                "candidate_manifest_sha256": self.candidate.manifest_sha256,
                "candidate_data_sha256": self.candidate.data_sha256,
                "zero_manifest_sha256": self.zero.manifest_sha256,
                "zero_data_sha256": self.zero.data_sha256,
            },
        )
