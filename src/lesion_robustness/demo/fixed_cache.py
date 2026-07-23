"""Strict, path-independent access to the immutable Loop206 fixed caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from lesion_robustness.packed_extra_channel import sha256_rgb_array
from lesion_robustness.demo.immutable_io import ImmutableSnapshot
from lesion_robustness.release_manifest import fixed_cache_projection


_RELEASE_FIXED_CACHE = fixed_cache_projection()
_CACHE_CONTRACT = _RELEASE_FIXED_CACHE["fixed_cache"]
CACHE_SCHEMA = str(_CACHE_CONTRACT["schema_version"])
ARTIFACT_TYPE = str(_CACHE_CONTRACT["artifact_type"])
DATASET_INDEX_SHA256 = str(_RELEASE_FIXED_CACHE["dataset_index"]["sha256"])
LIVE_CONFIG_SHA256 = str(_RELEASE_FIXED_CACHE["live_config"]["sha256"])
LIVE_CONFIG_SCHEMA = str(_RELEASE_FIXED_CACHE["live_config"]["schema_version"])
_PRODUCTION_PROVIDER_TOKEN = object()
_AUTHORIZED_SAMPLE_TOKEN = object()


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
        candidate = _CACHE_CONTRACT["candidate"]
        zero = _CACHE_CONTRACT["zero"]
        return cls(
            count=int(_CACHE_CONTRACT["count"]),
            shape=tuple(int(value) for value in _CACHE_CONTRACT["shape"]),
            candidate_manifest_sha256=str(candidate["manifest_sha256"]),
            candidate_data_sha256=str(candidate["data_sha256"]),
            zero_manifest_sha256=str(zero["manifest_sha256"]),
            zero_data_sha256=str(zero["data_sha256"]),
        )


@dataclass(frozen=True)
class _CacheChannels:
    group_key: str
    sample_id: str
    corruption: str
    input_rgb_sha256: str
    control_channel: np.ndarray
    candidate_channel: np.ndarray


@dataclass(frozen=True)
class _AuthorizedFixedSample:
    group_key: str
    sample_id: str
    corruption: str
    original_rgb: np.ndarray
    model_rgb: np.ndarray
    control_channel: np.ndarray
    candidate_channel: np.ndarray
    candidate_manifest_sha256: str
    candidate_data_sha256: str
    zero_manifest_sha256: str
    zero_data_sha256: str
    mask_sha256_raw: str
    mask_sha256_binary: str
    mask_sha256_runtime: str
    historical_cache_provenance_drift: bool
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _AUTHORIZED_SAMPLE_TOKEN:
            raise TypeError("fixed sample authorization must come from the production provider")


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
        manifest_snapshot = ImmutableSnapshot.read(self.manifest_path)
        if manifest_snapshot.sha256 != expected_manifest_sha256:
            raise ValueError(f"Loop206 {role} manifest hash mismatch")
        payload = json.loads(manifest_snapshot.text("ascii"))
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
        try:
            data_snapshot = ImmutableSnapshot.read(data_path)
        except FileNotFoundError as exc:
            raise ValueError(f"Loop206 {role} cache data is missing") from exc
        if data_snapshot.size != expected_size:
            raise ValueError(f"Loop206 {role} cache data size mismatch")
        actual_data_hash = data_snapshot.sha256
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
        self.manifest_sha256 = manifest_snapshot.sha256
        self._data_snapshot = data_snapshot
        self.data = np.memmap(
            self._data_snapshot.open(),
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
        if not _is_sha256(
            self.candidate.payload.get("provenance", {}).get("config_sha256", "")
        ):
            raise ValueError("Loop206 candidate cache provenance contract mismatch")

    def _load_channels(
        self,
        *,
        group_key: str,
        sample_id: str,
        corruption: str,
        input_rgb: np.ndarray,
    ) -> _CacheChannels:
        view = str(corruption).strip().lower()
        key = (str(group_key), view)
        if key not in self.candidate.lookup:
            known_groups = {group for group, _ in self.candidate.lookup}
            label = "group" if group_key not in known_groups else "corruption"
            raise KeyError(f"Loop206 fixed cache {label} binding mismatch")
        candidate_row = self.candidate.lookup[key]
        zero_row = self.zero.lookup.get(key)
        if zero_row is None or candidate_row != zero_row:
            raise ValueError("Loop206 candidate/zero row metadata mismatch")
        if str(sample_id) != str(candidate_row.get("sample_id", "")):
            raise ValueError("Loop206 fixed cache dataset row sample mismatch")
        actual_input_hash = sha256_rgb_array(input_rgb)
        if actual_input_hash != str(candidate_row["input_rgb_sha256"]):
            raise ValueError("Loop206 fixed cache input RGB hash mismatch")
        control = np.asarray(self.zero.data[int(zero_row["index"])], dtype=np.uint8).copy()
        candidate = np.asarray(
            self.candidate.data[int(candidate_row["index"])], dtype=np.uint8
        ).copy()
        return _CacheChannels(
            group_key=str(group_key),
            sample_id=str(sample_id),
            corruption=view,
            input_rgb_sha256=actual_input_hash,
            control_channel=control,
            candidate_channel=candidate,
        )

    def lookup_fixture(
        self,
        dataset_row: Mapping[str, Any],
        *,
        corruption: str,
        input_rgb: np.ndarray,
    ) -> _CacheChannels:
        """Exercise tiny cache fixtures; this method cannot authorize service inference."""

        if dataset_row.get("role") != "holdout" or int(dataset_row.get("fold", -1)) != 4:
            raise ValueError("Loop206 fixed cache accepts holdout rows only")
        return self._load_channels(
            group_key=str(dataset_row.get("group_key", "")),
            sample_id=str(dataset_row.get("sample_id", "")),
            corruption=corruption,
            input_rgb=input_rgb,
        )


def _safe_index_path(root: Path, relative: object) -> Path:
    candidate = (root / Path(str(relative))).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Loop206 fixed dataset path escape") from exc
    return candidate


def _fixed_presentation_rgb(
    decoded_source: np.ndarray, runtime_image: np.ndarray
) -> np.ndarray:
    source = np.asarray(decoded_source)
    runtime = np.asarray(runtime_image)
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or source.dtype != np.uint8
        or runtime.ndim != 3
        or runtime.shape[2] != 3
        or runtime.dtype != np.uint8
    ):
        raise ValueError("Loop206 fixed presentation requires RGB uint8 images")
    return np.ascontiguousarray(runtime).copy()


def _validate_live_preprocessing(
    live_config_path: str | Path, registry_preprocessing: Mapping[str, Any]
) -> None:
    import yaml

    snapshot = ImmutableSnapshot.read(live_config_path)
    if snapshot.sha256 != LIVE_CONFIG_SHA256:
        raise ValueError("Loop206 tracked live config hash mismatch")
    payload = yaml.safe_load(snapshot.text("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != LIVE_CONFIG_SCHEMA:
        raise ValueError("Loop206 tracked live config schema mismatch")
    live = payload.get("preprocessing")
    if not isinstance(live, dict):
        raise ValueError("Loop206 tracked live preprocessing is missing")
    for key, expected in live.items():
        if registry_preprocessing.get(key) != expected:
            raise ValueError("Loop206 registry/live preprocessing mismatch")


class ProductionFixedSampleProvider:
    def __init__(
        self,
        *,
        dataset_index: str | Path,
        dataset_roots: Sequence[str | Path],
        candidate_manifest: str | Path,
        zero_manifest: str | Path,
        live_config: str | Path,
        registry_preprocessing: Mapping[str, Any],
        corruption_configs: Mapping[str, Any],
        project_seed: int,
        _token: object,
    ) -> None:
        if _token is not _PRODUCTION_PROVIDER_TOKEN:
            raise TypeError("production fixed provider requires registry authorization")
        dataset_snapshot = ImmutableSnapshot.read(dataset_index)
        if dataset_snapshot.sha256 != DATASET_INDEX_SHA256:
            raise ValueError("Loop206 dataset index hash mismatch")
        _validate_live_preprocessing(live_config, registry_preprocessing)
        from lesion_robustness.demo.loop206_prior import load_dataset_index

        roots = tuple(Path(root).expanduser().resolve() for root in dataset_roots)
        _, holdout_rows, payload = load_dataset_index(
            dataset_index, dataset_roots=roots, _snapshot=dataset_snapshot
        )
        raw_rows = {
            str(row["group_key"]): row
            for row in payload["rows"]
            if row.get("role") == "holdout"
        }
        typed_rows = {row.group_key: row for row in holdout_rows}
        if raw_rows.keys() != typed_rows.keys():
            raise ValueError("Loop206 validated holdout index binding mismatch")
        identifiers: dict[str, str] = {}
        for group_key, row in typed_rows.items():
            for identifier in (group_key, row.sample_id):
                if identifier in identifiers:
                    raise ValueError("Loop206 fixed sample identifier is ambiguous")
                identifiers[identifier] = group_key
        self._roots = roots
        self._raw_rows = raw_rows
        self._typed_rows = typed_rows
        self._identifiers = identifiers
        self._preprocessing = deepcopy(dict(registry_preprocessing))
        self._corruption_configs = deepcopy(dict(corruption_configs))
        self._project_seed = int(project_seed)
        self._pair = FixedCachePair(
            candidate_manifest,
            zero_manifest,
            expectations=FixedCacheExpectations.loop206(),
        )
        historical_hash = str(
            self._pair.candidate.payload["provenance"]["config_sha256"]
        )
        self._historical_cache_provenance_drift = bool(
            historical_hash != LIVE_CONFIG_SHA256
        )
        self._token = _PRODUCTION_PROVIDER_TOKEN

    @property
    def is_production_authorized(self) -> bool:
        return self._token is _PRODUCTION_PROVIDER_TOKEN

    def authorize(self, identifier: str, *, corruption: str) -> _AuthorizedFixedSample:
        from lesion_robustness.corruptions import apply_corruption
        from lesion_robustness.demo.loop206_prior import deterministic_corruption_kwargs
        from lesion_robustness.image_utils import resize_image_and_mask
        from lesion_robustness.preprocessing import preprocess_image_from_config

        key = str(identifier)
        if key not in self._identifiers:
            raise KeyError("Loop206 fixed sample is not allowlisted")
        group_key = self._identifiers[key]
        typed = self._typed_rows[group_key]
        raw = self._raw_rows[group_key]
        root_index = int(raw["image_root"])
        if root_index not in range(len(self._roots)):
            raise ValueError("Loop206 fixed dataset root binding mismatch")
        image_path = _safe_index_path(self._roots[root_index], raw["image_relative"])
        image_snapshot = ImmutableSnapshot.read(image_path)
        original = image_snapshot.decode_rgb()
        if (
            image_snapshot.sha256 != raw["sha256_raw"]
            or image_snapshot.decoded_rgb_sha256(original) != raw["sha256_rgb"]
        ):
            raise ValueError("Loop206 fixed original image hash mismatch")
        resized, _ = resize_image_and_mask(
            original, None, self._pair.expectations.shape
        )
        if not np.array_equal(resized, typed.image):
            raise ValueError("Loop206 fixed decoded image binding mismatch")
        view = str(corruption).strip().lower()
        runtime = resized
        if view != "clean":
            if view not in self._corruption_configs:
                raise KeyError("Loop206 fixed cache corruption binding mismatch")
            kwargs = deterministic_corruption_kwargs(
                view,
                dict(self._corruption_configs[view]),
                base_seed=self._project_seed,
                index=int(typed.dataset_index),
            )
            runtime = apply_corruption(runtime, view, **kwargs)
        model_rgb = preprocess_image_from_config(
            runtime, deepcopy(self._preprocessing)
        )
        channels = self._pair._load_channels(
            group_key=typed.group_key,
            sample_id=typed.sample_id,
            corruption=view,
            input_rgb=model_rgb,
        )
        if typed.mask is None:
            raise ValueError("Loop206 fixed ground truth binding is missing")
        mask_sha256_raw = str(raw.get("mask_sha256_raw", ""))
        mask_sha256_binary = str(raw.get("mask_sha256_binary", ""))
        if not _is_sha256(mask_sha256_raw) or not _is_sha256(mask_sha256_binary):
            raise ValueError("Loop206 fixed ground truth hash binding mismatch")
        mask_sha256_runtime = ImmutableSnapshot.decoded_binary_mask_sha256(
            np.asarray(typed.mask, dtype=np.uint8)
        )
        return _AuthorizedFixedSample(
            group_key=typed.group_key,
            sample_id=typed.sample_id,
            corruption=view,
            original_rgb=_fixed_presentation_rgb(original, runtime),
            model_rgb=np.ascontiguousarray(model_rgb, dtype=np.uint8),
            control_channel=channels.control_channel,
            candidate_channel=channels.candidate_channel,
            candidate_manifest_sha256=self._pair.candidate.manifest_sha256,
            candidate_data_sha256=self._pair.candidate.data_sha256,
            zero_manifest_sha256=self._pair.zero.manifest_sha256,
            zero_data_sha256=self._pair.zero.data_sha256,
            mask_sha256_raw=mask_sha256_raw,
            mask_sha256_binary=mask_sha256_binary,
            mask_sha256_runtime=mask_sha256_runtime,
            historical_cache_provenance_drift=self._historical_cache_provenance_drift,
            _token=_AUTHORIZED_SAMPLE_TOKEN,
        )


def _build_production_provider(**kwargs: Any) -> ProductionFixedSampleProvider:
    return ProductionFixedSampleProvider(**kwargs, _token=_PRODUCTION_PROVIDER_TOKEN)
