"""Read-only packed binary extra-channel cache used by Loop206 pilots."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


PACKED_TYPES = frozenset({"packed_binary_cache", "loop206_contour_cache"})
PROTECTED_SPLITS = frozenset({"val", "test", "test_v3", "ph2", "external_audit"})


def is_packed_extra_channel_config(preprocessing: Mapping[str, Any] | None) -> bool:
    if not isinstance(preprocessing, Mapping):
        return False
    block = preprocessing.get("extra_channel", {})
    return (
        isinstance(block, Mapping)
        and block.get("enabled") is True
        and str(block.get("type", "")).strip().lower() in PACKED_TYPES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_path(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def sha256_rgb_array(image_rgb_u8: np.ndarray) -> str:
    image = np.asarray(image_rgb_u8)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("packed extra-channel input hash requires RGB uint8")
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


class PackedBinaryChannelCache:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="ascii"))
        if payload.get("artifact_type") != "loop206_packed_binary_channel":
            raise ValueError("packed extra-channel manifest has an unsupported artifact_type")
        if payload.get("status") != "passed":
            raise ValueError("packed extra-channel manifest is not complete")
        source_splits = set(payload.get("source_split_counts", {}))
        if source_splits != {"train"}:
            raise ValueError("packed extra-channel cache must contain train source rows only")
        allowed = {str(value).strip().lower() for value in payload.get("allowed_runtime_splits", [])}
        if not allowed or allowed & PROTECTED_SPLITS:
            raise ValueError("packed extra-channel runtime split policy is unsafe")
        self.allowed_runtime_splits = allowed
        self.count = int(payload["count"])
        self.height, self.width = map(int, payload["shape"])
        data_path = self.manifest_path.parent / payload["data"]["file"]
        if not data_path.is_file() or _sha256(data_path) != payload["data"]["sha256"]:
            raise ValueError("packed extra-channel data hash mismatch")
        if payload["data"].get("dtype") != "uint8":
            raise ValueError("packed extra-channel data must be uint8")
        self.data = np.memmap(
            data_path,
            mode="r",
            dtype=np.uint8,
            shape=(self.count, self.height, self.width),
        )
        self.lookup: dict[tuple[str, str], int] = {}
        self.input_rgb_sha256: dict[tuple[str, str], str] = {}
        for row in payload.get("rows", []):
            if str(row.get("source_split", "")).lower() != "train":
                raise ValueError("packed extra-channel row is not source_split=train")
            key = (
                _normalized_path(str(row.get("image_path", ""))),
                str(row.get("corruption", "")).strip().lower(),
            )
            index = int(row["index"])
            if not key[0] or not key[1] or not 0 <= index < self.count:
                raise ValueError("packed extra-channel row is malformed")
            if key in self.lookup:
                raise ValueError(f"duplicate packed extra-channel key: {key}")
            self.lookup[key] = index
            input_hash = str(row.get("input_rgb_sha256", "")).strip().lower()
            if input_hash:
                if len(input_hash) != 64 or any(
                    character not in "0123456789abcdef" for character in input_hash
                ):
                    raise ValueError("packed extra-channel input_rgb_sha256 is malformed")
                self.input_rgb_sha256[key] = input_hash
        if len(self.lookup) != self.count:
            raise ValueError("packed extra-channel row count mismatch")
        if self.input_rgb_sha256 and len(self.input_rgb_sha256) != self.count:
            raise ValueError("packed extra-channel input hashes must cover every row")

    def load(
        self,
        sample_key: str | Path,
        *,
        split: str,
        corruption: str,
        output_shape: tuple[int, int],
        input_rgb_u8: np.ndarray | None = None,
        require_input_sha256: bool = False,
    ) -> np.ndarray:
        normalized_split = str(split).strip().lower()
        if normalized_split not in self.allowed_runtime_splits:
            raise ValueError(f"packed extra-channel cache rejects runtime split {split!r}")
        key = (_normalized_path(sample_key), str(corruption).strip().lower())
        if key not in self.lookup:
            raise KeyError(f"packed extra-channel cache is missing {key}")
        expected_input_hash = self.input_rgb_sha256.get(key)
        if require_input_sha256 and expected_input_hash is None:
            raise ValueError("packed extra-channel cache lacks required input_rgb_sha256")
        if expected_input_hash is not None:
            if input_rgb_u8 is None:
                raise ValueError("packed extra-channel input hash cannot be verified")
            actual_input_hash = sha256_rgb_array(input_rgb_u8)
            if actual_input_hash != expected_input_hash:
                raise ValueError(
                    "packed extra-channel input RGB hash mismatch for "
                    f"corruption={key[1]!r}"
                )
        channel = np.asarray(self.data[self.lookup[key]], dtype=np.uint8)
        height, width = map(int, output_shape)
        if channel.shape != (height, width):
            channel = cv2.resize(channel, (width, height), interpolation=cv2.INTER_NEAREST)
        return (channel.astype(np.float32) / 255.0)[..., None]


@lru_cache(maxsize=8)
def load_packed_binary_channel_cache(manifest_path: str) -> PackedBinaryChannelCache:
    return PackedBinaryChannelCache(manifest_path)


def append_packed_binary_channel(
    image_rgb_u8: np.ndarray,
    preprocessing: Mapping[str, Any],
    *,
    sample_key: str,
    split: str,
    corruption: str = "clean",
) -> np.ndarray:
    if not is_packed_extra_channel_config(preprocessing):
        raise ValueError("packed extra-channel config is not enabled")
    image = np.asarray(image_rgb_u8)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("packed extra-channel input must be RGB uint8")
    block = preprocessing["extra_channel"]
    manifest = str(block.get("cache_manifest", "")).strip()
    if not manifest:
        raise ValueError("preprocessing.extra_channel.cache_manifest is required")
    cache = load_packed_binary_channel_cache(str(Path(manifest).expanduser().resolve()))
    extra = cache.load(
        sample_key,
        split=split,
        corruption=corruption,
        output_shape=image.shape[:2],
        input_rgb_u8=image,
        require_input_sha256=bool(block.get("require_input_sha256", False)),
    )
    return np.concatenate([image.astype(np.float32) / 255.0, extra], axis=2)
