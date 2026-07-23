"""Frozen RQ1-v2 corruption bytes and panel-cache contracts."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


CONDITIONS = (
    "clean",
    "low_brightness",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
)


@dataclass(frozen=True)
class ConditionPanel:
    """One immutable condition-byte panel shared by both model arms."""

    names: tuple[str, ...]
    values: tuple[np.ndarray, ...]
    hashes: Mapping[str, str]
    seeds: Mapping[str, int]

    @property
    def imp_inputs(self) -> tuple[np.ndarray, ...]:
        return self.values

    @property
    def nnunet_inputs(self) -> tuple[np.ndarray, ...]:
        return self.values

    @property
    def ordered_panel_sha256(self) -> str:
        return ordered_panel_sha256(self.names, self.values)


def _rgb(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("RGB input must be uint8 HxWx3")
    return np.ascontiguousarray(value)


def condition_seed(
    protocol_sha256: str, group_key: str, sample_id: str, condition: str
) -> int:
    """Derive the deterministic uint64 seed specified by the RQ1 protocol."""
    fields = (protocol_sha256, group_key, sample_id, condition)
    if any(not isinstance(field, str) for field in fields):
        raise TypeError("condition seed fields must be strings")
    payload = "|".join(fields).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _quantize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    value = np.clip(value, np.float32(0.0), np.float32(1.0))
    return np.floor(value * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)


def apply_condition(rgb: np.ndarray, condition: str, seed: int) -> np.ndarray:
    """Apply exactly one frozen condition, returning contiguous RGB uint8 bytes."""
    source = _rgb(rgb)
    if condition == "clean":
        return source.copy()
    if condition == "gaussian_blur":
        return np.ascontiguousarray(
            cv2.GaussianBlur(
                source,
                (0, 0),
                sigmaX=2.0,
                sigmaY=2.0,
                borderType=cv2.BORDER_REFLECT_101,
            )
        )
    if condition == "jpeg_compression":
        output = BytesIO()
        image = Image.fromarray(source, mode="RGB")
        image.save(
            output,
            format="JPEG",
            quality=30,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        with Image.open(BytesIO(output.getvalue())) as decoded:
            return np.ascontiguousarray(np.asarray(decoded.convert("RGB"), dtype=np.uint8))

    values = source.astype(np.float32) / np.float32(255.0)
    if condition == "low_brightness":
        values = values * np.float32(0.60)
    elif condition == "low_contrast":
        mean = values.mean(axis=(0, 1), dtype=np.float32)
        values = (values - mean) * np.float32(0.60) + mean
    elif condition == "gaussian_noise":
        generator = np.random.Generator(np.random.PCG64(seed))
        values = values + generator.standard_normal(values.shape, dtype=np.float32) * np.float32(0.05)
    else:
        raise ValueError(f"unknown RQ1-v2 condition: {condition}")
    return np.ascontiguousarray(_quantize(values))


def _protocol_sha256(protocol: Mapping[str, object] | str | Path) -> str:
    if isinstance(protocol, (str, Path)):
        return hashlib.sha256(Path(protocol).read_bytes()).hexdigest()
    explicit = protocol.get("protocol_sha256")
    if isinstance(explicit, str):
        if len(explicit) != 64 or any(char not in "0123456789abcdef" for char in explicit):
            raise ValueError("protocol_sha256 must be lowercase hexadecimal")
        return explicit
    canonical = (json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _row_value(row: Mapping[str, object] | object, key: str) -> str:
    if isinstance(row, Mapping):
        value = row.get(key)
    else:
        value = getattr(row, key, None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {key} is required")
    return value


def build_condition_panel(
    rgb: np.ndarray,
    row: Mapping[str, object] | object,
    protocol: Mapping[str, object] | str | Path,
) -> ConditionPanel:
    """Build one ordered, shared panel; no arm may re-apply conditions."""
    source = _rgb(rgb)
    protocol_sha = _protocol_sha256(protocol)
    group_key = _row_value(row, "group_key")
    sample_id = _row_value(row, "sample_id")
    conditions: Sequence[str] = CONDITIONS
    if isinstance(protocol, Mapping) and "conditions" in protocol:
        conditions = tuple(protocol["conditions"])  # type: ignore[arg-type]
    if tuple(conditions) != CONDITIONS:
        raise ValueError("RQ1-v2 condition order drift")
    values: list[np.ndarray] = []
    seeds: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for name in conditions:
        seed = condition_seed(protocol_sha, group_key, sample_id, name)
        value = apply_condition(source, name, seed)
        value.setflags(write=False)
        seeds[name] = seed
        hashes[name] = hashlib.sha256(value.tobytes()).hexdigest()
        values.append(value)
    return ConditionPanel(
        tuple(conditions),
        tuple(values),
        MappingProxyType(hashes),
        MappingProxyType(seeds),
    )


def _validated_input_hashes(panel: ConditionPanel) -> dict[str, str]:
    observed = {
        name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
        for name, value in zip(panel.names, panel.values)
    }
    if observed != dict(panel.hashes):
        raise ValueError("condition panel bytes do not match frozen hashes")
    return observed


def imp_input_hashes(panel: ConditionPanel) -> dict[str, str]:
    return _validated_input_hashes(panel)


def nnunet_input_hashes(panel: ConditionPanel) -> dict[str, str]:
    return _validated_input_hashes(panel)


def ordered_panel_sha256(names: Sequence[str], values: Sequence[np.ndarray]) -> str:
    if len(names) != len(values):
        raise ValueError("condition names and values length mismatch")
    digest = hashlib.sha256()
    for name, value in zip(names, values):
        encoded_name = name.encode("utf-8")
        encoded_value = _rgb(value).tobytes(order="C")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_value).to_bytes(8, "big"))
        digest.update(encoded_value)
    return digest.hexdigest()


def load_condition_golden(path: str | Path | None = None) -> dict[str, object]:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "rq1_v2_condition_golden.json"
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    payload["condition_uint64_seeds"] = {
        key: int(value) for key, value in payload["condition_uint64_seeds"].items()
    }
    return payload
