"""RQ1-v2 split capability, identity, and leakage audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


PROTOCOL_SCHEMA = "imp.rq1_v2.protocol.v1"
REPORT_SCHEMA = "imp.rq1_v2.data_integrity_report.v1"
INDEX_SCHEMA = "imp.rq1_v2.dataset_index.v1"
RQ1_V2_CAPABILITY = MappingProxyType({"train": 2008, "validation": 431})

_EXPECTED_PROTOCOL = {
    "research_question": "Under one Clean-v3 adaptive-validation identity set and original-image metric geometry, how do complete IMP MiT-B3 U-Net and nnU-Net v2 systems trade overlap and boundary quality across independent training runs?",
    "evidence_class": "adaptive_development_validation_rerun",
    "clean_v3_manifest_sha256": "4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102",
    "probability_restore": "bilinear_to_original_before_threshold",
    "primary_endpoint": "independent_arm_mean_robust_dice_delta_nnunet_minus_imp",
    "secondary_endpoint": "independent_arm_mean_robust_boundary_f1_delta_nnunet_minus_imp",
    "claim_limit": "adaptive validation; no protected-test or statistical-superiority claim",
}
_SEEDS = (206, 1206, 2206)
_CONDITIONS = (
    "clean",
    "low_brightness",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
)
_METRICS = ("dice", "iou", "precision", "recall", "boundary_f1", "hd95", "assd")
_PROTOCOL_FIELDS = frozenset(
    {
        "schema_version",
        "research_question",
        "evidence_class",
        "train_count",
        "validation_count",
        "test_v3_access",
        "clean_v3_manifest_sha256",
        "dataset_index_status",
        "dataset_index_sha256",
        "seeds",
        "conditions",
        "metrics",
        "boundary_tolerance_original_pixels",
        "probability_restore",
        "threshold",
        "primary_endpoint",
        "secondary_endpoint",
        "claim_limit",
    }
)
_IDENTITY_FIELDS = frozenset(
    {"train_ordered_identity_sha256", "validation_ordered_identity_sha256"}
)


@dataclass(frozen=True)
class Rq1Protocol:
    schema_version: str
    research_question: str
    evidence_class: str
    train_count: int
    validation_count: int
    test_v3_access: bool
    clean_v3_manifest_sha256: str
    dataset_index_status: str
    dataset_index_sha256: str | None
    seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    metrics: tuple[str, ...]
    boundary_tolerance_original_pixels: int
    probability_restore: str
    threshold: float
    primary_endpoint: str
    secondary_endpoint: str
    claim_limit: str
    train_ordered_identity_sha256: str | None = None
    validation_ordered_identity_sha256: str | None = None


@dataclass(frozen=True)
class DataRow:
    sample_id: str
    split: str
    group_key: str
    sha256_raw: str
    sha256_rgb: str
    source_dataset: str
    image_rgb: np.ndarray
    dataset_index_sha256: str
    clean_v3_manifest_sha256: str
    mask_sha256: str | None = None
    image_path: Path | None = None
    mask_path: Path | None = None


@dataclass(frozen=True)
class DataIntegrityReport:
    schema_version: str
    audit_id: str
    train_count: int
    validation_count: int
    train_ordered_identity_sha256: str
    validation_ordered_identity_sha256: str
    cross_split_groups: int
    cross_split_exact_rgb: int
    near_duplicate_candidate_count: int
    cross_split_near_rgb: int
    clean_v3_manifest_sha256: str
    dataset_index_status: str
    dataset_index_sha256: str
    test_v3_access: bool
    test_v3_open_count: int
    algorithms: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["canonical_report_sha256"] = canonical_report_sha256(payload)
        return payload


def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw.decode("ascii"), object_pairs_hook=_pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be duplicate-free ASCII JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_protocol_contract(
    protocol: Rq1Protocol, *, require_verified_index: bool
) -> Rq1Protocol:
    if protocol.schema_version != PROTOCOL_SCHEMA:
        raise ValueError("RQ1-v2 protocol schema_version drift")
    for field, expected in _EXPECTED_PROTOCOL.items():
        if getattr(protocol, field) != expected:
            raise ValueError(f"RQ1-v2 protocol {field} drift")
    if type(protocol.train_count) is not int or protocol.train_count != RQ1_V2_CAPABILITY["train"]:
        raise ValueError("RQ1-v2 protocol train_count must be 2008")
    if type(protocol.validation_count) is not int or protocol.validation_count != RQ1_V2_CAPABILITY["validation"]:
        raise ValueError("RQ1-v2 protocol validation_count must be 431")
    if protocol.test_v3_access is not False:
        raise ValueError("test-v3 must remain sealed")
    if protocol.seeds != _SEEDS:
        raise ValueError("RQ1-v2 protocol seeds drift")
    if protocol.conditions != _CONDITIONS:
        raise ValueError("RQ1-v2 protocol conditions drift")
    if protocol.metrics != _METRICS:
        raise ValueError("RQ1-v2 protocol metrics drift")
    if type(protocol.boundary_tolerance_original_pixels) is not int or protocol.boundary_tolerance_original_pixels != 2:
        raise ValueError("RQ1-v2 protocol boundary tolerance drift")
    if protocol.probability_restore != _EXPECTED_PROTOCOL["probability_restore"]:
        raise ValueError("RQ1-v2 protocol probability_restore drift")
    if type(protocol.threshold) is not float or protocol.threshold != 0.5:
        raise ValueError("RQ1-v2 protocol threshold drift")
    for field in ("primary_endpoint", "secondary_endpoint", "claim_limit"):
        if getattr(protocol, field) != _EXPECTED_PROTOCOL[field]:
            raise ValueError(f"RQ1-v2 protocol {field} drift")
    _sha256(protocol.clean_v3_manifest_sha256, "clean_v3_manifest_sha256")
    if protocol.dataset_index_status not in {"unresolved_blocked", "verified"}:
        raise ValueError("RQ1-v2 protocol dataset_index_status is invalid")
    if protocol.dataset_index_status == "unresolved_blocked":
        if protocol.dataset_index_sha256 is not None:
            raise ValueError("unresolved_blocked dataset index forbids a SHA-256")
        if require_verified_index:
            raise ValueError("RQ1-v2 dataset index is unresolved_blocked")
    else:
        if protocol.dataset_index_sha256 is None:
            raise ValueError("verified dataset index requires a SHA-256")
        _sha256(protocol.dataset_index_sha256, "dataset_index_sha256")
    if (protocol.train_ordered_identity_sha256 is None) != (
        protocol.validation_ordered_identity_sha256 is None
    ):
        raise ValueError("ordered identity SHA-256 values must be frozen together")
    if protocol.train_ordered_identity_sha256 is not None:
        _sha256(
            protocol.train_ordered_identity_sha256,
            "train_ordered_identity_sha256",
        )
        _sha256(
            protocol.validation_ordered_identity_sha256,
            "validation_ordered_identity_sha256",
        )
    return protocol


def load_protocol(path: str | Path) -> Rq1Protocol:
    protocol_path = Path(path)
    try:
        payload = _load_json_bytes(protocol_path.read_bytes(), "RQ1-v2 protocol")
    except OSError as exc:
        raise ValueError("RQ1-v2 protocol could not be opened") from exc
    fields = frozenset(payload)
    if fields not in (_PROTOCOL_FIELDS, _PROTOCOL_FIELDS | _IDENTITY_FIELDS):
        raise ValueError("RQ1-v2 protocol fields do not match the frozen schema")
    protocol = Rq1Protocol(
        schema_version=str(payload["schema_version"]),
        research_question=str(payload["research_question"]),
        evidence_class=str(payload["evidence_class"]),
        train_count=payload["train_count"],
        validation_count=payload["validation_count"],
        test_v3_access=payload["test_v3_access"],
        clean_v3_manifest_sha256=str(payload["clean_v3_manifest_sha256"]),
        dataset_index_status=str(payload["dataset_index_status"]),
        dataset_index_sha256=payload["dataset_index_sha256"],
        seeds=tuple(payload["seeds"]),
        conditions=tuple(payload["conditions"]),
        metrics=tuple(payload["metrics"]),
        boundary_tolerance_original_pixels=payload[
            "boundary_tolerance_original_pixels"
        ],
        probability_restore=str(payload["probability_restore"]),
        threshold=payload["threshold"],
        primary_endpoint=str(payload["primary_endpoint"]),
        secondary_endpoint=str(payload["secondary_endpoint"]),
        claim_limit=str(payload["claim_limit"]),
        train_ordered_identity_sha256=payload.get("train_ordered_identity_sha256"),
        validation_ordered_identity_sha256=payload.get(
            "validation_ordered_identity_sha256"
        ),
    )
    return _validate_protocol_contract(protocol, require_verified_index=False)


def protocol_payload(protocol: Rq1Protocol) -> dict[str, Any]:
    _validate_protocol_contract(protocol, require_verified_index=False)
    payload = asdict(protocol)
    payload["seeds"] = list(protocol.seeds)
    payload["conditions"] = list(protocol.conditions)
    payload["metrics"] = list(protocol.metrics)
    if protocol.train_ordered_identity_sha256 is None:
        payload.pop("train_ordered_identity_sha256")
        payload.pop("validation_ordered_identity_sha256")
    return payload


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def canonical_report_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("canonical_report_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _validate_task8_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("RQ1v2-"):
        raise ValueError(f"{label} must begin RQ1v2-")
    if not value or "|" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{label} is not canonical")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be ASCII") from exc
    return value


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError("authorized reference metadata could not be inspected") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    is_junction = getattr(os.path, "isjunction", None)
    return bool(
        path.is_symlink()
        or (reparse_flag and getattr(metadata, "st_file_attributes", 0) & reparse_flag)
        or (is_junction is not None and is_junction(path))
    )


def _resolve_relative(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a nonempty relative name")
    root_absolute = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(root_absolute / relative))
    try:
        candidate.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its authorized root") from exc
    current = root_absolute
    if _is_reparse_or_symlink(current):
        raise ValueError(f"{label} traverses a reparse or symlink")
    for part in Path(relative).parts:
        current /= part
        if _is_reparse_or_symlink(current):
            raise ValueError(f"{label} traverses a reparse or symlink")
    return candidate


def _row_file(payload: Mapping[str, Any], roots: tuple[Path, ...], kind: str) -> Path:
    root_value = payload.get(f"{kind}_root")
    if type(root_value) is not int or not 0 <= root_value < len(roots):
        raise ValueError(f"{kind} root capability is unavailable")
    return _resolve_relative(roots[root_value], payload.get(f"{kind}_relative"), f"{kind} reference")


def _decoded_rgb(raw: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(raw)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ValueError("authorized image decode failed") from exc
    result = np.ascontiguousarray(rgb)
    result.setflags(write=False)
    return result


def _rgb_sha256(rgb: np.ndarray) -> str:
    payload = f"{rgb.shape[0]}x{rgb.shape[1]}x3|".encode("ascii") + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _read_verified_row(
    raw: Mapping[str, Any],
    split: str,
    roots: tuple[Path, ...],
    index_sha256: str,
    manifest_sha256: str,
) -> DataRow:
    source = raw.get("source_dataset")
    sample_id = _validate_task8_id(raw.get("sample_id"), "authorized row sample_id")
    group_key = str(raw["group_key"])
    expected_raw = _sha256(raw.get("sha256_raw"), "sha256_raw")
    expected_rgb = _sha256(raw.get("sha256_rgb"), "sha256_rgb")
    expected_mask = _sha256(
        raw.get("mask_sha256", raw.get("mask_sha256_raw")), "mask_sha256"
    )
    image_path = _row_file(raw, roots, "image")
    mask_path = _row_file(raw, roots, "mask")
    try:
        image_bytes = image_path.read_bytes()
        mask_bytes = mask_path.read_bytes()
    except OSError as exc:
        raise ValueError("authorized image/mask bytes could not be opened") from exc
    if hashlib.sha256(image_bytes).hexdigest() != expected_raw:
        raise ValueError("authorized image raw SHA-256 mismatch")
    if hashlib.sha256(mask_bytes).hexdigest() != expected_mask:
        raise ValueError("authorized mask SHA-256 mismatch")
    rgb = _decoded_rgb(image_bytes)
    if _rgb_sha256(rgb) != expected_rgb:
        raise ValueError("authorized image decoded-RGB SHA-256 mismatch")
    return DataRow(
        sample_id=sample_id,
        split=split,
        group_key=group_key,
        sha256_raw=expected_raw,
        sha256_rgb=expected_rgb,
        source_dataset=source,
        image_rgb=rgb,
        dataset_index_sha256=index_sha256,
        clean_v3_manifest_sha256=manifest_sha256,
        mask_sha256=expected_mask,
        image_path=image_path,
        mask_path=mask_path,
    )


def _validate_selected_row_metadata(raw: Mapping[str, Any]) -> None:
    source = raw.get("source_dataset")
    if not isinstance(source, str) or not source:
        raise ValueError("authorized row source_dataset is missing")
    if source.casefold() == "ph2":
        raise ValueError("PH2 is not authorized for RQ1-v2")
    _validate_task8_id(raw.get("sample_id"), "authorized row sample_id")
    group_key = raw.get("group_key")
    if (
        not isinstance(group_key, str)
        or not group_key
        or "|" in group_key
        or "\n" in group_key
        or "\r" in group_key
    ):
        raise ValueError("authorized row group_key is not canonical")
    _sha256(raw.get("sha256_raw"), "sha256_raw")
    _sha256(raw.get("sha256_rgb"), "sha256_rgb")
    _sha256(raw.get("mask_sha256", raw.get("mask_sha256_raw")), "mask_sha256")
    for kind in ("image", "mask"):
        if type(raw.get(f"{kind}_root")) is not int:
            raise ValueError(f"{kind} root capability is unavailable")
        relative = raw.get(f"{kind}_relative")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{kind} reference must be a nonempty relative name")


def read_authorized_rows(
    index: str | Path,
    split: str,
    protocol: Rq1Protocol | None = None,
) -> Sequence[DataRow]:
    """Open only train/validation bytes; sealed split denial precedes index access."""
    if split in {"test", "test-v3", "test_v3"}:
        raise ValueError("test-v3 is sealed")
    if split not in RQ1_V2_CAPABILITY:
        raise ValueError(f"RQ1-v2 split {split!r} is not authorized")
    if protocol is None:
        raise ValueError("RQ1-v2 verified protocol is required before index access")
    _validate_protocol_contract(protocol, require_verified_index=True)

    index_path = Path(index)
    try:
        index_bytes = index_path.read_bytes()
    except OSError as exc:
        raise ValueError("authorized dataset index could not be opened") from exc
    index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    if index_sha256 != protocol.dataset_index_sha256:
        raise ValueError("dataset index SHA-256 mismatch")
    payload = _load_json_bytes(index_bytes, "RQ1-v2 dataset index")
    if payload.get("schema_version") != INDEX_SCHEMA:
        raise ValueError(f"dataset index must use {INDEX_SCHEMA}")
    manifest_sha256 = _sha256(
        payload.get("clean_v3_manifest_sha256"), "clean_v3_manifest_sha256"
    )
    if manifest_sha256 != protocol.clean_v3_manifest_sha256:
        raise ValueError("Clean-v3 manifest SHA-256 mismatch")
    rows_value = payload.get("rows")
    if not isinstance(rows_value, list):
        raise ValueError("dataset index rows must be a list")
    selected: list[Mapping[str, Any]] = []
    for value in rows_value:
        if not isinstance(value, Mapping):
            raise ValueError("dataset index row must be an object")
        if value.get("split") != split:
            continue
        _validate_selected_row_metadata(value)
        selected.append(value)
    sample_ids = [str(value["sample_id"]) for value in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("dataset index contains a duplicate sample_id")

    roots_value = payload.get("roots")
    if not isinstance(roots_value, list) or not roots_value:
        raise ValueError("dataset index has no authorized roots")
    if any(
        not isinstance(root, str) or not root or not Path(root).is_absolute()
        for root in roots_value
    ):
        raise ValueError("dataset index contains an invalid authorized root")
    roots = tuple(Path(root) for root in roots_value)
    opened: list[DataRow] = []
    for value in selected:
        opened.append(
            _read_verified_row(value, split, roots, index_sha256, manifest_sha256)
        )
    return tuple(opened)


def _identity_line(row: DataRow) -> bytes:
    _validate_task8_id(row.sample_id, "RQ1-v2 sample_id")
    values = (row.sample_id, row.group_key, row.sha256_raw, row.sha256_rgb)
    if any(not value or "|" in value or "\n" in value or "\r" in value for value in values):
        raise ValueError("RQ1-v2 identity fields must be nonempty and delimiter-free")
    for value in values[2:]:
        _sha256(value, "identity SHA-256")
    try:
        return ("|".join(values) + "\n").encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("RQ1-v2 identity fields must be ASCII") from exc


def ordered_identity_sha256(rows: Iterable[DataRow]) -> str:
    ordered = sorted(rows, key=lambda row: (row.sample_id, row.group_key))
    identities = [(row.sample_id, row.group_key) for row in ordered]
    if len(identities) != len(set(identities)):
        raise ValueError("RQ1-v2 ordered identity contains duplicates")
    digest = hashlib.sha256()
    for row in ordered:
        digest.update(_identity_line(row))
    return digest.hexdigest()


def phash63_luminance(rgb: np.ndarray) -> int:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    gray = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    low = cv2.dct(gray)[:8, :8].reshape(-1)[1:]
    median = float(np.median(low))
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bool(bit))
    return value


def ssim_luminance_256(first: np.ndarray, second: np.ndarray) -> float:
    def luminance(rgb: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        resized = image.convert("L").resize((256, 256), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.float64)

    left = luminance(first)
    right = luminance(second)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mu_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    sigma_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mu_left * mu_left
    sigma_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mu_right * mu_right
    covariance = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mu_left * mu_right
    numerator = (2.0 * mu_left * mu_right + c1) * (2.0 * covariance + c2)
    denominator = (mu_left * mu_left + mu_right * mu_right + c1) * (
        sigma_left + sigma_right + c2
    )
    return float(np.mean(numerator / denominator))


def _consensus(rows: Sequence[DataRow], field: str) -> str:
    values = {str(getattr(row, field)) for row in rows}
    if len(values) != 1:
        raise ValueError(f"RQ1-v2 rows disagree on {field}")
    return next(iter(values))


def audit_data(rows: Sequence[DataRow], protocol: Rq1Protocol) -> DataIntegrityReport:
    _validate_protocol_contract(protocol, require_verified_index=True)
    materialized = tuple(rows)
    if any(row.split not in RQ1_V2_CAPABILITY for row in materialized):
        raise ValueError("RQ1-v2 audit received an unauthorized split")
    if any(row.source_dataset.casefold() == "ph2" for row in materialized):
        raise ValueError("PH2 is not authorized for RQ1-v2")
    train = tuple(row for row in materialized if row.split == "train")
    validation = tuple(row for row in materialized if row.split == "validation")
    if (
        len(train) != RQ1_V2_CAPABILITY["train"]
        or len(validation) != RQ1_V2_CAPABILITY["validation"]
    ):
        raise ValueError(
            f"RQ1-v2 split count mismatch: train={len(train)} validation={len(validation)}"
        )
    for row in materialized:
        _validate_task8_id(row.sample_id, "RQ1-v2 sample_id")
        rgb = np.asarray(row.image_rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RQ1-v2 decoded RGB must be uint8 HxWx3")
        if _rgb_sha256(np.ascontiguousarray(rgb)) != row.sha256_rgb:
            raise ValueError("decoded-RGB SHA-256 mismatch")
    if len({row.sample_id for row in materialized}) != len(materialized):
        raise ValueError("RQ1-v2 sample_id values must be unique")
    manifest_sha256 = _consensus(materialized, "clean_v3_manifest_sha256")
    index_sha256 = _consensus(materialized, "dataset_index_sha256")
    if manifest_sha256 != protocol.clean_v3_manifest_sha256:
        raise ValueError("Clean-v3 manifest SHA-256 mismatch")
    if index_sha256 != protocol.dataset_index_sha256:
        raise ValueError("dataset index SHA-256 mismatch")

    train_groups = {row.group_key for row in train}
    validation_groups = {row.group_key for row in validation}
    cross_groups = len(train_groups & validation_groups)
    train_exact = {row.sha256_rgb for row in train}
    validation_exact = {row.sha256_rgb for row in validation}
    cross_exact = len(train_exact & validation_exact)
    train_phash = tuple((row, phash63_luminance(row.image_rgb)) for row in train)
    validation_phash = tuple(
        (row, phash63_luminance(row.image_rgb)) for row in validation
    )
    near_candidates = 0
    cross_near = 0
    for left, left_hash in train_phash:
        for right, right_hash in validation_phash:
            if left.sha256_rgb == right.sha256_rgb:
                continue
            if (left_hash ^ right_hash).bit_count() > 4:
                continue
            near_candidates += 1
            if ssim_luminance_256(left.image_rgb, right.image_rgb) >= 0.98:
                cross_near += 1
    if cross_groups or cross_exact or cross_near:
        raise ValueError(
            "cross-split leakage: "
            f"group={cross_groups} exact_rgb={cross_exact} near_rgb={cross_near}"
        )

    train_identity = ordered_identity_sha256(train)
    validation_identity = ordered_identity_sha256(validation)
    if (
        protocol.train_ordered_identity_sha256 is not None
        and train_identity != protocol.train_ordered_identity_sha256
    ):
        raise ValueError("train ordered identity SHA-256 mismatch")
    if (
        protocol.validation_ordered_identity_sha256 is not None
        and validation_identity != protocol.validation_ordered_identity_sha256
    ):
        raise ValueError("validation ordered identity SHA-256 mismatch")
    return DataIntegrityReport(
        schema_version=REPORT_SCHEMA,
        audit_id="RQ1v2-data-integrity",
        train_count=len(train),
        validation_count=len(validation),
        train_ordered_identity_sha256=train_identity,
        validation_ordered_identity_sha256=validation_identity,
        cross_split_groups=cross_groups,
        cross_split_exact_rgb=cross_exact,
        near_duplicate_candidate_count=near_candidates,
        cross_split_near_rgb=cross_near,
        clean_v3_manifest_sha256=manifest_sha256,
        dataset_index_status="verified",
        dataset_index_sha256=index_sha256,
        test_v3_access=False,
        test_v3_open_count=0,
        algorithms={
            "ordered_identity": "sample_id|group_key|sha256_raw|sha256_rgb\\n sorted by sample_id,group_key; ASCII; SHA-256",
            "group": "exact group_key crossing",
            "exact_rgb": "decoded RGB SHA-256 crossing",
            "near_candidate": "63-bit luminance pHash; Lanczos 32x32; DCT 8x8 excluding DC; Hamming <=4",
            "near_confirmation": "luminance SSIM at Lanczos 256x256; Gaussian 11x11 sigma 1.5; threshold >=0.98",
        },
    )


def freeze_protocol_identities(
    protocol: Rq1Protocol, report: DataIntegrityReport
) -> Rq1Protocol:
    _validate_protocol_contract(protocol, require_verified_index=True)
    return replace(
        protocol,
        train_ordered_identity_sha256=report.train_ordered_identity_sha256,
        validation_ordered_identity_sha256=report.validation_ordered_identity_sha256,
    )


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(payload)
