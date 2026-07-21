from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from lesion_robustness import loop204_protocol, loop205_protocol, loop206_active_contour
from lesion_robustness.config import load_config
from lesion_robustness.corruptions import apply_corruption, deterministic_corruption_kwargs
from lesion_robustness.demo.immutable_io import ImmutableSnapshot
from lesion_robustness.image_utils import resize_image_and_mask
from lesion_robustness.packed_extra_channel import sha256_rgb_array
from lesion_robustness.preprocessing import preprocess_image_from_config


ARTIFACT_SCHEMA = "loop206.demo.prior.v1"
RECEIPT_SCHEMA = "loop206.demo.prior_receipt.v1"
DATASET_INDEX_SCHEMA = "loop206.demo.dataset_index.v1"
CANDIDATE_CACHE_SCHEMA = "loop206.leakage_safe_pilot_cache.v2"
VIEWS = ("clean", "low_contrast", "gaussian_noise")
PROJECT_SEED = 206
EXPECTED_FIT_ROWS = 308
EXPECTED_HOLDOUT_ROWS = 76
EXPECTED_DATASET_ROWS = 384
EXPECTED_CACHE_ROWS = 536
EXPECTED_FOLD_COUNTS = {0: 77, 1: 77, 2: 77, 3: 77, 4: 76}
EXPECTED_BASE_THRESHOLD = 0.07500000000000001
EXPECTED_FIT_FOLD_THRESHOLDS = {
    0: 0.07500000000000001,
    1: 0.07500000000000001,
    2: 0.05,
    3: 0.07500000000000001,
}
LOCKED_CONFIG_NAME = "neutral_mid_30_s2"
DEFAULT_FROZEN_CONFIG = Path("configs/loop206/l206_control_train_screen_pilot20.yaml")


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(loop204_protocol.canonical_json_bytes(payload)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_active_config() -> loop206_active_contour.ActiveContourConfig:
    return next(
        config
        for config in loop206_active_contour.CANDIDATE_CONFIGS
        if config.name == LOCKED_CONFIG_NAME
    )


def _runtime_payload(
    *, image_size: tuple[int, int], frozen_config: str | Path | None = None
) -> dict[str, Any]:
    if frozen_config is None:
        preprocessing = deepcopy(loop204_protocol.BASE_PREPROCESSING)
        corruption_configs = {
            "low_contrast": {"factor": 0.5},
            "gaussian_noise": {"sigma": 0.05},
        }
    else:
        loaded = load_config(frozen_config)
        if int(loaded.get("project", {}).get("seed", -1)) != PROJECT_SEED:
            raise ValueError("Loop206 frozen config project seed mismatch")
        configured_size = tuple(int(value) for value in loaded.get("data", {}).get("image_size", ()))
        if configured_size != tuple(image_size):
            raise ValueError("Loop206 frozen config image size mismatch")
        if tuple(loaded.get("evaluation", {}).get("corruptions", ())) != VIEWS:
            raise ValueError("Loop206 frozen config corruption panel mismatch")
        preprocessing = deepcopy(dict(loaded.get("preprocessing", {})))
        preprocessing["extra_channel"] = {"enabled": False}
        if preprocessing != loop204_protocol.BASE_PREPROCESSING:
            raise ValueError("Loop206 frozen preprocessing mismatch")
        configured = loaded.get("robustness", {}).get("corruptions", {})
        corruption_configs = {name: dict(configured.get(name, {})) for name in VIEWS[1:]}
    return {
        "project_seed": PROJECT_SEED,
        "image_size": list(image_size),
        "views": list(VIEWS),
        "preprocessing": preprocessing,
        "corruption_configs": corruption_configs,
        "active_contour": asdict(_canonical_active_config()),
    }


def _current_code_hashes() -> dict[str, str]:
    return {
        "loop205_protocol": sha256_file(Path(loop205_protocol.__file__).resolve()),
        "loop206_active_contour": sha256_file(Path(loop206_active_contour.__file__).resolve()),
        "loop206_prior": sha256_file(Path(__file__).resolve()),
    }


@dataclass(frozen=True)
class PriorFitRow:
    sample_id: str
    group_key: str
    image: np.ndarray
    mask: np.ndarray
    dataset_index: int


@dataclass(frozen=True)
class PriorHoldoutRow:
    sample_id: str
    group_key: str
    image: np.ndarray
    dataset_index: int
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class ValidatedCandidateCache:
    payload: dict[str, Any]
    manifest_sha256: str
    data_snapshot: ImmutableSnapshot


@dataclass(frozen=True)
class Loop206Prior:
    regressor: Any
    selected_threshold: float
    loop205_config: dict[str, Any]
    loop206_config: dict[str, Any]
    manifest_sha256: str
    fit_group_sha256: str
    feature_names: tuple[str, ...]
    sklearn_version: str
    code_hashes: dict[str, str]
    parity_passed: bool
    schema_version: str = ARTIFACT_SCHEMA

    def _preprocess(
        self, image_rgb_u8: np.ndarray, *, corruption: str = "clean", dataset_index: int = 0
    ) -> np.ndarray:
        image = np.asarray(image_rgb_u8)
        expected = tuple(int(value) for value in self.loop206_config["image_size"])
        if image.shape != (*expected, 3) or image.dtype != np.uint8:
            raise ValueError(f"Loop206 prior requires RGB uint8 with shape {expected + (3,)}")
        name = str(corruption).strip().lower()
        if name not in VIEWS:
            raise ValueError(f"unsupported Loop206 prior view: {name}")
        runtime = image
        if name != "clean":
            kwargs = deterministic_corruption_kwargs(
                name,
                dict(self.loop206_config["corruption_configs"][name]),
                base_seed=int(self.loop206_config["project_seed"]),
                index=int(dataset_index),
            )
            runtime = apply_corruption(runtime, name, **kwargs)
        processed = preprocess_image_from_config(
            runtime, deepcopy(self.loop206_config["preprocessing"])
        )
        return np.ascontiguousarray(processed, dtype=np.uint8)

    def _predict_preprocessed(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        config = loop205_protocol.Loop205Config(**self.loop205_config)
        return loop205_protocol.predict_saliency_map(
            self.regressor, image_rgb_u8, config=config
        )

    def _contour_preprocessed(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        probability = self._predict_preprocessed(image_rgb_u8)
        contour, _ = loop206_active_contour.refine_active_contour(
            image_rgb_u8,
            probability,
            float(self.selected_threshold),
            config=_canonical_active_config(),
        )
        return np.asarray(contour, dtype=np.uint8) * 255

    def predict(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        if not self.parity_passed:
            raise RuntimeError("Loop206 candidate prior parity receipt has not passed")
        processed = self._preprocess(image_rgb_u8)
        return self._contour_preprocessed(processed)


def _validate_fit_rows(rows: Sequence[PriorFitRow]) -> tuple[int, int]:
    if not rows:
        raise ValueError("Loop206 prior fit rows are empty")
    groups = [row.group_key for row in rows]
    if any(not group for group in groups) or len(set(groups)) != len(groups):
        raise ValueError("Loop206 prior fit groups must be non-empty and unique")
    first_shape = np.asarray(rows[0].image).shape
    if len(first_shape) != 3 or first_shape[2] != 3:
        raise ValueError("Loop206 prior fit images must be RGB")
    image_size = (int(first_shape[0]), int(first_shape[1]))
    for row in rows:
        image = np.asarray(row.image)
        mask = np.asarray(row.mask)
        if image.shape != (*image_size, 3) or image.dtype != np.uint8:
            raise ValueError("Loop206 prior fit image shape/dtype mismatch")
        if mask.shape != image_size or not (mask > 0).any() or (mask > 0).all():
            raise ValueError("Loop206 prior fit mask is malformed")
        if int(row.dataset_index) < 0:
            raise ValueError("Loop206 prior dataset index must be non-negative")
    return image_size


def fit_deployment_prior(
    fit_rows: Sequence[PriorFitRow],
    *,
    n_jobs: int = 1,
    manifest_sha256: str | None = None,
    frozen_config: str | Path | None = None,
    parity_passed: bool = False,
) -> Loop206Prior:
    image_size = _validate_fit_rows(fit_rows)
    runtime_payload = _runtime_payload(image_size=image_size, frozen_config=frozen_config)
    rf_config = loop205_protocol.Loop205Config()
    batches: list[loop205_protocol.RegionFeatureBatch] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    corruptions: list[str] = []
    for row in fit_rows:
        binary_mask = np.asarray(row.mask) > 0
        for corruption in VIEWS:
            temporary_prior = Loop206Prior(
                regressor=None,
                selected_threshold=0.5,
                loop205_config=asdict(rf_config),
                loop206_config=runtime_payload,
                manifest_sha256="0" * 64,
                fit_group_sha256="0" * 64,
                feature_names=tuple(loop205_protocol.FEATURE_NAMES),
                sklearn_version="",
                code_hashes={},
                parity_passed=False,
            )
            image = temporary_prior._preprocess(
                row.image, corruption=corruption, dataset_index=row.dataset_index
            )
            batch = loop205_protocol.extract_region_features(image, config=rf_config)
            batches.append(batch)
            targets.append(loop205_protocol.compute_region_targets(batch, binary_mask))
            masks.append(binary_mask)
            corruptions.append(corruption)
    feature_matrix = np.concatenate([batch.matrix for batch in batches], axis=0)
    target_vector = np.concatenate(targets, axis=0)
    regressor = loop205_protocol.fit_region_forest(
        feature_matrix, target_vector, config=rf_config, n_jobs_cap=int(n_jobs)
    )
    oob_scores = np.asarray(regressor.oob_prediction_, dtype=np.float32).reshape(-1)
    if oob_scores.shape != target_vector.shape or not np.isfinite(oob_scores).all():
        raise RuntimeError("Loop206 prior OOB predictions are incomplete")
    cases: list[tuple[np.ndarray, np.ndarray, str]] = []
    offset = 0
    for batch, mask, corruption in zip(batches, masks, corruptions, strict=True):
        count = int(batch.matrix.shape[0])
        dense = loop205_protocol.dense_map_from_region_scores(
            batch, oob_scores[offset : offset + count]
        )
        cases.append((dense, mask, corruption))
        offset += count
    if offset != oob_scores.size:
        raise RuntimeError("Loop206 prior OOB calibration offset mismatch")
    calibration = loop205_protocol.select_train_only_threshold(
        cases, boundary_tolerance=rf_config.boundary_tolerance
    )
    import sklearn

    groups = sorted(row.group_key for row in fit_rows)
    manifest_hash = manifest_sha256 or _canonical_hash(
        [
            {
                "sample_id": row.sample_id,
                "group_key": row.group_key,
                "dataset_index": int(row.dataset_index),
            }
            for row in fit_rows
        ]
    )
    return Loop206Prior(
        regressor=regressor,
        selected_threshold=float(calibration["selected"]["threshold"]),
        loop205_config=asdict(rf_config),
        loop206_config=runtime_payload,
        manifest_sha256=manifest_hash,
        fit_group_sha256=_canonical_hash(groups),
        feature_names=tuple(loop205_protocol.FEATURE_NAMES),
        sklearn_version=str(sklearn.__version__),
        code_hashes=_current_code_hashes(),
        parity_passed=bool(parity_passed),
    )


def save_prior(prior: Loop206Prior, path: str | Path) -> None:
    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        joblib.dump(prior, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_loaded_prior(prior: Loop206Prior) -> None:
    import sklearn

    if prior.schema_version != ARTIFACT_SCHEMA:
        raise ValueError("Loop206 prior schema mismatch")
    if prior.feature_names != tuple(loop205_protocol.FEATURE_NAMES):
        raise ValueError("Loop206 prior feature names mismatch")
    if prior.loop205_config != asdict(loop205_protocol.Loop205Config()):
        raise ValueError("Loop206 prior Loop205 config mismatch")
    image_size = tuple(int(value) for value in prior.loop206_config.get("image_size", ()))
    if prior.loop206_config != _runtime_payload(image_size=image_size):
        raise ValueError("Loop206 prior runtime config mismatch")
    if prior.sklearn_version != str(sklearn.__version__):
        raise ValueError("Loop206 prior sklearn version mismatch")
    if prior.code_hashes != _current_code_hashes():
        raise ValueError("Loop206 prior code hash mismatch")
    if not _is_sha256(prior.manifest_sha256) or not _is_sha256(prior.fit_group_sha256):
        raise ValueError("Loop206 prior manifest/group hash mismatch")
    if not 0.0 < float(prior.selected_threshold) < 1.0:
        raise ValueError("Loop206 prior threshold mismatch")
    if int(getattr(prior.regressor, "n_features_in_", -1)) != len(prior.feature_names):
        raise ValueError("Loop206 prior regressor feature count mismatch")


def load_prior(
    path: str | Path,
    *,
    expected_sha256: str,
    _snapshot: ImmutableSnapshot | None = None,
) -> Loop206Prior:
    import joblib

    snapshot = _snapshot or ImmutableSnapshot.read(path)
    if snapshot.sha256 != str(expected_sha256).strip().lower():
        raise ValueError(
            f"Loop206 prior artifact SHA256 mismatch: expected {expected_sha256}, got {snapshot.sha256}"
        )
    loaded = joblib.load(snapshot.open())
    if not isinstance(loaded, Loop206Prior):
        raise ValueError("Loop206 prior artifact type mismatch")
    _validate_loaded_prior(loaded)
    return loaded


def _receipt_hash(payload: Mapping[str, Any]) -> str:
    return _canonical_hash(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )


def _load_passed_receipt(
    path: str | Path, *, _snapshot: ImmutableSnapshot | None = None
) -> dict[str, Any]:
    import sklearn

    snapshot = _snapshot or ImmutableSnapshot.read(path)
    payload = json.loads(snapshot.text("ascii"))
    if not isinstance(payload, dict) or payload.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("Loop206 deployment receipt schema mismatch")
    if payload.get("status") != "passed":
        raise ValueError("Loop206 deployment receipt status is not passed")
    if payload.get("content_sha256") != _receipt_hash(payload):
        raise ValueError("Loop206 deployment receipt content hash mismatch")
    for field in (
        "dataset_index_sha256",
        "candidate_manifest_sha256",
        "fit_group_sha256",
        "artifact_sha256",
    ):
        if not _is_sha256(payload.get(field)):
            raise ValueError(f"Loop206 deployment receipt {field} mismatch")
    parity = payload.get("parity")
    if not isinstance(parity, dict):
        raise ValueError("Loop206 deployment receipt parity is missing")
    if (
        parity.get("parity_passed") is not True
        or int(parity.get("expected", -1)) != EXPECTED_HOLDOUT_ROWS
        or int(parity.get("input_rgb_hash_matches", -1)) != EXPECTED_HOLDOUT_ROWS
        or int(parity.get("contour_byte_matches", -1)) != EXPECTED_HOLDOUT_ROWS
        or parity.get("mismatch_groups", []) != []
        or not _is_sha256(parity.get("candidate_data_sha256"))
    ):
        raise ValueError("Loop206 deployment receipt parity mismatch")
    if parity.get("candidate_manifest_sha256") != payload["candidate_manifest_sha256"]:
        raise ValueError("Loop206 deployment receipt candidate manifest binding mismatch")
    if int(payload.get("fit_groups", -1)) != EXPECTED_FIT_ROWS:
        raise ValueError("Loop206 deployment receipt fit group count mismatch")
    if payload.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError("Loop206 deployment receipt artifact schema mismatch")
    if payload.get("feature_names") != list(loop205_protocol.FEATURE_NAMES):
        raise ValueError("Loop206 deployment receipt feature names mismatch")
    if _canonical_hash(payload.get("loop205_config")) != _canonical_hash(
        asdict(loop205_protocol.Loop205Config())
    ):
        raise ValueError("Loop206 deployment receipt Loop205 config mismatch")
    runtime = payload.get("loop206_config")
    if not isinstance(runtime, dict):
        raise ValueError("Loop206 deployment receipt Loop206 config mismatch")
    image_size = tuple(int(value) for value in runtime.get("image_size", ()))
    if _canonical_hash(runtime) != _canonical_hash(_runtime_payload(image_size=image_size)):
        raise ValueError("Loop206 deployment receipt Loop206 config mismatch")
    if payload.get("sklearn_version") != str(sklearn.__version__):
        raise ValueError("Loop206 deployment receipt sklearn version mismatch")
    if payload.get("code_hashes") != _current_code_hashes():
        raise ValueError("Loop206 deployment receipt code hashes mismatch")
    try:
        threshold = float(payload.get("selected_threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Loop206 deployment receipt threshold mismatch") from exc
    if not 0.0 < threshold < 1.0:
        raise ValueError("Loop206 deployment receipt threshold mismatch")
    return payload


def load_deployment_prior(
    artifact_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_receipt_sha256: str,
) -> Loop206Prior:
    """Load the Task5 prior only after validating its immutable parity receipt."""

    prior, _ = load_deployment_prior_with_receipt_hash(
        artifact_path,
        receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    return prior


def load_deployment_prior_with_receipt_hash(
    artifact_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_receipt_sha256: str,
) -> tuple[Loop206Prior, str]:
    """Return the prior and hash of the exact receipt bytes that authorized it."""

    expected_receipt_hash = str(expected_receipt_sha256).strip().lower()
    if not _is_sha256(expected_receipt_hash):
        raise ValueError("Loop206 deployment receipt SHA256 pin is invalid")
    receipt_snapshot = ImmutableSnapshot.read(receipt_path)
    if receipt_snapshot.sha256 != expected_receipt_hash:
        raise ValueError("Loop206 deployment receipt SHA256 mismatch")
    receipt = _load_passed_receipt(receipt_path, _snapshot=receipt_snapshot)
    artifact_snapshot = ImmutableSnapshot.read(artifact_path)
    if artifact_snapshot.sha256 != receipt["artifact_sha256"]:
        raise ValueError("Loop206 deployment artifact hash mismatch")
    prior = load_prior(
        artifact_path,
        expected_sha256=receipt["artifact_sha256"],
        _snapshot=artifact_snapshot,
    )
    if prior.parity_passed is not True:
        raise ValueError("Loop206 deployment artifact parity flag mismatch")
    bindings = (
        (prior.schema_version, receipt.get("artifact_schema"), "artifact schema"),
        (prior.manifest_sha256, receipt.get("dataset_index_sha256"), "manifest hash"),
        (prior.fit_group_sha256, receipt.get("fit_group_sha256"), "group hash"),
        (prior.selected_threshold, receipt.get("selected_threshold"), "threshold"),
        (list(prior.feature_names), receipt.get("feature_names"), "feature names"),
        (prior.sklearn_version, receipt.get("sklearn_version"), "sklearn version"),
        (prior.code_hashes, receipt.get("code_hashes"), "code hashes"),
    )
    for actual, expected, label in bindings:
        if actual != expected:
            raise ValueError(f"Loop206 deployment receipt {label} mismatch")
    for actual, expected, label in (
        (prior.loop205_config, receipt.get("loop205_config"), "Loop205 config"),
        (prior.loop206_config, receipt.get("loop206_config"), "Loop206 config"),
    ):
        if _canonical_hash(actual) != _canonical_hash(expected):
            raise ValueError(f"Loop206 deployment receipt {label} mismatch")
    return prior, receipt_snapshot.sha256


def _safe_index_path(root: Path, relative: object) -> Path:
    text = str(relative)
    candidate = (root / Path(text)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Loop206 dataset path escapes root: {text}") from exc
    return candidate


def _default_dataset_roots(dataset_index: Path) -> list[Path]:
    roots: list[Path] = []
    configured = os.environ.get("IMP_LOOP206_DATA_ROOT", "").strip()
    if configured:
        roots.extend(Path(value) for value in configured.split(os.pathsep) if value)
    repo_root = dataset_index.resolve().parent.parent
    roots.extend(
        (
            repo_root / "demo_runtime" / "dataset",
            repo_root.parent / "datasets" / "loop206",
            repo_root.parent / "datasets",
        )
    )
    return [root.expanduser().resolve() for root in roots if root.expanduser().is_dir()]


def _load_verified_indexed_pair(
    image_path: Path, mask_path: Path, row: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    sample_id = str(row.get("sample_id", ""))
    image_snapshot = ImmutableSnapshot.read(image_path)
    if image_snapshot.sha256 != str(row.get("sha256_raw", "")):
        raise ValueError(f"Loop206 raw image hash mismatch: {sample_id}")
    image = image_snapshot.decode_rgb()
    if image_snapshot.decoded_rgb_sha256(image) != str(row.get("sha256_rgb", "")):
        raise ValueError(f"Loop206 RGB image hash mismatch: {sample_id}")
    mask_snapshot = ImmutableSnapshot.read(mask_path)
    if mask_snapshot.sha256 != str(row.get("mask_sha256_raw", "")):
        raise ValueError(f"Loop206 raw mask hash mismatch: {sample_id}")
    mask = mask_snapshot.decode_binary_mask()
    if mask_snapshot.decoded_binary_mask_sha256(mask) != str(
        row.get("mask_sha256_binary", "")
    ):
        raise ValueError(f"Loop206 binary mask hash mismatch: {sample_id}")
    return image, mask


def load_dataset_index(
    dataset_index: str | Path,
    *,
    dataset_roots: Sequence[str | Path] = (),
    _snapshot: ImmutableSnapshot | None = None,
) -> tuple[list[PriorFitRow], list[PriorHoldoutRow], dict[str, Any]]:
    index_path = Path(dataset_index).resolve()
    snapshot = _snapshot or ImmutableSnapshot.read(index_path)
    payload = json.loads(snapshot.text("ascii"))
    if payload.get("schema_version") != DATASET_INDEX_SCHEMA:
        raise ValueError("Loop206 dataset index schema mismatch")
    rows = list(payload.get("rows", []))
    try:
        declared_rows = int(payload.get("row_count", -1))
        declared_roots = int(payload.get("root_count", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Loop206 dataset row count/root count mismatch") from exc
    if (
        len(rows) != EXPECTED_DATASET_ROWS
        or declared_rows != len(rows)
        or int(payload.get("fit_count", -1)) != EXPECTED_FIT_ROWS
        or int(payload.get("holdout_count", -1)) != EXPECTED_HOLDOUT_ROWS
    ):
        raise ValueError("Loop206 dataset row count mismatch")
    if declared_roots < 1:
        raise ValueError("Loop206 dataset root count mismatch")
    role_counts = Counter(str(row.get("role", "")).strip().lower() for row in rows)
    if role_counts != Counter({"fit": EXPECTED_FIT_ROWS, "holdout": EXPECTED_HOLDOUT_ROWS}):
        raise ValueError("Loop206 dataset role count mismatch")
    fold_counts = Counter(int(row.get("fold", -1)) for row in rows)
    if fold_counts != Counter(EXPECTED_FOLD_COUNTS):
        raise ValueError("Loop206 dataset fold count mismatch")
    for row in rows:
        role = str(row.get("role", "")).strip().lower()
        fold = int(row.get("fold", -1))
        split = str(row.get("split", "")).strip().lower()
        if role == "fit" and (fold not in range(4) or split != "train"):
            raise ValueError("Loop206 dataset fold/role policy mismatch")
        if role == "holdout" and (fold != 4 or split != "train_screen_holdout"):
            raise ValueError("Loop206 dataset fold/role policy mismatch")
        if str(row.get("source_split", "")).strip().lower() != "train":
            raise ValueError("Loop206 dataset source split policy mismatch")
        try:
            references = (int(row["image_root"]), int(row["mask_root"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Loop206 dataset root reference mismatch") from exc
        if any(reference not in range(declared_roots) for reference in references):
            raise ValueError("Loop206 dataset root reference mismatch")
    roots = [Path(root).expanduser().resolve() for root in dataset_roots]
    if not roots:
        roots = _default_dataset_roots(index_path)
    if len(roots) < declared_roots:
        raise ValueError("Loop206 dataset roots are missing; set IMP_LOOP206_DATA_ROOT")
    roots = roots[:declared_roots]
    fit_rows: list[PriorFitRow] = []
    holdout_rows: list[PriorHoldoutRow] = []
    fit_index = 0
    holdout_index = 0
    groups: set[str] = set()
    for row in rows:
        role = str(row.get("role", "")).strip().lower()
        group = str(row.get("group_key", "")).strip()
        if role not in {"fit", "holdout"} or not group or group in groups:
            raise ValueError("Loop206 dataset index role/group mismatch")
        groups.add(group)
        image_root = int(row["image_root"])
        mask_root = int(row["mask_root"])
        if image_root not in range(len(roots)) or mask_root not in range(len(roots)):
            raise ValueError("Loop206 dataset index root reference mismatch")
        image_path = _safe_index_path(roots[image_root], row["image_relative"])
        mask_path = _safe_index_path(roots[mask_root], row["mask_relative"])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Loop206 indexed dataset file is missing: {image_path}")
        image, mask = _load_verified_indexed_pair(image_path, mask_path, row)
        resized_image, resized_mask = resize_image_and_mask(image, mask, (384, 384))
        assert resized_mask is not None
        if role == "fit":
            fit_rows.append(
                PriorFitRow(
                    sample_id=str(row["sample_id"]),
                    group_key=group,
                    image=np.asarray(resized_image, dtype=np.uint8),
                    mask=np.asarray(resized_mask, dtype=np.uint8),
                    dataset_index=fit_index,
                )
            )
            fit_index += 1
        else:
            holdout_rows.append(
                PriorHoldoutRow(
                    sample_id=str(row["sample_id"]),
                    group_key=group,
                    image=np.asarray(resized_image, dtype=np.uint8),
                    dataset_index=holdout_index,
                    mask=np.asarray(resized_mask, dtype=np.uint8),
                )
            )
            holdout_index += 1
    if len(fit_rows) != EXPECTED_FIT_ROWS or len(holdout_rows) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError("Loop206 resolved fit/holdout counts mismatch")
    return fit_rows, holdout_rows, payload


def validate_candidate_manifest(
    candidate_manifest: str | Path,
    *,
    expected_base_threshold: float = EXPECTED_BASE_THRESHOLD,
    frozen_config: str | Path = DEFAULT_FROZEN_CONFIG,
) -> ValidatedCandidateCache:
    manifest_path = Path(candidate_manifest).resolve()
    manifest_snapshot = ImmutableSnapshot.read(manifest_path)
    payload = json.loads(manifest_snapshot.text("ascii"))
    exact_fields = {
        "schema_version": CANDIDATE_CACHE_SCHEMA,
        "artifact_type": "loop206_packed_binary_channel",
        "status": "passed",
        "arm": "candidate",
        "count": EXPECTED_CACHE_ROWS,
        "shape": [384, 384],
        "source_row_count": EXPECTED_DATASET_ROWS,
        "fit_clean_rows": EXPECTED_FIT_ROWS,
        "holdout_rows_per_corruption": EXPECTED_HOLDOUT_ROWS,
        "source_split_counts": {"train": EXPECTED_CACHE_ROWS},
        "allowed_runtime_splits": ["train", "train_screen_holdout"],
        "runtime_split_counts": {"train": EXPECTED_FIT_ROWS, "train_screen_holdout": 228},
        "corruption_counts": {"clean": 384, "gaussian_noise": 76, "low_contrast": 76},
        "input_rgb_sha256_count": EXPECTED_CACHE_ROWS,
        "locked_active_contour_config": asdict(_canonical_active_config()),
    }
    for field, expected in exact_fields.items():
        if payload.get(field) != expected:
            raise ValueError(f"Loop206 candidate manifest {field} contract mismatch")
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("dtype") != "uint8":
        raise ValueError("Loop206 candidate manifest data dtype mismatch")
    data_file = str(data.get("file", ""))
    if not data_file or Path(data_file).name != data_file:
        raise ValueError("Loop206 candidate manifest data path mismatch")
    data_path = (manifest_path.parent / data_file).resolve()
    try:
        data_path.relative_to(manifest_path.parent)
    except ValueError as exc:
        raise ValueError("Loop206 candidate manifest data path escape") from exc
    expected_size = EXPECTED_CACHE_ROWS * 384 * 384
    try:
        data_snapshot = ImmutableSnapshot.read(data_path)
    except FileNotFoundError as exc:
        raise ValueError("Loop206 candidate manifest data size mismatch") from exc
    if data_snapshot.size != expected_size:
        raise ValueError("Loop206 candidate manifest data size mismatch")
    if not _is_sha256(data.get("sha256")) or data_snapshot.sha256 != data["sha256"]:
        raise ValueError("Loop206 candidate manifest data hash mismatch")

    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CACHE_ROWS:
        raise ValueError("Loop206 candidate manifest row count mismatch")
    rows_hash = _canonical_hash(rows)
    if payload.get("rows_sha256") != rows_hash:
        raise ValueError("Loop206 candidate manifest rows hash mismatch")
    indices: list[int] = []
    row_keys: set[tuple[str, str]] = set()
    image_keys: set[tuple[str, str]] = set()
    fit_fold_counts: Counter[int] = Counter()
    holdout_views: dict[str, set[str]] = defaultdict(set)
    holdout_indices: dict[str, int] = {}
    holdout_samples: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Loop206 candidate manifest row contract mismatch")
        try:
            raw_index = row["index"]
            raw_fold = row["fold"]
            raw_threshold = row["base_threshold"]
        except KeyError as exc:
            raise ValueError("Loop206 candidate manifest index/threshold mismatch") from exc
        if (
            type(raw_index) is not int
            or type(raw_fold) is not int
            or isinstance(raw_threshold, bool)
            or not isinstance(raw_threshold, (int, float))
        ):
            raise ValueError("Loop206 candidate manifest index/threshold type mismatch")
        index, fold, threshold = raw_index, raw_fold, float(raw_threshold)
        indices.append(index)
        group = str(row.get("group_key", ""))
        sample = str(row.get("sample_id", ""))
        image_path = str(row.get("image_path", ""))
        corruption = str(row.get("corruption", ""))
        if not group or not sample or not image_path or corruption not in VIEWS:
            raise ValueError("Loop206 candidate manifest row key mismatch")
        row_key = (group, corruption)
        image_key = (image_path, corruption)
        if row_key in row_keys or image_key in image_keys:
            raise ValueError("Loop206 candidate manifest row key uniqueness mismatch")
        row_keys.add(row_key)
        image_keys.add(image_key)
        if row.get("source_split") != "train":
            raise ValueError("Loop206 candidate manifest source split policy mismatch")
        if row.get("locked_config") != LOCKED_CONFIG_NAME:
            raise ValueError("Loop206 candidate manifest locked config mismatch")
        expected_threshold = (
            EXPECTED_FIT_FOLD_THRESHOLDS[fold]
            if fold in EXPECTED_FIT_FOLD_THRESHOLDS
            else float(expected_base_threshold)
        )
        if threshold != expected_threshold:
            raise ValueError("Loop206 candidate manifest threshold mismatch")
        if not _is_sha256(row.get("input_rgb_sha256")):
            raise ValueError("Loop206 candidate manifest input hash mismatch")
        if not isinstance(row.get("candidate_fallback_used"), bool):
            raise ValueError("Loop206 candidate manifest fallback policy mismatch")
        if not isinstance(row.get("candidate_fallback_reason"), str):
            raise ValueError("Loop206 candidate manifest fallback policy mismatch")
        if fold in range(4):
            if (
                corruption != "clean"
                or row.get("runtime_split") != "train"
                or row.get("holdout_dataset_index") is not None
            ):
                raise ValueError("Loop206 candidate manifest fit row policy mismatch")
            fit_fold_counts[fold] += 1
        elif fold == 4:
            if row.get("runtime_split") != "train_screen_holdout":
                raise ValueError("Loop206 candidate manifest holdout runtime policy mismatch")
            holdout_index = row.get("holdout_dataset_index")
            if type(holdout_index) is not int:
                raise ValueError("Loop206 candidate manifest holdout index mismatch")
            if holdout_index not in range(EXPECTED_HOLDOUT_ROWS):
                raise ValueError("Loop206 candidate manifest holdout index mismatch")
            if group in holdout_indices and holdout_indices[group] != holdout_index:
                raise ValueError("Loop206 candidate manifest holdout index mismatch")
            if group in holdout_samples and holdout_samples[group] != sample:
                raise ValueError("Loop206 candidate manifest holdout sample mismatch")
            holdout_indices[group] = holdout_index
            holdout_samples[group] = sample
            holdout_views[group].add(corruption)
        else:
            raise ValueError("Loop206 candidate manifest fold policy mismatch")
    if sorted(indices) != list(range(EXPECTED_CACHE_ROWS)) or len(set(indices)) != len(indices):
        raise ValueError("Loop206 candidate manifest index uniqueness/bounds mismatch")
    if fit_fold_counts != Counter({fold: 77 for fold in range(4)}):
        raise ValueError("Loop206 candidate manifest fit fold policy mismatch")
    if (
        len(holdout_views) != EXPECTED_HOLDOUT_ROWS
        or set(holdout_indices.values()) != set(range(EXPECTED_HOLDOUT_ROWS))
        or any(views != set(VIEWS) for views in holdout_views.values())
    ):
        raise ValueError("Loop206 candidate manifest holdout policy mismatch")

    provenance = payload.get("provenance")
    provenance_fields = (
        "builder_sha256",
        "config_sha256",
        "confirmatory_report_sha256",
        "loop204_protocol_sha256",
        "loop205_protocol_sha256",
        "loop206_protocol_sha256",
        "runtime_manifest_sha256",
        "source_manifest_sha256",
    )
    if not isinstance(provenance, dict) or any(
        not _is_sha256(provenance.get(field)) for field in provenance_fields
    ):
        raise ValueError("Loop206 candidate manifest provenance hash contract mismatch")
    expected_code_hashes = {
        "loop204_protocol_sha256": sha256_file(Path(loop204_protocol.__file__).resolve()),
        "loop205_protocol_sha256": sha256_file(Path(loop205_protocol.__file__).resolve()),
        "loop206_protocol_sha256": sha256_file(
            Path(loop206_active_contour.__file__).resolve()
        ),
    }
    if any(provenance[field] != expected for field, expected in expected_code_hashes.items()):
        raise ValueError("Loop206 candidate manifest provenance code hash mismatch")
    config_path = Path(frozen_config)
    if not config_path.is_file() or provenance["config_sha256"] != sha256_file(config_path):
        raise ValueError("Loop206 candidate manifest provenance config hash mismatch")
    frozen = load_config(config_path)
    expected_runtime_manifest_hash = str(
        frozen.get("data", {}).get("manifest_sha256", "")
    )
    if provenance["runtime_manifest_sha256"] != expected_runtime_manifest_hash:
        raise ValueError("Loop206 candidate manifest provenance runtime manifest hash mismatch")
    return ValidatedCandidateCache(
        payload=payload,
        manifest_sha256=manifest_snapshot.sha256,
        data_snapshot=data_snapshot,
    )


def verify_holdout_parity(
    prior: Loop206Prior,
    holdout_rows: Sequence[PriorHoldoutRow],
    candidate_manifest: str | Path,
    *,
    _validated_cache: ValidatedCandidateCache | None = None,
) -> dict[str, Any]:
    validated = _validated_cache or validate_candidate_manifest(
        candidate_manifest, expected_base_threshold=prior.selected_threshold
    )
    payload = validated.payload
    shape = tuple(int(value) for value in payload["shape"])
    count = int(payload["count"])
    clean_rows = {
        str(row["group_key"]): row
        for row in payload.get("rows", [])
        if int(row.get("fold", -1)) == 4 and row.get("corruption") == "clean"
    }
    if len(holdout_rows) != EXPECTED_HOLDOUT_ROWS or len(clean_rows) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError("Loop206 holdout clean parity row count mismatch")
    cache = np.memmap(
        validated.data_snapshot.open(),
        mode="r",
        dtype=np.uint8,
        shape=(count, *shape),
    )
    input_hash_matches = 0
    contour_matches = 0
    mismatches: list[str] = []
    for row in holdout_rows:
        expected = clean_rows.get(row.group_key)
        if expected is None or str(expected.get("sample_id")) != row.sample_id:
            mismatches.append(row.group_key)
            continue
        processed = prior._preprocess(
            row.image, corruption="clean", dataset_index=row.dataset_index
        )
        if sha256_rgb_array(processed) == str(expected.get("input_rgb_sha256", "")):
            input_hash_matches += 1
        else:
            mismatches.append(row.group_key)
            continue
        packed = prior._contour_preprocessed(processed)
        if np.array_equal(packed, np.asarray(cache[int(expected["index"])])):
            contour_matches += 1
        else:
            mismatches.append(row.group_key)
    del cache
    passed = (
        input_hash_matches == EXPECTED_HOLDOUT_ROWS
        and contour_matches == EXPECTED_HOLDOUT_ROWS
        and not mismatches
    )
    return {
        "expected": EXPECTED_HOLDOUT_ROWS,
        "input_rgb_hash_matches": input_hash_matches,
        "contour_byte_matches": contour_matches,
        "parity_passed": passed,
        "mismatch_groups": sorted(set(mismatches)),
        "candidate_manifest_sha256": validated.manifest_sha256,
        "candidate_data_sha256": validated.data_snapshot.sha256,
    }


def _receipt_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    content["content_sha256"] = _canonical_hash(content)
    return content


def _stage_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    content = _receipt_payload(payload)
    encoded = (
        json.dumps(content, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return content


def _publish_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one staged file without replacing a concurrent writer."""

    os.link(source, destination)


def _rollback_own_publish(staged: Path, published: Path) -> None:
    try:
        if staged.exists() and published.exists() and os.path.samefile(staged, published):
            published.unlink()
    except FileNotFoundError:
        pass


def build_prior_artifact(
    *,
    dataset_index: str | Path,
    candidate_manifest: str | Path,
    output: str | Path,
    receipt: str | Path,
    n_jobs: int = 4,
    dataset_roots: Sequence[str | Path] = (),
    frozen_config: str | Path = DEFAULT_FROZEN_CONFIG,
) -> dict[str, Any]:
    output_path = Path(output).resolve()
    receipt_path = Path(receipt).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(output_path.name + ".build.lock")
    lock_token = uuid.uuid4().hex
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"Loop206 prior build lock already exists: {lock_path}") from exc
    with os.fdopen(lock_fd, "w", encoding="ascii") as lock_handle:
        lock_handle.write(lock_token)
        lock_handle.flush()
        os.fsync(lock_handle.fileno())
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.build-", dir=output_path.parent)
    )
    staged_artifact = staging_root / output_path.name
    staged_receipt = staging_root / receipt_path.name
    base_receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "created_at": _utc_now(),
        "requested_output": str(output_path),
        "requested_receipt": str(receipt_path),
    }
    failure_details: dict[str, Any] = {}
    artifact_published = False
    try:
        if output_path.parent != receipt_path.parent:
            raise ValueError("Loop206 prior output and receipt must share one directory")
        if output_path.exists() or receipt_path.exists():
            raise FileExistsError("Loop206 prior output/receipt is immutable and already exists")
        dataset_snapshot = ImmutableSnapshot.read(dataset_index)
        validated_cache = validate_candidate_manifest(
            candidate_manifest,
            expected_base_threshold=EXPECTED_BASE_THRESHOLD,
            frozen_config=frozen_config,
        )
        base_receipt["dataset_index_sha256"] = dataset_snapshot.sha256
        base_receipt["candidate_manifest_sha256"] = validated_cache.manifest_sha256
        fit_rows, holdout_rows, _ = load_dataset_index(
            dataset_index, dataset_roots=dataset_roots, _snapshot=dataset_snapshot
        )
        prior = fit_deployment_prior(
            fit_rows,
            n_jobs=n_jobs,
            manifest_sha256=base_receipt["dataset_index_sha256"],
            frozen_config=frozen_config,
            parity_passed=False,
        )
        save_prior(prior, staged_artifact)
        parity = verify_holdout_parity(
            prior,
            holdout_rows,
            candidate_manifest,
            _validated_cache=validated_cache,
        )
        if not parity["parity_passed"]:
            failure_details = {
                "fit_groups": len(fit_rows),
                "fit_group_sha256": prior.fit_group_sha256,
                "selected_threshold": prior.selected_threshold,
                "sklearn_version": prior.sklearn_version,
                "code_hashes": prior.code_hashes,
                "parity": parity,
            }
            raise RuntimeError(
                "Loop206 holdout parity failed: "
                f"{parity['contour_byte_matches']}/{parity['expected']} contours"
            )
        prior = replace(prior, parity_passed=True)
        save_prior(prior, staged_artifact)
        artifact_sha256 = sha256_file(staged_artifact)
        passed_receipt = {
            **base_receipt,
            "status": "passed",
            "fit_groups": len(fit_rows),
            "fit_group_sha256": prior.fit_group_sha256,
            "selected_threshold": prior.selected_threshold,
            "artifact_sha256": artifact_sha256,
            "artifact_schema": prior.schema_version,
            "loop205_config": prior.loop205_config,
            "loop206_config": prior.loop206_config,
            "feature_names": list(prior.feature_names),
            "sklearn_version": prior.sklearn_version,
            "code_hashes": prior.code_hashes,
            "parity": parity,
        }
        published_receipt = _stage_receipt(staged_receipt, passed_receipt)
        _publish_no_replace(staged_artifact, output_path)
        artifact_published = True
        _publish_no_replace(staged_receipt, receipt_path)
        return published_receipt
    except Exception as exc:
        if artifact_published:
            _rollback_own_publish(staged_artifact, output_path)
        if not receipt_path.exists():
            failed_receipt = {
                **base_receipt,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                **failure_details,
            }
            failed_stage = staging_root / f"failed-{receipt_path.name}"
            try:
                _stage_receipt(failed_stage, failed_receipt)
                _publish_no_replace(failed_stage, receipt_path)
            except FileExistsError:
                pass
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        try:
            if lock_path.read_text(encoding="ascii") == lock_token:
                lock_path.unlink()
        except FileNotFoundError:
            pass
