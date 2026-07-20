from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lesion_robustness import loop204_protocol, loop205_protocol, loop206_active_contour
from lesion_robustness.config import load_config
from lesion_robustness.corruptions import apply_corruption, deterministic_corruption_kwargs
from lesion_robustness.data_manifest import sha256_rgb
from lesion_robustness.image_utils import read_mask, read_rgb, resize_image_and_mask
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
    parity_passed: bool = True,
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


def load_prior(path: str | Path, *, expected_sha256: str) -> Loop206Prior:
    import joblib

    source = Path(path)
    actual_sha256 = sha256_file(source)
    if actual_sha256 != str(expected_sha256).strip().lower():
        raise ValueError(
            f"Loop206 prior artifact SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    loaded = joblib.load(source)
    if not isinstance(loaded, Loop206Prior):
        raise ValueError("Loop206 prior artifact type mismatch")
    _validate_loaded_prior(loaded)
    return loaded


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


def load_dataset_index(
    dataset_index: str | Path, *, dataset_roots: Sequence[str | Path] = ()
) -> tuple[list[PriorFitRow], list[PriorHoldoutRow], dict[str, Any]]:
    index_path = Path(dataset_index).resolve()
    payload = json.loads(index_path.read_text(encoding="ascii"))
    if payload.get("schema_version") != DATASET_INDEX_SCHEMA:
        raise ValueError("Loop206 dataset index schema mismatch")
    rows = list(payload.get("rows", []))
    if (
        len(rows) != EXPECTED_DATASET_ROWS
        or int(payload.get("fit_count", -1)) != EXPECTED_FIT_ROWS
        or int(payload.get("holdout_count", -1)) != EXPECTED_HOLDOUT_ROWS
    ):
        raise ValueError("Loop206 dataset index counts mismatch")
    roots = [Path(root).expanduser().resolve() for root in dataset_roots]
    if not roots:
        roots = _default_dataset_roots(index_path)
    if len(roots) < int(payload.get("root_count", -1)):
        raise ValueError("Loop206 dataset roots are missing; set IMP_LOOP206_DATA_ROOT")
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
        if sha256_file(image_path) != str(row.get("sha256_raw", "")):
            raise ValueError(f"Loop206 raw image hash mismatch: {row['sample_id']}")
        if sha256_rgb(image_path) != str(row.get("sha256_rgb", "")):
            raise ValueError(f"Loop206 RGB image hash mismatch: {row['sample_id']}")
        image, mask = read_rgb(image_path), read_mask(mask_path)
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
                )
            )
            holdout_index += 1
    if len(fit_rows) != EXPECTED_FIT_ROWS or len(holdout_rows) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError("Loop206 resolved fit/holdout counts mismatch")
    return fit_rows, holdout_rows, payload


def verify_holdout_parity(
    prior: Loop206Prior,
    holdout_rows: Sequence[PriorHoldoutRow],
    candidate_manifest: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(candidate_manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    if (
        payload.get("schema_version") != CANDIDATE_CACHE_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("arm") != "candidate"
        or payload.get("data", {}).get("dtype") != "uint8"
    ):
        raise ValueError("Loop206 candidate cache manifest mismatch")
    shape = tuple(int(value) for value in payload.get("shape", ()))
    count = int(payload.get("count", -1))
    data_path = manifest_path.parent / str(payload.get("data", {}).get("file", ""))
    if shape != (384, 384) or count <= 0 or not data_path.is_file():
        raise ValueError("Loop206 candidate cache data contract mismatch")
    if sha256_file(data_path) != str(payload.get("data", {}).get("sha256", "")):
        raise ValueError("Loop206 candidate cache data SHA256 mismatch")
    clean_rows = {
        str(row["group_key"]): row
        for row in payload.get("rows", [])
        if int(row.get("fold", -1)) == 4 and row.get("corruption") == "clean"
    }
    if len(holdout_rows) != EXPECTED_HOLDOUT_ROWS or len(clean_rows) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError("Loop206 holdout clean parity row count mismatch")
    cache = np.memmap(data_path, mode="r", dtype=np.uint8, shape=(count, *shape))
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
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_data_sha256": sha256_file(data_path),
    }


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    content = dict(payload)
    content["content_sha256"] = _canonical_hash(content)
    encoded = (
        json.dumps(content, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("Loop206 prior output/receipt is immutable and already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    staged = output_path.with_name(output_path.name + ".parity-tmp")
    staged.unlink(missing_ok=True)
    base_receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "created_at": _utc_now(),
        "dataset_index_sha256": sha256_file(dataset_index),
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
    }
    try:
        fit_rows, holdout_rows, _ = load_dataset_index(
            dataset_index, dataset_roots=dataset_roots
        )
        prior = fit_deployment_prior(
            fit_rows,
            n_jobs=n_jobs,
            manifest_sha256=base_receipt["dataset_index_sha256"],
            frozen_config=frozen_config,
            parity_passed=False,
        )
        save_prior(prior, staged)
        parity = verify_holdout_parity(prior, holdout_rows, candidate_manifest)
        if not parity["parity_passed"]:
            _write_receipt(
                receipt_path,
                {
                    **base_receipt,
                    "status": "failed",
                    "fit_groups": len(fit_rows),
                    "fit_group_sha256": prior.fit_group_sha256,
                    "selected_threshold": prior.selected_threshold,
                    "sklearn_version": prior.sklearn_version,
                    "code_hashes": prior.code_hashes,
                    "parity": parity,
                },
            )
            raise RuntimeError(
                "Loop206 holdout parity failed: "
                f"{parity['contour_byte_matches']}/{parity['expected']} contours"
            )
        prior = replace(prior, parity_passed=True)
        save_prior(prior, staged)
        artifact_sha256 = sha256_file(staged)
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
        _write_receipt(receipt_path, passed_receipt)
        os.replace(staged, output_path)
        return passed_receipt
    except Exception as exc:
        staged.unlink(missing_ok=True)
        if not receipt_path.exists():
            _write_receipt(
                receipt_path,
                {
                    **base_receipt,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
