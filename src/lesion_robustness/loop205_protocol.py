"""Train-only Loop205 regional-saliency Phase-0 core.

This module is paper-inspired rather than an exact mDRFI/RBCS reproduction. It
fits only a classical region regressor and deliberately contains no student or
segmentation-model training entry point.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from lesion_robustness import loop204_protocol
from lesion_robustness.metrics import segmentation_metrics
from lesion_robustness.preprocessing import saliency_proxy


SEED = 205
CORRUPTIONS = ("clean", "low_contrast", "gaussian_noise")
FIXED_THRESHOLD = 0.5
CALIBRATION_FALLBACK_RATE_MAX = 0.20
THRESHOLD_GRID = (
    0.01,
    0.02,
    0.03,
    0.04,
    *tuple(float(value) for value in np.arange(0.05, 0.5001, 0.025)),
)
PROTECTED_SPLITS = frozenset(
    {
        "val",
        "validation",
        "test",
        "test_v3",
        "ph2",
        "external",
        "external_audit",
        "screen",
    }
)
GROUP_FIELDS = loop204_protocol.GROUP_FIELDS
SAMPLE_ID_FIELDS = loop204_protocol.SAMPLE_ID_FIELDS
CANDIDATE_MAP_NAME = "regional_saliency"
BASELINE_MAP_NAME = "saliency_proxy"
HARD_GATES = {
    "coverage_min_each_corruption": 0.97,
    "overall_auroc_delta_min": 0.04,
    "per_corruption_auroc_delta_min": 0.02,
    "bootstrap_auroc_delta_lower_strict_min": 0.0,
    "overall_auprc_delta_min": 0.03,
    "gaussian_noise_auprc_delta_min": 0.02,
    "support_area_ratio_max_overall": 0.60,
    "support_area_ratio_max_each_corruption": 0.65,
    "fallback_rate_max_each_corruption": 0.35,
    "clean_to_noise_auroc_degradation_max": 0.06,
    "boundary_f1_delta_min": 0.025,
    "dice_delta_min": 0.015,
    "invalid_border_rate_max": 0.05,
    "runtime_median_seconds_max": 1.5,
    "runtime_p95_seconds_max": 3.0,
}


class Loop205ProtocolError(ValueError):
    """Raised when a Loop205 safety, input, or metric invariant is violated."""


@dataclass(frozen=True)
class Loop205Config:
    """Frozen, practical Phase-0 SLIC and random-forest settings."""

    slic_scales: tuple[int, ...] = (64, 128)
    slic_compactness: float = 12.0
    slic_sigma: float = 1.0
    pseudo_background_border_fraction: float = 0.08
    pseudo_background_sigma: float = 0.25
    regional_contrast_spatial_sigma: float = 0.35
    rf_n_estimators: int = 96
    rf_max_depth: int | None = 12
    rf_min_samples_leaf: int = 3
    rf_max_features: float = 0.75
    rf_max_samples: float = 0.65
    rf_n_jobs_max: int = 4
    boundary_tolerance: int = 2


FEATURE_NAMES = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "lab_mean_l",
    "lab_mean_a",
    "lab_mean_b",
    "lab_std_l",
    "lab_std_a",
    "lab_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "hsv_std_h",
    "hsv_std_s",
    "hsv_std_v",
    "gray_mean",
    "gray_std",
    "gradient_mean",
    "gradient_std",
    "area_fraction",
    "centroid_y",
    "centroid_x",
    "border_distance",
    "boundary_connectivity",
    "regional_lab_contrast",
    "pseudo_background_similarity",
    "pseudo_background_distance",
)


@dataclass(frozen=True)
class RegionFeatureBatch:
    """Concatenated region features plus the label maps needed for densification."""

    matrix: np.ndarray
    feature_names: tuple[str, ...]
    label_maps: tuple[np.ndarray, ...]
    region_counts: tuple[int, ...]


def _text(value: object) -> str:
    return str(value or "").strip()


def _stable_digest(seed: int, *values: object) -> bytes:
    payload = "::".join([str(int(seed)), *(_text(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _sample_id(row: Mapping[str, object]) -> str:
    for field in SAMPLE_ID_FIELDS:
        value = _text(row.get(field))
        if value:
            return value
    image_path = _text(row.get("image_path"))
    if image_path:
        return image_path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    raise Loop205ProtocolError("manifest row has no stable sample identifier")


def _normalized_source(value: object) -> str:
    return "_".join(
        token
        for token in _text(value).lower().replace("-", "_").replace("/", "_").split("_")
        if token
    )


def _reject_protected_source(row: Mapping[str, object]) -> None:
    for field in ("split", "source_split", "dataset", "source_dataset", "cohort"):
        value = _normalized_source(row.get(field))
        tokens = set(value.split("_")) if value else set()
        if value in PROTECTED_SPLITS or "ph2" in tokens:
            raise Loop205ProtocolError(
                f"Loop205 train-only protocol rejects protected source {field}={value!r}"
            )


def validate_train_only_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Reject non-train lineage before inspecting or resolving any data path."""

    if not rows:
        raise Loop205ProtocolError("Loop205 train-only protocol received no rows")

    # The complete lineage check precedes all image_path/mask_path handling.
    for row in rows:
        split = _normalized_source(row.get("split"))
        if split != "train":
            raise Loop205ProtocolError(f"Loop205 rejects non-train split {split!r}")
        source_split = _normalized_source(row.get("source_split"))
        if source_split and source_split != "train":
            raise Loop205ProtocolError(
                f"Loop205 rejects non-train source_split {source_split!r}"
            )
        _reject_protected_source(row)

    seen_ids: set[str] = set()
    for row in rows:
        sample_id = _sample_id(row)
        if sample_id in seen_ids:
            raise Loop205ProtocolError(f"duplicate Loop205 sample identifier: {sample_id}")
        seen_ids.add(sample_id)
        for field in ("image_path", "mask_path"):
            if not _text(row.get(field)):
                raise Loop205ProtocolError(f"Loop205 row {sample_id!r} is missing {field}")
        if not any(_text(row.get(field)) for field in GROUP_FIELDS):
            raise Loop205ProtocolError(
                f"Loop205 row {sample_id!r} has no patient/group identity"
            )


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _group_components(rows: Sequence[Mapping[str, object]]) -> list[list[int]]:
    disjoint = _DisjointSet(len(rows))
    owners: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for field in GROUP_FIELDS:
            value = _text(row.get(field))
            if not value:
                continue
            key = (field, value)
            previous = owners.get(key)
            if previous is None:
                owners[key] = index
            else:
                disjoint.union(index, previous)

    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(disjoint.find(index), []).append(index)
    return list(components.values())


def assign_group_folds(
    rows: Sequence[Mapping[str, object]],
    *,
    n_folds: int = 5,
    seed: int = SEED,
) -> dict[str, int]:
    """Assign linked groups to deterministic, approximately balanced folds."""

    if int(n_folds) != 5:
        raise Loop205ProtocolError("Loop205 Phase-0 requires exactly five folds")
    validate_train_only_rows(rows)
    components = _group_components(rows)
    ranked_components: list[tuple[int, bytes, tuple[str, ...], list[int]]] = []
    for component in components:
        member_ids = tuple(sorted(_sample_id(rows[index]) for index in component))
        ranked_components.append(
            (-len(component), _stable_digest(seed, "group", *member_ids), member_ids, component)
        )
    ranked_components.sort(key=lambda item: (item[0], item[1], item[2]))

    fold_sizes = [0] * int(n_folds)
    assignments: dict[str, int] = {}
    for _, digest, _, component in ranked_components:
        fold = min(
            range(int(n_folds)),
            key=lambda candidate: (
                fold_sizes[candidate],
                _stable_digest(seed, "fold", digest.hex(), candidate),
                candidate,
            ),
        )
        for index in component:
            assignments[_sample_id(rows[index])] = fold
        fold_sizes[fold] += len(component)
    return assignments


def build_corruptions(
    image: np.ndarray, sample_id: str, *, seed: int = SEED
) -> dict[str, np.ndarray]:
    """Reuse Loop204's deterministic corruption implementation with seed 205."""

    try:
        return loop204_protocol.build_corruptions(image, sample_id, seed=seed)
    except loop204_protocol.Loop204ProtocolError as exc:
        raise Loop205ProtocolError(str(exc)) from exc


def _validate_config(config: Loop205Config) -> None:
    if not config.slic_scales or any(int(scale) < 2 for scale in config.slic_scales):
        raise Loop205ProtocolError("SLIC scales must contain region counts >= 2")
    if config.slic_compactness <= 0.0 or config.slic_sigma < 0.0:
        raise Loop205ProtocolError("SLIC compactness/sigma are invalid")
    if not 0.0 < config.pseudo_background_border_fraction <= 0.5:
        raise Loop205ProtocolError("pseudo-background border fraction must be in (0, 0.5]")
    if config.pseudo_background_sigma <= 0.0:
        raise Loop205ProtocolError("pseudo-background sigma must be positive")
    if config.regional_contrast_spatial_sigma <= 0.0:
        raise Loop205ProtocolError("regional contrast spatial sigma must be positive")
    if config.rf_n_estimators < 1 or config.rf_min_samples_leaf < 1:
        raise Loop205ProtocolError("random-forest size parameters must be positive")
    if (
        not 0.0 < config.rf_max_features <= 1.0
        or not 0.0 < config.rf_max_samples <= 1.0
        or config.rf_n_jobs_max < 1
    ):
        raise Loop205ProtocolError("random-forest feature/job parameters are invalid")
    if config.boundary_tolerance < 0:
        raise Loop205ProtocolError("boundary tolerance must be non-negative")


def _slic_labels(image: np.ndarray, n_segments: int, config: Loop205Config) -> np.ndarray:
    try:
        from skimage.segmentation import slic
    except ImportError as exc:  # pragma: no cover - depends on the runtime environment
        raise Loop205ProtocolError(
            "Loop205 SLIC feature extraction requires scikit-image"
        ) from exc

    labels = slic(
        image.astype(np.float32) / 255.0,
        n_segments=min(int(n_segments), int(image.shape[0] * image.shape[1])),
        compactness=float(config.slic_compactness),
        sigma=float(config.slic_sigma),
        start_label=0,
        channel_axis=-1,
        convert2lab=True,
        enforce_connectivity=True,
    )
    _, contiguous = np.unique(np.asarray(labels), return_inverse=True)
    return contiguous.reshape(image.shape[:2]).astype(np.int32, copy=False)


def _image_descriptors(image: np.ndarray, config: Loop205Config) -> dict[str, np.ndarray]:
    rgb = image.astype(np.float32) / 255.0
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    hsv_raw = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv = np.empty_like(hsv_raw, dtype=np.float32)
    hsv[..., 0] = hsv_raw[..., 0] / 179.0
    hsv[..., 1:] = hsv_raw[..., 1:] / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gradient_x, gradient_y).astype(np.float32)
    gradient_max = float(gradient.max())
    if gradient_max > 1e-8:
        gradient /= gradient_max

    height, width = image.shape[:2]
    band_width = max(
        1,
        int(round(min(height, width) * config.pseudo_background_border_fraction)),
    )
    border_band = np.zeros((height, width), dtype=bool)
    border_band[:band_width, :] = True
    border_band[-band_width:, :] = True
    border_band[:, :band_width] = True
    border_band[:, -band_width:] = True
    image_border = np.zeros((height, width), dtype=bool)
    image_border[[0, -1], :] = True
    image_border[:, [0, -1]] = True
    return {
        "rgb": rgb,
        "lab": lab,
        "hsv": hsv,
        "gray": gray,
        "gradient": gradient,
        "pseudo_background_lab": lab[border_band].mean(axis=0),
        "image_border": image_border,
    }


def _features_for_scale(
    labels: np.ndarray,
    descriptors: Mapping[str, np.ndarray],
    config: Loop205Config,
) -> np.ndarray:
    height, width = labels.shape
    count = int(labels.max()) + 1
    rgb = descriptors["rgb"]
    lab = descriptors["lab"]
    hsv = descriptors["hsv"]
    gray = descriptors["gray"]
    gradient = descriptors["gradient"]
    image_border = descriptors["image_border"].astype(bool, copy=False)
    pseudo_background_lab = descriptors["pseudo_background_lab"]

    areas = np.empty(count, dtype=np.float64)
    centroids = np.empty((count, 2), dtype=np.float64)
    lab_means = np.empty((count, 3), dtype=np.float64)
    rows: list[list[float]] = []
    for region in range(count):
        region_mask = labels == region
        ys, xs = np.where(region_mask)
        area_pixels = int(ys.size)
        if area_pixels == 0:
            raise Loop205ProtocolError("SLIC produced an empty region")
        area_fraction = area_pixels / float(height * width)
        centroid_y = (float(ys.mean()) + 0.5) / height
        centroid_x = (float(xs.mean()) + 0.5) / width
        border_distance = min(
            centroid_y,
            centroid_x,
            1.0 - centroid_y,
            1.0 - centroid_x,
        ) * 2.0
        boundary_connectivity = float(np.count_nonzero(region_mask & image_border)) / np.sqrt(
            area_pixels
        )
        rgb_values = rgb[region_mask]
        lab_values = lab[region_mask]
        hsv_values = hsv[region_mask]
        gray_values = gray[region_mask]
        gradient_values = gradient[region_mask]

        areas[region] = area_fraction
        centroids[region] = (centroid_y, centroid_x)
        lab_means[region] = lab_values.mean(axis=0)
        rows.append(
            [
                *rgb_values.mean(axis=0),
                *rgb_values.std(axis=0),
                *lab_means[region],
                *lab_values.std(axis=0),
                *hsv_values.mean(axis=0),
                *hsv_values.std(axis=0),
                float(gray_values.mean()),
                float(gray_values.std()),
                float(gradient_values.mean()),
                float(gradient_values.std()),
                area_fraction,
                centroid_y,
                centroid_x,
                border_distance,
                boundary_connectivity,
            ]
        )

    area_weights = areas / areas.sum()
    spatial_sigma = float(config.regional_contrast_spatial_sigma)
    background_sigma = float(config.pseudo_background_sigma)
    for region, row in enumerate(rows):
        color_distance = np.linalg.norm(lab_means - lab_means[region], axis=1) / np.sqrt(3.0)
        spatial_distance = np.linalg.norm(centroids - centroids[region], axis=1)
        weights = area_weights * np.exp(
            -(spatial_distance**2) / (2.0 * spatial_sigma**2)
        )
        weights[region] = 0.0
        regional_contrast = (
            0.0
            if float(weights.sum()) <= 1e-12
            else float(np.dot(weights, color_distance) / weights.sum())
        )
        background_distance = float(
            np.linalg.norm(lab_means[region] - pseudo_background_lab) / np.sqrt(3.0)
        )
        background_similarity = float(
            np.exp(-(background_distance**2) / (2.0 * background_sigma**2))
        )
        row.extend((regional_contrast, background_similarity, background_distance))

    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape != (count, len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise Loop205ProtocolError("regional feature matrix is malformed or non-finite")
    return matrix


def extract_region_features(
    image: np.ndarray, *, config: Loop205Config = Loop205Config()
) -> RegionFeatureBatch:
    """Extract multi-scale image-only regional features from RGB uint8 input."""

    _validate_config(config)
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise Loop205ProtocolError("Loop205 images must be RGB uint8")
    descriptors = _image_descriptors(rgb, config)
    label_maps: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    region_counts: list[int] = []
    for scale in config.slic_scales:
        labels = _slic_labels(rgb, int(scale), config)
        matrix = _features_for_scale(labels, descriptors, config)
        label_maps.append(labels)
        matrices.append(matrix)
        region_counts.append(matrix.shape[0])
    combined = np.concatenate(matrices, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(combined).all():
        raise Loop205ProtocolError("regional feature matrix contains non-finite values")
    return RegionFeatureBatch(
        matrix=combined,
        feature_names=FEATURE_NAMES,
        label_maps=tuple(label_maps),
        region_counts=tuple(region_counts),
    )


def compute_region_targets(batch: RegionFeatureBatch, mask: np.ndarray) -> np.ndarray:
    """Use a train mask only to assign soft lesion fractions to training regions."""

    target = np.asarray(mask) > 0
    if target.ndim != 2 or not batch.label_maps or target.shape != batch.label_maps[0].shape:
        raise Loop205ProtocolError("region target mask shape does not match the image")
    values: list[float] = []
    for labels, count in zip(batch.label_maps, batch.region_counts, strict=True):
        if labels.shape != target.shape:
            raise Loop205ProtocolError("multi-scale label maps have inconsistent shapes")
        for region in range(count):
            region_mask = labels == region
            values.append(float(target[region_mask].mean()))
    targets = np.asarray(values, dtype=np.float32)
    if targets.shape != (batch.matrix.shape[0],) or not np.isfinite(targets).all():
        raise Loop205ProtocolError("region targets are malformed or non-finite")
    return targets


def build_region_records(
    image: np.ndarray,
    train_mask: np.ndarray,
    *,
    config: Loop205Config = Loop205Config(),
) -> tuple[np.ndarray, np.ndarray]:
    """Build one image's feature/target records without mixing GT into features."""

    batch = extract_region_features(image, config=config)
    return batch.matrix, compute_region_targets(batch, train_mask)


def fit_region_forest(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    config: Loop205Config = Loop205Config(),
    n_jobs_cap: int = 1,
) -> Any:
    """Fit the deterministic classical region regressor for one OOF training fold."""

    _validate_config(config)
    matrix = np.asarray(features, dtype=np.float32)
    response = np.asarray(targets, dtype=np.float32).reshape(-1)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(FEATURE_NAMES)
        or matrix.shape[0] != response.size
        or matrix.shape[0] < 2
        or not np.isfinite(matrix).all()
        or not np.isfinite(response).all()
    ):
        raise Loop205ProtocolError("random-forest training records are malformed")
    if np.any((response < 0.0) | (response > 1.0)):
        raise Loop205ProtocolError("region targets must be in [0, 1]")
    if int(n_jobs_cap) < 1:
        raise Loop205ProtocolError("n_jobs_cap must be positive")
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:  # pragma: no cover - depends on the runtime environment
        raise Loop205ProtocolError(
            "Loop205 region regression requires scikit-learn"
        ) from exc

    n_jobs = min(int(n_jobs_cap), int(config.rf_n_jobs_max))
    regressor = RandomForestRegressor(
        n_estimators=int(config.rf_n_estimators),
        max_depth=config.rf_max_depth,
        min_samples_leaf=int(config.rf_min_samples_leaf),
        max_features=float(config.rf_max_features),
        max_samples=float(config.rf_max_samples),
        oob_score=True,
        random_state=SEED,
        n_jobs=n_jobs,
    )
    regressor.fit(matrix, response)
    return regressor


def dense_map_from_region_scores(
    batch: RegionFeatureBatch, region_scores: np.ndarray
) -> np.ndarray:
    """Average multi-scale region scores into a dense probability map."""

    predicted = np.asarray(region_scores, dtype=np.float32).reshape(-1)
    if predicted.shape != (batch.matrix.shape[0],) or not np.isfinite(predicted).all():
        raise Loop205ProtocolError("regional scores are malformed or non-finite")
    predicted = np.clip(predicted, 0.0, 1.0)
    scale_maps: list[np.ndarray] = []
    offset = 0
    for labels, count in zip(batch.label_maps, batch.region_counts, strict=True):
        region_scores = predicted[offset : offset + count]
        scale_maps.append(region_scores[labels])
        offset += count
    dense = np.mean(np.stack(scale_maps, axis=0), axis=0, dtype=np.float32)
    return np.clip(dense, 0.0, 1.0).astype(np.float32, copy=False)


def predict_saliency_map(
    regressor: Any,
    image: np.ndarray,
    *,
    config: Loop205Config = Loop205Config(),
) -> np.ndarray:
    """Predict and average regional scores into a dense held-out saliency map."""

    batch = extract_region_features(image, config=config)
    return dense_map_from_region_scores(batch, regressor.predict(batch.matrix))


def select_train_only_threshold(
    cases: Sequence[tuple[np.ndarray, np.ndarray, str]],
    *,
    threshold_grid: Sequence[float] = THRESHOLD_GRID,
    boundary_tolerance: int = 2,
) -> dict[str, Any]:
    """Select a support threshold from outer-train OOB maps only."""

    if not cases:
        raise Loop205ProtocolError("threshold calibration requires OOB train cases")
    if {str(case[2]) for case in cases} != set(CORRUPTIONS):
        raise Loop205ProtocolError("threshold calibration requires all matched corruptions")
    candidates: list[dict[str, Any]] = []
    for threshold in threshold_grid:
        value = float(threshold)
        if not 0.0 < value < 1.0:
            raise Loop205ProtocolError("threshold grid values must be in (0, 1)")
        by_corruption: dict[str, list[dict[str, float]]] = {name: [] for name in CORRUPTIONS}
        for probability, mask, corruption in cases:
            target = np.asarray(mask) > 0
            support = np.asarray(probability, dtype=np.float32) >= value
            if support.shape != target.shape or not np.isfinite(probability).all():
                raise Loop205ProtocolError("OOB threshold case is malformed")
            overlap = segmentation_metrics(
                support,
                target,
                include_boundary=True,
                boundary_tolerance=boundary_tolerance,
                empty_boundary_distance_policy="image_diagonal",
            )
            by_corruption[str(corruption)].append(
                {
                    "coverage": float(support[target].mean()),
                    "support": float(support.mean()),
                    "dice": float(overlap["dice"]),
                    "boundary_f1": float(overlap["boundary_f1"]),
                }
            )
        per_corruption = {}
        for name, rows in by_corruption.items():
            per_corruption[name] = {
                key: float(np.median([row[key] for row in rows]))
                for key in ("coverage", "support", "dice", "boundary_f1")
            }
            per_corruption[name]["fallback_rate"] = float(
                np.mean(
                    [
                        row["coverage"] < HARD_GATES["coverage_min_each_corruption"]
                        or row["support"]
                        > HARD_GATES["support_area_ratio_max_each_corruption"]
                        for row in rows
                    ]
                )
            )
        min_coverage = min(item["coverage"] for item in per_corruption.values())
        max_support = max(item["support"] for item in per_corruption.values())
        max_fallback_rate = max(
            item["fallback_rate"] for item in per_corruption.values()
        )
        median_dice = float(
            np.median([row["dice"] for rows in by_corruption.values() for row in rows])
        )
        median_bf1 = float(
            np.median(
                [row["boundary_f1"] for rows in by_corruption.values() for row in rows]
            )
        )
        feasible = (
            min_coverage >= HARD_GATES["coverage_min_each_corruption"]
            and max_support <= HARD_GATES["support_area_ratio_max_each_corruption"]
            and max_fallback_rate <= CALIBRATION_FALLBACK_RATE_MAX
        )
        candidates.append(
            {
                "threshold": value,
                "feasible": feasible,
                "min_coverage": min_coverage,
                "max_support": max_support,
                "max_fallback_rate": max_fallback_rate,
                "median_dice": median_dice,
                "median_boundary_f1": median_bf1,
                "per_corruption": per_corruption,
            }
        )
    feasible = [item for item in candidates if item["feasible"]]
    if feasible:
        selected = max(
            feasible,
            key=lambda item: (
                item["median_boundary_f1"],
                item["median_dice"],
                -item["max_fallback_rate"],
                -item["max_support"],
                item["threshold"],
            ),
        )
    else:
        # Fail-closed calibration prioritizes the preregistered coverage gate.
        selected = max(
            candidates,
            key=lambda item: (
                -item["max_fallback_rate"],
                item["min_coverage"],
                -max(
                    0.0,
                    item["max_support"]
                    - HARD_GATES["support_area_ratio_max_each_corruption"],
                ),
                item["median_boundary_f1"],
                item["median_dice"],
                -item["threshold"],
            ),
        )
    return {
        "selected": selected,
        "feasible_threshold_found": bool(feasible),
        "candidate_count": len(candidates),
        "grid": [float(value) for value in threshold_grid],
    }


def _normalize_map(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise Loop205ProtocolError("saliency maps must be finite HxW arrays")
    low, high = float(value.min()), float(value.max())
    if high - low <= 1e-8:
        return np.zeros_like(value, dtype=np.float32)
    return ((value - low) / (high - low)).astype(np.float32, copy=False)


def pixel_auroc(target: np.ndarray, score: np.ndarray) -> float:
    """Reuse Loop204's tie-aware pixel AUROC with Loop205 errors."""

    try:
        return loop204_protocol.pixel_auroc(target, score)
    except loop204_protocol.Loop204ProtocolError as exc:
        raise Loop205ProtocolError(str(exc)) from exc


def pixel_auprc(target: np.ndarray, score: np.ndarray) -> float:
    """Compute tie-aware pixel average precision (area under the PR step curve)."""

    labels = np.asarray(target, dtype=bool).reshape(-1)
    scores = np.asarray(score, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or labels.size == 0 or not np.isfinite(scores).all():
        raise Loop205ProtocolError("AUPRC target/score mismatch or non-finite score")
    positives = int(labels.sum())
    if positives == 0:
        raise Loop205ProtocolError("pixel AUPRC requires lesion pixels")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    true_positives = np.cumsum(sorted_labels, dtype=np.float64)
    false_positives = np.cumsum(~sorted_labels, dtype=np.float64)
    threshold_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    precision = true_positives[threshold_ends] / (
        true_positives[threshold_ends] + false_positives[threshold_ends]
    )
    recall = true_positives[threshold_ends] / positives
    recall_change = np.diff(np.r_[0.0, recall])
    return float(np.dot(recall_change, precision))


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    border = np.zeros(shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    return border


def evaluate_maps(
    mask: np.ndarray,
    candidate: np.ndarray,
    old_proxy: np.ndarray,
    *,
    fallback_used: bool = False,
    invalid_border: bool | None = None,
    generation_seconds: float = 0.0,
    threshold: float = FIXED_THRESHOLD,
    config: Loop205Config = Loop205Config(),
) -> dict[str, Any]:
    """Compare a held-out regional map with the existing saliency proxy."""

    _validate_config(config)
    target = np.asarray(mask) > 0
    candidate_map = np.asarray(candidate, dtype=np.float32)
    proxy_map = _normalize_map(old_proxy)
    if (
        target.ndim != 2
        or candidate_map.shape != target.shape
        or proxy_map.shape != target.shape
        or not np.isfinite(candidate_map).all()
    ):
        raise Loop205ProtocolError("metric mask and saliency maps must be finite and shape-matched")
    seconds = float(generation_seconds)
    if not np.isfinite(seconds) or seconds < 0.0:
        raise Loop205ProtocolError("generation_seconds must be finite and non-negative")
    candidate_map = np.clip(candidate_map, 0.0, 1.0)
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise Loop205ProtocolError("evaluation threshold must be in (0, 1)")
    candidate_support = candidate_map >= threshold
    proxy_support = proxy_map >= FIXED_THRESHOLD
    if not target.any() or target.all():
        raise Loop205ProtocolError("metrics require lesion and background pixels")

    candidate_overlap = segmentation_metrics(
        candidate_support,
        target,
        include_boundary=True,
        boundary_tolerance=config.boundary_tolerance,
        empty_boundary_distance_policy="image_diagonal",
    )
    proxy_overlap = segmentation_metrics(
        proxy_support,
        target,
        include_boundary=True,
        boundary_tolerance=config.boundary_tolerance,
        empty_boundary_distance_policy="image_diagonal",
    )
    candidate_metrics = {
        "pixel_auroc": pixel_auroc(target, candidate_map),
        "pixel_auprc": pixel_auprc(target, candidate_map),
        "dice": float(candidate_overlap["dice"]),
        "boundary_f1": float(candidate_overlap["boundary_f1"]),
    }
    proxy_metrics = {
        "pixel_auroc": pixel_auroc(target, proxy_map),
        "pixel_auprc": pixel_auprc(target, proxy_map),
        "dice": float(proxy_overlap["dice"]),
        "boundary_f1": float(proxy_overlap["boundary_f1"]),
    }
    border_support_fraction = float(candidate_support[_border_mask(target.shape)].mean())
    if invalid_border is None:
        invalid_border = border_support_fraction > 0.5
    support_area_ratio = float(candidate_support.mean())
    lesion_coverage = float(candidate_support[target].mean())
    derived_fallback = (
        not candidate_support.any()
        or lesion_coverage < HARD_GATES["coverage_min_each_corruption"]
        or support_area_ratio > HARD_GATES["support_area_ratio_max_each_corruption"]
    )
    return {
        "threshold": threshold,
        "metric_shape": [int(target.shape[0]), int(target.shape[1])],
        "maps": {
            BASELINE_MAP_NAME: proxy_metrics,
            CANDIDATE_MAP_NAME: candidate_metrics,
        },
        "auroc_delta": candidate_metrics["pixel_auroc"] - proxy_metrics["pixel_auroc"],
        "auprc_delta": candidate_metrics["pixel_auprc"] - proxy_metrics["pixel_auprc"],
        "dice_delta": candidate_metrics["dice"] - proxy_metrics["dice"],
        "boundary_f1_delta": candidate_metrics["boundary_f1"]
        - proxy_metrics["boundary_f1"],
        "lesion_coverage": lesion_coverage,
        "support_area_ratio": support_area_ratio,
        "fallback_used": bool(fallback_used or derived_fallback),
        "invalid_border": bool(invalid_border),
        "invalid_border_fraction": border_support_fraction,
        "generation_seconds": seconds,
    }


def evaluate_image(
    image: np.ndarray,
    mask: np.ndarray,
    candidate: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper that obtains the comparison proxy from the RGB image."""

    return evaluate_maps(mask, candidate, saliency_proxy(image), **kwargs)


def bootstrap_paired_auroc_delta(
    old_auroc: Sequence[float],
    candidate_auroc: Sequence[float],
    *,
    seed: int = SEED,
    iterations: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Reuse Loop204's paired bootstrap with Loop205's frozen seed."""

    try:
        return loop204_protocol.bootstrap_paired_auroc_delta(
            old_auroc,
            candidate_auroc,
            seed=seed,
            iterations=iterations,
            confidence=confidence,
        )
    except loop204_protocol.Loop204ProtocolError as exc:
        raise Loop205ProtocolError(str(exc)) from exc


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise Loop205ProtocolError(f"cannot summarize empty/non-finite metric {key}")
    return float(np.median(values))


def _p95(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise Loop205ProtocolError(f"cannot summarize empty/non-finite metric {key}")
    return float(np.quantile(values, 0.95))


def _map_metrics(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    maps = row.get("maps")
    if not isinstance(maps, Mapping) or not isinstance(maps.get(name), Mapping):
        raise Loop205ProtocolError(f"sample metric is missing map metrics for {name}")
    return maps[name]


def summarize_screen(
    sample_metrics: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    """Summarize matched clean/corrupted OOF observations and paired deltas."""

    expected = set(CORRUPTIONS)
    by_sample: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in sample_metrics:
        sample_id = _text(row.get("sample_id"))
        corruption = _text(row.get("corruption"))
        if not sample_id or corruption not in expected:
            raise Loop205ProtocolError("invalid sample metric identity/corruption")
        if corruption in by_sample.setdefault(sample_id, {}):
            raise Loop205ProtocolError(f"duplicate metric for {sample_id}/{corruption}")
        _map_metrics(row, CANDIDATE_MAP_NAME)
        _map_metrics(row, BASELINE_MAP_NAME)
        by_sample[sample_id][corruption] = row
    if len(by_sample) < 2:
        raise Loop205ProtocolError("Loop205 screen requires at least two train groups")
    for sample_id, views in by_sample.items():
        if set(views) != expected:
            raise Loop205ProtocolError(f"sample {sample_id!r} lacks matched corruptions")

    ordered_sample_ids = sorted(by_sample)
    corruption_summary: dict[str, Any] = {}
    for corruption in CORRUPTIONS:
        rows = [by_sample[sample_id][corruption] for sample_id in ordered_sample_ids]
        candidate = [_map_metrics(row, CANDIDATE_MAP_NAME) for row in rows]
        baseline = [_map_metrics(row, BASELINE_MAP_NAME) for row in rows]
        corruption_summary[corruption] = {
            "count": len(rows),
            "median_auroc": _median(candidate, "pixel_auroc"),
            "median_proxy_auroc": _median(baseline, "pixel_auroc"),
            "median_auroc_delta": _median(rows, "auroc_delta"),
            "median_auprc": _median(candidate, "pixel_auprc"),
            "median_proxy_auprc": _median(baseline, "pixel_auprc"),
            "median_auprc_delta": _median(rows, "auprc_delta"),
            "median_dice_delta": _median(rows, "dice_delta"),
            "median_boundary_f1_delta": _median(rows, "boundary_f1_delta"),
            "median_coverage": _median(rows, "lesion_coverage"),
            "median_support_area_ratio": _median(rows, "support_area_ratio"),
            "fallback_rate": float(np.mean([bool(row["fallback_used"]) for row in rows])),
            "invalid_border_rate": float(
                np.mean([bool(row["invalid_border"]) for row in rows])
            ),
            "runtime_median_seconds": _median(rows, "generation_seconds"),
            "runtime_p95_seconds": _p95(rows, "generation_seconds"),
        }

    all_rows = [
        by_sample[sample_id][corruption]
        for sample_id in ordered_sample_ids
        for corruption in CORRUPTIONS
    ]
    paired_old: list[float] = []
    paired_candidate: list[float] = []
    for sample_id in ordered_sample_ids:
        views = by_sample[sample_id]
        paired_old.append(
            float(
                np.mean(
                    [
                        _map_metrics(views[name], BASELINE_MAP_NAME)["pixel_auroc"]
                        for name in CORRUPTIONS
                    ]
                )
            )
        )
        paired_candidate.append(
            float(
                np.mean(
                    [
                        _map_metrics(views[name], CANDIDATE_MAP_NAME)["pixel_auroc"]
                        for name in CORRUPTIONS
                    ]
                )
            )
        )
    bootstrap = bootstrap_paired_auroc_delta(
        paired_old,
        paired_candidate,
        seed=SEED,
        iterations=bootstrap_iterations,
    )
    overall = {
        "sample_count": len(by_sample),
        "metric_row_count": len(all_rows),
        "median_auroc_delta": _median(all_rows, "auroc_delta"),
        "median_auprc_delta": _median(all_rows, "auprc_delta"),
        "median_dice_delta": _median(all_rows, "dice_delta"),
        "median_boundary_f1_delta": _median(all_rows, "boundary_f1_delta"),
        "median_coverage": _median(all_rows, "lesion_coverage"),
        "median_support_area_ratio": _median(all_rows, "support_area_ratio"),
        "fallback_rate": float(np.mean([bool(row["fallback_used"]) for row in all_rows])),
        "invalid_border_rate": float(
            np.mean([bool(row["invalid_border"]) for row in all_rows])
        ),
        "runtime_median_seconds": _median(all_rows, "generation_seconds"),
        "runtime_p95_seconds": _p95(all_rows, "generation_seconds"),
    }
    return {
        "matched_corruptions": list(CORRUPTIONS),
        "overall": overall,
        "corruptions": corruption_summary,
        "paired_auroc_delta_bootstrap": bootstrap,
    }


def evaluate_hard_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered independent-judge Phase-0 thresholds exactly."""

    overall = summary["overall"]
    corruptions = summary["corruptions"]
    bootstrap = summary["paired_auroc_delta_bootstrap"]
    clean_auroc = float(corruptions["clean"]["median_auroc"])
    noise_auroc = float(corruptions["gaussian_noise"]["median_auroc"])
    degradation = clean_auroc - noise_auroc

    results = {
        "coverage_each_corruption": {
            "passed": all(
                float(corruptions[name]["median_coverage"])
                >= HARD_GATES["coverage_min_each_corruption"]
                for name in CORRUPTIONS
            ),
            "observed": {
                name: float(corruptions[name]["median_coverage"]) for name in CORRUPTIONS
            },
            "required": ">= 0.97 each corruption",
        },
        "overall_auroc_delta": {
            "passed": float(overall["median_auroc_delta"])
            >= HARD_GATES["overall_auroc_delta_min"],
            "observed": float(overall["median_auroc_delta"]),
            "required": ">= 0.04",
        },
        "auroc_delta_each_corruption": {
            "passed": all(
                float(corruptions[name]["median_auroc_delta"])
                >= HARD_GATES["per_corruption_auroc_delta_min"]
                for name in CORRUPTIONS
            ),
            "observed": {
                name: float(corruptions[name]["median_auroc_delta"])
                for name in CORRUPTIONS
            },
            "required": ">= 0.02 each corruption",
        },
        "paired_auroc_delta_ci_lower": {
            "passed": float(bootstrap["lower"])
            > HARD_GATES["bootstrap_auroc_delta_lower_strict_min"],
            "observed": float(bootstrap["lower"]),
            "required": "> 0",
        },
        "overall_auprc_delta": {
            "passed": float(overall["median_auprc_delta"])
            >= HARD_GATES["overall_auprc_delta_min"],
            "observed": float(overall["median_auprc_delta"]),
            "required": ">= 0.03",
        },
        "gaussian_noise_auprc_delta": {
            "passed": float(corruptions["gaussian_noise"]["median_auprc_delta"])
            >= HARD_GATES["gaussian_noise_auprc_delta_min"],
            "observed": float(corruptions["gaussian_noise"]["median_auprc_delta"]),
            "required": ">= 0.02",
        },
        "support_area_ratio_overall": {
            "passed": float(overall["median_support_area_ratio"])
            <= HARD_GATES["support_area_ratio_max_overall"],
            "observed": float(overall["median_support_area_ratio"]),
            "required": "<= 0.60",
        },
        "support_area_ratio_each_corruption": {
            "passed": all(
                float(corruptions[name]["median_support_area_ratio"])
                <= HARD_GATES["support_area_ratio_max_each_corruption"]
                for name in CORRUPTIONS
            ),
            "observed": {
                name: float(corruptions[name]["median_support_area_ratio"])
                for name in CORRUPTIONS
            },
            "required": "<= 0.65 each corruption",
        },
        "fallback_each_corruption": {
            "passed": all(
                float(corruptions[name]["fallback_rate"])
                <= HARD_GATES["fallback_rate_max_each_corruption"]
                for name in CORRUPTIONS
            ),
            "observed": {
                name: float(corruptions[name]["fallback_rate"]) for name in CORRUPTIONS
            },
            "required": "<= 0.35 each corruption",
        },
        "clean_to_noise_auroc_degradation": {
            "passed": degradation
            <= HARD_GATES["clean_to_noise_auroc_degradation_max"],
            "observed": degradation,
            "required": "<= 0.06",
        },
        "boundary_f1_delta": {
            "passed": float(overall["median_boundary_f1_delta"])
            >= HARD_GATES["boundary_f1_delta_min"],
            "observed": float(overall["median_boundary_f1_delta"]),
            "required": ">= 0.025",
        },
        "dice_delta": {
            "passed": float(overall["median_dice_delta"])
            >= HARD_GATES["dice_delta_min"],
            "observed": float(overall["median_dice_delta"]),
            "required": ">= 0.015",
        },
        "invalid_border": {
            "passed": float(overall["invalid_border_rate"])
            <= HARD_GATES["invalid_border_rate_max"],
            "observed": float(overall["invalid_border_rate"]),
            "required": "<= 0.05",
        },
        "runtime_median": {
            "passed": float(overall["runtime_median_seconds"])
            <= HARD_GATES["runtime_median_seconds_max"],
            "observed": float(overall["runtime_median_seconds"]),
            "required": "<= 1.5 seconds/image",
        },
        "runtime_p95": {
            "passed": float(overall["runtime_p95_seconds"])
            <= HARD_GATES["runtime_p95_seconds_max"],
            "observed": float(overall["runtime_p95_seconds"]),
            "required": "<= 3.0 seconds/image",
        },
    }
    return {
        "passed": all(bool(result["passed"]) for result in results.values()),
        "results": results,
        "thresholds": dict(HARD_GATES),
    }
