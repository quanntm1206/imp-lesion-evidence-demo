from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


_LAID_IP_MODES = {"full", "matched_zero_control", "shuffled_control"}
_LAID_IP_STATISTICS_FIT_PASSES = ["residual_quantiles", "normalized_moments"]
_LAID_IP_FORBIDDEN_SHUFFLED_SPLITS = {"val", "test", "ph2", "external_audit"}
_CTO_SMP_MODELS = {
    "segformer_mit",
    "smp_unet_mit",
    "mit_unet",
    "segformer_unet",
    "smp",
    "smp_model",
    "segmentation_models_pytorch",
}


def validate_cto_boundary_config(config: dict[str, Any]) -> None:
    model = _mapping(config.get("model"))
    block = _mapping(model.get("cto_boundary"))
    if not block or block.get("enabled", False) is False:
        return
    if block.get("enabled") is not True:
        raise ValueError("model.cto_boundary.enabled must be true or false")
    if str(model.get("name", "")).lower() not in _CTO_SMP_MODELS:
        raise ValueError("model.cto_boundary requires an SMP U-Net style model")
    if int(model.get("in_channels", 3)) != 3:
        raise ValueError("model.cto_boundary requires direct three-channel RGB input")
    if bool(model.get("edge_aux", False)):
        raise ValueError("model.cto_boundary cannot be combined with model.edge_aux")
    if bool(_mapping(model.get("input_adapter")).get("enabled", False)):
        raise ValueError("model.cto_boundary cannot be combined with model.input_adapter")
    if bool(_mapping(model.get("image_gradient_shape_stream")).get("enabled", False)):
        raise ValueError(
            "model.cto_boundary cannot be combined with image_gradient_shape_stream"
        )
    if str(block.get("operator_mode", "")) not in {"learnable", "fixed_sobel"}:
        raise ValueError(
            "model.cto_boundary.operator_mode must be learnable or fixed_sobel"
        )
    _require_int(
        block.get("shallow_feature_index"),
        2,
        "model.cto_boundary.shallow_feature_index",
    )
    _require_int(
        block.get("high_feature_index"),
        5,
        "model.cto_boundary.high_feature_index",
    )
    _require_int(
        block.get("boundary_channels"),
        32,
        "model.cto_boundary.boundary_channels",
    )

    training = _mapping(config.get("training"))
    if training.get("loss") != "edge_aux_bce_focal_tversky_bdou":
        raise ValueError(
            "CTO candidates require training.loss=edge_aux_bce_focal_tversky_bdou"
        )
    if float(training.get("edge_aux_weight", -1.0)) != 0.10:
        raise ValueError("CTO candidates require training.edge_aux_weight=0.10")
    if training.get("edge_target_mode") != "inner_1px":
        raise ValueError("CTO candidates require training.edge_target_mode=inner_1px")
    _require_int(training.get("edge_kernel_size"), 3, "training.edge_kernel_size")
    if training.get("init_checkpoint"):
        if training.get("init_strict") is not False:
            raise ValueError("CTO base warm-start requires training.init_strict=false")
        allowed = set(training.get("init_allowed_missing_prefixes", []))
        required = {"boundary_extractor.", "boundary_injections."}
        if not required.issubset(allowed):
            raise ValueError(
                "CTO warm-start must allow only declared boundary module prefixes"
            )
    for incompatible in (
        "distillation",
        "consistency",
        "boundary_curriculum",
        "fixed_boundary_auxiliary",
    ):
        if bool(_mapping(training.get(incompatible)).get("enabled", False)):
            raise ValueError(f"CTO candidates cannot enable training.{incompatible}")
    monitor = _mapping(training.get("cto_boundary_monitor"))
    if monitor.get("enabled") is not True:
        raise ValueError("CTO candidates require training.cto_boundary_monitor.enabled=true")
    _require_int(monitor.get("check_epoch"), 5, "training.cto_boundary_monitor.check_epoch")
    if float(monitor.get("min_edge_loss_improvement", -1.0)) != 0.05:
        raise ValueError(
            "CTO candidates require min_edge_loss_improvement=0.05"
        )
    if float(monitor.get("min_injection_ratio", -1.0)) != 1e-3:
        raise ValueError("CTO candidates require min_injection_ratio=0.001")


def validate_fade_upsampling_config(config: dict[str, Any]) -> None:
    model = _mapping(config.get("model"))
    block = _mapping(model.get("fade_upsampling"))
    if not block or block.get("enabled", False) is False:
        return
    if block.get("enabled") is not True:
        raise ValueError("model.fade_upsampling.enabled must be true or false")
    if str(model.get("name", "")).lower() not in _CTO_SMP_MODELS:
        raise ValueError("model.fade_upsampling requires an SMP U-Net style model")
    if int(model.get("in_channels", 3)) != 3:
        raise ValueError("model.fade_upsampling requires direct three-channel RGB input")
    if bool(model.get("edge_aux", False)):
        raise ValueError("model.fade_upsampling cannot be combined with model.edge_aux")
    if bool(_mapping(model.get("input_adapter")).get("enabled", False)):
        raise ValueError("model.fade_upsampling cannot be combined with model.input_adapter")
    for incompatible in ("image_gradient_shape_stream", "cto_boundary"):
        if bool(_mapping(model.get(incompatible)).get("enabled", False)):
            raise ValueError(f"model.fade_upsampling cannot be combined with model.{incompatible}")
    if str(block.get("encoder_context_mode", "")) not in {"globalized", "local"}:
        raise ValueError("model.fade_upsampling.encoder_context_mode must be globalized or local")
    if int(block.get("kernel_size", 0)) not in {3, 5}:
        raise ValueError("model.fade_upsampling.kernel_size must be 3 or 5")
    _require_int(
        block.get("compressed_channels"),
        64,
        "model.fade_upsampling.compressed_channels",
    )
    if tuple(block.get("stage_indices", ())) != (0, 1, 2):
        raise ValueError("model.fade_upsampling.stage_indices must be [0, 1, 2]")

    training = _mapping(config.get("training"))
    if training.get("loss") != "bce_focal_tversky_bdou":
        raise ValueError("FADE-Lite candidates require training.loss=bce_focal_tversky_bdou")
    if training.get("init_checkpoint"):
        if training.get("init_strict") is not False:
            raise ValueError("FADE-Lite base warm-start requires training.init_strict=false")
        allowed = set(training.get("init_allowed_missing_prefixes", []))
        if "fade_upsamplers." not in allowed:
            raise ValueError("FADE-Lite warm-start must allow fade_upsamplers. missing keys")
    for incompatible in (
        "distillation",
        "consistency",
        "boundary_curriculum",
        "fixed_boundary_auxiliary",
    ):
        if bool(_mapping(training.get(incompatible)).get("enabled", False)):
            raise ValueError(f"FADE-Lite candidates cannot enable training.{incompatible}")
    monitor = _mapping(training.get("fade_upsampling_monitor"))
    if monitor.get("enabled") is not True:
        raise ValueError("FADE-Lite candidates require training.fade_upsampling_monitor.enabled=true")
    _require_int(monitor.get("check_epoch"), 5, "training.fade_upsampling_monitor.check_epoch")
    _require_int(monitor.get("minimum_active_stages"), 2, "training.fade_upsampling_monitor.minimum_active_stages")
    _require_int(
        monitor.get("minimum_active_detail_stages"),
        1,
        "training.fade_upsampling_monitor.minimum_active_detail_stages",
    )
    for name in ("active_scale_threshold", "minimum_residual_ratio", "minimum_kernel_variance"):
        if float(monitor.get(name, -1.0)) <= 0.0:
            raise ValueError(f"training.fade_upsampling_monitor.{name} must be positive")
    entropy = float(monitor.get("minimum_kernel_entropy", -1.0))
    if not 0.0 < entropy < 1.0:
        raise ValueError("training.fade_upsampling_monitor.minimum_kernel_entropy must be in (0, 1)")


def validate_wmren_inspired_haar_config(config: dict[str, Any]) -> None:
    model = _mapping(config.get("model"))
    block = _mapping(model.get("wmren_inspired_haar"))
    if not block or block.get("enabled", False) is False:
        return
    if block.get("enabled") is not True:
        raise ValueError("model.wmren_inspired_haar.enabled must be true or false")
    if str(model.get("name", "")).lower() not in _CTO_SMP_MODELS:
        raise ValueError("model.wmren_inspired_haar requires an SMP U-Net style model")
    if str(model.get("smp_encoder", "")).lower() != "mit_b3":
        raise ValueError("model.wmren_inspired_haar requires smp_encoder=mit_b3")
    if int(model.get("in_channels", 3)) != 3:
        raise ValueError("model.wmren_inspired_haar requires direct three-channel RGB input")
    if bool(model.get("edge_aux", False)):
        raise ValueError("model.wmren_inspired_haar cannot be combined with model.edge_aux")
    for incompatible in (
        "input_adapter",
        "image_gradient_shape_stream",
        "cto_boundary",
        "fade_upsampling",
    ):
        if bool(_mapping(model.get(incompatible)).get("enabled", False)):
            raise ValueError(
                f"model.wmren_inspired_haar cannot be combined with model.{incompatible}"
            )
    if tuple(block.get("feature_indices", ())) != (2, 3, 4):
        raise ValueError("model.wmren_inspired_haar.feature_indices must be [2, 3, 4]")
    _require_int(block.get("gate_channels"), 32, "model.wmren_inspired_haar.gate_channels")
    if float(block.get("detail_gain", -1.0)) != 0.25:
        raise ValueError("Loop202 requires model.wmren_inspired_haar.detail_gain=0.25")
    if float(block.get("alpha_limit", -1.0)) != 0.25:
        raise ValueError("Loop202 requires model.wmren_inspired_haar.alpha_limit=0.25")
    if block.get("check_finite") is not True:
        raise ValueError("Loop202 requires model.wmren_inspired_haar.check_finite=true")

    training = _mapping(config.get("training"))
    if training.get("loss") != "bce_focal_tversky_bdou":
        raise ValueError(
            "WMREN-inspired candidates require training.loss=bce_focal_tversky_bdou"
        )
    if training.get("init_checkpoint"):
        if training.get("init_strict") is not False:
            raise ValueError("WMREN-inspired warm-start requires training.init_strict=false")
        if set(training.get("init_allowed_missing_prefixes", [])) != {"haar_fusions."}:
            raise ValueError(
                "WMREN-inspired warm-start must allow only haar_fusions. missing keys"
            )
    for incompatible in (
        "distillation",
        "consistency",
        "boundary_curriculum",
        "fixed_boundary_auxiliary",
    ):
        if bool(_mapping(training.get(incompatible)).get("enabled", False)):
            raise ValueError(
                f"WMREN-inspired candidates cannot enable training.{incompatible}"
            )
    monitor = _mapping(training.get("wmren_haar_monitor"))
    if monitor.get("enabled") is not True:
        raise ValueError(
            "WMREN-inspired candidates require training.wmren_haar_monitor.enabled=true"
        )
    _require_int(monitor.get("check_epoch"), 5, "training.wmren_haar_monitor.check_epoch")
    _require_int(
        monitor.get("minimum_active_stages"),
        2,
        "training.wmren_haar_monitor.minimum_active_stages",
    )
    for name in (
        "active_scale_threshold",
        "minimum_region_residual_ratio",
        "minimum_detail_residual_ratio",
    ):
        if float(monitor.get(name, -1.0)) <= 0.0:
            raise ValueError(f"training.wmren_haar_monitor.{name} must be positive")
    entropy = float(monitor.get("minimum_gate_entropy", -1.0))
    if not 0.0 < entropy < 1.0:
        raise ValueError("training.wmren_haar_monitor.minimum_gate_entropy must be in (0, 1)")


def validate_boundary_refinement_config(
    config: dict[str, Any],
    *,
    purpose: str | None = None,
    check_paths: bool = False,
) -> None:
    block = config.get("boundary_refinement", {})
    if not block or not bool(block.get("enabled", False)):
        return
    candidate = str(block.get("candidate", ""))
    expected_channels = {"edge": 5, "structure": 8, "artifact": 9}
    if candidate not in expected_channels:
        raise ValueError("boundary_refinement.candidate must be edge, structure, or artifact")
    in_channels = int(block.get("in_channels", -1))
    if in_channels != expected_channels[candidate]:
        raise ValueError(f"boundary_refinement candidate={candidate} requires in_channels={expected_channels[candidate]}")
    if int(block.get("roi_size", 0)) != 512:
        raise ValueError("boundary_refinement.roi_size must be 512")
    margin_ratio = float(block.get("roi_margin_ratio", -1.0))
    if not 0.0 <= margin_ratio <= 0.5:
        raise ValueError("boundary_refinement.roi_margin_ratio must be in [0, 0.5]")
    if int(block.get("band_width_px", 0)) not in {8, 12}:
        raise ValueError("boundary_refinement.band_width_px must be 8 or 12")
    if int(block.get("base_channels", 0)) < 4:
        raise ValueError("boundary_refinement.base_channels must be at least 4")
    if str(block.get("coarse_fusion", "")) not in {"raw", "boundary_preserving", "logit_mean"}:
        raise ValueError("boundary_refinement.coarse_fusion must be raw, boundary_preserving, or logit_mean")
    if bool(block.get("target_derived_roi", False)):
        raise ValueError("boundary_refinement.target_derived_roi must be false")
    if bool(block.get("inpainting", False)):
        raise ValueError("boundary_refinement.inpainting must be false")
    manifest_path = str(block.get("cache_manifest", "")).strip()
    data_path = str(block.get("cache_data", "")).strip()
    if not manifest_path or not data_path:
        raise ValueError("boundary_refinement cache_manifest and cache_data are required")
    loss = block.get("loss", {})
    for name in ("boundary_weight", "edge_weight", "identity_weight"):
        if float(loss.get(name, -1.0)) < 0.0:
            raise ValueError(f"boundary_refinement.loss.{name} must be non-negative")
    if purpose not in {None, "train", "eval"}:
        raise ValueError("boundary refinement validation purpose must be train or eval")
    if not check_paths:
        return
    manifest = Path(manifest_path)
    data = Path(data_path)
    if not manifest.exists():
        raise FileNotFoundError(f"missing boundary refinement cache manifest: {manifest}")
    if not data.exists():
        raise FileNotFoundError(f"missing boundary refinement cache data: {data}")
    if purpose == "train":
        manifest_data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        if manifest_data.get("split") != "train" or manifest_data.get("purpose") != "training":
            raise ValueError("boundary refinement training cache is train split only")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_int(value: Any, expected: int, field: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{field} must be {expected}")


def validate_laid_ip_config(config: dict[str, Any]) -> None:
    preprocessing = _mapping(config.get("preprocessing"))
    laid_ip = _mapping(preprocessing.get("laid_ip"))
    if not laid_ip or laid_ip.get("enabled", False) is False:
        return
    if laid_ip.get("enabled") is not True:
        raise ValueError("preprocessing.laid_ip.enabled must be true or false")

    mode = str(laid_ip.get("mode", ""))
    if mode not in _LAID_IP_MODES:
        raise ValueError(
            "preprocessing.laid_ip.mode must be full, matched_zero_control, or shuffled_control"
        )

    statistics_fit = _mapping(laid_ip.get("statistics_fit"))
    if statistics_fit.get("passes") != _LAID_IP_STATISTICS_FIT_PASSES:
        raise ValueError(
            "preprocessing.laid_ip.statistics_fit.passes must be "
            "['residual_quantiles', 'normalized_moments']"
        )
    if not str(laid_ip.get("statistics_manifest", "")).strip():
        raise ValueError("preprocessing.laid_ip.statistics_manifest is required")

    model = _mapping(config.get("model"))
    _require_int(model.get("input_channels"), 6, "model.input_channels")
    _require_int(model.get("encoder_in_channels"), 3, "model.encoder_in_channels")

    adapter = model.get("input_adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("model.input_adapter is required for LAID-IP configs")
    if adapter.get("enabled") is not True:
        raise ValueError("model.input_adapter.enabled must be true")
    if adapter.get("initialization") != "rgb_identity_extra_zero":
        raise ValueError("model.input_adapter.initialization must be rgb_identity_extra_zero")

    if mode != "shuffled_control":
        return
    if not str(laid_ip.get("shuffled_mapping_manifest", "")).strip():
        raise ValueError("preprocessing.laid_ip.shuffled_mapping_manifest is required")

    data = _mapping(config.get("data"))
    evaluation = _mapping(config.get("evaluation"))
    split_values = [
        data.get("split"),
        data.get("train_split"),
        data.get("val_split"),
        data.get("test_split"),
        evaluation.get("split"),
        laid_ip.get("split"),
        laid_ip.get("target_split"),
        laid_ip.get("apply_split"),
    ]
    if any(str(split).lower() in _LAID_IP_FORBIDDEN_SHUFFLED_SPLITS for split in split_values if split):
        raise ValueError("shuffled_control is forbidden on val/test/PH2/external_audit")


def validate_packed_extra_channel_config(config: dict[str, Any]) -> None:
    preprocessing = _mapping(config.get("preprocessing"))
    block = _mapping(preprocessing.get("extra_channel"))
    if not block or block.get("enabled", False) is False:
        return
    channel_type = str(block.get("type", "")).strip().lower()
    if channel_type not in {"packed_binary_cache", "loop206_contour_cache"}:
        return
    if block.get("enabled") is not True:
        raise ValueError("preprocessing.extra_channel.enabled must be true")
    if not str(block.get("cache_manifest", "")).strip():
        raise ValueError("preprocessing.extra_channel.cache_manifest is required")
    if block.get("require_input_sha256") is not True:
        raise ValueError("packed extra channel requires exact input SHA-256 verification")
    model = _mapping(config.get("model"))
    _require_int(model.get("input_channels"), 4, "model.input_channels")
    _require_int(model.get("encoder_in_channels"), 3, "model.encoder_in_channels")
    adapter = _mapping(model.get("input_adapter"))
    if adapter.get("enabled") is not True:
        raise ValueError("packed extra channel requires model.input_adapter.enabled=true")
    if adapter.get("initialization") != "rgb_identity_extra_zero":
        raise ValueError("packed extra channel requires rgb_identity_extra_zero initialization")
    data = _mapping(config.get("data"))
    protected = {"val", "test", "test_v3", "ph2", "external_audit"}
    active_splits = {
        str(data.get(name, "")).strip().lower()
        for name in ("train_split", "val_split", "test_split")
        if str(data.get(name, "")).strip()
    }
    if active_splits & protected:
        raise ValueError("packed train-only extra channel config cannot name protected runtime splits")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _resolve_extends_path(config_path: Path, parent: str | Path) -> Path:
    parent_path = Path(parent)
    if parent_path.is_absolute():
        return parent_path
    candidates = [config_path.parent / parent_path]
    candidates.extend(ancestor / parent_path for ancestor in config_path.resolve().parents)
    candidates.append(Path.cwd() / parent_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return config_path.parent / parent_path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = _expand_environment(yaml.safe_load(config_path.read_text()) or {})
    parent = data.get("extends")
    if not parent:
        validate_boundary_refinement_config(data)
        validate_laid_ip_config(data)
        validate_packed_extra_channel_config(data)
        validate_cto_boundary_config(data)
        validate_fade_upsampling_config(data)
        validate_wmren_inspired_haar_config(data)
        return data
    parent_path = _resolve_extends_path(config_path, parent)
    merged = _deep_merge(load_config(parent_path), data)
    validate_boundary_refinement_config(merged)
    validate_laid_ip_config(merged)
    validate_packed_extra_channel_config(merged)
    validate_cto_boundary_config(merged)
    validate_fade_upsampling_config(merged)
    validate_wmren_inspired_haar_config(merged)
    return merged
