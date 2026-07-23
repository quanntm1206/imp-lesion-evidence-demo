"""Fail-closed contract entry point for prospective RQ1-v2 training.

This tracked scaffold validates identities and prerequisites. It intentionally
contains no model-training engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
SEEDS = {206, 1206, 2206}
CONDITIONS = (
    "clean",
    "low_brightness",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
)
MODEL_IDS = {
    "imp": "RQ1v2-IMP-MiT-B3-UNet",
    "nnunet": "RQ1v2-nnUNet-v2-2d",
}
PROTOCOL_IDENTITY = {
    "schema_version": "imp.rq1_v2.protocol.v1",
    "research_question": "Under one Clean-v3 adaptive-validation identity set and original-image metric geometry, how do complete IMP MiT-B3 U-Net and nnU-Net v2 systems trade overlap and boundary quality across independent training runs?",
    "evidence_class": "adaptive_development_validation_rerun",
    "train_count": 2008,
    "validation_count": 431,
    "test_v3_access": False,
    "clean_v3_manifest_sha256": "4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102",
    "seeds": [206, 1206, 2206],
    "conditions": ["clean", "low_brightness", "low_contrast", "gaussian_noise", "gaussian_blur", "jpeg_compression"],
    "metrics": ["dice", "iou", "precision", "recall", "boundary_f1", "hd95", "assd"],
    "boundary_tolerance_original_pixels": 2,
    "probability_restore": "bilinear_to_original_before_threshold",
    "threshold": 0.5,
    "primary_endpoint": "independent_arm_mean_robust_dice_delta_nnunet_minus_imp",
    "secondary_endpoint": "independent_arm_mean_robust_boundary_f1_delta_nnunet_minus_imp",
    "claim_limit": "adaptive validation; no protected-test or statistical-superiority claim",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PROTOCOL_KEYS = set(PROTOCOL_IDENTITY) | {"dataset_index_status", "dataset_index_sha256"}
CONFIG_KEYS = {
    "schema_version", "arm", "model_id", "run_id", "seed", "conditions",
    "geometry", "metric_contract", "training", "private_inputs", "output", "budget",
}


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys instead of overwriting."""


def _construct_unique_mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ContractError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class ContractError(ValueError):
    """A public contract or private prerequisite failed validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ContractError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        payload = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON contract: {path.name}") from exc
    if not isinstance(payload, Mapping):
        raise ContractError(f"JSON contract must be an object: {path.name}")
    return payload


def load_public_contract(protocol_path: Path, config_path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = _load_json(protocol_path)
    try:
        config = yaml.load(config_path.read_text(encoding="ascii"), Loader=_UniqueSafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read YAML config: {config_path.name}") from exc
    if not isinstance(config, Mapping):
        raise ContractError("config must be a YAML object")
    if set(protocol) != PROTOCOL_KEYS:
        raise ContractError("protocol schema fields drift")
    if set(config) != CONFIG_KEYS:
        raise ContractError("config schema fields drift")

    arm = config.get("arm")
    if config.get("schema_version") != "imp.rq1_v2.config.v1" or arm not in MODEL_IDS:
        raise ContractError("config schema or arm is invalid")
    if config.get("model_id") != MODEL_IDS[arm]:
        raise ContractError("prospective model_id drift")
    if config.get("seed") not in SEEDS:
        raise ContractError("config seed drift")
    if config.get("run_id") != f"{arm}_seed{config['seed']}":
        raise ContractError("config run_id drift")
    if config_path.stem != config["run_id"]:
        raise ContractError("config basename must match run_id")
    for key, expected in PROTOCOL_IDENTITY.items():
        if protocol.get(key) != expected:
            raise ContractError(f"protocol {key} drift")
    index_status = protocol.get("dataset_index_status")
    index_sha = protocol.get("dataset_index_sha256")
    if not ((index_status == "unresolved_blocked" and index_sha is None) or
            (index_status == "verified" and isinstance(index_sha, str) and SHA256.fullmatch(index_sha))):
        raise ContractError("protocol dataset index status/digest drift")
    if tuple(config.get("conditions", ())) != CONDITIONS:
        raise ContractError("six-condition contract drift")
    if tuple(protocol.get("conditions", ())) != CONDITIONS:
        raise ContractError("protocol/config condition mismatch")
    protocol_seeds = protocol.get("seeds", ())
    if not isinstance(protocol_seeds, (list, tuple)) or config.get("seed") not in protocol_seeds:
        raise ContractError("protocol/config seed mismatch")

    geometry = config.get("geometry")
    expected_geometry = {
        "input_hw": [384, 384],
        "probability_restore": "bilinear_to_original_before_threshold",
        "metric_geometry": "original_image",
        "threshold": 0.5,
    }
    if geometry != expected_geometry:
        raise ContractError("original-image geometry contract drift")
    if protocol.get("probability_restore") != geometry["probability_restore"] or protocol.get("threshold") != geometry["threshold"]:
        raise ContractError("protocol/config geometry mismatch")

    metric = config.get("metric_contract")
    if not isinstance(metric, Mapping):
        raise ContractError("metric contract is missing")
    if set(metric) != {"source", "source_sha256", "metrics", "boundary_tolerance_original_pixels", "empty_mask_policy"}:
        raise ContractError("metric contract schema drift")
    source = metric.get("source")
    if source != "src/lesion_robustness/research/rq1_metrics.py":
        raise ContractError("metric source binding drift")
    source_path = ROOT / source
    if not source_path.is_file() or metric.get("source_sha256") != _sha256(source_path):
        raise ContractError("metric source SHA-256 drift")
    if metric.get("boundary_tolerance_original_pixels") != 2:
        raise ContractError("boundary tolerance drift")
    if metric.get("empty_mask_policy") != "both_empty_perfect;one_empty_diagonal_penalty":
        raise ContractError("empty-mask policy drift")
    if tuple(metric.get("metrics", ())) != tuple(protocol.get("metrics", ())):
        raise ContractError("metric set mismatch")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ContractError("training controls are missing")
    locked = {
        "epochs": 100,
        "num_workers": 0,
        "amp": True,
        "checkpoint_selection": "final_epoch_only",
        "deterministic": True,
    }
    if any(training.get(key) != value for key, value in locked.items()):
        raise ContractError("training control drift")
    private_inputs = config.get("private_inputs")
    output = config.get("output")
    if not isinstance(private_inputs, Mapping) or not isinstance(output, Mapping):
        raise ContractError("input/output contract is missing")
    if set(private_inputs) != {"dataset_index_env", "experiment_manifest_env", "training_input_env", "training_input_artifact_key"}:
        raise ContractError("private input schema drift")
    if set(output) != {"checkpoint_env", "checkpoint_template", "checkpoint_selection"}:
        raise ContractError("output checkpoint schema drift")
    if private_inputs.get("training_input_env") == output.get("checkpoint_env"):
        raise ContractError("training input and output checkpoint must be separate")
    expected_private = {
        "dataset_index_env": "IMP_CLEAN_V3_INDEX",
        "experiment_manifest_env": "IMP_RQ1_V2_EXPERIMENT_INPUT",
        "training_input_env": "IMP_RQ1_V2_IMP_INITIALIZATION" if arm == "imp" else "IMP_RQ1_V2_NNUNET_INITIALIZATION",
        "training_input_artifact_key": "artifact_file_sha256" if arm == "imp" else "plans_sha256",
    }
    if any(private_inputs.get(key) != expected for key, expected in expected_private.items()):
        raise ContractError("private input environment mapping drift")
    if output.get("checkpoint_env") != "IMP_RQ1_V2_OUTPUT_ROOT":
        raise ContractError("output checkpoint environment mapping drift")
    if output.get("checkpoint_template") != "{run_id}/final.pt":
        raise ContractError("output checkpoint template drift")
    expected_controls = (
        {"model": "segmentation_models_pytorch.Unet", "encoder_name": "timm-mit_b3", "encoder_weights": "imagenet",
         "batch_size": 4, "optimizer": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.0001,
         "scheduler": "cosine", "loss": "bce_focal_tversky_bdou", "augmentation": "clean_v3_imp_recipe_v1",
         "initialization": "imagenet_state_from_experiment_input_manifest"}
        if arm == "imp" else
        {"configuration": "2d", "trainer": "nnUNetTrainer_100epochs", "fold": 0, "batch_size": 2,
         "optimizer": "nnunet_default", "learning_rate": "nnunet_default", "weight_decay": "nnunet_default",
         "scheduler": "nnunet_default", "loss": "nnUNetTrainer_100epochs_default_v2_8_1",
         "augmentation": "nnUNetTrainer_100epochs_default_v2_8_1",
         "initialization": "nnUNetTrainer_100epochs_default_v2_8_1"}
    )
    if any(training.get(key) != expected for key, expected in expected_controls.items()):
        raise ContractError("arm training control drift")
    expected_training_keys = set(locked) | set(expected_controls)
    if set(training) != expected_training_keys:
        raise ContractError("arm training schema drift")
    if output.get("checkpoint_selection") != "final_epoch_only":
        raise ContractError("output checkpoint policy drift")
    budget = config.get("budget")
    expected_budget = {"max_wall_hours": 24, "max_checkpoint_bytes": 2147483648,
                       "max_job_storage_bytes": 25000000000}
    if not isinstance(budget, Mapping) or set(budget) != set(expected_budget) or any(
        budget.get(key) != value for key, value in expected_budget.items()
    ):
        raise ContractError("resource budget drift")
    return protocol, config


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _missing(paths: Mapping[str, Path | None]) -> list[str]:
    return sorted(name for name, path in paths.items() if path is None or not path.is_file())


def _protocol_identity_projection(protocol: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"dataset_index_status", "dataset_index_sha256", "protocol_sha256",
                "condition_contract_sha256", "metric_contract_sha256"}
    return {key: value for key, value in protocol.items() if key not in excluded}


def canonical_checkpoint_relative(config: Mapping[str, Any]) -> Path:
    try:
        relative = Path(config["output"]["checkpoint_template"].format(run_id=config["run_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("checkpoint path contract is invalid") from exc
    if relative.is_absolute() or relative.parts != (config["run_id"], "final.pt"):
        raise ContractError("checkpoint path must be {run_id}/final.pt")
    return relative


def _load_imp_tensor_state(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise ContractError("IMP tensor-state artifact cannot be loaded in the declared hash domain") from exc
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ContractError("IMP tensor-state artifact must contain a nonempty state mapping")
    return payload


def validate_frozen_experiment_trust(
    experiment_path: Path,
    *,
    parent_release: Path,
    data_manifest: Path,
    imp_input_artifact: Path,
    nnunet_input_artifact: Path,
    nnunet_checkpoint: Path,
    protocol: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Validate the frozen prospective trust chain without opening dataset rows."""
    try:
        experiment = _load_json(experiment_path)
    except ContractError as exc:
        return None, str(exc)
    expected_keys = {
        "schema_version", "parent_release_manifest_sha256", "public_protocol_projection", "protocol", "data_report",
        "model_artifacts", "runtimes", "configs",
    }
    if set(experiment) != expected_keys or experiment.get("schema_version") != "imp.rq1_v2.experiment_input.v1":
        return None, "frozen experiment manifest schema drift"
    if experiment.get("parent_release_manifest_sha256") != _sha256(parent_release):
        return None, "parent release manifest binding drift"
    projection = experiment.get("public_protocol_projection")
    if (not isinstance(projection, Mapping) or
            set(projection) != {"schema_version", "identity"} or
            projection.get("schema_version") != "imp.rq1_v2.public_protocol_projection.v1" or
            not isinstance(projection.get("identity"), Mapping)):
        return None, "public protocol projection schema drift"
    try:
        tracked_protocol = _load_json(ROOT / "experiments" / "rq1_v2" / "protocol.json")
    except ContractError as exc:
        return None, str(exc)
    enriched_protocol = experiment.get("protocol")
    scientific_keys = {"protocol_sha256", "condition_contract_sha256", "metric_contract_sha256"}
    if (not isinstance(enriched_protocol, Mapping) or
            set(enriched_protocol) != PROTOCOL_KEYS | scientific_keys or
            any(not SHA256.fullmatch(str(enriched_protocol.get(key, ""))) for key in scientific_keys)):
        return None, "enriched verified protocol schema drift"
    identity = _protocol_identity_projection(tracked_protocol)
    if (projection["identity"] != identity or
            _protocol_identity_projection(enriched_protocol) != identity or
            _protocol_identity_projection(protocol) != identity):
        return None, "public protocol identity projection drift"
    verified_protocol = {key: enriched_protocol[key] for key in PROTOCOL_KEYS}
    if protocol.get("dataset_index_status") == "verified" and dict(protocol) != verified_protocol:
        return None, "caller verified protocol binding drift"
    try:
        from freeze_rq1_v2_artifacts import _reject_private_or_blocked
        _reject_private_or_blocked(experiment)
        _reject_private_or_blocked(_load_json(parent_release))
    except (ImportError, ContractError, ValueError) as exc:
        return None, f"private/path scrub failed: {exc}"
    if (verified_protocol.get("dataset_index_status") != "verified" or
            verified_protocol.get("dataset_index_sha256") != _sha256(data_manifest)):
        return None, "verified data manifest binding missing"

    report = experiment.get("data_report")
    report_keys = {
        "schema_version", "audit_id", "train_count", "validation_count",
        "train_ordered_identity_sha256", "validation_ordered_identity_sha256",
        "cross_split_groups", "cross_split_exact_rgb", "near_duplicate_candidate_count",
        "cross_split_near_rgb", "clean_v3_manifest_sha256", "dataset_index_status",
        "dataset_index_sha256", "test_v3_access", "test_v3_open_count", "algorithms",
        "canonical_report_sha256",
    }
    if not isinstance(report, Mapping) or set(report) != report_keys:
        return None, "verified data report schema drift"
    expected_algorithms = {
        "ordered_identity": "sample_id|group_key|sha256_raw|sha256_rgb\\n sorted by sample_id,group_key; ASCII; SHA-256",
        "group": "exact group_key crossing",
        "exact_rgb": "decoded RGB SHA-256 crossing",
        "near_candidate": "63-bit luminance pHash; Lanczos 32x32; DCT 8x8 excluding DC; Hamming <=4",
        "near_confirmation": "luminance SSIM at Lanczos 256x256; Gaussian 11x11 sigma 1.5; threshold >=0.98",
    }
    hash_report_fields = (
        "train_ordered_identity_sha256", "validation_ordered_identity_sha256",
        "clean_v3_manifest_sha256", "dataset_index_sha256", "canonical_report_sha256",
    )
    integer_report_fields = (
        "train_count", "validation_count", "cross_split_groups", "cross_split_exact_rgb",
        "near_duplicate_candidate_count", "cross_split_near_rgb", "test_v3_open_count",
    )
    if (any(not isinstance(report.get(key), str) or not SHA256.fullmatch(report[key]) for key in hash_report_fields) or
            any(isinstance(report.get(key), bool) or not isinstance(report.get(key), int) or report[key] < 0
                for key in integer_report_fields) or
            report.get("schema_version") != "imp.rq1_v2.data_integrity_report.v1" or
            report.get("audit_id") != "RQ1v2-data-integrity" or report.get("algorithms") != expected_algorithms or
            report.get("dataset_index_status") != "verified" or
            report.get("dataset_index_sha256") != verified_protocol["dataset_index_sha256"] or
            report.get("train_count") != 2008 or report.get("validation_count") != 431 or
            report.get("test_v3_access") is not False or report.get("test_v3_open_count") != 0 or
            report.get("cross_split_groups") != 0 or report.get("cross_split_exact_rgb") != 0 or
            report.get("cross_split_near_rgb") != 0 or
            report.get("clean_v3_manifest_sha256") != PROTOCOL_IDENTITY["clean_v3_manifest_sha256"]):
        return None, "verified data report admission failed"
    try:
        from lesion_robustness.research.rq1_data import canonical_report_sha256
        if report.get("canonical_report_sha256") != canonical_report_sha256(report):
            return None, "verified data report digest drift"
    except (ImportError, TypeError, ValueError):
        return None, "verified data report digest unavailable"

    model_artifacts = experiment.get("model_artifacts")
    if (not isinstance(model_artifacts, Mapping) or
            set(model_artifacts) != {"schema_version", "imp", "nnunet"} or
            model_artifacts.get("schema_version") != "imp.rq1_v2.model_artifacts.v1"):
        return None, "model artifact metadata missing"
    imp = model_artifacts.get("imp")
    nnunet = model_artifacts.get("nnunet")
    if (not isinstance(imp, Mapping) or
            set(imp) != {"imagenet_pretrained_state_sha256", "artifact_file_sha256"}):
        return None, "IMP artifact metadata drift"
    nn_keys = {"checkpoint_sha256", "plans_sha256", "fingerprint_sha256", "dataset_sha256", "container_image_digest"}
    if not isinstance(nnunet, Mapping) or set(nnunet) != nn_keys:
        return None, "nnU-Net artifact metadata drift"
    if imp.get("artifact_file_sha256") != _sha256(imp_input_artifact):
        return None, "IMP initialization raw artifact bytes drift"
    try:
        from freeze_rq1_v2_artifacts import tensor_state_sha256
        if imp.get("imagenet_pretrained_state_sha256") != tensor_state_sha256(_load_imp_tensor_state(imp_input_artifact)):
            return None, "IMP initialization tensor-state drift"
    except (ImportError, ContractError, ValueError) as exc:
        return None, f"IMP tensor-state trust validation failed: {exc}"
    if nnunet.get("plans_sha256") != _sha256(nnunet_input_artifact):
        return None, "nnU-Net input artifact bytes drift"
    if nnunet.get("checkpoint_sha256") != _sha256(nnunet_checkpoint):
        return None, "nnU-Net checkpoint bytes drift"

    configs = experiment.get("configs")
    expected_configs: dict[str, list[Mapping[str, Any]]] = {"imp": [], "nnunet": []}
    try:
        for arm in expected_configs:
            for seed in sorted(SEEDS):
                _, config = load_public_contract(
                    ROOT / "experiments" / "rq1_v2" / "protocol.json",
                    ROOT / "experiments" / "rq1_v2" / "configs" / f"{arm}_seed{seed}.yaml",
                )
                expected_configs[arm].append(config)
    except ContractError as exc:
        return None, str(exc)
    if configs != expected_configs:
        return None, "frozen six-config binding drift"

    runtimes = experiment.get("runtimes")
    if not isinstance(runtimes, Mapping) or set(runtimes) != {"imp", "nnunet"}:
        return None, "runtime metadata missing"
    runtime_keys = {"arm", "container_image_digest", "dependency_lock_sha256", "shared_contract",
                    "config_family_sha256", "scientific_contract_sha256s", "artifact_sha256s"}
    expected_scientific = {
        "protocol_sha256": enriched_protocol["protocol_sha256"],
        "condition_contract_sha256": enriched_protocol["condition_contract_sha256"],
        "metric_contract_sha256": enriched_protocol["metric_contract_sha256"],
        "data_report_sha256": report["canonical_report_sha256"],
    }
    for arm in ("imp", "nnunet"):
        runtime = runtimes.get(arm)
        if not isinstance(runtime, Mapping) or set(runtime) != runtime_keys or runtime.get("arm") != arm:
            return None, f"{arm} runtime schema drift"
        if not SHA256.fullmatch(str(runtime.get("dependency_lock_sha256", ""))):
            return None, f"{arm} dependency lock drift"
        if runtime.get("scientific_contract_sha256s") != expected_scientific:
            return None, f"{arm} scientific protocol/report binding drift"
        expected_refs = ({name: imp[name] for name in ("imagenet_pretrained_state_sha256", "artifact_file_sha256")}
                         if arm == "imp" else {key: nnunet[key] for key in
                                                ("checkpoint_sha256", "plans_sha256", "fingerprint_sha256", "dataset_sha256")})
        if runtime.get("artifact_sha256s") != expected_refs:
            return None, f"{arm} runtime/model artifact binding drift"
    if runtimes["imp"]["dependency_lock_sha256"] == runtimes["nnunet"]["dependency_lock_sha256"]:
        return None, "arm dependency locks must differ"
    try:
        from freeze_rq1_v2_artifacts import (
            RuntimeManifest,
            _validate_configs,
            _validate_model_artifact_metadata,
            _validate_runtimes,
            config_family_sha256,
        )
        runtime_objects = {
            arm: RuntimeManifest(
                arm=runtime["arm"],
                container_image_digest=runtime["container_image_digest"],
                dependency_lock_sha256=runtime["dependency_lock_sha256"],
                shared_contract=runtime["shared_contract"],
                config_family_sha256=runtime["config_family_sha256"],
                scientific_contract_sha256s=runtime["scientific_contract_sha256s"],
                artifact_sha256s=runtime["artifact_sha256s"],
            )
            for arm, runtime in runtimes.items()
        }
        _validate_model_artifact_metadata(model_artifacts)
        _validate_configs(configs)
        _validate_runtimes(runtime_objects)
        for arm, runtime in runtime_objects.items():
            family_hashes = {config_family_sha256(config) for config in configs[arm]}
            if family_hashes != {runtime.config_family_sha256}:
                return None, f"{arm} runtime config family mismatch"
        if nnunet["container_image_digest"] != runtime_objects["nnunet"].container_image_digest:
            return None, "nnunet model/runtime container image mismatch"
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        return None, f"frozen runtime/model trust validation failed: {exc}"
    return experiment, None


def run_contract(args: argparse.Namespace, *, operation: str) -> int:
    try:
        protocol, config = load_public_contract(args.protocol, args.config)
    except ContractError as exc:
        _emit({"status": "blocked_contract_drift", "reason": str(exc), "data_open_count": 0,
               "engine_available": False})
        return 2

    base = {
        "operation": operation,
        "arm": config["arm"],
        "model_id": config["model_id"],
        "seed": config["seed"],
        "engine_available": False,
        "data_open_count": 0,
        "checkpoint_relative_path": canonical_checkpoint_relative(config).as_posix(),
    }
    if args.dry_run:
        _emit({**base, "status": "dry_run", "private_artifacts_checked": False})
        return 0

    if (operation == "train" and args.input_artifact is not None and
            args.output_checkpoint is not None and
            args.input_artifact.resolve() == args.output_checkpoint.resolve()):
        _emit({**base, "status": "blocked_contract_drift",
               "reason": "training input and output checkpoint must be separate"})
        return 2

    required = {
        "data_manifest": args.data_manifest,
        "experiment_manifest": args.experiment_manifest,
        "input_artifact": args.input_artifact,
        "parent_release": args.parent_release,
        "imp_input_artifact": args.imp_input_artifact,
        "nnunet_input_artifact": args.nnunet_input_artifact,
        "nnunet_checkpoint": args.nnunet_checkpoint,
    }
    missing = _missing(required)
    if operation == "train" and args.output_checkpoint is None:
        missing.append("output_checkpoint")
    if missing:
        _emit({**base, "status": "blocked_missing_prerequisite",
               "missing_prerequisites": sorted(set(missing))})
        return 2

    if not isinstance(args.input_artifact_sha256, str) or not SHA256.fullmatch(args.input_artifact_sha256):
        _emit({**base, "status": "blocked_missing_prerequisite",
               "missing_prerequisites": ["input_artifact_sha256"]})
        return 2
    actual_input_sha = _sha256(args.input_artifact)
    if actual_input_sha != args.input_artifact_sha256:
        _emit({**base, "status": "blocked_artifact_drift", "artifact": "input_artifact"})
        return 2
    checkpoint_relative = canonical_checkpoint_relative(config)
    if operation == "train":
        if tuple(args.output_checkpoint.parts[-len(checkpoint_relative.parts):]) != checkpoint_relative.parts:
            _emit({**base, "status": "blocked_checkpoint_path_drift",
                   "reason": "output checkpoint must match config run_id/checkpoint_template"})
            return 2
    elif tuple(args.input_artifact.parts[-len(checkpoint_relative.parts):]) != checkpoint_relative.parts:
        _emit({**base, "status": "blocked_checkpoint_path_drift",
               "reason": "evaluation checkpoint must match config run_id/checkpoint_template"})
        return 2
    if operation == "train":
        trusted_input = args.imp_input_artifact if config["arm"] == "imp" else args.nnunet_input_artifact
        if args.input_artifact.resolve() != trusted_input.resolve():
            _emit({**base, "status": "blocked_artifact_drift", "artifact": "training_input_binding"})
            return 2

    # The prospective protocol must be admitted before the private index opens.
    if protocol.get("dataset_index_status") != "verified" or not SHA256.fullmatch(str(protocol.get("dataset_index_sha256", ""))):
        _emit({**base, "status": "blocked_unverified_data_manifest"})
        return 2
    experiment, trust_error = validate_frozen_experiment_trust(
        args.experiment_manifest,
        parent_release=args.parent_release,
        data_manifest=args.data_manifest,
        imp_input_artifact=args.imp_input_artifact,
        nnunet_input_artifact=args.nnunet_input_artifact,
        nnunet_checkpoint=args.nnunet_checkpoint,
        protocol=protocol,
    )
    if experiment is None:
        _emit({**base, "status": "blocked_untrusted_experiment", "reason": trust_error,
               "data_open_count": 0})
        return 2

    actual_data_sha = _sha256(args.data_manifest)
    if actual_data_sha != protocol["dataset_index_sha256"]:
        _emit({**base, "status": "blocked_artifact_drift", "artifact": "data_manifest",
               "data_open_count": 1})
        return 2
    if args.preflight_only:
        if operation == "evaluate":
            _emit({**base, "status": "blocked_untrusted_output_checkpoint",
                   "reason": "contract-only release has no frozen evaluation-output trust chain",
                   "data_open_count": 1})
            return 2
        _emit({**base, "status": "preflight_passed", "data_open_count": 1,
               "input_artifact_sha256": actual_input_sha, "data_manifest_sha256": actual_data_sha})
        return 0

    _emit({**base, "status": "blocked_engine_not_in_release",
           "reason": "tracked repository provides a contract-only scaffold; no training/evaluation engine"})
    return 2


def build_parser(description: str, *, operation: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument("--input-artifact", type=Path)
    parser.add_argument("--input-artifact-sha256")
    parser.add_argument("--parent-release", type=Path)
    parser.add_argument("--imp-input-artifact", type=Path)
    parser.add_argument("--nnunet-input-artifact", type=Path)
    parser.add_argument("--nnunet-checkpoint", type=Path)
    if operation == "train":
        parser.add_argument("--output-checkpoint", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser("Validate the contract-only RQ1-v2 training surface", operation="train")
    return run_contract(parser.parse_args(), operation="train")


if __name__ == "__main__":
    raise SystemExit(main())
