"""Validate six RQ1-v2 job receipts without performing deferred analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import yaml


SEEDS = (206, 1206, 2206)
EXPECTED = {(arm, seed) for arm in ("imp", "nnunet") for seed in SEEDS}
MODEL_IDS = {"imp": "RQ1v2-IMP-MiT-B3-UNet", "nnunet": "RQ1v2-nnUNet-v2-2d"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
HASH_FIELDS = (
    "config_sha256",
    "protocol_sha256",
    "experiment_manifest_sha256",
    "data_manifest_sha256",
    "dependency_lock_sha256",
    "input_artifact_sha256",
    "output_checkpoint_sha256",
    "metric_source_sha256",
)
METRICS = {"dice", "iou", "precision", "recall", "boundary_f1", "hd95", "assd"}
ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_rq1_v2 import canonical_checkpoint_relative, validate_frozen_experiment_trust  # noqa: E402
EXPECTED_RECEIPT_KEYS = {
    "schema_version", "job_id", "arm", "seed", "model_id", "status",
    *HASH_FIELDS, "metrics",
}


class _UniqueSafeLoader(yaml.SafeLoader):
    """Reject duplicate config keys rather than silently choosing one."""


def _construct_unique_mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _pending(completed: int = 0) -> dict[str, Any]:
    return {
        "schema_version": "imp.rq1_v2.public_results.v1",
        "status": "pending/unverified",
        "p1_status": "not_promoted",
        "metrics": [],
        "completed_jobs": completed,
        "required_jobs": 6,
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def _valid_receipt(payload: Mapping[str, Any]) -> bool:
    try:
        arm = payload["arm"]
        seed = payload["seed"]
        metrics = payload["metrics"]
        if set(payload) != EXPECTED_RECEIPT_KEYS:
            return False
        if payload["schema_version"] != "imp.rq1_v2.job_receipt.v1":
            return False
        if (arm, seed) not in EXPECTED or payload["job_id"] != f"{arm}_seed{seed}":
            return False
        if payload["model_id"] != MODEL_IDS[arm] or payload["status"] != "validated":
            return False
        if any(not isinstance(payload[name], str) or not SHA256.fullmatch(payload[name]) for name in HASH_FIELDS):
            return False
        if not isinstance(metrics, Mapping) or set(metrics) != METRICS:
            return False
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               or (key in {"dice", "iou", "precision", "recall", "boundary_f1"} and not 0.0 <= value <= 1.0)
               or (key in {"hd95", "assd"} and value < 0.0)
               for key, value in metrics.items()):
            return False
    except (KeyError, TypeError):
        return False
    return True


def _load_trusted_experiment(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def summarize(receipts_dir: Path, *, data_manifest: Path | None = None,
              experiment_manifest: Path | None = None, parent_release: Path | None = None,
              imp_input_artifact: Path | None = None,
              nnunet_input_artifact: Path | None = None,
              nnunet_checkpoint: Path | None = None,
              checkpoint_dir: Path | None = None) -> tuple[dict[str, Any], int]:
    paths = sorted(receipts_dir.glob("*.json")) if receipts_dir.is_dir() else []
    receipts: list[Mapping[str, Any]] = []
    malformed = False
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=_unique_pairs)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            malformed = True
            continue
        if not isinstance(payload, Mapping) or not _valid_receipt(payload):
            malformed = True
            continue
        receipts.append(payload)
    unique = {(row["arm"], row["seed"]) for row in receipts}
    completed = len(unique)
    if malformed or len(receipts) != 6 or unique != EXPECTED:
        return _pending(completed), 2

    trust_paths = (data_manifest, experiment_manifest, parent_release, imp_input_artifact,
                   nnunet_input_artifact, nnunet_checkpoint)
    if any(path is None or not path.is_file() for path in trust_paths) or checkpoint_dir is None or not checkpoint_dir.is_dir():
        return _pending(completed), 2
    protocol_path = ROOT / "experiments" / "rq1_v2" / "protocol.json"
    trusted_protocol = _load_trusted_experiment(protocol_path)
    if trusted_protocol is None:
        return _pending(completed), 2
    experiment, trust_error = validate_frozen_experiment_trust(
        experiment_manifest,
        parent_release=parent_release,
        data_manifest=data_manifest,
        imp_input_artifact=imp_input_artifact,
        nnunet_input_artifact=nnunet_input_artifact,
        nnunet_checkpoint=nnunet_checkpoint,
        protocol=trusted_protocol,
    )
    if experiment is None or trust_error is not None:
        return _pending(completed), 2
    experiment_sha = hashlib.sha256(experiment_manifest.read_bytes()).hexdigest()
    data_sha = hashlib.sha256(data_manifest.read_bytes()).hexdigest()
    enriched_protocol = experiment.get("protocol")
    protocol_sha = enriched_protocol.get("protocol_sha256") if isinstance(enriched_protocol, Mapping) else None
    if any(row["experiment_manifest_sha256"] != experiment_sha or row["data_manifest_sha256"] != data_sha
           for row in receipts):
        return _pending(completed), 2
    data_report = experiment.get("data_report")
    if not isinstance(data_report, Mapping) or data_report.get("dataset_index_sha256") != data_sha:
        return _pending(completed), 2

    if any(row["protocol_sha256"] != protocol_sha for row in receipts):
        return _pending(completed), 2
    for row in receipts:
        config_path = ROOT / "experiments" / "rq1_v2" / "configs" / f"{row['arm']}_seed{row['seed']}.yaml"
        if not config_path.is_file() or hashlib.sha256(config_path.read_bytes()).hexdigest() != row["config_sha256"]:
            return _pending(completed), 2
        try:
            config = yaml.load(config_path.read_text(encoding="ascii"), Loader=_UniqueSafeLoader)
        except (OSError, UnicodeError, yaml.YAMLError, ValueError):
            return _pending(completed), 2
        if not isinstance(config, Mapping):
            return _pending(completed), 2
        experiment_configs = experiment.get("configs")
        arm_configs = experiment_configs.get(row["arm"]) if isinstance(experiment_configs, Mapping) else None
        if not isinstance(arm_configs, list) or config not in arm_configs:
            return _pending(completed), 2
        metric_source = config.get("metric_contract", {}).get("source_sha256") if isinstance(config.get("metric_contract"), Mapping) else None
        if row["metric_source_sha256"] != metric_source:
            return _pending(completed), 2
        expected_key = config.get("private_inputs", {}).get("training_input_artifact_key") if isinstance(config.get("private_inputs"), Mapping) else None
        artifacts = experiment.get("model_artifacts", {}).get(row["arm"]) if isinstance(experiment.get("model_artifacts"), Mapping) else None
        if not isinstance(artifacts, Mapping) or not isinstance(expected_key, str) or row["input_artifact_sha256"] != artifacts.get(expected_key):
            return _pending(completed), 2
        runtime = experiment.get("runtimes", {}).get(row["arm"]) if isinstance(experiment.get("runtimes"), Mapping) else None
        if not isinstance(runtime, Mapping) or row["dependency_lock_sha256"] != runtime.get("dependency_lock_sha256"):
            return _pending(completed), 2
        try:
            relative_checkpoint = canonical_checkpoint_relative(config)
        except ValueError:
            return _pending(completed), 2
        checkpoint = checkpoint_dir / relative_checkpoint
        if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row["output_checkpoint_sha256"]:
            return _pending(completed), 2

    common_fields = ("protocol_sha256", "experiment_manifest_sha256", "data_manifest_sha256", "metric_source_sha256")
    if any(len({row[field] for row in receipts}) != 1 for field in common_fields):
        return _pending(completed), 2
    if len({row["output_checkpoint_sha256"] for row in receipts}) != 6:
        return _pending(completed), 2
    if len({row["config_sha256"] for row in receipts}) != 6:
        return _pending(completed), 2
    locks = {arm: {row["dependency_lock_sha256"] for row in receipts if row["arm"] == arm}
             for arm in MODEL_IDS}
    if any(len(values) != 1 for values in locks.values()) or locks["imp"] == locks["nnunet"]:
        return _pending(completed), 2

    # Job-level scalar receipts do not authenticate per-case evaluation bytes.
    # Promotion remains blocked until the deferred analysis trust chain exists.
    pending = _pending(6)
    pending["trust_chain_status"] = "validated_inputs_analysis_deferred"
    return pending, 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a complete six-job RQ1-v2 receipt set")
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument("--parent-release", type=Path)
    parser.add_argument("--imp-input-artifact", type=Path)
    parser.add_argument("--nnunet-input-artifact", type=Path)
    parser.add_argument("--nnunet-checkpoint", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload, status = summarize(
        args.receipts,
        data_manifest=args.data_manifest,
        experiment_manifest=args.experiment_manifest,
        parent_release=args.parent_release,
        imp_input_artifact=args.imp_input_artifact,
        nnunet_input_artifact=args.nnunet_input_artifact,
        nnunet_checkpoint=args.nnunet_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
    )
    _write(args.output, payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
