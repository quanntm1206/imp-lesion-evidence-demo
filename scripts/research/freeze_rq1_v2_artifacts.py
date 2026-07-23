"""Fail-closed RQ1-v2 artifact freezer interfaces.

This module deliberately builds in-memory manifests only.  Production values
must be supplied by the private artifact and runtime capture workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARMS = ("imp", "nnunet")
_RQ1_SEEDS = frozenset({206, 1206, 2206})
_NNUNET_IDENTITIES = {
    "checkpoint_sha256": "3814716033afd464dacc573f92a5a44ff20eb7f2163d99b4f16ecff8aa278ea2",
    "plans_sha256": "b60e4defd229b03f7064dc5b66123545c91cdaa44c09d990b86690a94e1e08a7",
    "fingerprint_sha256": "931da8aae52ffecd726d5928009ebdcae7002e24b035fad89177e0bc81dba85c",
    "dataset_sha256": "eb33bcbad9d8d5c96168b3c12171392ffabf63ba4cbff4f2bf4badc98bf6487a",
}
_SCIENTIFIC_CONTRACT_KEYS = frozenset({
    "protocol_sha256", "data_report_sha256", "condition_contract_sha256", "metric_contract_sha256",
})
_ARTIFACT_KEYS = {
    "imp": frozenset({"imagenet_pretrained_state_sha256", "artifact_file_sha256"}),
    "nnunet": frozenset(_NNUNET_IDENTITIES),
}


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Tensor objects are represented by stable metadata and their raw bytes.
    try:
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        return {"__tensor__": True, "dtype": str(array.dtype), "shape": list(array.shape),
                "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    except (AttributeError, TypeError, RuntimeError):
        raise ValueError("tensor state contains an unsupported value") from None


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the sole accepted ASCII representation for frozen JSON."""
    _reject_private_or_blocked(payload)
    return _canonical_bytes(payload)


def write_immutable_json(path: Path, payload: Any) -> bytes:
    """Create a canonical JSON file, accepting only byte-identical reruns."""
    if not isinstance(path, Path) or not path.parent.is_dir():
        raise ValueError("immutable JSON parent directory must exist")
    serialized = canonical_json_bytes(payload)
    if path.exists():
        if not path.is_file() or path.read_bytes() != serialized:
            raise ValueError("existing immutable JSON drift; refusing overwrite")
    else:
        path.write_bytes(serialized)
    return serialized


def _sha(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a sha256 image digest")


def _walk(value: Any, depth: int = 0) -> Any:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_lower = str(key).lower()
            if (("release_manifest_sha256" in key_lower and not
                 (depth == 0 and key_lower == "parent_release_manifest_sha256")) or
                    "current_release" in key_lower or
                    "future_release" in key_lower or key_lower == "experiment_input_manifest_sha256"):
                raise ValueError("release manifest reference is forbidden")
            yield key
            yield from _walk(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item, depth + 1)
    else:
        yield value


def _reject_private_or_blocked(value: Any) -> None:
    for item in _walk(value):
        if not isinstance(item, str):
            continue
        lower = item.lower().replace("\\", "/")
        if "unresolved_blocked" in lower:
            raise ValueError("data integrity is unresolved_blocked")
        if (lower.startswith(("/", "file:")) or re.match(r"^[a-z]:/+", lower) or
                "/private/" in lower or "private://" in lower):
            raise ValueError("private paths are forbidden")


def tensor_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash an ordered, typed representation of a nonempty state dictionary."""
    if not isinstance(state, Mapping) or not state:
        raise ValueError("pretrained tensor state must be nonempty")
    return hashlib.sha256(_canonical_bytes(state)).hexdigest()


def config_family_sha256(config: Mapping[str, Any]) -> str:
    if not isinstance(config, Mapping) or not config:
        raise ValueError("config family must be nonempty")
    family = {key: value for key, value in config.items() if key not in {"seed", "run_id"}}
    if not family:
        raise ValueError("config family cannot contain only seed/run_id")
    _reject_private_or_blocked(family)
    return hashlib.sha256(_canonical_bytes(family)).hexdigest()


@dataclass(frozen=True)
class RuntimeManifest:
    arm: str
    container_image_digest: str
    dependency_lock_sha256: str
    shared_contract: str
    config_family_sha256: str
    scientific_contract_sha256s: Mapping[str, str]
    artifact_sha256s: Mapping[str, str]

    def to_dict(self) -> dict[str, str]:
        return {"arm": self.arm, "container_image_digest": self.container_image_digest,
                "dependency_lock_sha256": self.dependency_lock_sha256,
                "shared_contract": self.shared_contract,
                "config_family_sha256": self.config_family_sha256,
                "scientific_contract_sha256s": dict(sorted(self.scientific_contract_sha256s.items())),
                "artifact_sha256s": dict(sorted(self.artifact_sha256s.items()))}


@dataclass(frozen=True)
class FrozenArtifacts:
    model_artifacts: Mapping[str, Any]
    imp_state: Mapping[str, Any]
    runtimes: Mapping[str, RuntimeManifest]


@dataclass(frozen=True)
class ExperimentInputManifest:
    parent_release_manifest_sha256: str
    protocol: Mapping[str, Any]
    data_report: Mapping[str, Any]
    model_artifacts: Mapping[str, Any]
    runtimes: Mapping[str, RuntimeManifest]
    configs: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Canonical payload; intentionally excludes a manifest self-hash."""
        public_protocol_identity = {
            key: value for key, value in self.protocol.items()
            if key not in {"protocol_sha256", "condition_contract_sha256", "metric_contract_sha256",
                           "dataset_index_status", "dataset_index_sha256"}
        }
        return {"schema_version": "imp.rq1_v2.experiment_input.v1",
                "parent_release_manifest_sha256": self.parent_release_manifest_sha256,
                "public_protocol_projection": {
                    "schema_version": "imp.rq1_v2.public_protocol_projection.v1",
                    "identity": _canonical(public_protocol_identity),
                },
                "protocol": _canonical(self.protocol), "data_report": _canonical(self.data_report),
                "model_artifacts": _canonical(self.model_artifacts),
                "runtimes": {arm: self.runtimes[arm].to_dict() for arm in sorted(self.runtimes)},
                "configs": _canonical(self.configs)}


def experiment_input_sha256(manifest: ExperimentInputManifest | Mapping[str, Any]) -> str:
    """Digest an input manifest externally; never inject a self-reference."""
    payload = manifest.to_dict() if isinstance(manifest, ExperimentInputManifest) else manifest
    _reject_private_or_blocked(payload)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_runtime_manifest(arm: str, image_digest: str, dependency_lock: str,
                           config_family: Mapping[str, Any],
                           scientific_contract_sha256s: Mapping[str, str] | None = None,
                           artifact_sha256s: Mapping[str, str] | None = None) -> RuntimeManifest:
    if arm not in _ARMS:
        raise ValueError("unknown arm")
    _digest(image_digest, "container image")
    _sha(dependency_lock, "dependency lock")
    if not isinstance(scientific_contract_sha256s, Mapping) or set(scientific_contract_sha256s) != _SCIENTIFIC_CONTRACT_KEYS:
        raise ValueError("all explicit scientific contract digests are required")
    for name, digest in scientific_contract_sha256s.items():
        _sha(digest, name)
    if not isinstance(artifact_sha256s, Mapping) or set(artifact_sha256s) != _ARTIFACT_KEYS[arm]:
        raise ValueError(f"{arm} runtime artifact references are incomplete")
    for name, digest in artifact_sha256s.items():
        _sha(digest, name)
        if arm == "nnunet" and digest != _NNUNET_IDENTITIES[name]:
            raise ValueError(f"nnunet runtime {name} drift")
    family_hash = config_family_sha256(config_family)
    shared_contract = hashlib.sha256(canonical_json_bytes(scientific_contract_sha256s)).hexdigest()
    return RuntimeManifest(arm, image_digest, dependency_lock, shared_contract, family_hash,
                           dict(scientific_contract_sha256s), dict(artifact_sha256s))


def _validate_runtimes(runtimes: Mapping[str, RuntimeManifest]) -> None:
    if set(runtimes) != set(_ARMS):
        raise ValueError("runtime manifests require imp and nnunet arms")
    for arm in _ARMS:
        runtime = runtimes[arm]
        if not isinstance(runtime, RuntimeManifest) or runtime.arm != arm:
            raise ValueError("runtime arm separation is invalid")
        _digest(runtime.container_image_digest, "container image")
        _sha(runtime.dependency_lock_sha256, "dependency lock")
        _sha(runtime.shared_contract, "shared contract")
        _sha(runtime.config_family_sha256, "config family")
        if set(runtime.scientific_contract_sha256s) != _SCIENTIFIC_CONTRACT_KEYS:
            raise ValueError("runtime scientific contract is incomplete")
        for name, digest in runtime.scientific_contract_sha256s.items():
            _sha(digest, name)
        if runtime.shared_contract != hashlib.sha256(canonical_json_bytes(runtime.scientific_contract_sha256s)).hexdigest():
            raise ValueError("runtime shared contract digest drift")
        if set(runtime.artifact_sha256s) != _ARTIFACT_KEYS[arm]:
            raise ValueError(f"{arm} runtime artifact references are incomplete")
        for name, digest in runtime.artifact_sha256s.items():
            _sha(digest, name)
            if arm == "nnunet" and digest != _NNUNET_IDENTITIES[name]:
                raise ValueError(f"nnunet runtime {name} drift")
    if runtimes["imp"].container_image_digest == runtimes["nnunet"].container_image_digest:
        raise ValueError("runtime arm separation requires distinct images")
    if runtimes["imp"].dependency_lock_sha256 == runtimes["nnunet"].dependency_lock_sha256:
        raise ValueError("runtime arm separation requires distinct locks")
    if runtimes["imp"].shared_contract != runtimes["nnunet"].shared_contract:
        raise ValueError("runtime shared contract mismatch")


def _scientific_projection(protocol: Mapping[str, Any], data_report: Mapping[str, Any]) -> dict[str, str]:
    try:
        report_sha256 = data_report.get("canonical_report_sha256", data_report.get("report_sha256"))
        projection = {
            "protocol_sha256": protocol["protocol_sha256"],
            "condition_contract_sha256": protocol["condition_contract_sha256"],
            "metric_contract_sha256": protocol["metric_contract_sha256"],
            "data_report_sha256": report_sha256,
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("scientific contract projection is incomplete") from exc
    for name, digest in projection.items():
        _sha(digest, name)
    return projection


def _validate_configs(configs: Mapping[str, Any]) -> None:
    if set(configs) != set(_ARMS) or any(not isinstance(configs[arm], list) or len(configs[arm]) != 3 for arm in _ARMS):
        raise ValueError("six configs require three per arm")
    for arm in _ARMS:
        seeds: set[Any] = set()
        for config in configs[arm]:
            if not isinstance(config, Mapping):
                raise ValueError("configs require final_epoch_only")
            training = config.get("training")
            selection = config.get("checkpoint_selection")
            if selection is None and isinstance(training, Mapping):
                selection = training.get("checkpoint_selection")
            if selection != "final_epoch_only":
                raise ValueError("configs require final_epoch_only")
            if config.get("seed") in seeds:
                raise ValueError("configs require unique seeds per arm")
            seeds.add(config.get("seed"))
        if seeds != _RQ1_SEEDS:
            raise ValueError("configs require exact seeds 206, 1206, 2206")
        families = {config_family_sha256(config) for config in configs[arm]}
        if len(families) != 1:
            raise ValueError("configs require seed-neutral family equality per arm")
    _reject_private_or_blocked(configs)


def _validate_model_artifact_metadata(model_artifacts: Mapping[str, Any]) -> None:
    try:
        _sha(model_artifacts["imp"]["imagenet_pretrained_state_sha256"], "pretrained tensor state")
        _sha(model_artifacts["imp"]["artifact_file_sha256"], "pretrained artifact file")
        nnunet = model_artifacts["nnunet"]
        for name in ("checkpoint_sha256", "plans_sha256", "fingerprint_sha256", "dataset_sha256"):
            _sha(nnunet[name], name)
            if nnunet[name] != _NNUNET_IDENTITIES[name]:
                raise ValueError(f"nnunet {name} drift")
        _digest(nnunet["container_image_digest"], "nnunet container image")
    except (KeyError, TypeError) as exc:
        raise ValueError("missing model artifact") from exc
    _reject_private_or_blocked(model_artifacts)


def validate_artifacts(frozen: FrozenArtifacts, *, imp_state: Mapping[str, Any] | None = None) -> None:
    if not isinstance(frozen, FrozenArtifacts):
        raise ValueError("FrozenArtifacts is required")
    actual = tensor_state_sha256(frozen.imp_state if imp_state is None else imp_state)
    try:
        expected = frozen.model_artifacts["imp"]["imagenet_pretrained_state_sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("missing imp pretrained artifact") from exc
    _sha(expected, "pretrained tensor state")
    if actual != expected:
        raise ValueError("pretrained weight drift")
    _validate_model_artifact_metadata(frozen.model_artifacts)
    _validate_runtimes(frozen.runtimes)
    expected_refs = {
        "imp": {name: frozen.model_artifacts["imp"][name] for name in _ARTIFACT_KEYS["imp"]},
        "nnunet": {name: frozen.model_artifacts["nnunet"][name] for name in _NNUNET_IDENTITIES},
    }
    for arm, refs in expected_refs.items():
        if dict(frozen.runtimes[arm].artifact_sha256s) != refs:
            raise ValueError(f"{arm} runtime/model artifact reference mismatch")


def build_experiment_input_manifest(parent: Path, protocol: Mapping[str, Any], data_report: Mapping[str, Any],
                                    model_artifacts: Mapping[str, Any], runtimes: Mapping[str, RuntimeManifest],
                                    configs: Mapping[str, Any]) -> ExperimentInputManifest:
    if not isinstance(parent, Path) or not parent.is_file():
        raise ValueError("parent release manifest is required")
    _reject_private_or_blocked((protocol, data_report, model_artifacts, runtimes, configs))
    _validate_model_artifact_metadata(model_artifacts)
    _validate_runtimes(runtimes)
    _validate_configs(configs)
    projection = _scientific_projection(protocol, data_report)
    for arm, runtime in runtimes.items():
        if dict(runtime.scientific_contract_sha256s) != projection:
            raise ValueError(f"{arm} runtime scientific contract mismatch")
        family_hashes = {config_family_sha256(config) for config in configs[arm]}
        if family_hashes != {runtime.config_family_sha256}:
            raise ValueError(f"{arm} runtime config family mismatch")
    if model_artifacts["nnunet"]["container_image_digest"] != runtimes["nnunet"].container_image_digest:
        raise ValueError("nnunet model/runtime container image mismatch")
    model_refs = {
        "imp": {name: model_artifacts["imp"][name] for name in _ARTIFACT_KEYS["imp"]},
        "nnunet": {name: model_artifacts["nnunet"][name] for name in _NNUNET_IDENTITIES},
    }
    for arm in _ARMS:
        if dict(runtimes[arm].artifact_sha256s) != model_refs[arm]:
            raise ValueError(f"{arm} runtime artifact reference mismatch")
    return ExperimentInputManifest(hashlib.sha256(parent.read_bytes()).hexdigest(), protocol, data_report,
                                   model_artifacts, runtimes, configs)
