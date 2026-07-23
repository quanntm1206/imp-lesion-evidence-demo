"""Canonical release identity and deterministic consumer projections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


MANIFEST_SCHEMA = "imp.release.manifest.v1"
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_KEYS = frozenset(
    {"claim_policies", "comparisons", "models", "provenance", "public_sample_contract", "public_sample_selection", "public_samples", "rq1_v2", "schema_version"}
)
_MODEL_IDS = frozenset(
    {"L191-C0-clean-v3-IMP-control", "L192-nnUNet-v2-raw-100ep", "L206-control-s206", "L206-contour-channel-s206"}
)
_POLICIES = {
    "operational_only": {"clinical_use": False, "scientific_comparison": False},
    "protected_validation_descriptive": {"clinical_use": False, "protected_test_claim_allowed": False, "scientific_comparison": True},
    "train_screen_ablation": {"clinical_use": False, "scientific_comparison": False},
}
_PUBLIC_TRAINING_EXPOSURE = {
    "L206-control-s206": "excluded_from_308_fit_in_76_group_train_screen_holdout",
    "L192-nnUNet-v2-raw-100ep": "included_in_clean_v3_2008_training_rows",
}
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "release" / "imp_release_manifest.json"


@dataclass(frozen=True)
class RuntimeIdentity:
    model_id: str
    checkpoint_sha256: str
    data: Mapping[str, Any]


@dataclass(frozen=True)
class Comparison:
    id: str
    left_model_id: str
    right_model_id: str
    claim_policy: str
    scope: str


@dataclass(frozen=True)
class LicenseEvidence:
    clean_v3_manifest_sha256: str
    csv_row_number: int
    raw_csv_row_sha256: str


@dataclass(frozen=True)
class PublicSample:
    sample_id: str
    group_key: str
    source_dataset: str
    sha256_raw: str
    sha256_rgb: str
    source_page: str
    license_id: str
    license_evidence: LicenseEvidence
    training_exposure: Mapping[str, str]
    ground_truth_used: bool
    ground_truth_not_loaded: bool


@dataclass(frozen=True)
class PublicSamples:
    selection: Mapping[str, Any]
    samples: tuple[PublicSample, ...]


@dataclass(frozen=True)
class PublicSampleRole:
    sample_id: str
    group_key: str
    sha256_raw: str
    sha256_rgb: str


@dataclass(frozen=True)
class PublicSampleContract:
    state: str
    roles: Mapping[str, PublicSampleRole]


@dataclass(frozen=True)
class ReleaseManifest:
    schema_version: str
    models: Mapping[str, RuntimeIdentity]
    comparisons: tuple[Comparison, ...]
    claim_policies: Mapping[str, Mapping[str, Any]]
    public_sample_selection: Mapping[str, Any]
    public_samples: PublicSamples
    public_sample_contract: PublicSampleContract
    provenance: Mapping[str, Any]
    rq1_v2: Mapping[str, Any]
    path: Path
    digest: str

    def model(self, model_id: str) -> RuntimeIdentity:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise ValueError(f"unknown release model: {model_id}") from exc

    def comparison(self, comparison_id: str) -> Comparison:
        for comparison in self.comparisons:
            if comparison.id == comparison_id:
                return comparison
        raise ValueError(f"unknown release comparison: {comparison_id}")

    def comparison_mapping(self, comparison_id: str) -> dict[str, str]:
        comparison = self.comparison(comparison_id)
        return {
            "id": comparison.id,
            "left_model_id": comparison.left_model_id,
            "right_model_id": comparison.right_model_id,
            "claim_policy": comparison.claim_policy,
            "scope": comparison.scope,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"release manifest {label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f"release manifest {label} keys mismatch")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"release manifest {label} SHA-256 is invalid")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"release manifest {label} is invalid")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"release manifest {label} is invalid") from exc
    return value


def _release_path(value: object, label: str) -> str:
    text = _text(value, f"{label} path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or "\\" in text
        or ":" in text
        or ".." in path.parts
    ):
        raise ValueError(f"release manifest {label} path is invalid")
    return text


def _fixed_cache_arm(value: object, label: str) -> dict[str, Any]:
    arm = _mapping(value, label)
    _exact_keys(arm, {"data_sha256", "manifest_path", "manifest_sha256"}, label)
    _release_path(arm.get("manifest_path"), f"{label} manifest")
    _sha256(arm.get("manifest_sha256"), f"{label}.manifest_sha256")
    _sha256(arm.get("data_sha256"), f"{label}.data_sha256")
    return dict(arm)


def _public_sample_selection(value: object) -> dict[str, Any]:
    public_sample = _mapping(value, "public_sample_selection")
    _exact_keys(
        public_sample,
        {"dataset_index", "fixed_cache", "live_config", "ordered_universe_sha256"},
        "public_sample_selection",
    )
    _sha256(
        public_sample.get("ordered_universe_sha256"),
        "public_sample_selection.ordered_universe_sha256",
    )

    dataset_index = _mapping(public_sample.get("dataset_index"), "dataset_index")
    _exact_keys(dataset_index, {"path", "sha256"}, "dataset_index")
    _release_path(dataset_index.get("path"), "dataset_index")
    _sha256(dataset_index.get("sha256"), "dataset_index.sha256")

    live_config = _mapping(public_sample.get("live_config"), "live_config")
    _exact_keys(live_config, {"path", "schema_version", "sha256"}, "live_config")
    _release_path(live_config.get("path"), "live_config")
    _text(live_config.get("schema_version"), "live_config schema")
    _sha256(live_config.get("sha256"), "live_config.sha256")

    fixed_cache = _mapping(public_sample.get("fixed_cache"), "fixed_cache")
    _exact_keys(
        fixed_cache,
        {"artifact_type", "candidate", "count", "schema_version", "shape", "zero"},
        "fixed_cache",
    )
    _text(fixed_cache.get("schema_version"), "fixed_cache schema")
    _text(fixed_cache.get("artifact_type"), "fixed_cache artifact_type")
    count = fixed_cache.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("release manifest fixed_cache count is invalid")
    shape = fixed_cache.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape)
    ):
        raise ValueError("release manifest fixed_cache shape is invalid")
    candidate = _fixed_cache_arm(fixed_cache.get("candidate"), "fixed_cache candidate")
    zero = _fixed_cache_arm(fixed_cache.get("zero"), "fixed_cache zero")
    if candidate["manifest_path"] == zero["manifest_path"]:
        raise ValueError("release manifest fixed_cache arm paths are invalid")
    return dict(public_sample)


def _public_samples(value: object) -> PublicSamples:
    payload = _mapping(value, "public_samples")
    _exact_keys(payload, {"selection", "samples"}, "public_samples")
    selection = _mapping(payload.get("selection"), "public_samples selection")
    _exact_keys(
        selection,
        {"universe", "universe_count", "dataset_index_sha256", "ordered_universe_sha256", "rule"},
        "public_samples selection",
    )
    if selection.get("universe") != "loop206_train_screen_holdout_clean":
        raise ValueError("release manifest public sample universe mismatch")
    if selection.get("rule") != "explicit_roles_A_B_boundary_after_index_hash":
        raise ValueError("release manifest public sample rule mismatch")
    if type(selection.get("universe_count")) is not int or selection["universe_count"] != 76:
        raise ValueError("release manifest public sample universe count mismatch")
    _sha256(selection.get("dataset_index_sha256"), "public_samples selection.dataset_index_sha256")
    _sha256(selection.get("ordered_universe_sha256"), "public_samples selection.ordered_universe_sha256")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != 2:
        raise ValueError("release manifest public sample count mismatch")
    samples: list[PublicSample] = []
    for value in raw_samples:
        sample = _mapping(value, "public sample")
        _exact_keys(
            sample,
            {"sample_id", "group_key", "source_dataset", "sha256_raw", "sha256_rgb", "source_page", "license_id", "license_evidence", "training_exposure", "ground_truth_used", "ground_truth_not_loaded"},
            "public sample",
        )
        sample_id = _text(sample.get("sample_id"), "public sample sample_id")
        group_key = _text(sample.get("group_key"), "public sample group_key")
        source_dataset = _text(sample.get("source_dataset"), "public sample source_dataset")
        source_page = _text(sample.get("source_page"), "public sample source_page")
        if source_dataset != "isic2018" or source_page != "https://challenge.isic-archive.com/data/":
            raise ValueError("release manifest public sample source mismatch")
        if sample.get("license_id") != "CC-0":
            raise ValueError("release manifest public sample license mismatch")
        raw_hash = _sha256(sample.get("sha256_raw"), "public sample sha256_raw")
        rgb_hash = _sha256(sample.get("sha256_rgb"), "public sample sha256_rgb")
        evidence = _mapping(sample.get("license_evidence"), "public sample license_evidence")
        _exact_keys(evidence, {"clean_v3_manifest_sha256", "csv_row_number", "raw_csv_row_sha256"}, "public sample license_evidence")
        if type(evidence.get("csv_row_number")) is not int or evidence["csv_row_number"] < 2:
            raise ValueError("release manifest public sample CSV row mismatch")
        license_evidence = LicenseEvidence(
            _sha256(evidence.get("clean_v3_manifest_sha256"), "public sample clean_v3_manifest_sha256"),
            evidence["csv_row_number"],
            _sha256(evidence.get("raw_csv_row_sha256"), "public sample raw_csv_row_sha256"),
        )
        exposure = _mapping(sample.get("training_exposure"), "public sample training_exposure")
        if dict(exposure) != _PUBLIC_TRAINING_EXPOSURE:
            raise ValueError("release manifest public sample training exposure mismatch")
        if sample.get("ground_truth_used") is not False or sample.get("ground_truth_not_loaded") is not True:
            raise ValueError("release manifest public sample ground truth mismatch")
        samples.append(PublicSample(sample_id, group_key, source_dataset, raw_hash, rgb_hash, source_page, "CC-0", license_evidence, dict(exposure), False, True))
    if tuple(item.sample_id for item in samples) != ("ISIC_0000050", "ISIC_0012690"):
        raise ValueError("release manifest public sample IDs mismatch")
    return PublicSamples(dict(selection), tuple(samples))


def _public_sample_contract(value: object, samples: PublicSamples) -> PublicSampleContract:
    payload = _mapping(value, "public_sample_contract")
    _exact_keys(payload, {"roles", "state"}, "public_sample_contract")
    if payload.get("state") != "verified":
        raise ValueError("release manifest public sample contract is blocked")
    raw_roles = _mapping(payload.get("roles"), "public_sample_contract roles")
    if tuple(raw_roles) != ("A", "B", "boundary"):
        raise ValueError("release manifest public sample contract roles mismatch")
    roles: dict[str, PublicSampleRole] = {}
    for name, expected_id in (("A", "ISIC_0000050"), ("B", "ISIC_0012690"), ("boundary", "ISIC_0016069")):
        role = _mapping(raw_roles.get(name), f"public sample role {name}")
        _exact_keys(role, {"group_key", "sample_id", "sha256_raw", "sha256_rgb"}, f"public sample role {name}")
        item = PublicSampleRole(
            _text(role.get("sample_id"), f"public sample role {name} sample_id"),
            _text(role.get("group_key"), f"public sample role {name} group_key"),
            _sha256(role.get("sha256_raw"), f"public sample role {name} sha256_raw"),
            _sha256(role.get("sha256_rgb"), f"public sample role {name} sha256_rgb"),
        )
        if item.sample_id != expected_id:
            raise ValueError("release manifest public sample contract roles mismatch")
        roles[name] = item
    for role, sample in zip((roles["A"], roles["B"]), samples.samples):
        if (role.sample_id, role.group_key, role.sha256_raw, role.sha256_rgb) != (sample.sample_id, sample.group_key, sample.sha256_raw, sample.sha256_rgb):
            raise ValueError("public sample provenance contract binding mismatch")
    return PublicSampleContract("verified", roles)


def _identity(model_id: str, value: object) -> RuntimeIdentity:
    payload = _mapping(value, f"model {model_id}")
    common = {"checkpoint_sha256", "display_name", "evidence", "verification_status"}
    if model_id == "L191-C0-clean-v3-IMP-control":
        _exact_keys(payload, common | {"runtime"}, f"model {model_id}")
        if payload.get("checkpoint_sha256") != "unverified":
            raise ValueError("release manifest unverified IMP checkpoint mismatch")
        runtime = _mapping(payload.get("runtime"), f"model {model_id} runtime")
        _exact_keys(runtime, {"preprocessing"}, f"model {model_id} runtime")
    elif model_id == "L192-nnUNet-v2-raw-100ep":
        _exact_keys(payload, common | {"runtime"}, f"model {model_id}")
        runtime = _mapping(payload.get("runtime"), f"model {model_id} runtime")
        _exact_keys(runtime, {"device", "preprocessing", "protocol"}, f"model {model_id} runtime")
    else:
        _exact_keys(payload, common | {"checkpoint_env"}, f"model {model_id}")
    checkpoint = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or (
        checkpoint != "unverified" and _HEX_64.fullmatch(checkpoint) is None
    ):
        raise ValueError(f"release manifest model {model_id} has invalid checkpoint pin")
    return RuntimeIdentity(model_id, checkpoint, dict(payload))


def load_release_manifest(path: str | Path = DEFAULT_MANIFEST) -> ReleaseManifest:
    manifest_path = Path(path)
    try:
        bytes_value = manifest_path.read_bytes()
        text = bytes_value.decode("ascii")
        raw = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("release manifest is unreadable") from exc
    payload = _mapping(raw, "root")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    if bytes_value != canonical:
        raise ValueError("release manifest bytes are not canonical")
    _exact_keys(payload, _MANIFEST_KEYS, "root")
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("release manifest schema mismatch")
    if "release_manifest_sha256" in payload:
        raise ValueError("release manifest must not self-hash")
    models_raw = _mapping(payload.get("models"), "models")
    models = {model_id: _identity(model_id, value) for model_id, value in models_raw.items()}
    if set(models) != _MODEL_IDS:
        raise ValueError("release manifest model set mismatch")
    policies = _mapping(payload.get("claim_policies"), "claim_policies")
    if dict(policies) != _POLICIES:
        raise ValueError("release manifest claim policy mismatch")
    comparisons_raw = payload.get("comparisons")
    if not isinstance(comparisons_raw, list):
        raise ValueError("release manifest comparisons must be an array")
    comparisons: list[Comparison] = []
    for value in comparisons_raw:
        item = _mapping(value, "comparison")
        fields = ("id", "left_model_id", "right_model_id", "claim_policy", "scope")
        _exact_keys(item, set(fields), "comparison")
        if any(not isinstance(item.get(field), str) or not item[field] for field in fields):
            raise ValueError("release manifest comparison is invalid")
        if item["left_model_id"] not in models or item["right_model_id"] not in models:
            raise ValueError("release manifest comparison references an unknown model")
        if item["claim_policy"] not in policies:
            raise ValueError("release manifest comparison has unknown claim policy")
        comparisons.append(Comparison(*(str(item[field]) for field in fields)))
    if len(comparisons) != 3 or {item.id for item in comparisons} != {"live_demo", "paper_rq1", "paper_rq2"}:
        raise ValueError("release manifest comparison set mismatch")
    live = next(item for item in comparisons if item.id == "live_demo")
    if (live.left_model_id, live.right_model_id) != ("L206-control-s206", "L192-nnUNet-v2-raw-100ep"):
        raise ValueError("release manifest live comparison mismatch")
    rq1 = next(item for item in comparisons if item.id == "paper_rq1")
    rq2 = next(item for item in comparisons if item.id == "paper_rq2")
    if (rq1.left_model_id, rq1.right_model_id, rq1.claim_policy) != ("L191-C0-clean-v3-IMP-control", "L192-nnUNet-v2-raw-100ep", "protected_validation_descriptive"):
        raise ValueError("release manifest RQ1 comparison mismatch")
    if (rq2.left_model_id, rq2.right_model_id, rq2.claim_policy) != ("L206-control-s206", "L206-contour-channel-s206", "train_screen_ablation"):
        raise ValueError("release manifest RQ2 comparison mismatch")
    public_sample = _public_sample_selection(payload.get("public_sample_selection"))
    public_samples = _public_samples(payload.get("public_samples"))
    public_sample_contract = _public_sample_contract(payload.get("public_sample_contract"), public_samples)
    if (
        public_sample["dataset_index"]["sha256"] != public_samples.selection["dataset_index_sha256"]
        or public_sample["ordered_universe_sha256"] != public_samples.selection["ordered_universe_sha256"]
    ):
        raise ValueError("release manifest public sample selection mismatch")
    rq1_v2 = _mapping(payload.get("rq1_v2"), "rq1_v2")
    _exact_keys(rq1_v2, {"report_ref", "status"}, "rq1_v2")
    provenance = _mapping(payload.get("provenance"), "provenance")
    _exact_keys(provenance, {"sidecar"}, "provenance")
    sidecar = _mapping(provenance.get("sidecar"), "sidecar provenance")
    _exact_keys(
        sidecar,
        {
            "checkpoint_size",
            "dataset_sha256",
            "dataset_size",
            "fingerprint_sha256",
            "fingerprint_size",
            "plans_sha256",
            "plans_size",
            "recovery_receipt_sha256",
            "runtime_git_commit",
            "runtime_status",
            "runtime_version",
        },
        "sidecar provenance",
    )
    for field in (
        "dataset_sha256",
        "fingerprint_sha256",
        "plans_sha256",
        "recovery_receipt_sha256",
    ):
        _sha256(sidecar[field], f"sidecar.{field}")
    for field in ("checkpoint_size", "dataset_size", "fingerprint_size", "plans_size"):
        if isinstance(sidecar[field], bool) or not isinstance(sidecar[field], int) or sidecar[field] < 1:
            raise ValueError(f"release manifest sidecar.{field} is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", str(sidecar["runtime_git_commit"])):
        raise ValueError("release manifest sidecar runtime commit is invalid")
    return ReleaseManifest(
        MANIFEST_SCHEMA, models, tuple(comparisons), dict(policies),
        dict(public_sample), public_samples, public_sample_contract, dict(provenance), dict(rq1_v2), manifest_path, sha256_file(manifest_path),
    )


def _model_mapping(identity: RuntimeIdentity) -> dict[str, Any]:
    return {"model_id": identity.model_id, **dict(identity.data)}


def _projection_base(manifest: ReleaseManifest) -> dict[str, Any]:
    return {"release_manifest_sha256": manifest.digest}


def runtime_projection(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = load_release_manifest(path)
    live = manifest.comparison("live_demo")
    return {
        **_projection_base(manifest), "comparison": manifest.comparison_mapping("live_demo"),
        "imp": _model_mapping(manifest.model(live.left_model_id)),
        "nnunet": _model_mapping(manifest.model(live.right_model_id)),
        "paper_rq1_notice": "Live demo only; paper RQ1 uses Loop191 versus Loop192",
    }


def registry_projection(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = load_release_manifest(path)
    return {
        **_projection_base(manifest), "schema_version": "loop206.demo.models.v1",
        "control": _model_mapping(manifest.model("L206-control-s206")),
        "candidate": _model_mapping(manifest.model("L206-contour-channel-s206")),
        "prior_env": "IMP_LOOP206_PRIOR", "prior_receipt_env": "IMP_LOOP206_PRIOR_RECEIPT",
        "prior_receipt_sha256": None,
    }


def paper_projection(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = load_release_manifest(path)
    return {**_projection_base(manifest), "comparisons": [manifest.comparison_mapping("paper_rq1"), manifest.comparison_mapping("paper_rq2")]}


def deck_projection(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return paper_projection(path)


def fixed_cache_projection(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = load_release_manifest(path)
    selection = manifest.public_sample_selection
    dataset_index = _mapping(selection.get("dataset_index"), "dataset_index")
    live_config = _mapping(selection.get("live_config"), "live_config")
    fixed_cache = _mapping(selection.get("fixed_cache"), "fixed_cache")
    candidate = _mapping(fixed_cache.get("candidate"), "fixed_cache candidate")
    zero = _mapping(fixed_cache.get("zero"), "fixed_cache zero")
    return {
        **_projection_base(manifest),
        "ordered_universe_sha256": selection["ordered_universe_sha256"],
        "dataset_index": dict(dataset_index),
        "live_config": dict(live_config),
        "fixed_cache": {
            "schema_version": fixed_cache["schema_version"],
            "artifact_type": fixed_cache["artifact_type"],
            "count": fixed_cache["count"],
            "shape": list(fixed_cache["shape"]),
            "candidate": dict(candidate),
            "zero": dict(zero),
        },
    }


def launcher_projection(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = load_release_manifest(path)
    return {**runtime_projection(path), "sidecar": dict(_mapping(manifest.provenance.get("sidecar"), "sidecar provenance"))}


def sidecar_model_manifest_projection(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = load_release_manifest(path)
    live = manifest.comparison("live_demo")
    nnunet = manifest.model(live.right_model_id)
    sidecar = _mapping(manifest.provenance.get("sidecar"), "sidecar provenance")
    return {
        "schema_version": "imp.nnunet.model-manifest.v1",
        "release_manifest_sha256": manifest.digest,
        "model_id": nnunet.model_id,
        "runtime": {
            "distribution": "nnunetv2",
            "version": sidecar["runtime_version"],
            "recovered_git_commit": sidecar["runtime_git_commit"],
            "environment_status": sidecar["runtime_status"],
        },
        "input": {"layout": "CZYX", "channels": 3, "spacing": [999.0, 1.0, 1.0]},
        "artifacts": {
            "checkpoint_final.pth": {
                "sha256": nnunet.checkpoint_sha256,
                "size": sidecar["checkpoint_size"],
            },
            "dataset.json": {
                "sha256": sidecar["dataset_sha256"],
                "size": sidecar["dataset_size"],
            },
            "dataset_fingerprint.json": {
                "sha256": sidecar["fingerprint_sha256"],
                "size": sidecar["fingerprint_size"],
            },
            "plans.json": {
                "sha256": sidecar["plans_sha256"],
                "size": sidecar["plans_size"],
            },
        },
    }


def live_demo_receipt_projection(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return _projection_base(load_release_manifest(path))


def validate_projection(artifact: str | Path, manifest_path: str | Path = DEFAULT_MANIFEST) -> None:
    manifest = load_release_manifest(manifest_path)
    try:
        payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("release projection is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("release_manifest_sha256") != manifest.digest:
        raise ValueError("release projection manifest digest mismatch")
    name = Path(artifact).name
    if name == "model_registry.example.json":
        if payload != registry_projection(manifest.path):
            raise ValueError("release projection exact payload mismatch")
        return
    if name == "model_manifest.example.json":
        if payload != sidecar_model_manifest_projection(manifest.path):
            raise ValueError("release projection exact payload mismatch")
        return
    if name == "fixed_cache_projection.json":
        if payload != fixed_cache_projection(manifest.path):
            raise ValueError("release projection exact payload mismatch")
        return
    if name in {"runtime_identity.json", "runtime_projection.json"}:
        if payload != runtime_projection(manifest.path):
            raise ValueError("release projection exact payload mismatch")
        return
    if payload == runtime_projection(manifest.path):
        return
    if payload.get("schema_version") == "imp.evidence_registry.v1":
        from lesion_robustness.evidence_registry import validate_registry

        validate_registry(payload)
        return
    if name == "content.json":
        if payload.get("release_comparisons") != deck_projection(manifest.path)["comparisons"]:
            raise ValueError("release projection exact payload mismatch")
        return
    raise ValueError("release projection schema is unknown")
