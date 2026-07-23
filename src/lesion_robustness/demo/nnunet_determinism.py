"""Same-client interleaved nnU-Net determinism probe and evidence writer."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from lesion_robustness.demo.dual_live_protocol import rgb_sha256, validate_rgb
from lesion_robustness.demo.nnunet_client import NnUNetClient
from lesion_robustness.demo.runtime_identity import CHECKPOINT_SHA256, MODEL_ID
from lesion_robustness.release_manifest import DEFAULT_MANIFEST, load_release_manifest


SCHEMA_VERSION = "imp.nnunet.determinism.v1"
RUNTIME_MANIFEST_SCHEMA = "imp.nnunet.determinism.runtime.v1"
REQUIRED_STAGE_HASHES = (
    "preprocess",
    "workers",
    "cuda",
    "framework",
    "accumulation",
    "postprocess",
    "serialization",
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_CANONICAL_INPUTS = {
    "A": ("ISIC_0000050", "68fa0dd008c8ac3e301be0495c00ee2df0ece31216165da7c62e441d71b835aa"),
    "B": ("ISIC_0012690", "0282de65b80464fce23b16995187bb10a6e89b52858b9408ea8b58ac183f2e9e"),
}


class _ProbeResult(Protocol):
    input_sha256: str
    mask_sha256: str
    model_id: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class DeterminismProbe:
    input_hashes_a: Sequence[str]
    mask_hashes_a: Sequence[str]
    input_hashes_b: Sequence[str]
    mask_hashes_b: Sequence[str]
    runtime_identity: tuple[str, str]
    repetitions: int
    restart_count: int


@dataclass(frozen=True)
class DeterminismDiagnosis:
    probe: DeterminismProbe
    run_id: str
    runtime_manifest_sha256: str | None = None
    stage_hashes: Mapping[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class DeterminismBinding:
    """Caller-provided binding description; never sufficient for promotion."""

    role_a: str
    sample_id_a: str
    rgb_sha256_a: str
    role_b: str
    sample_id_b: str
    rgb_sha256_b: str
    runtime_manifest_path: Path
    stage_artifact_paths: Mapping[str, Path]


@dataclass(frozen=True)
class _VerifiedDeterminismBinding:
    probe_snapshot: tuple[object, ...]
    release_manifest_path: Path
    release_manifest_sha256: str
    dataset_index_path: Path
    dataset_index_sha256: str
    runtime_manifest_path: Path
    runtime_manifest_sha256: str
    stage_artifact_paths: Mapping[str, Path]
    stage_hashes: Mapping[str, str]
    role_a: str = "A"
    sample_id_a: str = _CANONICAL_INPUTS["A"][0]
    rgb_sha256_a: str = _CANONICAL_INPUTS["A"][1]
    role_b: str = "B"
    sample_id_b: str = _CANONICAL_INPUTS["B"][0]
    rgb_sha256_b: str = _CANONICAL_INPUTS["B"][1]


@dataclass(frozen=True)
class _IssuedBindingAuthority:
    token: _VerifiedDeterminismBinding
    token_snapshot: tuple[object, ...]
    root: Path
    probe_snapshot: tuple[object, ...]
    release_manifest_path: Path
    release_manifest_sha256: str
    dataset_index_path: Path
    dataset_index_sha256: str
    runtime_manifest_path: Path
    runtime_manifest_sha256: str
    stage_artifact_paths: tuple[tuple[str, Path], ...]
    stage_hashes: tuple[tuple[str, str], ...]
    input_projection: tuple[tuple[str, str], ...]


_ISSUED_BINDINGS: dict[int, _IssuedBindingAuthority] = {}
_ISSUED_PROBES: dict[
    int,
    tuple[DeterminismProbe, tuple[object, ...], Mapping[str, str]],
] = {}


def _request_id(label: str, repetition: int) -> str:
    return hashlib.sha256(f"{label}:{repetition}".encode("ascii")).hexdigest()[:32]


def run_interleaved_probe(
    client: NnUNetClient,
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    repetitions: int = 3,
) -> DeterminismProbe:
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("determinism repetitions must be positive")

    a = validate_rgb(sample_a).copy(order="C")
    b = validate_rgb(sample_b).copy(order="C")
    input_hashes_a: list[str] = []
    mask_hashes_a: list[str] = []
    input_hashes_b: list[str] = []
    mask_hashes_b: list[str] = []
    identities: list[tuple[str, str]] = []

    for repetition in range(repetitions):
        for label, image, input_hashes, mask_hashes in (
            ("a", a, input_hashes_a, mask_hashes_a),
            ("b", b, input_hashes_b, mask_hashes_b),
        ):
            result = client.predict(_request_id(label, repetition), image.copy(order="C"))
            input_hashes.append(result.input_sha256)
            mask_hashes.append(result.mask_sha256)
            identities.append((result.model_id, result.checkpoint_sha256))

    identity = identities[0] if identities and len(set(identities)) == 1 else ("", "")
    probe = DeterminismProbe(
        tuple(input_hashes_a),
        tuple(mask_hashes_a),
        tuple(input_hashes_b),
        tuple(mask_hashes_b),
        identity,
        repetitions,
        0,
    )
    _ISSUED_PROBES[id(probe)] = (
        probe,
        _probe_snapshot(probe),
        {"A": rgb_sha256(a), "B": rgb_sha256(b)},
    )
    return probe


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _failed(probe: DeterminismProbe) -> bool:
    count = probe.repetitions
    identity = probe.runtime_identity
    values = (
        probe.input_hashes_a,
        probe.mask_hashes_a,
        probe.input_hashes_b,
        probe.mask_hashes_b,
    )
    try:
        lengths_valid = all(len(value) == count for value in values)
    except TypeError:
        return True
    return (
        isinstance(count, bool)
        or count != 3
        or probe.restart_count != 0
        or not lengths_valid
        or any(not _is_sha256(digest) for value in values for digest in value)
        or not isinstance(identity, tuple)
        or len(identity) != 2
        or identity != (MODEL_ID, CHECKPOINT_SHA256)
        or not _is_sha256(identity[1])
        or any(len(set(value)) != 1 for value in values)
        or probe.input_hashes_a[0] == probe.input_hashes_b[0]
        or probe.mask_hashes_a[0] == probe.mask_hashes_b[0]
    )


def validate_determinism_probe(probe: DeterminismProbe) -> None:
    if not isinstance(probe, DeterminismProbe) or _failed(probe):
        raise ValueError("determinism gate failed")


def diagnose_determinism(
    probe: DeterminismProbe,
    *,
    run_id: str = "",
    runtime_manifest_sha256: str | None = None,
    stage_hashes: Mapping[str, str | None] | None = None,
) -> DeterminismDiagnosis:
    return DeterminismDiagnosis(
        probe,
        run_id,
        runtime_manifest_sha256,
        dict(stage_hashes or {}),
    )


def _first_divergent_stage(probe: DeterminismProbe) -> str | None:
    values = (
        ("a_input", probe.input_hashes_a),
        ("a_mask", probe.mask_hashes_a),
        ("b_input", probe.input_hashes_b),
        ("b_mask", probe.mask_hashes_b),
    )
    try:
        lengths_bad = any(len(value) != probe.repetitions for _, value in values)
    except TypeError:
        lengths_bad = True
    if probe.repetitions != 3 or probe.restart_count != 0 or lengths_bad:
        return "gate"
    if probe.runtime_identity != (MODEL_ID, CHECKPOINT_SHA256):
        return "runtime_identity"
    for label, value in values:
        if any(not _is_sha256(digest) for digest in value) or len(set(value)) != 1:
            return label
    if probe.input_hashes_a[0] == probe.input_hashes_b[0]:
        return "input_collision"
    if probe.mask_hashes_a[0] == probe.mask_hashes_b[0]:
        return "mask_collision"
    return None


def _binding_token_snapshot(
    binding: object,
) -> tuple[object, ...] | None:
    if type(binding) is not _VerifiedDeterminismBinding:
        return None
    try:
        if (
            not isinstance(binding.stage_artifact_paths, Mapping)
            or set(binding.stage_artifact_paths) != set(REQUIRED_STAGE_HASHES)
            or not isinstance(binding.stage_hashes, Mapping)
            or set(binding.stage_hashes) != set(REQUIRED_STAGE_HASHES)
        ):
            return None
        return (
            binding.probe_snapshot,
            binding.release_manifest_path,
            binding.release_manifest_sha256,
            binding.dataset_index_path,
            binding.dataset_index_sha256,
            binding.runtime_manifest_path,
            binding.runtime_manifest_sha256,
            tuple(
                (name, binding.stage_artifact_paths[name])
                for name in REQUIRED_STAGE_HASHES
            ),
            tuple((name, binding.stage_hashes[name]) for name in REQUIRED_STAGE_HASHES),
            binding.role_a,
            binding.sample_id_a,
            binding.rgb_sha256_a,
            binding.role_b,
            binding.sample_id_b,
            binding.rgb_sha256_b,
        )
    except Exception:
        return None


def _issued_binding_authority(
    binding: object | None,
) -> _IssuedBindingAuthority | None:
    if type(binding) is not _VerifiedDeterminismBinding:
        return None
    authority = _ISSUED_BINDINGS.get(id(binding))
    if (
        authority is None
        or authority.token is not binding
        or _binding_token_snapshot(binding) != authority.token_snapshot
    ):
        return None
    return authority


def _binding_failure(
    probe: DeterminismProbe,
    input_binding: object | None,
    authority: _IssuedBindingAuthority | None,
    runtime_manifest_sha256: str | None,
    stage_hashes: Mapping[str, str | None],
) -> str | None:
    if not _is_sha256(runtime_manifest_sha256):
        return "runtime_manifest"
    if set(stage_hashes) != set(REQUIRED_STAGE_HASHES):
        return next(
            (name for name in REQUIRED_STAGE_HASHES if name not in stage_hashes),
            "stage_hashes",
        )
    invalid_stage = next(
        (name for name in REQUIRED_STAGE_HASHES if not _is_sha256(stage_hashes[name])),
        None,
    )
    if invalid_stage is not None:
        return invalid_stage
    if input_binding is None:
        return "input_binding"
    if authority is None:
        return "artifact_binding"
    if authority.probe_snapshot != _probe_snapshot(probe):
        return "artifact_binding"
    if (
        authority.runtime_manifest_sha256 != runtime_manifest_sha256
        or dict(authority.stage_hashes) != dict(stage_hashes)
    ):
        return "artifact_binding"
    try:
        release_path = _contained_regular_file(
            authority.root, authority.release_manifest_path
        )
        dataset_path = _contained_regular_file(
            authority.root, authority.dataset_index_path
        )
        runtime_root = authority.root / "demo_runtime"
        runtime_path = _contained_regular_file(
            runtime_root, authority.runtime_manifest_path
        )
        if _sha256_path(release_path) != authority.release_manifest_sha256:
            return "artifact_binding"
        if _sha256_path(dataset_path) != authority.dataset_index_sha256:
            return "artifact_binding"
        if _sha256_path(runtime_path) != runtime_manifest_sha256:
            return "artifact_binding"
        artifact_paths = dict(authority.stage_artifact_paths)
        for name in REQUIRED_STAGE_HASHES:
            artifact = _contained_regular_file(runtime_root, artifact_paths[name])
            if _sha256_path(artifact) != stage_hashes[name]:
                return "artifact_binding"
    except (OSError, TypeError, ValueError):
        return "artifact_binding"
    return None


def _binding_projection(
    authority: _IssuedBindingAuthority | None,
) -> Mapping[str, object] | None:
    if authority is None:
        return None
    return dict(authority.input_projection)


def _canonical(payload: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("determinism diagnostic is not canonical JSON") from exc


def _is_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & _REPARSE_ATTRIBUTE)


def _probe_snapshot(probe: DeterminismProbe) -> tuple[object, ...]:
    return (
        tuple(probe.input_hashes_a),
        tuple(probe.mask_hashes_a),
        tuple(probe.input_hashes_b),
        tuple(probe.mask_hashes_b),
        probe.runtime_identity,
        probe.repetitions,
        probe.restart_count,
    )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contained_regular_file(root: Path, path: Path) -> Path:
    root = root.resolve(strict=True)
    raw = Path(path)
    if ".." in raw.parts:
        raise ValueError("determinism artifact path escapes authority")
    candidate = raw if raw.is_absolute() else root / raw
    for item in (candidate, *candidate.parents):
        if item == root.parent:
            break
        if item.exists() and _is_reparse(item):
            raise ValueError("determinism artifact reparse path is forbidden")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("determinism artifact path escapes authority") from exc
    if not resolved.is_file():
        raise ValueError("determinism artifact must be a regular file")
    return resolved


def create_verified_determinism_binding(
    probe: DeterminismProbe,
    *,
    runtime_manifest_path: Path,
) -> object:
    """Validate repository authorities, then issue an opaque writer binding."""
    validate_determinism_probe(probe)
    issued_probe = _ISSUED_PROBES.get(id(probe))
    if (
        issued_probe is None
        or issued_probe[0] is not probe
        or issued_probe[1] != _probe_snapshot(probe)
        or dict(issued_probe[2])
        != {role: _CANONICAL_INPUTS[role][1] for role in ("A", "B")}
    ):
        raise ValueError("probe is not bound to canonical input bytes")
    root = DEFAULT_MANIFEST.resolve(strict=True).parents[1]
    release_path = _contained_regular_file(root, DEFAULT_MANIFEST)
    release = load_release_manifest(release_path)
    roles = release.public_sample_contract.roles
    if release.public_sample_contract.state != "verified" or any(
        (
            roles[role].sample_id,
            roles[role].sha256_rgb,
        )
        != _CANONICAL_INPUTS[role]
        for role in ("A", "B")
    ):
        raise ValueError("canonical public sample contract mismatch")
    if (
        tuple(probe.input_hashes_a) != (_CANONICAL_INPUTS["A"][1],) * 3
        or tuple(probe.input_hashes_b) != (_CANONICAL_INPUTS["B"][1],) * 3
        or probe.runtime_identity != (MODEL_ID, CHECKPOINT_SHA256)
        or release.model(MODEL_ID).checkpoint_sha256 != CHECKPOINT_SHA256
    ):
        raise ValueError("probe is not bound to canonical inputs and runtime")

    selection = release.public_sample_selection["dataset_index"]
    dataset_path = _contained_regular_file(root, Path(selection["path"]))
    dataset_digest = _sha256_path(dataset_path)
    if dataset_digest != selection["sha256"]:
        raise ValueError("dataset index binding mismatch")

    runtime_path = _contained_regular_file(root / "demo_runtime", runtime_manifest_path)
    runtime_bytes = runtime_path.read_bytes()
    try:
        runtime = json.loads(runtime_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runtime manifest is invalid") from exc
    if runtime_bytes != _canonical(runtime):
        raise ValueError("runtime manifest is not canonical")
    expected_keys = {
        "schema_version", "release_manifest_sha256", "dataset_index_sha256",
        "model_id", "checkpoint_sha256", "samples", "stage_artifacts",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != expected_keys:
        raise ValueError("runtime manifest schema mismatch")
    samples = runtime.get("samples")
    if (
        runtime["schema_version"] != RUNTIME_MANIFEST_SCHEMA
        or runtime["release_manifest_sha256"] != release.digest
        or runtime["dataset_index_sha256"] != dataset_digest
        or runtime["model_id"] != MODEL_ID
        or runtime["checkpoint_sha256"] != CHECKPOINT_SHA256
        or samples != {
            role: {"sample_id": _CANONICAL_INPUTS[role][0], "rgb_sha256": _CANONICAL_INPUTS[role][1]}
            for role in ("A", "B")
        }
    ):
        raise ValueError("runtime manifest authority mismatch")
    artifacts = runtime.get("stage_artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(REQUIRED_STAGE_HASHES):
        raise ValueError("runtime manifest stage schema mismatch")
    artifact_paths: dict[str, Path] = {}
    stage_hashes: dict[str, str] = {}
    for name in REQUIRED_STAGE_HASHES:
        entry = artifacts[name]
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError("runtime manifest stage schema mismatch")
        path_value = entry["path"]
        digest = entry["sha256"]
        if not isinstance(path_value, str) or not _is_sha256(digest):
            raise ValueError("runtime manifest stage binding mismatch")
        artifact = _contained_regular_file(root / "demo_runtime", Path(path_value))
        if _sha256_path(artifact) != digest:
            raise ValueError("runtime manifest stage binding mismatch")
        artifact_paths[name] = artifact
        stage_hashes[name] = digest
    binding = _VerifiedDeterminismBinding(
        _probe_snapshot(probe),
        release_path,
        release.digest,
        dataset_path,
        dataset_digest,
        runtime_path,
        hashlib.sha256(runtime_bytes).hexdigest(),
        artifact_paths,
        stage_hashes,
    )
    token_snapshot = _binding_token_snapshot(binding)
    if token_snapshot is None:
        raise ValueError("verified binding token is invalid")
    authority = _IssuedBindingAuthority(
        binding,
        token_snapshot,
        root,
        _probe_snapshot(probe),
        release_path,
        release.digest,
        dataset_path,
        dataset_digest,
        runtime_path,
        hashlib.sha256(runtime_bytes).hexdigest(),
        tuple((name, artifact_paths[name]) for name in REQUIRED_STAGE_HASHES),
        tuple((name, stage_hashes[name]) for name in REQUIRED_STAGE_HASHES),
        (
            ("role_a", "A"),
            ("sample_id_a", _CANONICAL_INPUTS["A"][0]),
            ("rgb_sha256_a", _CANONICAL_INPUTS["A"][1]),
            ("role_b", "B"),
            ("sample_id_b", _CANONICAL_INPUTS["B"][0]),
            ("rgb_sha256_b", _CANONICAL_INPUTS["B"][1]),
        ),
    )
    _ISSUED_BINDINGS[id(binding)] = authority
    return binding


def _validated_output_path(path: Path, run_id: str) -> Path:
    raw = Path(path)
    if ".." in raw.parts or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("determinism diagnostic path is invalid")
    absolute = Path(os.path.abspath(raw))
    if (
        absolute.name != "diagnostic.json"
        or absolute.parent.name != run_id
        or absolute.parent.parent.name != SCHEMA_VERSION
        or absolute.parents[2].name != "diagnostics"
        or absolute.parents[3].name != "demo_runtime"
    ):
        raise ValueError("determinism diagnostic path is invalid")
    base = absolute.parents[2]
    if not base.is_dir():
        raise ValueError("determinism diagnostic base directory is missing")
    for ancestor in (absolute, *absolute.parents):
        if ancestor.exists() and _is_reparse(ancestor):
            raise ValueError("determinism diagnostic symlink or reparse path is forbidden")
    for directory in (absolute.parent.parent, absolute.parent):
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass
        if not directory.is_dir() or _is_reparse(directory):
            raise ValueError("determinism diagnostic symlink or reparse path is forbidden")
    return absolute


def _write_exclusive(path: Path, encoded: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        if _is_reparse(path):
            raise ValueError("determinism diagnostic symlink or reparse path is forbidden")
        if path.read_bytes() != encoded:
            raise ValueError("determinism diagnostic byte drift")
        return
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("determinism diagnostic write failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_determinism_diagnostic(
    path: Path,
    probe: DeterminismProbe | DeterminismDiagnosis,
    *,
    run_id: str | None = None,
    runtime_manifest_sha256: str | None = None,
    stage_hashes: Mapping[str, str | None] | None = None,
    input_binding: object | None = None,
) -> str:
    """Write one canonical diagnostic under the schema/run-ID tail."""
    if isinstance(probe, DeterminismDiagnosis):
        diagnosis = probe
        probe = diagnosis.probe
        run_id = diagnosis.run_id if run_id is None else run_id
        if runtime_manifest_sha256 is None:
            runtime_manifest_sha256 = diagnosis.runtime_manifest_sha256
        if stage_hashes is None:
            stage_hashes = diagnosis.stage_hashes
    if run_id is None:
        raise ValueError("determinism diagnostic run_id is required")
    output = _validated_output_path(Path(path), run_id)
    stages = dict(stage_hashes or {})
    probe_failure = _first_divergent_stage(probe)
    authority = _issued_binding_authority(input_binding)
    binding_failure = _binding_failure(
        probe,
        input_binding,
        authority,
        runtime_manifest_sha256,
        stages,
    )
    eligible = probe_failure is None and binding_failure is None
    payload: dict[str, object] = {
        "a_input_hashes": list(probe.input_hashes_a),
        "a_mask_hashes": list(probe.mask_hashes_a),
        "b_input_hashes": list(probe.input_hashes_b),
        "b_mask_hashes": list(probe.mask_hashes_b),
        "claim_status": "eligible" if eligible else "unpromoted",
        "first_divergent_stage": probe_failure or binding_failure,
        "gate_status": "passed" if eligible else "failed",
        "input_binding": _binding_projection(authority),
        "repetitions": probe.repetitions,
        "restart_count": probe.restart_count,
        "run_id": run_id,
        "runtime_identity": list(probe.runtime_identity),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "schema_version": SCHEMA_VERSION,
        "stage_hashes": stages,
    }
    encoded = _canonical(payload)
    _write_exclusive(output, encoded)
    return hashlib.sha256(encoded).hexdigest()


def missing_runtime_prerequisites(
    root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    tool_finder: Callable[[str], str | None] = shutil.which,
) -> tuple[str, ...]:
    """Inventory private runtime inputs without binding ports or starting services."""
    env = os.environ if environ is None else environ
    root = Path(root)
    missing: list[str] = []
    if not (root / ".venv-win/Scripts/python.exe").is_file():
        missing.append("python")
    if tool_finder("docker") is None or tool_finder("nvidia-smi") is None:
        missing.append("docker_cuda")
    sidecar_bundle = root / "demo_runtime/nnunet/recovered-container-final2"
    if not sidecar_bundle.is_dir() or not (sidecar_bundle / "recovery_receipt.json").is_file():
        missing.append("sidecar_bundle")
    checkpoint_defaults = (
        "runs/loop206-control-train-screen-pilot20-checkpoints/best.pt",
        "runs/loop206-contour-channel-train-screen-pilot20-checkpoints/best.pt",
    )
    checkpoint_env = (
        env.get("IMP_LOOP206_CONTROL_CHECKPOINT"),
        env.get("IMP_LOOP206_CANDIDATE_CHECKPOINT"),
    )
    for index, default in enumerate(checkpoint_defaults):
        selected = Path(checkpoint_env[index] or default)
        candidate = selected if selected.is_absolute() else root / selected
        if not candidate.is_file():
            missing.append(
                "imp_control_checkpoint" if index == 0 else "imp_candidate_checkpoint"
            )
    required = {
        "dataset_index": "demo_runtime/loop206_dataset_index.json",
        "candidate_cache_manifest": ".artifacts/preprocessing_search/loop206_leac_drlse/pilot_cache_v2_candidate/manifest.json",
        "zero_control_cache_manifest": ".artifacts/preprocessing_search/loop206_leac_drlse/pilot_cache_v2_zero_control/manifest.json",
        "release_manifest": "release/imp_release_manifest.json",
        "evidence_registry": "demo/data/evidence_registry.json",
    }
    for label, relative in required.items():
        if not (root / relative).is_file():
            missing.append(label)
    clean_v3 = Path(env.get("IMP_CLEAN_V3_MANIFEST", "data/splits/clean_v3_manifest.csv"))
    clean_v3_path = clean_v3 if clean_v3.is_absolute() else root / clean_v3
    if not clean_v3_path.is_file():
        missing.append("clean_v3_manifest")
    return tuple(missing)
