from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import lesion_robustness.demo.nnunet_determinism as determinism_module
from lesion_robustness.demo.nnunet_determinism import (
    create_verified_determinism_binding,
    DeterminismBinding,
    DeterminismProbe,
    REQUIRED_STAGE_HASHES,
    run_interleaved_probe,
    validate_determinism_probe,
    write_determinism_diagnostic,
)
from lesion_robustness.demo.runtime_identity import CHECKPOINT_SHA256, MODEL_ID


SAMPLE_A = np.zeros((2, 2, 3), dtype=np.uint8)
SAMPLE_B = np.ones((2, 2, 3), dtype=np.uint8)
RUN_ID = "20260723t000000000z"
ROOT = Path(__file__).resolve().parents[2]
DATASET_INDEX = ROOT / "demo_runtime/loop206_dataset_index.json"
requires_external_runtime_assets = pytest.mark.skipif(
    not DATASET_INDEX.is_file(),
    reason="external runtime assets; local release gate required",
)
CANONICAL_RGB_A = "68fa0dd008c8ac3e301be0495c00ee2df0ece31216165da7c62e441d71b835aa"
CANONICAL_RGB_B = "0282de65b80464fce23b16995187bb10a6e89b52858b9408ea8b58ac183f2e9e"


@dataclass(frozen=True)
class FakeResult:
    input_sha256: str
    mask_sha256: str
    model_id: str = MODEL_ID
    checkpoint_sha256: str = CHECKPOINT_SHA256


class FakeClient:
    def __init__(
        self,
        *,
        drift_mask: str | None = None,
        drift_input: str | None = None,
        drift_identity: bool = False,
    ) -> None:
        self.drift_mask = drift_mask
        self.drift_input = drift_input
        self.drift_identity = drift_identity
        self.calls: list[tuple[str, bytes]] = []

    def predict(self, request_id: str, image: np.ndarray) -> FakeResult:
        contiguous = np.ascontiguousarray(image)
        label = "a" if int(contiguous.sum()) == 0 else "b"
        self.calls.append((request_id, contiguous.tobytes()))
        sequence = len(self.calls)
        input_bytes = contiguous.tobytes() + (
            bytes([sequence]) if self.drift_input == label else b""
        )
        mask_bytes = bytes([int(contiguous.sum()) % 256]) + (
            bytes([sequence]) if self.drift_mask == label else b"\x00"
        )
        model_id = "wrong-model" if self.drift_identity and sequence == 2 else MODEL_ID
        return FakeResult(
            hashlib.sha256(input_bytes).hexdigest(),
            hashlib.sha256(mask_bytes).hexdigest(),
            model_id,
            CHECKPOINT_SHA256,
        )


def _stage_hashes() -> dict[str, str]:
    return {name: hashlib.sha256(name.encode("ascii")).hexdigest() for name in REQUIRED_STAGE_HASHES}


def _path(root: Path, run_id: str = RUN_ID) -> Path:
    base = root / "demo_runtime" / "diagnostics"
    base.mkdir(parents=True, exist_ok=True)
    return base / "imp.nnunet.determinism.v1" / run_id / "diagnostic.json"


def _arbitrary_binding(root: Path) -> tuple[DeterminismBinding, str, dict[str, str]]:
    runtime = root / "runtime.json"
    runtime.write_text('{"schema_version":"caller.controlled"}\n', encoding="ascii")
    artifacts: dict[str, Path] = {}
    for name in REQUIRED_STAGE_HASHES:
        artifact = root / f"{name}.bin"
        artifact.write_bytes(name.encode("ascii"))
        artifacts[name] = artifact
    runtime_digest = hashlib.sha256(runtime.read_bytes()).hexdigest()
    stage_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in artifacts.items()
    }
    return (
        DeterminismBinding(
            role_a="A",
            sample_id_a="ISIC_0000050",
            rgb_sha256_a="68fa0dd008c8ac3e301be0495c00ee2df0ece31216165da7c62e441d71b835aa",
            role_b="B",
            sample_id_b="ISIC_0012690",
            rgb_sha256_b="0282de65b80464fce23b16995187bb10a6e89b52858b9408ea8b58ac183f2e9e",
            runtime_manifest_path=runtime,
            stage_artifact_paths=artifacts,
        ),
        runtime_digest,
        stage_hashes,
    )


def _verified_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DeterminismProbe, object, str, dict[str, str], dict[str, Path], Path]:
    authority_root = tmp_path / "authority"
    release_path = authority_root / "release" / "imp_release_manifest.json"
    release_path.parent.mkdir(parents=True)
    release_bytes = (ROOT / "release" / "imp_release_manifest.json").read_bytes()
    release_path.write_bytes(release_bytes)
    release_payload = json.loads(release_bytes)

    dataset_relative = Path(
        release_payload["public_sample_selection"]["dataset_index"]["path"]
    )
    dataset_path = authority_root / dataset_relative
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_bytes((ROOT / dataset_relative).read_bytes())
    dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    runtime_root = authority_root / "demo_runtime"
    runtime_dir = runtime_root / "issued"
    runtime_dir.mkdir(parents=True)
    artifact_paths: dict[str, Path] = {}
    stage_hashes: dict[str, str] = {}
    stage_artifacts: dict[str, dict[str, str]] = {}
    for name in REQUIRED_STAGE_HASHES:
        artifact = runtime_dir / f"{name}.bin"
        artifact.write_bytes(f"issued:{name}".encode("ascii"))
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        artifact_paths[name] = artifact
        stage_hashes[name] = digest
        stage_artifacts[name] = {
            "path": artifact.relative_to(runtime_root).as_posix(),
            "sha256": digest,
        }

    runtime_payload = {
        "schema_version": "imp.nnunet.determinism.runtime.v1",
        "release_manifest_sha256": hashlib.sha256(release_bytes).hexdigest(),
        "dataset_index_sha256": dataset_digest,
        "model_id": MODEL_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "samples": {
            "A": {"sample_id": "ISIC_0000050", "rgb_sha256": CANONICAL_RGB_A},
            "B": {"sample_id": "ISIC_0012690", "rgb_sha256": CANONICAL_RGB_B},
        },
        "stage_artifacts": stage_artifacts,
    }
    runtime_path = runtime_dir / "runtime.json"
    runtime_path.write_bytes(
        (
            json.dumps(
                runtime_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("ascii")
    )
    runtime_digest = hashlib.sha256(runtime_path.read_bytes()).hexdigest()

    probe = DeterminismProbe(
        (CANONICAL_RGB_A,) * 3,
        ("a" * 64,) * 3,
        (CANONICAL_RGB_B,) * 3,
        ("b" * 64,) * 3,
        (MODEL_ID, CHECKPOINT_SHA256),
        3,
        0,
    )
    determinism_module._ISSUED_PROBES[id(probe)] = (
        probe,
        determinism_module._probe_snapshot(probe),
        {"A": CANONICAL_RGB_A, "B": CANONICAL_RGB_B},
    )
    monkeypatch.setattr(determinism_module, "DEFAULT_MANIFEST", release_path)
    binding = create_verified_determinism_binding(
        probe,
        runtime_manifest_path=Path("issued/runtime.json"),
    )
    return probe, binding, runtime_digest, stage_hashes, artifact_paths, runtime_path


def test_probe_public_constructor_is_exactly_frozen_seven_fields() -> None:
    assert tuple(field.name for field in fields(DeterminismProbe)) == (
        "input_hashes_a",
        "mask_hashes_a",
        "input_hashes_b",
        "mask_hashes_b",
        "runtime_identity",
        "repetitions",
        "restart_count",
    )


def test_interleaved_probe_uses_exact_request_ids_and_one_client_order() -> None:
    client = FakeClient()
    probe = run_interleaved_probe(client, SAMPLE_A, SAMPLE_B, repetitions=3)

    assert [request_id for request_id, _ in client.calls] == [
        hashlib.sha256(f"{label}:{repetition}".encode("ascii")).hexdigest()[:32]
        for repetition in range(3)
        for label in ("a", "b")
    ]
    assert [payload for _, payload in client.calls] == [
        payload
        for _ in range(3)
        for payload in (SAMPLE_A.tobytes(), SAMPLE_B.tobytes())
    ]
    assert probe.input_hashes_a == (probe.input_hashes_a[0],) * 3
    assert probe.mask_hashes_a == (probe.mask_hashes_a[0],) * 3
    assert probe.input_hashes_b == (probe.input_hashes_b[0],) * 3
    assert probe.mask_hashes_b == (probe.mask_hashes_b[0],) * 3
    assert probe.input_hashes_a[0] != probe.input_hashes_b[0]
    assert probe.mask_hashes_a[0] != probe.mask_hashes_b[0]
    assert probe.runtime_identity == (MODEL_ID, CHECKPOINT_SHA256)
    assert probe.restart_count == 0
    validate_determinism_probe(probe)


@pytest.mark.parametrize("label", ["a", "b"])
def test_interleaved_probe_rejects_input_drift(label: str) -> None:
    with pytest.raises(ValueError, match="determinism gate failed"):
        validate_determinism_probe(
            run_interleaved_probe(FakeClient(drift_input=label), SAMPLE_A, SAMPLE_B)
        )


@pytest.mark.parametrize("label", ["a", "b"])
def test_interleaved_probe_rejects_mask_drift(label: str) -> None:
    with pytest.raises(ValueError, match="determinism gate failed"):
        validate_determinism_probe(
            run_interleaved_probe(FakeClient(drift_mask=label), SAMPLE_A, SAMPLE_B)
        )


def test_run_marks_mid_probe_identity_drift_invalid() -> None:
    probe = run_interleaved_probe(FakeClient(drift_identity=True), SAMPLE_A, SAMPLE_B)
    assert probe.runtime_identity != (MODEL_ID, CHECKPOINT_SHA256)
    with pytest.raises(ValueError, match="determinism gate failed"):
        validate_determinism_probe(probe)


def test_validator_rejects_stable_wrong_runtime_identity() -> None:
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    wrong = replace(probe, runtime_identity=("wrong-model", "f" * 64))
    with pytest.raises(ValueError, match="determinism gate failed"):
        validate_determinism_probe(wrong)


@pytest.mark.parametrize(
    "mutation",
    [
        {"repetitions": 2},
        {"restart_count": 1},
        {"input_hashes_a": ("A" * 64,) * 3},
        {"mask_hashes_b": ("f" * 63,) * 3},
    ],
)
def test_validator_rejects_wrong_counts_restart_and_hash_format(mutation: dict[str, object]) -> None:
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    with pytest.raises(ValueError, match="determinism gate failed"):
        validate_determinism_probe(replace(probe, **mutation))


def test_b_only_drift_is_named_in_diagnostic(tmp_path: Path) -> None:
    probe = run_interleaved_probe(FakeClient(drift_mask="b"), SAMPLE_A, SAMPLE_B)
    path = _path(tmp_path)
    write_determinism_diagnostic(
        path,
        probe,
        run_id=RUN_ID,
        runtime_manifest_sha256="a" * 64,
        stage_hashes=_stage_hashes(),
    )
    packet = json.loads(path.read_text(encoding="ascii"))
    assert packet["first_divergent_stage"] == "b_mask"
    assert packet["gate_status"] == "failed"
    assert packet["claim_status"] == "unpromoted"


def test_diagnostic_writer_requires_all_bindings_for_eligibility(tmp_path: Path) -> None:
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    incomplete_path = _path(tmp_path, "incomplete")
    write_determinism_diagnostic(incomplete_path, probe, run_id="incomplete")
    incomplete = json.loads(incomplete_path.read_text(encoding="ascii"))
    assert incomplete["gate_status"] == "failed"
    assert incomplete["claim_status"] == "unpromoted"

    path = _path(tmp_path)
    digest = write_determinism_diagnostic(
        path,
        probe,
        run_id=RUN_ID,
        runtime_manifest_sha256="a" * 64,
        stage_hashes=_stage_hashes(),
    )
    raw = path.read_bytes()
    packet = json.loads(raw)
    assert raw.endswith(b"\n")
    assert digest == hashlib.sha256(raw).hexdigest()
    assert packet["claim_status"] == "unpromoted"
    assert packet["gate_status"] == "failed"
    assert packet["a_input_hashes"] == list(probe.input_hashes_a)
    assert packet["a_mask_hashes"] == list(probe.mask_hashes_a)
    assert packet["b_input_hashes"] == list(probe.input_hashes_b)
    assert packet["b_mask_hashes"] == list(probe.mask_hashes_b)
    assert write_determinism_diagnostic(
        path,
        probe,
        run_id=RUN_ID,
        runtime_manifest_sha256="a" * 64,
        stage_hashes=_stage_hashes(),
    ) == digest


def test_synthetic_arrays_and_placeholder_artifact_hashes_are_unpromoted(tmp_path: Path) -> None:
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    path = _path(tmp_path, "synthetic")
    write_determinism_diagnostic(
        path,
        probe,
        run_id="synthetic",
        runtime_manifest_sha256="a" * 64,
        stage_hashes=_stage_hashes(),
    )
    packet = json.loads(path.read_text(encoding="ascii"))
    assert packet["claim_status"] == "unpromoted"
    assert packet["first_divergent_stage"] == "input_binding"


@pytest.mark.parametrize("binding_kind", ["public_constructor", "mapping"])
def test_caller_forged_binding_and_probe_attribute_cannot_become_eligible(
    tmp_path: Path, binding_kind: str
) -> None:
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    object.__setattr__(
        probe,
        "_observed_rgb_sha256",
        {
            "A": "68fa0dd008c8ac3e301be0495c00ee2df0ece31216165da7c62e441d71b835aa",
            "B": "0282de65b80464fce23b16995187bb10a6e89b52858b9408ea8b58ac183f2e9e",
        },
    )
    binding, runtime_digest, stages = _arbitrary_binding(tmp_path)
    supplied: object = binding if binding_kind == "public_constructor" else {
        field.name: getattr(binding, field.name) for field in fields(binding)
    }
    path = _path(tmp_path, binding_kind)

    write_determinism_diagnostic(
        path,
        probe,
        run_id=binding_kind,
        runtime_manifest_sha256=runtime_digest,
        stage_hashes=stages,
        input_binding=supplied,
    )

    packet = json.loads(path.read_text(encoding="ascii"))
    assert packet["gate_status"] == "failed"
    assert packet["claim_status"] == "unpromoted"
    assert packet["first_divergent_stage"] == "artifact_binding"


def test_caller_constructed_probe_cannot_enter_verified_binding_factory() -> None:
    forged = DeterminismProbe(
        ("68fa0dd008c8ac3e301be0495c00ee2df0ece31216165da7c62e441d71b835aa",) * 3,
        ("a" * 64,) * 3,
        ("0282de65b80464fce23b16995187bb10a6e89b52858b9408ea8b58ac183f2e9e",) * 3,
        ("b" * 64,) * 3,
        (MODEL_ID, CHECKPOINT_SHA256),
        3,
        0,
    )

    with pytest.raises(ValueError, match="canonical input bytes"):
        create_verified_determinism_binding(
            forged,
            runtime_manifest_path=Path("forged-runtime.json"),
        )


@requires_external_runtime_assets
def test_untampered_verified_binding_uses_issued_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe, binding, runtime_digest, stages, _artifacts, _runtime = _verified_binding(
        tmp_path, monkeypatch
    )
    path = _path(tmp_path, "issued-authority")

    write_determinism_diagnostic(
        path,
        probe,
        run_id="issued-authority",
        runtime_manifest_sha256=runtime_digest,
        stage_hashes=stages,
        input_binding=binding,
    )

    packet = json.loads(path.read_text(encoding="ascii"))
    assert packet["claim_status"] == "eligible"
    assert packet["input_binding"] == {
        "role_a": "A",
        "sample_id_a": "ISIC_0000050",
        "rgb_sha256_a": CANONICAL_RGB_A,
        "role_b": "B",
        "sample_id_b": "ISIC_0012690",
        "rgb_sha256_b": CANONICAL_RGB_B,
    }


@pytest.mark.parametrize(
    "tamper_kind",
    [
        "stage_path_mapping",
        "runtime_path",
        "stage_hash_mapping",
        "projected_role",
        "malformed_mapping",
        "extra_stage_path_key",
        "extra_stage_hash_key",
        "live_reparse",
    ],
)
@requires_external_runtime_assets
def test_post_issuance_tamper_is_always_unpromoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    probe, binding, runtime_digest, stages, artifacts, runtime_path = _verified_binding(
        tmp_path, monkeypatch
    )
    supplied_stages = dict(stages)
    if tamper_kind == "stage_path_mapping":
        outside = tmp_path / "outside-stage.bin"
        outside.write_bytes(artifacts["preprocess"].read_bytes())
        binding.stage_artifact_paths["preprocess"] = outside
    elif tamper_kind == "runtime_path":
        outside = tmp_path / "outside-runtime.json"
        outside.write_bytes(runtime_path.read_bytes())
        object.__setattr__(binding, "runtime_manifest_path", outside)
    elif tamper_kind == "stage_hash_mapping":
        artifacts["preprocess"].write_bytes(b"caller-mutated-stage")
        mutated_digest = hashlib.sha256(artifacts["preprocess"].read_bytes()).hexdigest()
        binding.stage_hashes["preprocess"] = mutated_digest
        supplied_stages["preprocess"] = mutated_digest
    elif tamper_kind == "projected_role":
        object.__setattr__(binding, "sample_id_a", "caller-forged-sample")
    elif tamper_kind == "malformed_mapping":
        object.__setattr__(binding, "stage_artifact_paths", None)
    elif tamper_kind == "extra_stage_path_key":
        binding.stage_artifact_paths["caller_extra"] = artifacts["preprocess"]
    elif tamper_kind == "extra_stage_hash_key":
        binding.stage_hashes["caller_extra"] = stages["preprocess"]
    elif tamper_kind == "live_reparse":
        reparse_path = artifacts["preprocess"]
        monkeypatch.setattr(
            determinism_module,
            "_is_reparse",
            lambda path: Path(path) == reparse_path,
        )

    run_id = f"tamper-{tamper_kind}"
    path = _path(tmp_path, run_id)
    write_determinism_diagnostic(
        path,
        probe,
        run_id=run_id,
        runtime_manifest_sha256=runtime_digest,
        stage_hashes=supplied_stages,
        input_binding=binding,
    )

    packet = json.loads(path.read_text(encoding="ascii"))
    assert packet["gate_status"] == "failed"
    assert packet["claim_status"] == "unpromoted"
    assert packet["first_divergent_stage"] == "artifact_binding"
    if tamper_kind != "live_reparse":
        assert packet["input_binding"] is None


def test_arbitrary_escape_and_reparse_artifact_paths_cannot_become_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    binding, runtime_digest, stages = _arbitrary_binding(tmp_path)
    escaped = replace(
        binding,
        runtime_manifest_path=tmp_path / "contained" / ".." / "runtime.json",
    )
    monkeypatch.setattr(
        determinism_module,
        "_is_reparse",
        lambda path: Path(path).name == "serialization.bin",
    )
    path = _path(tmp_path, "unsafe-artifacts")

    write_determinism_diagnostic(
        path,
        probe,
        run_id="unsafe-artifacts",
        runtime_manifest_sha256=runtime_digest,
        stage_hashes=stages,
        input_binding=escaped,
    )

    packet = json.loads(path.read_text(encoding="ascii"))
    assert packet["claim_status"] == "unpromoted"
    assert packet["first_divergent_stage"] == "artifact_binding"


def test_canonical_role_mismatch_cannot_bind_probe(tmp_path: Path) -> None:
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    path = _path(tmp_path, "role-mismatch")
    write_determinism_diagnostic(
        path,
        probe,
        run_id="role-mismatch",
        runtime_manifest_sha256="a" * 64,
        stage_hashes=_stage_hashes(),
        input_binding={
            "role_a": "ISIC_0012690",
            "role_b": "ISIC_0000050",
            "rgb_sha256_a": "68fa0dd008c8ac3e301be0495c00ee2df0ece31216165da7c62e441d71b835aa",
            "rgb_sha256_b": "0282de65b80464fce23b16995187bb10a6e89b52858b9408ea8b58ac183f2e9e",
        },
    )
    packet = json.loads(path.read_text(encoding="ascii"))
    assert packet["claim_status"] == "unpromoted"
    assert packet["first_divergent_stage"] == "artifact_binding"


@pytest.mark.parametrize("invalid_runtime", ["A" * 64, "a" * 63])
def test_diagnostic_writer_rejects_invalid_runtime_hash_for_eligibility(
    tmp_path: Path, invalid_runtime: str
) -> None:
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    path = _path(tmp_path, invalid_runtime[:8])
    write_determinism_diagnostic(
        path,
        probe,
        run_id=invalid_runtime[:8],
        runtime_manifest_sha256=invalid_runtime,
        stage_hashes=_stage_hashes(),
    )
    packet = json.loads(path.read_text(encoding="ascii"))
    assert packet["gate_status"] == "failed"
    assert packet["claim_status"] == "unpromoted"
    assert packet["first_divergent_stage"] == "runtime_manifest"


def test_diagnostic_writer_rejects_invalid_stage_hash_for_eligibility(tmp_path: Path) -> None:
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    stages = _stage_hashes()
    stages["serialization"] = "B" * 64
    path = _path(tmp_path, "invalidstage")
    write_determinism_diagnostic(
        path,
        probe,
        run_id="invalidstage",
        runtime_manifest_sha256="a" * 64,
        stage_hashes=stages,
    )
    packet = json.loads(path.read_text(encoding="ascii"))
    assert packet["gate_status"] == "failed"
    assert packet["claim_status"] == "unpromoted"
    assert packet["first_divergent_stage"] == "serialization"


def test_diagnostic_writer_rejects_escape_and_conflicting_write(tmp_path: Path) -> None:
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    with pytest.raises(ValueError, match="diagnostic path"):
        write_determinism_diagnostic(
            tmp_path / "diagnostic.json",
            probe,
            run_id=RUN_ID,
            runtime_manifest_sha256="a" * 64,
            stage_hashes=_stage_hashes(),
        )
    wrong_base = tmp_path / "diagnostics"
    wrong_base.mkdir()
    wrong_tail = wrong_base / "imp.nnunet.determinism.v1" / RUN_ID / "diagnostic.json"
    with pytest.raises(ValueError, match="diagnostic path"):
        write_determinism_diagnostic(
            wrong_tail,
            probe,
            run_id=RUN_ID,
            runtime_manifest_sha256="a" * 64,
            stage_hashes=_stage_hashes(),
        )

    path = _path(tmp_path)
    write_determinism_diagnostic(
        path,
        probe,
        run_id=RUN_ID,
        runtime_manifest_sha256="a" * 64,
        stage_hashes=_stage_hashes(),
    )
    changed = _stage_hashes()
    changed[REQUIRED_STAGE_HASHES[-1]] = "c" * 64
    with pytest.raises(ValueError, match="byte drift"):
        write_determinism_diagnostic(
            path,
            probe,
            run_id=RUN_ID,
            runtime_manifest_sha256="a" * 64,
            stage_hashes=changed,
        )


def test_diagnostic_writer_rejects_reparse_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        determinism_module,
        "_is_reparse",
        lambda path: Path(path) == tmp_path,
    )
    probe = run_interleaved_probe(FakeClient(), SAMPLE_A, SAMPLE_B)
    with pytest.raises(ValueError, match="reparse|symlink"):
        write_determinism_diagnostic(
            _path(tmp_path),
            probe,
            run_id=RUN_ID,
            runtime_manifest_sha256="a" * 64,
            stage_hashes=_stage_hashes(),
        )
