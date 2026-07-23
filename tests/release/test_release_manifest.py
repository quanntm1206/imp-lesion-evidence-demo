from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.release.build_release_manifest import verify_public_sample_evidence

from lesion_robustness.release_manifest import (
    MANIFEST_SCHEMA,
    deck_projection,
    fixed_cache_projection,
    launcher_projection,
    load_release_manifest,
    paper_projection,
    registry_projection,
    runtime_projection,
    sidecar_model_manifest_projection,
    sha256_file,
    validate_projection,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "release" / "imp_release_manifest.json"
BASE_ROOT = ROOT.parents[1]


def _write_canonical_manifest(path: Path, payload: object) -> None:
    path.write_bytes(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    )


def test_manifest_declares_exact_current_lanes_without_self_hash() -> None:
    manifest = load_release_manifest(MANIFEST)

    assert manifest.schema_version == MANIFEST_SCHEMA
    assert manifest.comparison("live_demo").left_model_id == "L206-control-s206"
    assert manifest.comparison("live_demo").right_model_id == "L192-nnUNet-v2-raw-100ep"
    assert manifest.comparison("paper_rq1").left_model_id == "L191-C0-clean-v3-IMP-control"
    assert manifest.comparison("paper_rq2").right_model_id == "L206-contour-channel-s206"
    assert "release_manifest_sha256" not in json.loads(MANIFEST.read_text("ascii"))
    assert "model_manifest_sha256" not in manifest.provenance["sidecar"]


def test_all_projections_embed_the_canonical_manifest_digest() -> None:
    manifest = load_release_manifest(MANIFEST)
    digest = sha256_file(MANIFEST)

    assert runtime_projection()["release_manifest_sha256"] == digest
    assert launcher_projection()["release_manifest_sha256"] == digest
    assert registry_projection()["release_manifest_sha256"] == digest
    assert paper_projection()["release_manifest_sha256"] == digest
    assert deck_projection()["release_manifest_sha256"] == digest
    assert fixed_cache_projection()["release_manifest_sha256"] == digest
    assert runtime_projection()["comparison"] == manifest.comparison_mapping("live_demo")


def test_fixed_cache_projection_exposes_exact_release_contract() -> None:
    digest = sha256_file(MANIFEST)

    assert fixed_cache_projection() == {
        "release_manifest_sha256": digest,
        "ordered_universe_sha256": "3e0d7784845ab3b6eb87c5c6ef5f22d34061543f91b9c10d2b66924f48a5c25a",
        "dataset_index": {
            "path": "demo_runtime/loop206_dataset_index.json",
            "sha256": "e88a3cc144b799d214f40b85064665d3348bc8bac3ead549f80b96d436f69fc3",
        },
        "live_config": {
            "path": "configs/demo/loop206_live.yaml",
            "schema_version": "loop206.demo.live.v1",
            "sha256": "e3110561451dc735f996a564ad12202811266b805696a919a20784602f8f4903",
        },
        "fixed_cache": {
            "schema_version": "loop206.leakage_safe_pilot_cache.v2",
            "artifact_type": "loop206_packed_binary_channel",
            "count": 536,
            "shape": [384, 384],
            "candidate": {
                "manifest_path": ".artifacts/preprocessing_search/loop206_leac_drlse/pilot_cache_v2_candidate/manifest.json",
                "manifest_sha256": "48e48290507eff6e4da8357e3310db9305a920f731c5b49890851d058d892255",
                "data_sha256": "3f49e43524772b9eee17a146ff47cb15361cf78b2ce77f8c5b25c46b8f019ebb",
            },
            "zero": {
                "manifest_path": ".artifacts/preprocessing_search/loop206_leac_drlse/pilot_cache_v2_zero_control/manifest.json",
                "manifest_sha256": "b92bd22e5425354b46bc019f3ab6d3daddc24568670717be2654c8938894c0da",
                "data_sha256": "c8f67865341c41e506c41f9ef3221861d2c4a12f771c7eee4159886fc718fa18",
            },
        },
    }


def test_live_config_bytes_match_the_canonical_release_pin() -> None:
    projection = fixed_cache_projection()["live_config"]
    path = ROOT / projection["path"]

    assert sha256_file(path) == projection["sha256"]


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_manifest_rejects_fixed_cache_key_drift(tmp_path: Path, mutation: str) -> None:
    payload = json.loads(MANIFEST.read_text("ascii"))
    fixed_cache = payload["public_sample_selection"]["fixed_cache"]
    if mutation == "missing":
        del fixed_cache["candidate"]["data_sha256"]
    else:
        fixed_cache["candidate"]["unexpected"] = "forged"
    forged = tmp_path / f"fixed-cache-{mutation}.json"
    _write_canonical_manifest(forged, payload)

    with pytest.raises(ValueError, match="keys mismatch"):
        load_release_manifest(forged)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ordered_universe_sha256", "not-a-hash", "SHA-256"),
        ("dataset_index.path", "../escaped.json", "path"),
        ("live_config.schema_version", "", "schema"),
        ("fixed_cache.count", True, "count"),
        ("fixed_cache.shape", [384, 0], "shape"),
        ("fixed_cache.candidate.manifest_sha256", "f" * 63, "SHA-256"),
    ],
)
def test_manifest_rejects_malformed_fixed_cache_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = json.loads(MANIFEST.read_text("ascii"))
    target = payload["public_sample_selection"]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    forged = tmp_path / f"malformed-{field.replace('.', '-')}.json"
    _write_canonical_manifest(forged, payload)

    with pytest.raises(ValueError, match=message):
        load_release_manifest(forged)


@pytest.mark.parametrize(
    "field",
    [
        "dataset_index.sha256",
        "live_config.sha256",
        "fixed_cache.candidate.manifest_sha256",
        "fixed_cache.zero.data_sha256",
    ],
)
def test_fixed_cache_projection_rejects_forged_current_digest_stale_pin(
    tmp_path: Path, field: str,
) -> None:
    artifact = tmp_path / "fixed_cache_projection.json"
    payload = fixed_cache_projection()
    target = payload
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = "0" * 64
    artifact.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ValueError, match="exact payload"):
        validate_projection(artifact, MANIFEST)


def test_projection_validation_fails_closed_on_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(runtime_projection()), encoding="ascii")
    validate_projection(artifact, MANIFEST)
    payload = json.loads(artifact.read_text("ascii"))
    payload["release_manifest_sha256"] = "0" * 64
    artifact.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ValueError, match="manifest digest"):
        validate_projection(artifact, MANIFEST)


def test_projection_rejects_forged_current_digest_payload(tmp_path: Path) -> None:
    artifact = tmp_path / "model_registry.example.json"
    payload = registry_projection()
    payload["control"]["model_id"] = "forged-current-release-model"
    artifact.write_text(json.dumps(payload, sort_keys=True), encoding="ascii")

    with pytest.raises(ValueError, match="exact payload"):
        validate_projection(artifact, MANIFEST)


def test_manifest_rejects_noncanonical_bytes_and_missing_policy_shape(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text("ascii"))
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="canonical"):
        load_release_manifest(pretty)

    payload["claim_policies"]["operational_only"] = {"clinical_use": False}
    forged = tmp_path / "forged.json"
    forged.write_bytes(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    )
    with pytest.raises(ValueError, match="claim policy"):
        load_release_manifest(forged)


def test_python_sidecar_and_evidence_consumers_do_not_duplicate_release_pins() -> None:
    manifest = load_release_manifest(MANIFEST)
    pins = (
        manifest.model("L192-nnUNet-v2-raw-100ep").model_id,
        manifest.model("L192-nnUNet-v2-raw-100ep").checkpoint_sha256,
    )
    for path in (
        ROOT / "sidecar/nnunet/predictor.py",
        ROOT / "scripts/demo/verify_nnunet_bundle.py",
        ROOT / "src/lesion_robustness/evidence_registry.py",
    ):
        assert not any(pin in path.read_text("utf-8") for pin in pins), path


def test_sidecar_model_manifest_is_exact_generated_semantic_projection(
    tmp_path: Path,
) -> None:
    checked_in = ROOT / "sidecar/nnunet/model_manifest.example.json"
    assert json.loads(checked_in.read_text("utf-8")) == sidecar_model_manifest_projection()
    validate_projection(checked_in, MANIFEST)

    forged = tmp_path / "model_manifest.example.json"
    payload = sidecar_model_manifest_projection()
    payload["runtime"]["version"] = "forged-current-digest"
    forged.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="exact payload"):
        validate_projection(forged, MANIFEST)


def test_sha256_file_matches_canonical_bytes() -> None:
    assert sha256_file(MANIFEST) == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def test_literal_consumers_use_manifest_projections() -> None:
    manifest = load_release_manifest(MANIFEST)
    identities = (
        manifest.model("L191-C0-clean-v3-IMP-control").model_id,
        manifest.model("L192-nnUNet-v2-raw-100ep").model_id,
        manifest.model("L206-control-s206").model_id,
        manifest.model("L206-contour-channel-s206").model_id,
        manifest.model("L192-nnUNet-v2-raw-100ep").checkpoint_sha256,
    )
    consumers = (
        ROOT / "src/lesion_robustness/demo/dual_live_protocol.py",
        ROOT / "src/lesion_robustness/demo/model_service.py",
        ROOT / "scripts/demo/run_sidecar.ps1",
        ROOT / "scripts/demo/run_demo.ps1",
        ROOT / "scripts/demo/run_tunnel.ps1",
        ROOT / "scripts/paper/build_clean_v3_tables.py",
    )
    for consumer in consumers:
        text = consumer.read_text(encoding="utf-8")
        assert not any(value in text for value in identities), consumer


def test_public_builder_verifies_canonical_selection_license_and_training_membership() -> None:
    verify_public_sample_evidence(
        MANIFEST,
        dataset_index=BASE_ROOT / "demo_runtime/loop206_dataset_index.json",
        clean_v3_manifest=BASE_ROOT / "data/splits/clean_v3_manifest.csv",
    )


@pytest.mark.parametrize(
    ("model_id", "forged"),
    [
        ("L206-control-s206", "excluded_but_not_from_the_pinned_holdout"),
        ("L192-nnUNet-v2-raw-100ep", "included_but_not_in_clean_v3_training"),
    ],
)
def test_public_builder_rejects_forged_training_exposure(
    tmp_path: Path, model_id: str, forged: str
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="ascii"))
    payload["public_samples"]["samples"][0]["training_exposure"][model_id] = forged
    manifest = tmp_path / "forged-exposure.json"
    _write_canonical_manifest(manifest, payload)

    with pytest.raises(ValueError, match="training exposure"):
        verify_public_sample_evidence(
            manifest,
            dataset_index=BASE_ROOT / "demo_runtime/loop206_dataset_index.json",
            clean_v3_manifest=BASE_ROOT / "data/splits/clean_v3_manifest.csv",
        )


def test_public_builder_rejects_selected_index_binding_drift(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="ascii"))
    payload["public_samples"]["samples"][0]["group_key"] = "component:" + "0" * 64
    manifest = tmp_path / "forged-index-binding.json"
    _write_canonical_manifest(manifest, payload)

    with pytest.raises(ValueError, match="public sample provenance"):
        verify_public_sample_evidence(
            manifest,
            dataset_index=BASE_ROOT / "demo_runtime/loop206_dataset_index.json",
            clean_v3_manifest=BASE_ROOT / "data/splits/clean_v3_manifest.csv",
        )
