from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lesion_robustness.demo.immutable_io import ImmutableSnapshot


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper/clean_v3_loop206/figures"
REGISTRY = ROOT / "demo/data/evidence_registry.json"
DATASET_INDEX = ROOT / "demo_runtime/loop206_dataset_index.json"
PROVENANCE = (
    ROOT
    / ".artifacts/preprocessing_search/clean_v3_manifest/clean_v3_manifest.preview.csv"
)
SELECTED_IDS = {"ISIC_0000050", "ISIC_0012690", "ISIC_0016069"}
requires_external_runtime_assets = pytest.mark.skipif(
    not (DATASET_INDEX.is_file() and PROVENANCE.is_file()),
    reason="external runtime assets; local release gate required",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_module("task10_generator", FIGURES / "generate_task10_figures.py")
capture = _load_module("task10_capture", FIGURES / "capture_qualitative_demo.py")


def _registry_hash(payload: dict) -> str:
    unsigned = deepcopy(payload)
    unsigned.pop("registry_sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256((encoded + "\n").encode("ascii")).hexdigest()


def _write_registry(tmp_path: Path, payload: dict) -> Path:
    payload["registry_sha256"] = _registry_hash(payload)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    return path


def _selected_index_rows() -> list[dict]:
    payload = json.loads(DATASET_INDEX.read_text(encoding="ascii"))
    return sorted(
        [row for row in payload["rows"] if row.get("sample_id") in SELECTED_IDS],
        key=lambda row: row["sample_id"],
    )


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "provenance.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _provenance_rows() -> list[dict]:
    with PROVENANCE.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_delta_values_are_extracted_from_exact_registry_observation_and_comparison() -> None:
    evidence = generator.load_loop206_delta_evidence(REGISTRY)
    assert evidence.registry_sha256 == (
        "f6ed2eace90c49ee1b9f0c122e736920791b6301035bf8905c6a0ce27b755f32"
    )
    assert evidence.dice.point_delta == pytest.approx(-0.03129624395473221)
    assert evidence.dice.ci95 == pytest.approx(
        (-0.049121296024302145, -0.015627817085354864)
    )
    assert evidence.boundary_f1.point_delta == pytest.approx(-0.01465831334754726)
    assert evidence.boundary_f1.ci95 == pytest.approx(
        (-0.030758654691150956, 0.0010438469457382654)
    )


def test_delta_extraction_fails_closed_when_comparison_mismatches_observation(
    tmp_path: Path,
) -> None:
    payload = json.loads(REGISTRY.read_text(encoding="ascii"))
    comparison = next(
        row
        for row in payload["comparisons"]
        if row["comparison_id"] == "L206-contour-minus-control"
    )
    comparison["point_delta"] += 0.01
    path = _write_registry(tmp_path, payload)
    with pytest.raises(ValueError, match="comparison.*observation"):
        generator.load_loop206_delta_evidence(path)


def test_delta_extraction_fails_closed_when_exact_observation_is_missing(
    tmp_path: Path,
) -> None:
    payload = json.loads(REGISTRY.read_text(encoding="ascii"))
    payload["observations"] = [
        row for row in payload["observations"] if row["model_id"] != "L206-contour-vs-control"
    ]
    path = _write_registry(tmp_path, payload)
    with pytest.raises(ValueError, match="observation.*missing or duplicated"):
        generator.load_loop206_delta_evidence(path)


def test_delta_extraction_rejects_registry_hash_mismatch(tmp_path: Path) -> None:
    payload = json.loads(REGISTRY.read_text(encoding="ascii"))
    payload["registry_sha256"] = "0" * 64
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="registry hash mismatch"):
        generator.load_loop206_delta_evidence(path)


@requires_external_runtime_assets
def test_provenance_authorizes_selected_identity_hash_license_and_gt() -> None:
    authorization = capture.load_display_authorization(
        PROVENANCE,
        _selected_index_rows(),
        expected_sha256=capture.PROVENANCE_MANIFEST_SHA256,
    )
    assert authorization == {
        "schema_version": "loop206.qualitative_display_authorization.v1",
        "provenance_manifest_sha256": capture.PROVENANCE_MANIFEST_SHA256,
        "dataset_license": "legacy_isic_challenge_terms",
        "image_license": "CC-0",
        "mask_variant": "challenge_ground_truth",
        "identity_field": "isic_image_id",
        "hash_binding": "sha256_raw+sha256_rgb+mask_sha256_raw+mask_sha256_binary",
        "mask_bindings_sha256": authorization["mask_bindings_sha256"],
        "authorized_sample_count": 3,
    }
    assert len(authorization["mask_bindings_sha256"]) == 64
    assert not any("path" in key for key in authorization)


@requires_external_runtime_assets
def test_provenance_rejects_missing_selected_sample(tmp_path: Path) -> None:
    rows = [row for row in _provenance_rows() if row["isic_image_id"] != "ISIC_0012690"]
    path = _write_csv(tmp_path, rows)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="missing or duplicated"):
        capture.load_display_authorization(
            path, _selected_index_rows(), expected_sha256=expected
        )


@requires_external_runtime_assets
def test_provenance_rejects_unaccepted_image_license(tmp_path: Path) -> None:
    rows = _provenance_rows()
    row = next(value for value in rows if value["isic_image_id"] == "ISIC_0012690")
    row["image_license"] = "unaccepted"
    path = _write_csv(tmp_path, rows)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="image license"):
        capture.load_display_authorization(
            path, _selected_index_rows(), expected_sha256=expected
        )


@requires_external_runtime_assets
def test_provenance_rejects_manifest_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        capture.load_display_authorization(
            PROVENANCE, _selected_index_rows(), expected_sha256="0" * 64
        )


@requires_external_runtime_assets
def test_provenance_rejects_missing_mask_digest_binding() -> None:
    selected = _selected_index_rows()
    selected[0].pop("mask_sha256_raw", None)
    with pytest.raises(ValueError, match="mask hash binding"):
        capture.load_display_authorization(
            PROVENANCE,
            selected,
            expected_sha256=capture.PROVENANCE_MANIFEST_SHA256,
        )


def test_provenance_manifest_is_hashed_and_parsed_from_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = []
    provenance_rows = []
    for index in range(3):
        sample_id = f"sample-{index}"
        selected.append(
            {
                "sample_id": sample_id,
                "group_key": f"group-{index}",
                "source_dataset": "ISIC2017",
                "source_split": "train",
                "sha256_raw": str(index + 1) * 64,
                "sha256_rgb": str(index + 4) * 64,
                "mask_sha256_raw": "a" * 64,
                "mask_sha256_binary": "b" * 64,
            }
        )
        provenance_rows.append(
            {
                "isic_image_id": sample_id,
                "original_id": sample_id,
                "source_dataset": "ISIC2017",
                "split": "train",
                "dataset_license": "legacy_isic_challenge_terms",
                "image_license": "CC-0",
                "mask_variant": "challenge_ground_truth",
                "sha256_raw": str(index + 1) * 64,
                "sha256_rgb": str(index + 4) * 64,
            }
        )
    path = _write_csv(tmp_path, provenance_rows)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    real_open = Path.open
    opens = 0

    def counting_open(self: Path, *args, **kwargs):
        nonlocal opens
        if self == path:
            opens += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)

    capture.load_display_authorization(path, selected, expected_sha256=expected)

    assert opens == 1


def test_generator_passes_verified_bundle_snapshot_to_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "bundle.npz"
    bundle_path.write_bytes(b"bundle")
    snapshot = ImmutableSnapshot.from_bytes(b"bundle")
    evidence = generator.Loop206DeltaEvidence(
        registry_sha256="c" * 64,
        dice=generator.MetricDelta(-0.1, (-0.2, -0.05)),
        boundary_f1=generator.MetricDelta(-0.01, (-0.02, 0.01)),
    )
    captured = {}

    monkeypatch.setattr(generator, "_style", lambda: None)
    monkeypatch.setattr(generator, "load_loop206_delta_evidence", lambda _path: evidence)
    monkeypatch.setattr(
        generator.ImmutableSnapshot,
        "read",
        classmethod(lambda _cls, path: snapshot if Path(path) == bundle_path else None),
    )

    def fake_load_bundle(value, *, expected_registry_sha256):
        captured["value"] = value
        captured["registry"] = expected_registry_sha256
        return {}, [], {}

    monkeypatch.setattr(generator, "_load_bundle", fake_load_bundle)
    monkeypatch.setattr(generator, "_build_delta", lambda *_args: None)
    monkeypatch.setattr(generator, "_build_qualitative", lambda *_args: None)

    generator.main(
        [
            "--evidence-registry",
            str(REGISTRY),
            "--receipt-bundle",
            str(bundle_path),
            "--expected-receipt-bundle-sha256",
            snapshot.sha256,
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert captured == {"value": snapshot, "registry": "c" * 64}


def test_capture_selection_uses_masks_from_verified_index_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_rows = []
    holdout = []
    for index in range(76):
        sample_id = f"sample-{index:02d}"
        group_key = f"group-{index:02d}"
        mask = np.full((384, 384), index % 2, dtype=np.uint8)
        raw_rows.append(
            {
                "sample_id": sample_id,
                "group_key": group_key,
                "role": "holdout",
                "split": "train_screen_holdout",
                "source_split": "train",
                "mask_root": 999,
                "mask_relative": "must-not-be-reopened.png",
            }
        )
        holdout.append(SimpleNamespace(group_key=group_key, mask=mask))
    monkeypatch.setattr(
        capture,
        "load_dataset_index",
        lambda *_args, **_kwargs: ([], holdout, {"rows": raw_rows}),
    )

    selected, masks = capture.load_qualitative_selection(
        tmp_path / "index.json", (tmp_path,)
    )

    assert [row["sample_id"] for row in selected] == [
        "sample-00",
        "sample-38",
        "sample-75",
    ]
    np.testing.assert_array_equal(masks[1], holdout[38].mask)


def test_exact_caption_is_attached_to_each_of_15_modality_subplots() -> None:
    figure, axes = plt.subplots(3, 5)
    try:
        assert generator.caption_all_modality_panels(axes) == 15
        for axis in axes.flat:
            matching = [
                text
                for text in axis.texts
                if text.get_text() == "illustrative; not protected-test evidence"
            ]
            assert len(matching) == 1
            assert matching[0].get_transform() == axis.transAxes
            assert matching[0].get_position()[1] < 0
    finally:
        plt.close(figure)


def test_drawio_uses_exact_loop191_model_label() -> None:
    source = (FIGURES / "evidence_pipeline.drawio").read_text(encoding="utf-8")
    assert (
        'value="Loop191 IMP-SegFormer-B3 (MiT-B3 U-Net implementation)"'
        in source
    )


def test_manifest_binds_full_capture_render_chain_and_external_hashes() -> None:
    manifest_path = ROOT / "paper/clean_v3_loop206/artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    qualitative = manifest["figures"]["qualitative_demo"]
    receipts_path = ROOT / "paper/clean_v3_loop206" / qualitative["receipt_path"]
    receipts = json.loads(receipts_path.read_text(encoding="ascii"))

    assert len(qualitative["generation_chain"]) == 2
    capture_command, render_command = qualitative["generation_chain"]
    assert "--provenance-manifest <PROVENANCE_MANIFEST>" in capture_command
    assert "--output <AUTHORIZED_RECEIPT_BUNDLE>" in capture_command
    assert "--evidence-registry <EVIDENCE_REGISTRY>" in render_command
    assert "--receipt-bundle <AUTHORIZED_RECEIPT_BUNDLE>" in render_command
    assert qualitative["external_runtime_bundle_sha256"] == receipts[
        "runtime_bundle_sha256"
    ]
    assert qualitative["provenance_manifest_sha256"] == receipts[
        "display_authorization"
    ]["provenance_manifest_sha256"]
    authorization = receipts["display_authorization"]
    assert authorization["hash_binding"] == (
        "sha256_raw+sha256_rgb+mask_sha256_raw+mask_sha256_binary"
    )
    assert len(authorization["mask_bindings_sha256"]) == 64
    for receipt in receipts["receipts"]:
        assert receipt["display_authorization"] == authorization
        assert set(receipt["ground_truth_binding"]) == {
            "mask_sha256_raw",
            "mask_sha256_binary",
            "mask_sha256_runtime",
        }
    assert qualitative["provenance_receipt_sha256"] == hashlib.sha256(
        receipts_path.read_bytes()
    ).hexdigest()
    assert qualitative["evidence_registry_sha256"] == manifest[
        "evidence_registry_sha256"
    ]
    for figure in manifest["figures"].values():
        artifact = ROOT / "paper/clean_v3_loop206" / figure["path"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == figure["sha256"]
    for path_key, hash_key in (
        ("generation_source_path", "generation_source_sha256"),
        ("capture_source_path", "capture_source_sha256"),
    ):
        artifact = ROOT / "paper/clean_v3_loop206" / qualitative[path_key]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == qualitative[hash_key]
    assert len(qualitative["external_runtime_bundle_sha256"]) == 64
    assert len(qualitative["provenance_manifest_sha256"]) == 64


@requires_external_runtime_assets
def test_external_provenance_manifest_matches_recorded_hash() -> None:
    manifest = json.loads(
        (ROOT / "paper/clean_v3_loop206/artifact_manifest.json").read_text(
            encoding="ascii"
        )
    )
    qualitative = manifest["figures"]["qualitative_demo"]

    assert qualitative["provenance_manifest_sha256"] == hashlib.sha256(
        PROVENANCE.read_bytes()
    ).hexdigest()
