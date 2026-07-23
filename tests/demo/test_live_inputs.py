from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from lesion_robustness.demo.immutable_io import ImmutableSnapshot
from lesion_robustness.demo.live_inputs import (
    LiveInputEvidence,
    load_public_live_samples,
    recompute_public_selection,
    validate_live_input_evidence,
)
from lesion_robustness.release_manifest import load_release_manifest


ROOT = Path(__file__).resolve().parents[2]


def _windows_path(*parts: str) -> Path:
    drive, *segments = parts
    return Path(drive + ":" + "\\" + "\\".join(segments))


CANONICAL_INDEX = _windows_path("E", "0. IMP", "demo_runtime", "loop206_dataset_index.json")
DATASET_ROOT = _windows_path("E", "datasets")


def _rgb_hash(image: np.ndarray) -> str:
    return ImmutableSnapshot.decoded_rgb_sha256(image)


def _row(sample_id: str, group_key: str, raw: str, rgb: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "group_key": group_key,
        "role": "holdout",
        "split": "train_screen_holdout",
        "source_split": "train",
        "image_root": 0,
        "image_relative": f"images/{sample_id}.png",
        "mask_root": 0,
        "mask_relative": "forbidden-mask.png",
        "sha256_raw": raw,
        "sha256_rgb": rgb,
    }


def _ordered_hash(rows: list[dict[str, object]]) -> str:
    payload = "".join(
        f"{row['sample_id']}|{row['group_key']}|{row['sha256_raw']}|{row['sha256_rgb']}\n"
        for row in sorted(rows, key=lambda value: (str(value["sample_id"]), str(value["group_key"])))
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def public_fixture(tmp_path: Path, *, drift: str | None = None) -> tuple[object, Path, Path]:
    root = tmp_path / "dataset"
    image_root = root / "images"
    image_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    selected = ("ISIC_0000050", "ISIC_0012690")
    boundary = "ISIC_0016069"
    identifiers = (selected[0], *(f"ISIC_{index:07d}" for index in range(100, 173)), selected[1], boundary)
    for index, sample_id in enumerate(identifiers):
        image = np.full((5, 7, 3), index, dtype=np.uint8)
        path = image_root / f"{sample_id}.png"
        Image.fromarray(image, mode="RGB").save(path)
        rows.append(_row(sample_id, f"component:{index:064x}", hashlib.sha256(path.read_bytes()).hexdigest(), _rgb_hash(image)))
    index = tmp_path / "index.json"
    payload = {"schema_version": "loop206.demo.dataset_index.v1", "root_count": 1, "rows": rows}
    index.write_text(json.dumps(payload), encoding="ascii")
    index_hash = hashlib.sha256(index.read_bytes()).hexdigest()
    selected_rows = [row for row in rows if row["sample_id"] in selected]
    samples = []
    license_rows = (
        (50, "a9e8cb35e2c8b81cdb7ea893906057071b53bf40d279b05db1826ba6d2434669"),
        (1534, "56dc36553f48698fb7073f1ec60dc6457df1c2dc8968f01017c2e0c9b54ca0a9"),
    )
    for row, (row_number, row_hash) in zip(selected_rows, license_rows):
        samples.append(
            SimpleNamespace(
                sample_id=row["sample_id"], group_key=row["group_key"], source_dataset="isic2018",
                sha256_raw=row["sha256_raw"], sha256_rgb=row["sha256_rgb"],
                source_page="https://challenge.isic-archive.com/data/", license_id="CC-0",
                license_evidence=SimpleNamespace(clean_v3_manifest_sha256="4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102", csv_row_number=row_number, raw_csv_row_sha256=row_hash),
                training_exposure={"L206-control-s206": "excluded_from_308_fit_in_76_group_train_screen_holdout", "L192-nnUNet-v2-raw-100ep": "included_in_clean_v3_2008_training_rows"},
                ground_truth_used=False, ground_truth_not_loaded=True,
            )
        )
    registry = SimpleNamespace(
        digest="c" * 64,
        public_samples=SimpleNamespace(
            selection=SimpleNamespace(universe="loop206_train_screen_holdout_clean", universe_count=76, dataset_index_sha256=index_hash, ordered_universe_sha256=_ordered_hash(rows), rule="explicit_roles_A_B_boundary_after_index_hash"),
            samples=tuple(samples),
        ),
        public_sample_contract=SimpleNamespace(
            state="verified",
            roles={
                name: SimpleNamespace(**{field: row[field] for field in ("sample_id", "group_key", "sha256_raw", "sha256_rgb")})
                for name, row in (("A", rows[0]), ("B", next(row for row in rows if row["sample_id"] == selected[1])), ("boundary", rows[-1]))
            },
        ),
    )
    if drift is not None:
        target: object = registry.public_samples.selection if drift == "ordered_universe_sha256" else samples[0]
        if drift in {"raw_csv_row_sha256", "clean_v3_manifest_sha256"}:
            target = samples[0].license_evidence
        setattr(target, drift, "0" * 64 if "sha256" in drift else "forged")
    return registry, index, root


def test_public_loader_requires_index_identity_raw_and_rgb_hashes(tmp_path: Path) -> None:
    registry = load_release_manifest(ROOT / "release/imp_release_manifest.json")
    index = CANONICAL_INDEX
    samples = load_public_live_samples(registry, index, [DATASET_ROOT])
    assert set(samples) == {"ISIC_0000050", "ISIC_0012690"}
    assert all(sample.evidence.kind == "public_sample" for sample in samples.values())
    assert all(sample.evidence.ground_truth_used is False for sample in samples.values())
    assert all(sample.evidence.ground_truth_not_loaded is True for sample in samples.values())


def test_public_selection_recomputes_exact_universe_rule_and_hash(tmp_path: Path) -> None:
    registry = load_release_manifest(ROOT / "release/imp_release_manifest.json")
    index = CANONICAL_INDEX
    selected = recompute_public_selection(index, registry)
    assert selected.sample_ids == ("ISIC_0000050", "ISIC_0012690")
    assert selected.universe_count == 76
    assert selected.ordered_universe_sha256 == registry.public_samples.selection["ordered_universe_sha256"]


def test_public_contract_rejects_boundary_as_live_B_array_probe(tmp_path: Path) -> None:
    registry, index, _root = public_fixture(tmp_path)
    boundary = registry.public_sample_contract.roles["boundary"]
    registry.public_samples.samples = (
        registry.public_samples.samples[0],
        SimpleNamespace(
            sample_id=boundary.sample_id,
            group_key=boundary.group_key,
            sha256_raw=boundary.sha256_raw,
            sha256_rgb=boundary.sha256_rgb,
        ),
    )

    with pytest.raises(ValueError, match="public sample provenance"):
        recompute_public_selection(index, registry)


def test_public_contract_rejects_swapped_live_roles_even_when_manifest_object_is_forged() -> None:
    manifest = load_release_manifest(ROOT / "release/imp_release_manifest.json")
    roles = dict(manifest.public_sample_contract.roles)
    roles["A"], roles["B"] = roles["B"], roles["A"]
    forged = manifest.__class__(
        manifest.schema_version, manifest.models, manifest.comparisons,
        manifest.claim_policies, manifest.public_sample_selection,
        manifest.public_samples,
        manifest.public_sample_contract.__class__("verified", roles),
        manifest.provenance, manifest.rq1_v2, manifest.path, manifest.digest,
    )
    with pytest.raises(ValueError, match="public sample provenance"):
        recompute_public_selection(ROOT / "demo_runtime/loop206_dataset_index.json", forged)


def test_public_contract_rejects_missing_canonical_manifest_digest(tmp_path: Path) -> None:
    manifest = SimpleNamespace(path=ROOT / "release/imp_release_manifest.json", digest=None)
    with pytest.raises(ValueError, match="public sample provenance"):
        recompute_public_selection(tmp_path / "missing-index.json", manifest)


def test_public_contract_rejects_forged_aligned_role_and_sample_order(tmp_path: Path) -> None:
    registry, index, _root = public_fixture(tmp_path)
    registry.public_samples.samples = tuple(reversed(registry.public_samples.samples))
    roles = registry.public_sample_contract.roles
    roles["A"], roles["B"] = roles["B"], roles["A"]
    # The legacy loader accepted this self-consistent but non-canonical object.
    with pytest.raises(ValueError, match="public sample provenance"):
        recompute_public_selection(index, registry)


def test_manifest_public_samples_have_pinned_license_and_training_exposure() -> None:
    manifest = load_release_manifest(ROOT / "release/imp_release_manifest.json")
    assert tuple(sample.sample_id for sample in manifest.public_samples.samples) == ("ISIC_0000050", "ISIC_0012690")
    for sample in manifest.public_samples.samples:
        assert sample.license_id == "CC-0"
        assert sample.license_evidence.clean_v3_manifest_sha256 == "4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102"
        assert sample.training_exposure["L192-nnUNet-v2-raw-100ep"].startswith("included")


def test_public_sample_contract_binds_live_roles_and_boundary_by_hash() -> None:
    manifest = load_release_manifest(ROOT / "release/imp_release_manifest.json")

    assert manifest.public_sample_contract.state == "verified"
    assert tuple(manifest.public_sample_contract.roles) == ("A", "B", "boundary")
    assert manifest.public_sample_contract.roles["A"].sample_id == "ISIC_0000050"
    assert manifest.public_sample_contract.roles["B"].sample_id == "ISIC_0012690"
    assert manifest.public_sample_contract.roles["boundary"].sample_id == "ISIC_0016069"
    assert manifest.public_sample_contract.roles["B"].sha256_rgb == (
        "0282de65b80464fce23b16995187bb10a6e89b52858b9408ea8b58ac183f2e9e"
    )


def test_public_loader_never_reads_or_returns_a_mask(tmp_path: Path) -> None:
    registry = load_release_manifest(ROOT / "release/imp_release_manifest.json")
    index = CANONICAL_INDEX
    root = DATASET_ROOT
    forbidden = (root / "forbidden-mask.png").resolve()
    original_open = Path.open
    original_resolve = Path.resolve

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path == forbidden:
            raise AssertionError("public loader opened a mask")
        return original_open(path, *args, **kwargs)

    def guarded_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if str(path).endswith("forbidden-mask.png"):
            raise AssertionError("public loader resolved a mask")
        return original_resolve(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(Path, "open", guarded_open)
        monkeypatch.setattr(Path, "resolve", guarded_resolve)
        samples = load_public_live_samples(registry, index, [root])
    assert all(not hasattr(sample, "mask") for sample in samples.values())


@pytest.mark.parametrize("field", ["group_key", "sha256_raw", "sha256_rgb", "raw_csv_row_sha256", "clean_v3_manifest_sha256", "ordered_universe_sha256"])
def test_public_loader_fails_closed_on_provenance_drift(tmp_path: Path, field: str) -> None:
    registry, index, root = public_fixture(tmp_path, drift=field)
    with pytest.raises(ValueError, match="public sample provenance"):
        load_public_live_samples(registry, index, [root])


@pytest.mark.parametrize(
    ("kind", "evidence_class"),
    [
        ("synthetic", "illustrative_arbitrary_upload_no_ground_truth"),
        ("arbitrary_upload", "illustrative_public_sample_no_ground_truth"),
        ("unknown", "illustrative_synthetic_no_ground_truth"),
    ],
)
def test_live_evidence_rejects_unknown_or_mismatched_kind_class(
    kind: str, evidence_class: str
) -> None:
    evidence = SimpleNamespace(
        kind=kind,
        evidence_class=evidence_class,
        rgb_sha256="a" * 64,
        sample_id=None,
        source_dataset=None,
        source_page=None,
        image_license=None,
        training_exposure={},
        ground_truth_used=False,
        ground_truth_not_loaded=True,
    )

    with pytest.raises(ValueError, match="live input evidence"):
        validate_live_input_evidence(evidence)


@pytest.mark.parametrize("kind", ["synthetic", "arbitrary_upload"])
def test_nonpublic_evidence_forbids_public_metadata(kind: str) -> None:
    evidence_class = {
        "synthetic": "illustrative_synthetic_no_ground_truth",
        "arbitrary_upload": "illustrative_arbitrary_upload_no_ground_truth",
    }[kind]
    evidence = SimpleNamespace(
        kind=kind,
        evidence_class=evidence_class,
        rgb_sha256="a" * 64,
        sample_id="ISIC_0000050",
        source_dataset="isic2018",
        source_page="https://challenge.isic-archive.com/data/",
        image_license="CC-0",
        training_exposure={"forged": "included"},
        ground_truth_used=False,
        ground_truth_not_loaded=True,
    )

    with pytest.raises(ValueError, match="live input evidence"):
        validate_live_input_evidence(evidence)


def test_public_evidence_requires_exact_metadata_and_exposure() -> None:
    evidence = LiveInputEvidence(
        "public_sample",
        "illustrative_public_sample_no_ground_truth",
        "a" * 64,
        "ISIC_0000050",
        "isic2018",
        "https://challenge.isic-archive.com/data/",
        "CC-0",
        {
            "L206-control-s206": "excluded_from_308_fit_in_76_group_train_screen_holdout",
            "L192-nnUNet-v2-raw-100ep": "included_in_clean_v3_2008_training_rows",
        },
        False,
        True,
    )

    assert validate_live_input_evidence(evidence) is evidence


def test_actual_canonical_public_samples_match_base_metadata_and_images() -> None:
    manifest = load_release_manifest(ROOT / "release/imp_release_manifest.json")
    index = CANONICAL_INDEX
    selected = recompute_public_selection(index, manifest)
    samples = load_public_live_samples(manifest, index, [DATASET_ROOT])

    assert selected.sample_ids == ("ISIC_0000050", "ISIC_0012690")
    assert selected.universe_count == 76
    assert tuple(samples) == selected.sample_ids
    assert tuple(sample.evidence.rgb_sha256 for sample in samples.values()) == (
        "68fa0dd008c8ac3e301be0495c00ee2df0ece31216165da7c62e441d71b835aa",
        "0282de65b80464fce23b16995187bb10a6e89b52858b9408ea8b58ac183f2e9e",
    )
