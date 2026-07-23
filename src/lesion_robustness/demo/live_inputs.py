"""Hash-bound live input evidence without any ground-truth access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
from typing import Literal

import numpy as np

from lesion_robustness.demo.immutable_io import ImmutableSnapshot
from lesion_robustness.demo.dual_live_protocol import rgb_sha256
from lesion_robustness.release_manifest import load_release_manifest


_LICENSE_ROWS = {
    "ISIC_0000050": (50, "a9e8cb35e2c8b81cdb7ea893906057071b53bf40d279b05db1826ba6d2434669"),
    "ISIC_0012690": (1534, "56dc36553f48698fb7073f1ec60dc6457df1c2dc8968f01017c2e0c9b54ca0a9"),
}
_CLEAN_V3_MANIFEST_SHA256 = "4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102"
_CANONICAL_RELEASE_MANIFEST_SHA256 = "435606d5adc296be57405c65a9c725af3dff96c15f9aabf7ac0924d06387a264"
_CANONICAL_ROLE_PINS = {
    "A": (
        "ISIC_0000050",
        "component:359df3ad7a38f96ab2127e2c90303fbe922d7c3a062eb5ff17cf4e9c5e895462",
        "731bceb0bc06e03a6ffdd7b6f47a4d7b8f9c30709984fa382ccd39a2ad0a236c",
        "68fa0dd008c8ac3e301be0495c00ee2df0ece31216165da7c62e441d71b835aa",
    ),
    "B": (
        "ISIC_0012690",
        "component:dee626c85dc2b7aeb500dfd33907a6197d7ec338dfc97641f871146d0e303bb1",
        "5b66c0ba91fd0315113502a43d72be4a46639f8afbff48378d828a2a1cdb79cc",
        "0282de65b80464fce23b16995187bb10a6e89b52858b9408ea8b58ac183f2e9e",
    ),
    "boundary": (
        "ISIC_0016069",
        "component:6564144ba6288d70fabb922a64e0fe0132c67658a9f1433fca5e8cd6b055d3f1",
        "13a8a85ac23a5dedf2e4b1480d31e98020ad5dfffb0d65dd66cb398fc6354851",
        "f4021e5c8e09869b30a131c8f27473efd63814570b2175d5e993187db13e8a5c",
    ),
}
_TRAINING_EXPOSURE = {
    "L206-control-s206": "excluded_from_308_fit_in_76_group_train_screen_holdout",
    "L192-nnUNet-v2-raw-100ep": "included_in_clean_v3_2008_training_rows",
}
_EVIDENCE_CLASSES = {
    "synthetic": "illustrative_synthetic_no_ground_truth",
    "public_sample": "illustrative_public_sample_no_ground_truth",
    "arbitrary_upload": "illustrative_arbitrary_upload_no_ground_truth",
}
_EVIDENCE_FIELDS = frozenset(
    {
        "kind",
        "evidence_class",
        "rgb_sha256",
        "sample_id",
        "source_dataset",
        "source_page",
        "image_license",
        "training_exposure",
        "ground_truth_used",
        "ground_truth_not_loaded",
    }
)


@dataclass(frozen=True)
class LiveInputEvidence:
    kind: Literal["synthetic", "public_sample", "arbitrary_upload"]
    evidence_class: str
    rgb_sha256: str
    sample_id: str | None
    source_dataset: str | None
    source_page: str | None
    image_license: str | None
    training_exposure: Mapping[str, str]
    ground_truth_used: Literal[False]
    ground_truth_not_loaded: Literal[True]


@dataclass(frozen=True)
class LiveSample:
    label: str
    image: np.ndarray
    evidence: LiveInputEvidence


@dataclass(frozen=True)
class PublicSelection:
    sample_ids: tuple[str, ...]
    universe_count: int
    ordered_universe_sha256: str


def _provenance_error() -> ValueError:
    return ValueError("public sample provenance validation failed")


def _attribute(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise _provenance_error()
    return value


def _evidence_error() -> ValueError:
    return ValueError("live input evidence validation failed")


def validate_live_input_evidence(evidence: object) -> object:
    """Validate the exact public/synthetic/upload evidence union."""
    if isinstance(evidence, Mapping):
        fields = set(evidence)
    else:
        try:
            fields = set(vars(evidence))
        except TypeError as exc:
            raise _evidence_error() from exc
    if fields != _EVIDENCE_FIELDS:
        raise _evidence_error()
    kind = _attribute(evidence, "kind")
    evidence_class = _attribute(evidence, "evidence_class")
    if kind not in _EVIDENCE_CLASSES or evidence_class != _EVIDENCE_CLASSES[kind]:
        raise _evidence_error()
    try:
        _sha256(_attribute(evidence, "rgb_sha256"))
    except ValueError as exc:
        raise _evidence_error() from exc
    if (
        _attribute(evidence, "ground_truth_used") is not False
        or _attribute(evidence, "ground_truth_not_loaded") is not True
    ):
        raise _evidence_error()
    exposure = _attribute(evidence, "training_exposure")
    public_metadata = tuple(
        _attribute(evidence, field)
        for field in ("sample_id", "source_dataset", "source_page", "image_license")
    )
    if kind == "public_sample":
        sample_id, source_dataset, source_page, image_license = public_metadata
        if (
            sample_id not in _LICENSE_ROWS
            or source_dataset != "isic2018"
            or source_page != "https://challenge.isic-archive.com/data/"
            or image_license != "CC-0"
            or not isinstance(exposure, Mapping)
            or dict(exposure) != _TRAINING_EXPOSURE
        ):
            raise _evidence_error()
    elif public_metadata != (None, None, None, None) or not isinstance(exposure, Mapping) or dict(exposure):
        raise _evidence_error()
    return evidence


def _safe_image_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise _provenance_error()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise _provenance_error() from exc
    return candidate


def _public_contract(release_manifest: object) -> tuple[object, tuple[object, ...], object]:
    # A caller-supplied object is only an adapter: authority is the canonical
    # manifest bytes, whose digest is pinned independently of the object.
    manifest_path = _attribute(release_manifest, "path")
    digest = _attribute(release_manifest, "digest")
    if not isinstance(manifest_path, (str, Path)):
        raise _provenance_error()
    try:
        path = Path(manifest_path)
        bytes_value = path.read_bytes()
        actual_digest = hashlib.sha256(bytes_value).hexdigest()
        if actual_digest != _CANONICAL_RELEASE_MANIFEST_SHA256 or digest != actual_digest:
            raise _provenance_error()
        canonical = load_release_manifest(path)
    except (OSError, ValueError) as exc:
        raise _provenance_error() from exc

    def projection(value: object) -> tuple[object, ...]:
        samples = _attribute(_attribute(value, "public_samples"), "samples")
        roles = _attribute(_attribute(value, "public_sample_contract"), "roles")
        if not isinstance(samples, (tuple, list)) or not isinstance(roles, Mapping):
            raise _provenance_error()
        sample_projection = tuple(
            tuple(_attribute(sample, field) for field in ("sample_id", "group_key", "sha256_raw", "sha256_rgb"))
            for sample in samples
        )
        role_projection = tuple(
            (name, tuple(_attribute(roles.get(name), field) for field in ("sample_id", "group_key", "sha256_raw", "sha256_rgb")))
            for name in ("A", "B", "boundary")
        )
        return (
            _attribute(value, "digest"),
            _attribute(_attribute(value, "public_sample_contract"), "state"),
            sample_projection,
            role_projection,
        )

    if projection(release_manifest) != projection(canonical):
        raise _provenance_error()
    release_manifest = canonical
    public = _attribute(release_manifest, "public_samples")
    selection = _attribute(public, "selection")
    samples = _attribute(public, "samples")
    contract = _attribute(release_manifest, "public_sample_contract")
    roles = _attribute(contract, "roles")
    if (
        selection is None or not isinstance(samples, (tuple, list)) or len(samples) != 2
        or _attribute(contract, "state") != "verified" or not isinstance(roles, Mapping)
        or tuple(roles) != ("A", "B", "boundary")
    ):
        raise _provenance_error()
    for name, expected in _CANONICAL_ROLE_PINS.items():
        role = _attribute(roles, name)
        if tuple(_attribute(role, field) for field in ("sample_id", "group_key", "sha256_raw", "sha256_rgb")) != expected:
            raise _provenance_error()
    return selection, tuple(samples), roles


def _selection_rows(index_path: Path) -> tuple[dict[str, object], ...]:
    try:
        import json

        payload = json.loads(index_path.read_text(encoding="ascii"))
        rows = payload["rows"]
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise _provenance_error() from exc
    if not isinstance(rows, list):
        raise _provenance_error()
    selected: list[dict[str, object]] = []
    for value in rows:
        if not isinstance(value, dict):
            raise _provenance_error()
        if (
            value.get("role") == "holdout"
            and value.get("split") == "train_screen_holdout"
            and value.get("source_split") == "train"
        ):
            for field in ("sample_id", "group_key", "sha256_raw", "sha256_rgb", "image_root", "image_relative"):
                if field not in value:
                    raise _provenance_error()
            _sha256(value["sha256_raw"])
            _sha256(value["sha256_rgb"])
            selected.append(value)
    return tuple(sorted(selected, key=lambda row: (str(row["sample_id"]), str(row["group_key"]))))


def recompute_public_selection(dataset_index: str | Path, release_manifest: object) -> PublicSelection:
    selection, samples, roles = _public_contract(release_manifest)
    index_path = Path(dataset_index)
    try:
        index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise _provenance_error() from exc
    if not hmac.compare_digest(index_hash, _sha256(_attribute(selection, "dataset_index_sha256"))):
        raise _provenance_error()
    rows = _selection_rows(index_path)
    universe_count = _attribute(selection, "universe_count")
    if (
        _attribute(selection, "universe") != "loop206_train_screen_holdout_clean"
        or _attribute(selection, "rule") != "explicit_roles_A_B_boundary_after_index_hash"
        or type(universe_count) is not int
        or len(rows) != universe_count
    ):
        raise _provenance_error()
    ordered = "".join(
        f"{row['sample_id']}|{row['group_key']}|{row['sha256_raw']}|{row['sha256_rgb']}\n"
        for row in rows
    )
    ordered_hash = hashlib.sha256(ordered.encode("ascii")).hexdigest()
    if not hmac.compare_digest(ordered_hash, _sha256(_attribute(selection, "ordered_universe_sha256"))):
        raise _provenance_error()
    by_id = {str(row["sample_id"]): row for row in rows}
    for name in ("A", "B", "boundary"):
        role = roles[name]
        row = by_id.get(str(_attribute(role, "sample_id")))
        if row is None or any(_attribute(role, field) != row[field] for field in ("sample_id", "group_key", "sha256_raw", "sha256_rgb")):
            raise _provenance_error()
    selected_ids = (str(_attribute(roles["A"], "sample_id")), str(_attribute(roles["B"], "sample_id")))
    if selected_ids != tuple(str(_attribute(sample, "sample_id")) for sample in samples):
        raise _provenance_error()
    for sample, role in zip(samples, (roles["A"], roles["B"])):
        row = by_id[str(_attribute(role, "sample_id"))]
        if any(
            _attribute(sample, field) != row[field]
            for field in ("sample_id", "group_key", "sha256_raw", "sha256_rgb")
        ):
            raise _provenance_error()
    return PublicSelection(selected_ids, universe_count, ordered_hash)


def _validate_sample_metadata(sample: object, row: Mapping[str, object]) -> None:
    exact = (
        ("sample_id", row["sample_id"]),
        ("group_key", row["group_key"]),
        ("sha256_raw", row["sha256_raw"]),
        ("sha256_rgb", row["sha256_rgb"]),
        ("source_dataset", "isic2018"),
        ("source_page", "https://challenge.isic-archive.com/data/"),
        ("license_id", "CC-0"),
        ("ground_truth_used", False),
        ("ground_truth_not_loaded", True),
    )
    if any(_attribute(sample, field) != expected for field, expected in exact):
        raise _provenance_error()
    license_evidence = _attribute(sample, "license_evidence")
    expected_license = _LICENSE_ROWS.get(str(_attribute(sample, "sample_id")))
    if expected_license is None or _attribute(license_evidence, "csv_row_number") != expected_license[0]:
        raise _provenance_error()
    if _attribute(license_evidence, "clean_v3_manifest_sha256") != _CLEAN_V3_MANIFEST_SHA256 or _attribute(license_evidence, "raw_csv_row_sha256") != expected_license[1]:
        raise _provenance_error()
    exposure = _attribute(sample, "training_exposure")
    if not isinstance(exposure, Mapping) or dict(exposure) != _TRAINING_EXPOSURE:
        raise _provenance_error()


def load_public_live_samples(
    release_manifest: object, dataset_index: str | Path, roots: Sequence[str | Path]
) -> dict[str, LiveSample]:
    selection, records, _roles = _public_contract(release_manifest)
    selection_result = recompute_public_selection(dataset_index, release_manifest)
    rows = {str(row["sample_id"]): row for row in _selection_rows(Path(dataset_index))}
    resolved_roots = tuple(Path(root).expanduser().resolve() for root in roots)
    loaded: dict[str, LiveSample] = {}
    for record in records:
        sample_id = str(_attribute(record, "sample_id"))
        row = rows.get(sample_id)
        if row is None:
            raise _provenance_error()
        _validate_sample_metadata(record, row)
        root_index = row["image_root"]
        if type(root_index) is not int or root_index not in range(len(resolved_roots)):
            raise _provenance_error()
        path = _safe_image_path(resolved_roots[root_index], row["image_relative"])
        try:
            snapshot = ImmutableSnapshot.read(path)
            image = snapshot.decode_rgb()
        except (OSError, ValueError) as exc:
            raise _provenance_error() from exc
        if not hmac.compare_digest(snapshot.sha256, str(row["sha256_raw"])) or not hmac.compare_digest(snapshot.decoded_rgb_sha256(image), str(row["sha256_rgb"])):
            raise _provenance_error()
        exposure = dict(_attribute(record, "training_exposure"))
        evidence = LiveInputEvidence("public_sample", "illustrative_public_sample_no_ground_truth", rgb_sha256(image), sample_id, str(_attribute(record, "source_dataset")), str(_attribute(record, "source_page")), str(_attribute(record, "license_id")), exposure, False, True)
        validate_live_input_evidence(evidence)
        loaded[sample_id] = LiveSample(f"{sample_id} / public; L206 excluded; L192 included; no ground truth", np.ascontiguousarray(image), evidence)
    if tuple(loaded) != selection_result.sample_ids or _attribute(selection, "ordered_universe_sha256") != selection_result.ordered_universe_sha256:
        raise _provenance_error()
    return loaded


def synthetic_evidence(image: np.ndarray) -> LiveInputEvidence:
    evidence = LiveInputEvidence("synthetic", "illustrative_synthetic_no_ground_truth", rgb_sha256(image), None, None, None, None, {}, False, True)
    validate_live_input_evidence(evidence)
    return evidence


def upload_evidence(image: np.ndarray) -> LiveInputEvidence:
    evidence = LiveInputEvidence("arbitrary_upload", "illustrative_arbitrary_upload_no_ground_truth", rgb_sha256(image), None, None, None, None, {}, False, True)
    validate_live_input_evidence(evidence)
    return evidence
