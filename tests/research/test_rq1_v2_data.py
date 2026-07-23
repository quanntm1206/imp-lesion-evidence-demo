from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from lesion_robustness.research import rq1_data as rq1
from lesion_robustness.research.rq1_data import (
    DataIntegrityReport,
    DataRow,
    audit_data,
    canonical_report_sha256,
    load_protocol,
    ordered_identity_sha256,
    read_authorized_rows,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments" / "rq1_v2" / "protocol.json"
TRAIN_COUNT = 2008
VALIDATION_COUNT = 431
INDEX_DIGEST = "d" * 64
CLEAN_MANIFEST_DIGEST = "4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102"


def _sha256_rgb(rgb: np.ndarray) -> str:
    payload = f"{rgb.shape[0]}x{rgb.shape[1]}x3|".encode("ascii") + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _verified_protocol(index: Path, *, digest: str | None = None):
    return replace(
        load_protocol(PROTOCOL),
        dataset_index_status="verified",
        dataset_index_sha256=digest or hashlib.sha256(index.read_bytes()).hexdigest(),
    )


def _row(
    sample_id: str,
    split: str,
    group: str,
    rgb: np.ndarray,
    *,
    index_digest: str = INDEX_DIGEST,
) -> DataRow:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    return DataRow(
        sample_id=sample_id,
        split=split,
        group_key=group,
        sha256_raw=hashlib.sha256((sample_id + "-raw").encode("ascii")).hexdigest(),
        sha256_rgb=_sha256_rgb(rgb),
        source_dataset="isic2018",
        image_rgb=rgb,
        dataset_index_sha256=index_digest,
        clean_v3_manifest_sha256=CLEAN_MANIFEST_DIGEST,
    )


def _protocol(rows: tuple[DataRow, ...]):
    base = load_protocol(PROTOCOL)
    train = tuple(row for row in rows if row.split == "train")
    validation = tuple(row for row in rows if row.split == "validation")
    return replace(
        base,
        dataset_index_status="verified",
        dataset_index_sha256=rows[0].dataset_index_sha256,
        train_ordered_identity_sha256=ordered_identity_sha256(train),
        validation_ordered_identity_sha256=ordered_identity_sha256(validation),
    )


@lru_cache(maxsize=1)
def _clean_rows() -> tuple[DataRow, ...]:
    generator = np.random.default_rng(206)
    rows: list[DataRow] = []
    for index in range(TRAIN_COUNT):
        rows.append(
            _row(
                f"RQ1v2-train-{index:04d}",
                "train",
                f"group-train-{index:04d}",
                generator.integers(0, 256, (12, 12, 3), dtype=np.uint8),
            )
        )
    for index in range(VALIDATION_COUNT):
        rows.append(
            _row(
                f"RQ1v2-validation-{index:04d}",
                "validation",
                f"group-validation-{index:04d}",
                generator.integers(0, 256, (12, 12, 3), dtype=np.uint8),
            )
        )
    return tuple(rows)


def test_ordered_identity_uses_exact_four_field_ascii_record() -> None:
    rows = _clean_rows()
    expected = "".join(
        f"{row.sample_id}|{row.group_key}|{row.sha256_raw}|{row.sha256_rgb}\n"
        for row in sorted(rows, key=lambda item: (item.sample_id, item.group_key))
    )

    assert ordered_identity_sha256(reversed(rows)) == hashlib.sha256(expected.encode("ascii")).hexdigest()


def test_data_audit_accepts_clean_fixture_and_emits_path_free_report() -> None:
    rows = _clean_rows()
    report = audit_data(rows, _protocol(rows))
    payload = report.to_dict()

    assert report.train_count == TRAIN_COUNT
    assert report.validation_count == VALIDATION_COUNT
    assert report.cross_split_groups == 0
    assert report.cross_split_exact_rgb == 0
    assert report.cross_split_near_rgb == 0
    assert report.test_v3_open_count == 0
    assert payload["canonical_report_sha256"] == canonical_report_sha256(payload)
    assert "path" not in json.dumps(payload, sort_keys=True).lower()


@pytest.mark.parametrize("drift", ["group", "exact_rgb", "near_rgb"])
def test_data_audit_rejects_cross_split_leakage(drift: str) -> None:
    rows = list(_clean_rows())
    validation_index = TRAIN_COUNT
    if drift == "group":
        rows[validation_index] = replace(
            rows[validation_index], group_key=rows[0].group_key
        )
    elif drift == "exact_rgb":
        rows[validation_index] = replace(
            rows[validation_index],
            image_rgb=rows[0].image_rgb,
            sha256_rgb=rows[0].sha256_rgb,
        )
    else:
        near = rows[0].image_rgb.copy()
        near[0, 0, 0] ^= np.uint8(1)
        rows[validation_index] = replace(
            rows[validation_index], image_rgb=near, sha256_rgb=_sha256_rgb(near)
        )

    with pytest.raises(ValueError, match="cross-split leakage"):
        audit_data(tuple(rows), _protocol(tuple(rows)))


def test_data_audit_rejects_phash_candidate_below_ssim_threshold() -> None:
    base = np.full((48, 48, 3), 100, dtype=np.uint8)
    different = np.full((48, 48, 3), 155, dtype=np.uint8)

    assert (rq1.phash63_luminance(base) ^ rq1.phash63_luminance(different)).bit_count() <= 4
    assert rq1.ssim_luminance_256(base, different) < 0.98


def test_data_audit_rejects_decoded_rgb_hash_drift() -> None:
    rows = list(_clean_rows())
    rows[1] = replace(rows[1], sha256_rgb="f" * 64)

    with pytest.raises(ValueError, match="decoded-RGB SHA-256 mismatch"):
        audit_data(tuple(rows), _protocol(tuple(rows)))


def test_reader_opens_only_authorized_split_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    rgb = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:8, 3:10] = 255
    image_path = data_root / "train.png"
    mask_path = data_root / "train-mask.png"
    Image.fromarray(rgb, mode="RGB").save(image_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    image_raw = hashlib.sha256(image_path.read_bytes()).hexdigest()
    mask_raw = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": "imp.rq1_v2.dataset_index.v1",
        "clean_v3_manifest_sha256": "4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102",
        "roots": [str(data_root)],
        "rows": [
            {
                "sample_id": "RQ1v2-train-a",
                "split": "train",
                "group_key": "group-a",
                "source_dataset": "isic2018",
                "image_root": 0,
                "image_relative": image_path.name,
                "mask_root": 0,
                "mask_relative": mask_path.name,
                "sha256_raw": image_raw,
                "sha256_rgb": _sha256_rgb(rgb),
                "mask_sha256": mask_raw,
            },
            {
                "sample_id": "RQ1v2-test-a",
                "split": "test",
                "group_key": "sealed-group",
                "source_dataset": "ph2",
                "image_root": 0,
                "image_relative": "../sealed-test-v3.png",
                "mask_root": 0,
                "mask_relative": "../sealed-test-v3-mask.png",
                "sha256_raw": "0" * 64,
                "sha256_rgb": "1" * 64,
                "mask_sha256": "2" * 64,
            },
        ],
    }
    index = tmp_path / "index.json"
    index.write_text(json.dumps(payload), encoding="ascii")

    protocol = _verified_protocol(index)
    rows = read_authorized_rows(index, "train", protocol)

    assert len(rows) == 1
    assert rows[0].sample_id == "RQ1v2-train-a"
    assert np.array_equal(rows[0].image_rgb, rgb)


def test_reader_rejects_ph2_before_resolving_referenced_files(tmp_path: Path) -> None:
    payload = {
        "schema_version": "imp.rq1_v2.dataset_index.v1",
        "clean_v3_manifest_sha256": "4e86d251231bc105167910c6b5a41fc29900dff41342405657a5c2eccb67b102",
        "roots": [str(tmp_path)],
        "rows": [
            {
                "sample_id": "RQ1v2-train-ph2",
                "split": "train",
                "group_key": "group-ph2",
                "source_dataset": "PH2",
                "image_relative": "../must-not-resolve.png",
            }
        ],
    }
    index = tmp_path / "index.json"
    index.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ValueError, match="PH2 is not authorized"):
        protocol = _verified_protocol(index)
        read_authorized_rows(index, "train", protocol)


def _blocked_row(sample_id: str = "RQ1v2-train-blocked") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": "train",
        "group_key": "group-blocked",
        "source_dataset": "isic2018",
        "image_root": 0,
        "image_relative": "../must-not-resolve.png",
        "mask_root": 0,
        "mask_relative": "../must-not-resolve-mask.png",
        "sha256_raw": "0" * 64,
        "sha256_rgb": "1" * 64,
        "mask_sha256": "2" * 64,
    }


def _write_index(
    tmp_path: Path,
    rows: list[object],
    *,
    manifest_digest: str = CLEAN_MANIFEST_DIGEST,
    roots: list[str] | None = None,
) -> Path:
    index = tmp_path / "index.json"
    payload = {
        "schema_version": "imp.rq1_v2.dataset_index.v1",
        "clean_v3_manifest_sha256": manifest_digest,
        "roots": roots or [str(tmp_path)],
        "rows": rows,
    }
    index.write_text(json.dumps(payload), encoding="ascii")
    return index


def test_reader_rejects_wrong_index_pin_before_referenced_path_resolution(
    tmp_path: Path,
) -> None:
    index = _write_index(tmp_path, [_blocked_row()])

    with pytest.raises(ValueError, match="dataset index SHA-256 mismatch"):
        read_authorized_rows(index, "train", _verified_protocol(index, digest="0" * 64))


def test_reader_rejects_manifest_pin_before_referenced_path_resolution(
    tmp_path: Path,
) -> None:
    index = _write_index(tmp_path, [_blocked_row()], manifest_digest="0" * 64)

    with pytest.raises(ValueError, match="Clean-v3 manifest SHA-256 mismatch"):
        read_authorized_rows(index, "train", _verified_protocol(index))


def test_reader_rejects_selected_row_path_escape(tmp_path: Path) -> None:
    index = _write_index(tmp_path, [_blocked_row()])

    with pytest.raises(ValueError, match="escapes its authorized root"):
        read_authorized_rows(index, "train", _verified_protocol(index))


def test_reader_rejects_duplicate_selected_row_before_path_resolution(
    tmp_path: Path,
) -> None:
    index = _write_index(tmp_path, [_blocked_row(), _blocked_row()])

    with pytest.raises(ValueError, match="duplicate sample_id"):
        read_authorized_rows(index, "train", _verified_protocol(index))


def test_reader_rejects_malformed_selected_row_before_path_resolution(
    tmp_path: Path,
) -> None:
    malformed = _blocked_row()
    malformed.pop("sha256_raw")
    index = _write_index(tmp_path, [malformed])

    with pytest.raises(ValueError, match="sha256_raw"):
        read_authorized_rows(index, "train", _verified_protocol(index))


def test_reader_rejects_noncanonical_task8_id_before_path_resolution(
    tmp_path: Path,
) -> None:
    index = _write_index(tmp_path, [_blocked_row("ISIC_0000001")])

    with pytest.raises(ValueError, match="must begin RQ1v2-"):
        read_authorized_rows(index, "train", _verified_protocol(index))


def test_audit_rejects_noncanonical_task8_id() -> None:
    rows = list(_clean_rows())
    rows[0] = replace(rows[0], sample_id="ISIC_0000001")

    with pytest.raises(ValueError, match="must begin RQ1v2-"):
        audit_data(tuple(rows), _protocol(tuple(rows)))


def test_reader_rejects_symlinked_selected_file_when_supported(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not-opened")
    link = root / "linked.png"
    try:
        os.symlink(outside, link)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    row = _blocked_row()
    row["image_relative"] = link.name
    row["mask_relative"] = link.name
    index = _write_index(tmp_path, [row], roots=[str(root)])

    with pytest.raises(ValueError, match="reparse|symlink"):
        read_authorized_rows(index, "train", _verified_protocol(index))


def test_phash63_matches_independent_orthonormal_dct_reference() -> None:
    gray = np.random.default_rng(1206).integers(0, 256, (32, 32), dtype=np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    size = 32
    positions = np.arange(size, dtype=np.float64)
    frequencies = np.arange(8, dtype=np.float64)[:, None]
    basis = np.sqrt(2.0 / size) * np.cos(
        np.pi * (2.0 * positions + 1.0) * frequencies / (2.0 * size)
    )
    basis[0] = np.sqrt(1.0 / size)
    low = (basis @ gray.astype(np.float64) @ basis.T).reshape(-1)[1:]
    median = float(np.median(low))
    expected = 0
    for bit in low > median:
        expected = (expected << 1) | int(bool(bit))

    assert rq1.phash63_luminance(rgb) == expected


def test_ssim_matches_independent_constant_luminance_reference() -> None:
    first_value = 100
    second_value = 155
    first = np.full((19, 23, 3), first_value, dtype=np.uint8)
    second = np.full((31, 17, 3), second_value, dtype=np.uint8)
    c1 = (0.01 * 255.0) ** 2
    expected = (2.0 * first_value * second_value + c1) / (
        first_value**2 + second_value**2 + c1
    )

    assert rq1.ssim_luminance_256(first, second) == pytest.approx(expected, abs=1e-12)


def test_canonical_report_has_fixed_golden_digest() -> None:
    report = DataIntegrityReport(
        schema_version="imp.rq1_v2.data_integrity_report.v1",
        audit_id="RQ1v2-golden-report",
        train_count=2008,
        validation_count=431,
        train_ordered_identity_sha256="1" * 64,
        validation_ordered_identity_sha256="2" * 64,
        cross_split_groups=0,
        cross_split_exact_rgb=0,
        near_duplicate_candidate_count=3,
        cross_split_near_rgb=0,
        clean_v3_manifest_sha256=CLEAN_MANIFEST_DIGEST,
        dataset_index_status="verified",
        dataset_index_sha256="d" * 64,
        test_v3_access=False,
        test_v3_open_count=0,
        algorithms={"identity": "golden"},
    )

    assert report.to_dict()["canonical_report_sha256"] == (
        "c063e3091c21a3414fda500dcd33a244036ea88878a8f183d5c103cc7eecb6ca"
    )
