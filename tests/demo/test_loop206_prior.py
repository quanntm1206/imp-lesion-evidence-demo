from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lesion_robustness.demo.loop206_prior import (
    build_prior_artifact,
    PriorFitRow,
    fit_deployment_prior,
    load_prior,
    save_prior,
    sha256_file,
)
from lesion_robustness import loop204_protocol
import lesion_robustness.demo.loop206_prior as prior_module


def _validated_candidate_stub(path: Path):
    data = b""
    return prior_module.ValidatedCandidateCache(
        payload={},
        manifest_sha256=sha256_file(path),
        data_snapshot=prior_module.ImmutableSnapshot.from_bytes(data),
    )


@pytest.fixture
def tiny_fit_rows() -> list[PriorFitRow]:
    rows: list[PriorFitRow] = []
    for index in range(2):
        image = np.full((32, 32, 3), 35 + index * 20, dtype=np.uint8)
        image[8:24, 7:25, 0] = 180
        image[8:24, 7:25, 1] = 90 + index * 10
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 7:25] = 255
        rows.append(
            PriorFitRow(
                sample_id=f"tiny-{index}",
                group_key=f"group-{index}",
                image=image,
                mask=mask,
                dataset_index=index,
            )
        )
    return rows


def test_prior_round_trip_is_deterministic(
    tiny_fit_rows: list[PriorFitRow], tmp_path: Path
) -> None:
    prior = fit_deployment_prior(tiny_fit_rows, n_jobs=1, parity_passed=True)
    path = tmp_path / "prior.joblib"
    save_prior(prior, path)
    loaded = load_prior(path, expected_sha256=sha256_file(path))
    first = loaded.predict(tiny_fit_rows[0].image)
    second = loaded.predict(tiny_fit_rows[0].image.copy())
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8
    assert set(np.unique(first)).issubset({0, 255})


def test_prior_fit_defaults_to_parity_not_passed(
    tiny_fit_rows: list[PriorFitRow],
) -> None:
    prior = fit_deployment_prior(tiny_fit_rows, n_jobs=1)

    assert prior.parity_passed is False


def _replace_after_first_binary_read(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    replacement: bytes,
) -> None:
    original_open = Path.open
    replaced = False

    class ReplacingHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):
            nonlocal replaced
            result = self._handle.__exit__(exc_type, exc, traceback)
            if not replaced:
                replaced = True
                with original_open(target, "wb") as output:
                    output.write(replacement)
            return result

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def guarded_open(path: Path, *args, **kwargs):
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        handle = original_open(path, *args, **kwargs)
        if Path(path).resolve() == target.resolve() and mode == "rb" and not replaced:
            return ReplacingHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", guarded_open)


def _loaded_bytes(source) -> bytes:
    if hasattr(source, "data"):
        return bytes(source.data)
    if hasattr(source, "read"):
        return source.read()
    return Path(source).read_bytes()


def test_prior_hash_and_joblib_load_use_the_same_captured_bytes(
    tiny_fit_rows: list[PriorFitRow],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import joblib

    prior = fit_deployment_prior(tiny_fit_rows, n_jobs=1, parity_passed=False)
    artifact = tmp_path / "prior.joblib"
    save_prior(prior, artifact)
    expected_bytes = artifact.read_bytes()
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    _replace_after_first_binary_read(monkeypatch, artifact, b"replacement-prior")
    observed: list[bytes] = []

    def fake_load(source, *args, **kwargs):
        observed.append(_loaded_bytes(source))
        return prior

    monkeypatch.setattr(joblib, "load", fake_load)
    monkeypatch.setattr(prior_module, "_validate_loaded_prior", lambda _prior: None)

    loaded = load_prior(artifact, expected_sha256=expected_hash)

    assert loaded is prior
    assert observed == [expected_bytes]


def test_candidate_channel_requires_parity_receipt(
    tiny_fit_rows: list[PriorFitRow],
) -> None:
    prior = replace(
        fit_deployment_prior(tiny_fit_rows, n_jobs=1),
        parity_passed=False,
    )
    with pytest.raises(RuntimeError, match="parity"):
        prior.predict(np.zeros((32, 32, 3), dtype=np.uint8))


def test_load_rejects_artifact_hash_mismatch(
    tiny_fit_rows: list[PriorFitRow], tmp_path: Path
) -> None:
    path = tmp_path / "prior.joblib"
    save_prior(fit_deployment_prior(tiny_fit_rows, n_jobs=1), path)
    with pytest.raises(ValueError, match="artifact SHA256"):
        load_prior(path, expected_sha256="0" * 64)


def test_parity_failure_deletes_artifact_and_records_diagnostics(
    tiny_fit_rows: list[PriorFitRow], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import lesion_robustness.demo.loop206_prior as module

    index = tmp_path / "index.json"
    candidate = tmp_path / "candidate.json"
    index.write_text("{}", encoding="ascii")
    candidate.write_text("{}", encoding="ascii")
    prior = fit_deployment_prior(tiny_fit_rows, n_jobs=1, parity_passed=False)
    monkeypatch.setattr(
        module,
        "load_dataset_index",
        lambda *_args, **_kwargs: (tiny_fit_rows, [object()], {}),
    )
    monkeypatch.setattr(
        module,
        "validate_candidate_manifest",
        lambda *_args, **_kwargs: _validated_candidate_stub(candidate),
    )
    monkeypatch.setattr(module, "fit_deployment_prior", lambda *_args, **_kwargs: prior)
    monkeypatch.setattr(
        module,
        "verify_holdout_parity",
        lambda *_args, **_kwargs: {
            "expected": 76,
            "input_rgb_hash_matches": 76,
            "contour_byte_matches": 0,
            "parity_passed": False,
            "mismatch_groups": ["group-0"],
        },
    )
    output = tmp_path / "prior.joblib"
    receipt = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="parity failed"):
        build_prior_artifact(
            dataset_index=index,
            candidate_manifest=candidate,
            output=output,
            receipt=receipt,
        )
    assert not output.exists()
    assert not output.with_name(output.name + ".parity-tmp").exists()
    payload = json.loads(receipt.read_text(encoding="ascii"))
    assert payload["status"] == "failed"
    assert payload["selected_threshold"] == prior.selected_threshold
    assert payload["parity"]["input_rgb_hash_matches"] == 76
    assert payload["parity"]["contour_byte_matches"] == 0


def _candidate_rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(308):
        rows.append(
            {
                "index": index,
                "sample_id": f"fit-{index}",
                "group_key": f"fit-group-{index}",
                "image_path": f"/legacy/fit-{index}.jpg",
                "corruption": "clean",
                "source_split": "train",
                "runtime_split": "train",
                "fold": index // 77,
                "holdout_dataset_index": None,
                "input_rgb_sha256": "a" * 64,
                "base_threshold": (
                    0.05 if index // 77 == 2 else 0.07500000000000001
                ),
                "locked_config": "neutral_mid_30_s2",
                "candidate_fallback_used": False,
                "candidate_fallback_reason": "none",
            }
        )
    next_index = len(rows)
    for holdout_index in range(76):
        for corruption in ("clean", "low_contrast", "gaussian_noise"):
            rows.append(
                {
                    "index": next_index,
                    "sample_id": f"holdout-{holdout_index}",
                    "group_key": f"holdout-group-{holdout_index}",
                    "image_path": f"/legacy/holdout-{holdout_index}.jpg",
                    "corruption": corruption,
                    "source_split": "train",
                    "runtime_split": "train_screen_holdout",
                    "fold": 4,
                    "holdout_dataset_index": holdout_index,
                    "input_rgb_sha256": "b" * 64,
                    "base_threshold": 0.07500000000000001,
                    "locked_config": "neutral_mid_30_s2",
                    "candidate_fallback_used": False,
                    "candidate_fallback_reason": "none",
                }
            )
            next_index += 1
    return rows


@pytest.fixture
def candidate_manifest(tmp_path: Path) -> tuple[Path, dict]:
    rows = _candidate_rows()
    data_path = tmp_path / "contours.uint8.mmap"
    with data_path.open("wb") as handle:
        handle.truncate(536 * 384 * 384)
    payload = {
        "schema_version": "loop206.leakage_safe_pilot_cache.v2",
        "artifact_type": "loop206_packed_binary_channel",
        "status": "passed",
        "arm": "candidate",
        "count": 536,
        "shape": [384, 384],
        "source_row_count": 384,
        "fit_clean_rows": 308,
        "holdout_rows_per_corruption": 76,
        "source_split_counts": {"train": 536},
        "allowed_runtime_splits": ["train", "train_screen_holdout"],
        "runtime_split_counts": {"train": 308, "train_screen_holdout": 228},
        "corruption_counts": {"clean": 384, "gaussian_noise": 76, "low_contrast": 76},
        "input_rgb_sha256_count": 536,
        "rows_sha256": loop204_protocol.sha256_bytes(
            loop204_protocol.canonical_json_bytes(rows)
        ),
        "locked_active_contour_config": prior_module.asdict(
            prior_module._canonical_active_config()
        ),
        "data": {
            "file": data_path.name,
            "dtype": "uint8",
            "sha256": sha256_file(data_path),
        },
        "provenance": {
            "builder_sha256": "c" * 64,
            "config_sha256": sha256_file(
                "configs/loop206/l206_control_train_screen_pilot20.yaml"
            ),
            "confirmatory_report_sha256": "d" * 64,
            "loop204_protocol_sha256": sha256_file(
                prior_module.Path(prior_module.loop204_protocol.__file__).resolve()
            ),
            "loop205_protocol_sha256": sha256_file(
                prior_module.Path(prior_module.loop205_protocol.__file__).resolve()
            ),
            "loop206_protocol_sha256": sha256_file(
                prior_module.Path(prior_module.loop206_active_contour.__file__).resolve()
            ),
            "runtime_manifest_sha256": prior_module.load_config(
                "configs/loop206/l206_control_train_screen_pilot20.yaml"
            )["data"]["manifest_sha256"],
            "source_manifest_sha256": "f" * 64,
        },
        "rows": rows,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    return path, payload


@pytest.mark.parametrize("forgery", ["provenance", "index", "threshold"])
def test_candidate_manifest_rejects_forged_contract(
    candidate_manifest: tuple[Path, dict], forgery: str
) -> None:
    path, original = candidate_manifest
    payload = deepcopy(original)
    if forgery == "provenance":
        payload["provenance"]["loop205_protocol_sha256"] = "0" * 64
    elif forgery == "index":
        payload["rows"][-1]["index"] = 0
    else:
        payload["rows"][0]["base_threshold"] = 0.1
    payload["rows_sha256"] = loop204_protocol.sha256_bytes(
        loop204_protocol.canonical_json_bytes(payload["rows"])
    )
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match=forgery):
        prior_module.validate_candidate_manifest(
            path, expected_base_threshold=0.07500000000000001
        )


def test_validated_candidate_cache_owns_the_verified_data_snapshot(
    candidate_manifest: tuple[Path, dict],
) -> None:
    path, payload = candidate_manifest
    expected_manifest = path.read_bytes()
    data_path = path.parent / payload["data"]["file"]
    expected_data_hash = sha256_file(data_path)

    validated = prior_module.validate_candidate_manifest(
        path, expected_base_threshold=0.07500000000000001
    )
    data_path.write_bytes(b"replaced-after-validation")

    assert validated.manifest_sha256 == hashlib.sha256(expected_manifest).hexdigest()
    assert validated.data_snapshot.sha256 == expected_data_hash
    assert validated.data_snapshot.size == 536 * 384 * 384
    assert validated.data_snapshot.is_file_backed


def test_holdout_parity_uses_validated_cache_snapshot_not_reopened_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    contour = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    data_snapshot = prior_module.ImmutableSnapshot.from_bytes(contour.tobytes())
    manifest_bytes = b'{"validated":true}'
    validated = prior_module.ValidatedCandidateCache(
        payload={
            "shape": [2, 2],
            "count": 1,
            "rows": [
                {
                    "index": 0,
                    "fold": 4,
                    "corruption": "clean",
                    "group_key": "holdout-group",
                    "sample_id": "holdout-sample",
                    "input_rgb_sha256": prior_module.sha256_rgb_array(processed),
                }
            ],
        },
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        data_snapshot=data_snapshot,
    )
    mutable_data = tmp_path / "contours.uint8.mmap"
    mutable_data.write_bytes(bytes([99]) * 4)
    monkeypatch.setattr(
        prior_module,
        "validate_candidate_manifest",
        lambda *_args, **_kwargs: validated,
    )
    monkeypatch.setattr(prior_module, "EXPECTED_HOLDOUT_ROWS", 1)

    class FakePrior:
        selected_threshold = 0.07500000000000001

        def _preprocess(self, image, *, corruption, dataset_index):
            return processed

        def _contour_preprocessed(self, image):
            return contour

    holdout = prior_module.PriorHoldoutRow(
        sample_id="holdout-sample",
        group_key="holdout-group",
        image=processed,
        dataset_index=0,
    )

    result = prior_module.verify_holdout_parity(
        FakePrior(), [holdout], tmp_path / "manifest.json"
    )

    assert result["parity_passed"] is True
    assert result["candidate_manifest_sha256"] == validated.manifest_sha256
    assert result["candidate_data_sha256"] == data_snapshot.sha256


def _dataset_payload() -> dict:
    rows = []
    for index in range(384):
        role = "holdout" if index >= 308 else "fit"
        fold = 4 if role == "holdout" else index // 77
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "group_key": f"group-{index}",
                "role": role,
                "fold": fold,
                "split": "train_screen_holdout" if role == "holdout" else "train",
                "source_split": "train",
                "image_root": 0,
                "mask_root": 0,
                "image_relative": f"images/{index}.png",
                "mask_relative": f"masks/{index}.png",
                "sha256_raw": "0" * 64,
                "sha256_rgb": "1" * 64,
            }
        )
    return {
        "schema_version": "loop206.demo.dataset_index.v1",
        "root_count": 1,
        "row_count": 384,
        "fit_count": 308,
        "holdout_count": 76,
        "rows": rows,
    }


@pytest.mark.parametrize("inconsistency", ["row_count", "fold", "root"])
def test_dataset_index_rejects_internal_inconsistency(
    tmp_path: Path, inconsistency: str
) -> None:
    payload = _dataset_payload()
    roots = [tmp_path / "root", tmp_path / "extra"]
    roots[0].mkdir()
    roots[1].mkdir()
    if inconsistency == "row_count":
        payload["row_count"] = 383
    elif inconsistency == "fold":
        payload["rows"][0]["fold"] = 4
    else:
        payload["rows"][0]["image_root"] = 1
    path = tmp_path / "index.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match=inconsistency.replace("_", " ")):
        prior_module.load_dataset_index(path, dataset_roots=roots)


def test_dataset_index_rejects_path_escape(tmp_path: Path) -> None:
    payload = _dataset_payload()
    payload["rows"][0]["image_relative"] = "../escape.png"
    root = tmp_path / "root"
    root.mkdir()
    path = tmp_path / "index.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="escapes root"):
        prior_module.load_dataset_index(path, dataset_roots=[root])


def test_indexed_mask_hash_and_decode_use_one_captured_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 1
    Image.fromarray(image, mode="RGB").save(image_path)
    Image.fromarray(mask * 255, mode="L").save(mask_path)
    image_snapshot = prior_module.ImmutableSnapshot.read(image_path)
    mask_snapshot = prior_module.ImmutableSnapshot.read(mask_path)
    row = {
        "sample_id": "sample",
        "sha256_raw": image_snapshot.sha256,
        "sha256_rgb": image_snapshot.decoded_rgb_sha256(image),
        "mask_sha256_raw": mask_snapshot.sha256,
        "mask_sha256_binary": mask_snapshot.decoded_binary_mask_sha256(mask),
    }
    _replace_after_first_binary_read(monkeypatch, mask_path, b"replacement")

    loaded_image, loaded_mask = prior_module._load_verified_indexed_pair(
        image_path, mask_path, row
    )

    np.testing.assert_array_equal(loaded_image, image)
    np.testing.assert_array_equal(loaded_mask, mask)


def _passed_receipt(prior, artifact: Path) -> dict:
    payload = {
        "schema_version": prior_module.RECEIPT_SCHEMA,
        "created_at": "2026-07-20T00:00:00Z",
        "status": "passed",
        "dataset_index_sha256": prior.manifest_sha256,
        "candidate_manifest_sha256": "c" * 64,
        "fit_groups": 308,
        "fit_group_sha256": prior.fit_group_sha256,
        "selected_threshold": prior.selected_threshold,
        "artifact_sha256": sha256_file(artifact),
        "artifact_schema": prior.schema_version,
        "loop205_config": prior.loop205_config,
        "loop206_config": prior.loop206_config,
        "feature_names": list(prior.feature_names),
        "sklearn_version": prior.sklearn_version,
        "code_hashes": prior.code_hashes,
        "parity": {
            "expected": 76,
            "input_rgb_hash_matches": 76,
            "contour_byte_matches": 76,
            "parity_passed": True,
            "mismatch_groups": [],
            "candidate_manifest_sha256": "c" * 64,
            "candidate_data_sha256": "d" * 64,
        },
    }
    payload["content_sha256"] = prior_module._canonical_hash(payload)
    return payload


def test_deployment_loader_binds_passed_receipt(
    tiny_fit_rows: list[PriorFitRow], tmp_path: Path
) -> None:
    prior = fit_deployment_prior(tiny_fit_rows, n_jobs=1, parity_passed=True)
    artifact = tmp_path / "prior.joblib"
    save_prior(prior, artifact)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(_passed_receipt(prior, artifact)), encoding="ascii")
    loaded = prior_module.load_deployment_prior(
        artifact,
        receipt,
        expected_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
    )
    assert loaded.parity_passed is True


def test_deployment_receipt_hash_describes_the_loaded_receipt_bytes(
    tiny_fit_rows: list[PriorFitRow],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = fit_deployment_prior(tiny_fit_rows, n_jobs=1, parity_passed=True)
    artifact = tmp_path / "prior.joblib"
    save_prior(prior, artifact)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(_passed_receipt(prior, artifact)), encoding="ascii")
    expected_receipt_bytes = receipt.read_bytes()
    expected_receipt_sha256 = hashlib.sha256(expected_receipt_bytes).hexdigest()
    _replace_after_first_binary_read(monkeypatch, receipt, b'{"status":"replaced"}')

    loaded, receipt_sha256 = prior_module.load_deployment_prior_with_receipt_hash(
        artifact,
        receipt,
        expected_receipt_sha256=expected_receipt_sha256,
    )

    assert loaded.parity_passed is True
    assert receipt_sha256 == hashlib.sha256(expected_receipt_bytes).hexdigest()


@pytest.mark.parametrize("forgery", ["content", "artifact", "group", "parity"])
def test_deployment_loader_rejects_forged_receipt(
    tiny_fit_rows: list[PriorFitRow], tmp_path: Path, forgery: str
) -> None:
    prior = fit_deployment_prior(tiny_fit_rows, n_jobs=1, parity_passed=True)
    artifact = tmp_path / "prior.joblib"
    save_prior(prior, artifact)
    payload = _passed_receipt(prior, artifact)
    if forgery == "content":
        payload["content_sha256"] = "0" * 64
    elif forgery == "artifact":
        payload["artifact_sha256"] = "0" * 64
    elif forgery == "group":
        payload["fit_group_sha256"] = "0" * 64
    else:
        payload["parity"]["contour_byte_matches"] = 75
    if forgery != "content":
        payload.pop("content_sha256")
        payload["content_sha256"] = prior_module._canonical_hash(payload)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match=forgery):
        prior_module.load_deployment_prior(
            artifact,
            receipt,
            expected_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        )


def test_receipt_pin_mismatch_prevents_parse_and_unpickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    artifact = tmp_path / "prior.joblib"
    receipt.write_text('{"status":"forged"}', encoding="ascii")
    artifact.write_bytes(b"forged-prior")
    effects: list[str] = []
    monkeypatch.setattr(
        prior_module,
        "_load_passed_receipt",
        lambda *_args, **_kwargs: effects.append("parsed"),
    )
    import joblib

    monkeypatch.setattr(joblib, "load", lambda *_args, **_kwargs: effects.append("unpickled"))

    with pytest.raises(ValueError, match="receipt SHA256"):
        prior_module.load_deployment_prior_with_receipt_hash(
            artifact,
            receipt,
            expected_receipt_sha256="0" * 64,
        )

    assert effects == []


def test_receipt_publish_race_rolls_back_own_artifact(
    tiny_fit_rows: list[PriorFitRow], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "index.json"
    candidate = tmp_path / "candidate.json"
    index.write_text("{}", encoding="ascii")
    candidate.write_text("{}", encoding="ascii")
    prior = fit_deployment_prior(tiny_fit_rows, n_jobs=1, parity_passed=False)
    monkeypatch.setattr(
        prior_module,
        "load_dataset_index",
        lambda *_args, **_kwargs: (tiny_fit_rows, [object()], {}),
    )
    monkeypatch.setattr(
        prior_module,
        "validate_candidate_manifest",
        lambda *_args, **_kwargs: _validated_candidate_stub(candidate),
    )
    monkeypatch.setattr(
        prior_module, "fit_deployment_prior", lambda *_args, **_kwargs: prior
    )
    monkeypatch.setattr(
        prior_module,
        "verify_holdout_parity",
        lambda *_args, **_kwargs: {
            "expected": 76,
            "input_rgb_hash_matches": 76,
            "contour_byte_matches": 76,
            "parity_passed": True,
            "mismatch_groups": [],
            "candidate_manifest_sha256": sha256_file(candidate),
            "candidate_data_sha256": "d" * 64,
        },
    )
    output = tmp_path / "prior.joblib"
    receipt = tmp_path / "receipt.json"
    original_publish = prior_module._publish_no_replace

    def race(source: Path, destination: Path) -> None:
        if destination == receipt:
            destination.write_bytes(b"racer")
        original_publish(source, destination)

    monkeypatch.setattr(prior_module, "_publish_no_replace", race)
    with pytest.raises(FileExistsError):
        build_prior_artifact(
            dataset_index=index,
            candidate_manifest=candidate,
            output=output,
            receipt=receipt,
        )
    assert not output.exists()
    assert receipt.read_bytes() == b"racer"
    assert not output.with_name(output.name + ".build.lock").exists()


def test_preflight_failure_writes_structured_failed_receipt(tmp_path: Path) -> None:
    missing_index = tmp_path / "missing-index.json"
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="ascii")
    output = tmp_path / "prior.joblib"
    receipt = tmp_path / "receipt.json"
    with pytest.raises(FileNotFoundError):
        build_prior_artifact(
            dataset_index=missing_index,
            candidate_manifest=candidate,
            output=output,
            receipt=receipt,
        )
    payload = json.loads(receipt.read_text(encoding="ascii"))
    assert payload["status"] == "failed"
    assert payload["error_type"] == "FileNotFoundError"
    assert payload["content_sha256"] == prior_module._canonical_hash(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    assert not output.exists()


def test_frozen_runtime_config_resolves_exact_contract() -> None:
    frozen_config = "configs/loop206/l206_control_train_screen_pilot20.yaml"
    payload = prior_module._runtime_payload(
        image_size=(384, 384),
        frozen_config=frozen_config,
    )
    assert payload["project_seed"] == 206
    assert payload["views"] == ["clean", "low_contrast", "gaussian_noise"]
    assert payload["active_contour"]["name"] == "neutral_mid_30_s2"
    assert prior_module._canonical_hash(prior_module.load_config(frozen_config)) == (
        "95eba3167a841187da5fbbe6f5a5f93a406ff5fec97907829fcf7297dcf4e39a"
    )
