from __future__ import annotations

from dataclasses import replace
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
    prior = fit_deployment_prior(tiny_fit_rows, n_jobs=1)
    path = tmp_path / "prior.joblib"
    save_prior(prior, path)
    loaded = load_prior(path, expected_sha256=sha256_file(path))
    first = loaded.predict(tiny_fit_rows[0].image)
    second = loaded.predict(tiny_fit_rows[0].image.copy())
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8
    assert set(np.unique(first)).issubset({0, 255})


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
