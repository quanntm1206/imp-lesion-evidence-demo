from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import importlib.util

import yaml
import pytest


ROOT = Path(__file__).resolve().parents[2]
SUMMARIZE = ROOT / "scripts" / "research" / "summarize_rq1_v2.py"
MANIFEST = ROOT / "results" / "rq1_v2" / "result_manifest.json"
README = ROOT / "results" / "rq1_v2" / "README.md"
SEEDS = (206, 1206, 2206)


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SUMMARIZE), *(str(value) for value in args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _trusted_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    data = tmp_path / "data-manifest.json"
    data.write_text('{"rows":[]}', encoding="ascii")
    protocol = ROOT / "experiments" / "rq1_v2" / "protocol.json"
    protocol_payload = json.loads(protocol.read_text(encoding="ascii"))
    lock = {arm: hashlib.sha256(f"lock:{arm}".encode("ascii")).hexdigest() for arm in ("imp", "nnunet")}
    input_hash = {arm: hashlib.sha256(f"input:{arm}".encode("ascii")).hexdigest() for arm in ("imp", "nnunet")}
    experiment_payload = {
        "schema_version": "imp.rq1_v2.experiment_input.v1",
        "protocol": protocol_payload,
        "configs": {
            arm: [
                yaml.safe_load((ROOT / "experiments" / "rq1_v2" / "configs" / f"{arm}_seed{seed}.yaml").read_text(encoding="ascii"))
                for seed in SEEDS
            ]
            for arm in ("imp", "nnunet")
        },
        "runtimes": {arm: {"dependency_lock_sha256": lock[arm]} for arm in ("imp", "nnunet")},
        "data_report": {"dataset_index_sha256": hashlib.sha256(data.read_bytes()).hexdigest()},
        "model_artifacts": {
            "imp": {"imagenet_pretrained_state_sha256": input_hash["imp"]},
            "nnunet": {"plans_sha256": input_hash["nnunet"]},
        },
    }
    experiment = tmp_path / "experiment-input.json"
    experiment.write_text(json.dumps(experiment_payload, sort_keys=True), encoding="ascii")
    return data, experiment, input_hash


def _write_receipt(path: Path, arm: str, seed: int, data: Path, experiment: Path, input_hash: dict[str, str]) -> None:
    model = "RQ1v2-IMP-MiT-B3-UNet" if arm == "imp" else "RQ1v2-nnUNet-v2-2d"
    digest = lambda value: hashlib.sha256(value.encode("ascii")).hexdigest()
    config = ROOT / "experiments" / "rq1_v2" / "configs" / f"{arm}_seed{seed}.yaml"
    protocol = ROOT / "experiments" / "rq1_v2" / "protocol.json"
    metric_source_sha256 = yaml.safe_load(config.read_text(encoding="ascii"))["metric_contract"]["source_sha256"]
    payload = {
        "schema_version": "imp.rq1_v2.job_receipt.v1",
        "job_id": f"{arm}_seed{seed}",
        "arm": arm,
        "seed": seed,
        "model_id": model,
        "status": "validated",
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "experiment_manifest_sha256": hashlib.sha256(experiment.read_bytes()).hexdigest(),
        "data_manifest_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "dependency_lock_sha256": digest(f"lock:{arm}"),
        "input_artifact_sha256": input_hash[arm],
        "output_checkpoint_sha256": digest(f"checkpoint:{arm}:{seed}"),
        "metric_source_sha256": metric_source_sha256,
        "metrics": {
            "dice": 0.5 + seed / 100000.0,
            "iou": 0.4,
            "precision": 0.6,
            "recall": 0.7,
            "boundary_f1": 0.4,
            "hd95": 4.0,
            "assd": 1.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")


def test_tracked_result_manifest_is_truthful_pending_state() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="ascii"))
    assert payload == {
        "schema_version": "imp.rq1_v2.public_results.v1",
        "status": "pending/unverified",
        "p1_status": "not_promoted",
        "metrics": [],
        "completed_jobs": 0,
        "required_jobs": 6,
    }
    text = README.read_text(encoding="utf-8").lower()
    assert "pending/unverified" in text
    assert "checkpoint" in text and "dataset" in text
    assert "superiority" in text and "not" in text


def test_incomplete_receipt_set_stays_pending_and_has_no_metrics(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    data, experiment, input_hash = _trusted_inputs(tmp_path)
    _write_receipt(receipts / "imp-206.json", "imp", 206, data, experiment, input_hash)
    output = tmp_path / "summary.json"
    result = _run("--receipts", receipts, "--data-manifest", data, "--experiment-manifest", experiment, "--output", output)
    assert result.returncode == 2
    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["status"] == "pending/unverified"
    assert payload["p1_status"] == "not_promoted"
    assert payload["metrics"] == []
    assert payload["completed_jobs"] == 1


def test_six_synthetic_receipts_never_promote_without_frozen_trust_chain(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    data, experiment, input_hash = _trusted_inputs(tmp_path)
    for arm in ("imp", "nnunet"):
        for seed in SEEDS:
            _write_receipt(receipts / f"{arm}-{seed}.json", arm, seed, data, experiment, input_hash)
    output = tmp_path / "summary.json"
    result = _run("--receipts", receipts, "--data-manifest", data, "--experiment-manifest", experiment, "--output", output)
    assert result.returncode == 2
    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["status"] == "pending/unverified"
    assert payload["p1_status"] == "not_promoted"
    assert payload["completed_jobs"] == 6
    assert payload["metrics"] == []
    serialized = json.dumps(payload).lower()
    assert "p_value" not in serialized
    assert "superior" not in serialized
    assert "significant" not in serialized


def test_duplicate_or_drifted_job_blocks_promotion(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    data, experiment, input_hash = _trusted_inputs(tmp_path)
    for arm in ("imp", "nnunet"):
        for seed in SEEDS:
            _write_receipt(receipts / f"{arm}-{seed}.json", arm, seed, data, experiment, input_hash)
    _write_receipt(receipts / "duplicate.json", "imp", 206, data, experiment, input_hash)
    output = tmp_path / "summary.json"
    result = _run("--receipts", receipts, "--data-manifest", data, "--experiment-manifest", experiment, "--output", output)
    assert result.returncode == 2
    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["status"] == "pending/unverified"
    assert payload["metrics"] == []


def test_six_syntactic_receipts_without_trusted_manifests_do_not_promote(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    data, experiment, input_hash = _trusted_inputs(tmp_path)
    for arm in ("imp", "nnunet"):
        for seed in SEEDS:
            _write_receipt(receipts / f"{arm}-{seed}.json", arm, seed, data, experiment, input_hash)
    output = tmp_path / "summary.json"
    result = _run("--receipts", receipts, "--output", output)
    assert result.returncode == 2
    assert json.loads(output.read_text(encoding="ascii"))["metrics"] == []


def test_receipt_config_digest_drift_blocks_promotion(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    data, experiment, input_hash = _trusted_inputs(tmp_path)
    for arm in ("imp", "nnunet"):
        for seed in SEEDS:
            path = receipts / f"{arm}-{seed}.json"
            _write_receipt(path, arm, seed, data, experiment, input_hash)
    drifted = receipts / "imp-206.json"
    payload = json.loads(drifted.read_text(encoding="ascii"))
    payload["config_sha256"] = "0" * 64
    drifted.write_text(json.dumps(payload), encoding="ascii")
    output = tmp_path / "summary.json"
    result = _run("--receipts", receipts, "--data-manifest", data, "--experiment-manifest", experiment, "--output", output)
    assert result.returncode == 2
    assert json.loads(output.read_text(encoding="ascii"))["metrics"] == []


@pytest.mark.parametrize(
    "field",
    [
        "experiment_manifest_sha256",
        "data_manifest_sha256",
        "dependency_lock_sha256",
        "input_artifact_sha256",
        "metric_source_sha256",
    ],
)
def test_trusted_receipt_binding_drift_blocks_promotion(tmp_path: Path, field: str) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    data, experiment, input_hash = _trusted_inputs(tmp_path)
    for arm in ("imp", "nnunet"):
        for seed in SEEDS:
            _write_receipt(receipts / f"{arm}-{seed}.json", arm, seed, data, experiment, input_hash)
    drifted = receipts / "imp-206.json"
    payload = json.loads(drifted.read_text(encoding="ascii"))
    payload[field] = "0" * 64
    drifted.write_text(json.dumps(payload), encoding="ascii")
    output = tmp_path / "summary.json"
    result = _run("--receipts", receipts, "--data-manifest", data,
                  "--experiment-manifest", experiment, "--output", output)
    assert result.returncode == 2
    assert json.loads(output.read_text(encoding="ascii"))["metrics"] == []


def test_receipt_with_inferential_extra_field_blocks_promotion(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    data, experiment, input_hash = _trusted_inputs(tmp_path)
    for arm in ("imp", "nnunet"):
        for seed in SEEDS:
            path = receipts / f"{arm}-{seed}.json"
            _write_receipt(path, arm, seed, data, experiment, input_hash)
    drifted = receipts / "nnunet-2206.json"
    payload = json.loads(drifted.read_text(encoding="ascii"))
    payload["p_value"] = 0.01
    drifted.write_text(json.dumps(payload), encoding="ascii")
    output = tmp_path / "summary.json"
    result = _run("--receipts", receipts, "--data-manifest", data, "--experiment-manifest", experiment, "--output", output)
    assert result.returncode == 2
    assert json.loads(output.read_text(encoding="ascii"))["metrics"] == []


def test_reused_output_checkpoint_hash_blocks_promotion(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    data, experiment, input_hash = _trusted_inputs(tmp_path)
    for arm in ("imp", "nnunet"):
        for seed in SEEDS:
            _write_receipt(receipts / f"{arm}-{seed}.json", arm, seed, data, experiment, input_hash)
    first = json.loads((receipts / "imp-206.json").read_text(encoding="ascii"))
    second_path = receipts / "imp-1206.json"
    second = json.loads(second_path.read_text(encoding="ascii"))
    second["output_checkpoint_sha256"] = first["output_checkpoint_sha256"]
    second_path.write_text(json.dumps(second), encoding="ascii")
    output = tmp_path / "summary.json"
    result = _run("--receipts", receipts, "--data-manifest", data,
                  "--experiment-manifest", experiment, "--output", output)
    assert result.returncode == 2
    assert json.loads(output.read_text(encoding="ascii"))["metrics"] == []


def test_summarizer_yaml_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("summarize_rq1_v2", SUMMARIZE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="duplicate YAML key"):
        yaml.load("a: 1\na: 2\n", Loader=module._UniqueSafeLoader)
