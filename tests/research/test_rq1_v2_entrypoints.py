from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "scripts" / "research" / "train_rq1_v2.py"
EVALUATE = ROOT / "scripts" / "research" / "evaluate_rq1_v2.py"
SUMMARIZE = ROOT / "scripts" / "research" / "summarize_rq1_v2.py"
CONFIG_DIR = ROOT / "experiments" / "rq1_v2" / "configs"
PROTOCOL = ROOT / "experiments" / "rq1_v2" / "protocol.json"

SEEDS = (206, 1206, 2206)
CONDITIONS = (
    "clean",
    "low_brightness",
    "low_contrast",
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
)


def _run(script: Path, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(value) for value in args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _configs() -> list[dict[str, object]]:
    payloads = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="ascii"))
        payload["_path"] = path
        payloads.append(payload)
    return payloads


def test_six_canonical_configs_lock_scientific_and_compute_contract() -> None:
    configs = _configs()
    assert len(configs) == 6
    assert {(c["arm"], c["seed"]) for c in configs} == {
        (arm, seed) for arm in ("imp", "nnunet") for seed in SEEDS
    }
    assert {c["model_id"] for c in configs} == {
        "RQ1v2-IMP-MiT-B3-UNet",
        "RQ1v2-nnUNet-v2-2d",
    }

    metric_source = ROOT / "src" / "lesion_robustness" / "research" / "rq1_metrics.py"
    import hashlib

    metric_sha = hashlib.sha256(metric_source.read_bytes()).hexdigest()
    for config in configs:
        assert config["schema_version"] == "imp.rq1_v2.config.v1"
        assert tuple(config["conditions"]) == CONDITIONS
        assert config["geometry"] == {
            "input_hw": [384, 384],
            "probability_restore": "bilinear_to_original_before_threshold",
            "metric_geometry": "original_image",
            "threshold": 0.5,
        }
        assert config["metric_contract"] == {
            "source": "src/lesion_robustness/research/rq1_metrics.py",
            "source_sha256": metric_sha,
            "metrics": ["dice", "iou", "precision", "recall", "boundary_f1", "hd95", "assd"],
            "boundary_tolerance_original_pixels": 2,
            "empty_mask_policy": "both_empty_perfect;one_empty_diagonal_penalty",
        }
        assert config["training"]["epochs"] == 100
        assert config["training"]["checkpoint_selection"] == "final_epoch_only"
        assert config["training"]["deterministic"] is True
        assert config["training"]["num_workers"] == 0
        assert config["training"]["amp"] is True
        assert set(config["training"]) >= {
            "optimizer",
            "learning_rate",
            "weight_decay",
            "scheduler",
            "loss",
            "augmentation",
            "initialization",
        }
        assert config["budget"] == {
            "max_wall_hours": 24,
            "max_checkpoint_bytes": 2147483648,
            "max_job_storage_bytes": 25000000000,
        }
        inputs = config["private_inputs"]
        output = config["output"]
        assert inputs["dataset_index_env"] == "IMP_CLEAN_V3_INDEX"
        assert inputs["experiment_manifest_env"] == "IMP_RQ1_V2_EXPERIMENT_INPUT"
        assert inputs["training_input_env"] != output["checkpoint_env"]
        assert inputs["training_input_artifact_key"]
        assert output["checkpoint_selection"] == "final_epoch_only"

    imp = next(c for c in configs if c["arm"] == "imp")
    assert imp["training"]["model"] == "segmentation_models_pytorch.Unet"
    assert imp["training"]["encoder_name"] == "timm-mit_b3"
    assert imp["training"]["batch_size"] == 4
    nnunet = next(c for c in configs if c["arm"] == "nnunet")
    assert nnunet["training"]["trainer"] == "nnUNetTrainer_100epochs"
    assert nnunet["training"]["configuration"] == "2d"
    assert nnunet["training"]["fold"] == 0
    assert nnunet["training"]["batch_size"] == 2


@pytest.mark.parametrize("script", [TRAIN, EVALUATE])
def test_dry_run_validates_contract_without_private_data(script: Path) -> None:
    config = CONFIG_DIR / "imp_seed206.yaml"
    result = _run(script, "--protocol", PROTOCOL, "--config", config, "--dry-run")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["model_id"] == "RQ1v2-IMP-MiT-B3-UNet"
    assert payload["data_open_count"] == 0
    assert payload["engine_available"] is False


@pytest.mark.parametrize("script", [TRAIN, EVALUATE])
def test_direct_entrypoint_rejects_config_renamed_away_from_run_id(
    script: Path, tmp_path: Path
) -> None:
    renamed = tmp_path / "renamed.yaml"
    renamed.write_bytes((CONFIG_DIR / "imp_seed206.yaml").read_bytes())

    result = _run(script, "--protocol", PROTOCOL, "--config", renamed, "--dry-run")

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked_contract_drift"
    assert "basename" in payload["reason"].lower()


@pytest.mark.parametrize("script", [TRAIN, EVALUATE])
def test_preflight_missing_private_inputs_exits_two_before_data_open(script: Path, tmp_path: Path) -> None:
    config = CONFIG_DIR / "imp_seed206.yaml"
    missing = tmp_path / "must-not-open.json"
    result = _run(
        script,
        "--protocol",
        PROTOCOL,
        "--config",
        config,
        "--data-manifest",
        missing,
        "--input-artifact",
        tmp_path / "missing-input.bin",
        "--preflight-only",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked_missing_prerequisite"
    assert payload["data_open_count"] == 0
    assert payload["missing_prerequisites"]


def test_train_rejects_same_input_and_output_checkpoint_before_engine_use(tmp_path: Path) -> None:
    private = tmp_path / "same.pt"
    private.write_bytes(b"not-a-model")
    result = _run(
        TRAIN,
        "--protocol",
        PROTOCOL,
        "--config",
        CONFIG_DIR / "imp_seed206.yaml",
        "--data-manifest",
        private,
        "--input-artifact",
        private,
        "--output-checkpoint",
        private,
        "--preflight-only",
    )
    assert result.returncode == 2
    assert "separate" in result.stdout.lower()


def test_preflight_rejects_input_artifact_hash_drift_before_data_open(tmp_path: Path) -> None:
    artifact = tmp_path / "input.pt"
    artifact.write_bytes(b"bound-input")
    data = tmp_path / "data.json"
    data.write_text("{}", encoding="ascii")
    experiment = tmp_path / "experiment.json"
    experiment.write_text('{"schema_version":"imp.rq1_v2.experiment_input.v1"}', encoding="ascii")
    result = _run(
        TRAIN,
        "--protocol",
        PROTOCOL,
        "--config",
        CONFIG_DIR / "imp_seed206.yaml",
        "--data-manifest",
        data,
        "--experiment-manifest",
        experiment,
        "--parent-release",
        data,
        "--imp-input-artifact",
        artifact,
        "--nnunet-input-artifact",
        artifact,
        "--nnunet-checkpoint",
        artifact,
        "--input-artifact",
        artifact,
        "--input-artifact-sha256",
        "0" * 64,
        "--output-checkpoint",
        tmp_path / "output.pt",
        "--preflight-only",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked_artifact_drift"
    assert payload["artifact"] == "input_artifact"
    assert payload["data_open_count"] == 0


def test_normal_invocation_never_claims_training_engine(tmp_path: Path) -> None:
    result = _run(
        TRAIN,
        "--protocol",
        PROTOCOL,
        "--config",
        CONFIG_DIR / "imp_seed206.yaml",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["engine_available"] is False
    assert payload["status"].startswith("blocked_")


def test_powershell_runner_missing_private_environment_is_fail_closed() -> None:
    script = ROOT / "scripts" / "research" / "reproduce_paper_results.ps1"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-PreflightOnly"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked_missing_prerequisite"
    assert payload["data_open_count"] == 0


def test_powershell_runner_accepts_absolute_protocol_and_rejects_filename_substitution(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "research" / "reproduce_paper_results.ps1"
    absolute = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-DryRun", "-Protocol", str(PROTOCOL)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert absolute.returncode == 0, absolute.stderr

    configs = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, configs)
    (configs / "imp_seed206.yaml").rename(configs / "imp_seed999.yaml")
    substituted = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-DryRun", "-ConfigDirectory", str(configs)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert substituted.returncode == 2
    assert "exact six" in substituted.stderr.lower()


def test_python_entrypoints_are_import_safe() -> None:
    for script in (TRAIN, EVALUATE, SUMMARIZE):
        spec = importlib.util.spec_from_file_location(script.stem, script)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def test_dry_run_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    source = CONFIG_DIR / "imp_seed206.yaml"
    candidate = tmp_path / "duplicate.yaml"
    candidate.write_text(
        source.read_text(encoding="ascii").replace("seed: 206\n", "seed: 206\nseed: 206\n"),
        encoding="ascii",
    )
    result = _run(TRAIN, "--protocol", PROTOCOL, "--config", candidate, "--dry-run")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked_contract_drift"
    assert "duplicate" in payload["reason"].lower()


def test_dry_run_rejects_protocol_count_or_identity_drift(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="ascii"))
    payload["train_count"] = 2007
    candidate = tmp_path / "protocol-drift.json"
    candidate.write_text(json.dumps(payload), encoding="ascii")
    result = _run(TRAIN, "--protocol", candidate, "--config", CONFIG_DIR / "imp_seed206.yaml", "--dry-run")
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "blocked_contract_drift"


def test_dry_run_rejects_private_environment_mapping_drift(tmp_path: Path) -> None:
    candidate = tmp_path / "config-drift.yaml"
    candidate.write_text(
        (CONFIG_DIR / "imp_seed206.yaml").read_text(encoding="ascii").replace(
            "dataset_index_env: IMP_CLEAN_V3_INDEX", "dataset_index_env: WRONG_INDEX"
        ),
        encoding="ascii",
    )
    result = _run(TRAIN, "--protocol", PROTOCOL, "--config", candidate, "--dry-run")
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "blocked_contract_drift"


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ("run_id: imp_seed206", "run_id: forged-run"),
        ('checkpoint_template: "{run_id}/final.pt"', 'checkpoint_template: "forged.pt"'),
        ("max_wall_hours: 24", "max_wall_hours: 25"),
    ],
)
def test_dry_run_rejects_run_output_or_budget_drift(tmp_path: Path, needle: str, replacement: str) -> None:
    candidate = tmp_path / "config-drift.yaml"
    candidate.write_text(
        (CONFIG_DIR / "imp_seed206.yaml").read_text(encoding="ascii").replace(needle, replacement),
        encoding="ascii",
    )
    result = _run(TRAIN, "--protocol", PROTOCOL, "--config", candidate, "--dry-run")
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "blocked_contract_drift"


def test_dry_run_rejects_extra_config_or_protocol_fields(tmp_path: Path) -> None:
    config = tmp_path / "extra-config.yaml"
    config.write_text((CONFIG_DIR / "imp_seed206.yaml").read_text(encoding="ascii") + "extra: forged\n", encoding="ascii")
    result = _run(TRAIN, "--protocol", PROTOCOL, "--config", config, "--dry-run")
    assert result.returncode == 2

    payload = json.loads(PROTOCOL.read_text(encoding="ascii"))
    payload["extra"] = "forged"
    protocol = tmp_path / "extra-protocol.json"
    protocol.write_text(json.dumps(payload), encoding="ascii")
    result = _run(TRAIN, "--protocol", protocol, "--config", CONFIG_DIR / "imp_seed206.yaml", "--dry-run")
    assert result.returncode == 2


@pytest.mark.parametrize("script", [TRAIN, EVALUATE])
def test_preflight_rejects_minimal_forged_experiment_and_arbitrary_checkpoint(script: Path, tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    data.write_text('{"rows":[]}', encoding="ascii")
    protocol_payload = json.loads(PROTOCOL.read_text(encoding="ascii"))
    protocol_payload["dataset_index_status"] = "verified"
    protocol_payload["dataset_index_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_payload), encoding="ascii")
    artifact = tmp_path / "imp_seed206" / "final.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"arbitrary checkpoint")
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="ascii")
    forged = tmp_path / "experiment.json"
    forged.write_text('{"schema_version":"imp.rq1_v2.experiment_input.v1"}', encoding="ascii")
    args: list[object] = [
        "--protocol", protocol,
        "--config", CONFIG_DIR / "imp_seed206.yaml",
        "--data-manifest", data,
        "--experiment-manifest", forged,
        "--parent-release", parent,
        "--imp-input-artifact", artifact,
        "--nnunet-input-artifact", artifact,
        "--nnunet-checkpoint", artifact,
        "--input-artifact", artifact,
        "--input-artifact-sha256", hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "--preflight-only",
    ]
    if script == TRAIN:
        args.extend(("--output-checkpoint", tmp_path / "runner-output" / "imp_seed206" / "final.pt"))
    result = _run(script, *args)
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked_untrusted_experiment"
    assert payload["data_open_count"] == 0
