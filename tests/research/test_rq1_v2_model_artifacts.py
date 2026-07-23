from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))

import pytest
import yaml
import freeze_rq1_v2_artifacts as freezer

from freeze_rq1_v2_artifacts import (
    FrozenArtifacts,
    build_experiment_input_manifest,
    build_runtime_manifest,
    canonical_json_bytes,
    config_family_sha256,
    experiment_input_sha256,
    tensor_state_sha256,
    validate_artifacts,
    write_immutable_json,
)
from train_rq1_v2 import run_contract, validate_frozen_experiment_trust
from summarize_rq1_v2 import summarize
from lesion_robustness.research.rq1_data import canonical_report_sha256


def _contracts() -> dict[str, str]:
    return {
        "protocol_sha256": "1" * 64,
        "condition_contract_sha256": "2" * 64,
        "metric_contract_sha256": "3" * 64,
        "data_report_sha256": "f" * 64,
    }


def _protocol() -> dict[str, str]:
    return {"schema_version": "protocol", "protocol_sha256": "1" * 64,
            "condition_contract_sha256": "2" * 64, "metric_contract_sha256": "3" * 64}


def _fixture() -> FrozenArtifacts:
    state = {"encoder.weight": [[1.0, 2.0], [3.0, 4.0]], "encoder.bias": [0.0, 1.0]}
    model = {
        "schema_version": "imp.rq1_v2.model_artifacts.v1",
        "imp": {
            "imagenet_pretrained_state_sha256": tensor_state_sha256(state),
            "artifact_file_sha256": "a" * 64,
        },
        "nnunet": {
            "checkpoint_sha256": "3814716033afd464dacc573f92a5a44ff20eb7f2163d99b4f16ecff8aa278ea2",
            "plans_sha256": "b60e4defd229b03f7064dc5b66123545c91cdaa44c09d990b86690a94e1e08a7",
            "fingerprint_sha256": "931da8aae52ffecd726d5928009ebdcae7002e24b035fad89177e0bc81dba85c",
            "dataset_sha256": "eb33bcbad9d8d5c96168b3c12171392ffabf63ba4cbff4f2bf4badc98bf6487a",
            "container_image_digest": "sha256:" + "d" * 64,
        },
    }
    contracts = _contracts()
    imp_family = {"arch": "mit_b3", "checkpoint_selection": "final_epoch_only"}
    nn_family = {"arch": "nnunetv2", "checkpoint_selection": "final_epoch_only"}
    imp_runtime = build_runtime_manifest(
        "imp", "sha256:" + "b" * 64, "c" * 64, imp_family, contracts,
        {name: model["imp"][name] for name in ("imagenet_pretrained_state_sha256", "artifact_file_sha256")},
    )
    nn_runtime = build_runtime_manifest(
        "nnunet", "sha256:" + "d" * 64, "e" * 64, nn_family, contracts,
        {name: model["nnunet"][name] for name in ("checkpoint_sha256", "plans_sha256", "fingerprint_sha256", "dataset_sha256")},
    )
    return FrozenArtifacts(model, state, {"imp": imp_runtime, "nnunet": nn_runtime})


def test_imp_pretrained_tensor_state_is_bound_and_nonempty():
    frozen = _fixture()
    assert len(frozen.model_artifacts["imp"]["imagenet_pretrained_state_sha256"]) == 64
    drift = {"encoder.weight": [[1.0, 2.0], [3.0, 9.0]], "encoder.bias": [0.0, 1.0]}
    with pytest.raises(ValueError, match="pretrained weight drift"):
        validate_artifacts(frozen, imp_state=drift)


def test_nnunet_plans_fingerprint_dataset_and_container_are_bound():
    n = _fixture().model_artifacts["nnunet"]
    assert n["plans_sha256"] == "b60e4defd229b03f7064dc5b66123545c91cdaa44c09d990b86690a94e1e08a7"
    assert n["fingerprint_sha256"] == "931da8aae52ffecd726d5928009ebdcae7002e24b035fad89177e0bc81dba85c"
    assert n["dataset_sha256"] == "eb33bcbad9d8d5c96168b3c12171392ffabf63ba4cbff4f2bf4badc98bf6487a"
    assert n["container_image_digest"].startswith("sha256:")


def test_runtime_manifests_are_arm_specific_but_share_scientific_contract():
    f = _fixture()
    imp, nnunet = f.runtimes["imp"], f.runtimes["nnunet"]
    assert imp.arm == "imp" and nnunet.arm == "nnunet"
    assert imp.container_image_digest != nnunet.container_image_digest
    assert imp.dependency_lock_sha256 != nnunet.dependency_lock_sha256
    assert imp.shared_contract == nnunet.shared_contract


def test_experiment_input_manifest_binds_parent_without_self_reference(tmp_path: Path):
    parent = tmp_path / "parent.json"
    parent.write_text('{"schema_version":"imp.release.manifest.v1"}\n', encoding="ascii")
    f = _fixture()
    configs = {
        "imp": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "mit_b3"} for s in (206, 1206, 2206)],
        "nnunet": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "nnunetv2"} for s in (206, 1206, 2206)],
    }
    m = build_experiment_input_manifest(parent, _protocol(), {"status": "verified", "report_sha256": "f" * 64}, f.model_artifacts, f.runtimes, configs)
    assert m.parent_release_manifest_sha256 == hashlib.sha256(parent.read_bytes()).hexdigest()
    assert "experiment_input_manifest_sha256" not in m.to_dict()


def test_config_family_changes_only_with_non_seed_fields():
    a = {"seed": 206, "arch": "mit_b3", "checkpoint_selection": "final_epoch_only"}
    b = {"seed": 1206, "arch": "mit_b3", "checkpoint_selection": "final_epoch_only"}
    c = {"seed": 1206, "arch": "other", "checkpoint_selection": "final_epoch_only"}
    assert config_family_sha256(a) == config_family_sha256(b)
    assert config_family_sha256(a) != config_family_sha256(c)


def test_tensor_state_rejects_empty():
    with pytest.raises(ValueError, match="nonempty"):
        tensor_state_sha256({})


def test_unresolved_data_blocks_input_manifest(tmp_path: Path):
    f = _fixture()
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="ascii")
    with pytest.raises(ValueError, match="data integrity"):
        build_experiment_input_manifest(parent, _protocol(), {"dataset_index_status": "unresolved_blocked"}, f.model_artifacts, f.runtimes, {})


def test_exact_rq1_seed_set_is_required(tmp_path: Path):
    f = _fixture()
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="ascii")
    configs = {
        "imp": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "mit_b3"} for s in (206, 1206, 9999)],
        "nnunet": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "nnunetv2"} for s in (206, 1206, 2206)],
    }
    with pytest.raises(ValueError, match="exact seeds"):
        build_experiment_input_manifest(parent, _protocol(), {"status": "verified", "report_sha256": "f" * 64}, f.model_artifacts, f.runtimes, configs)


def test_nnunet_recovered_identity_drift_is_rejected():
    f = _fixture()
    model = deepcopy(f.model_artifacts)
    model["nnunet"]["plans_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="plans_sha256 drift"):
        validate_artifacts(FrozenArtifacts(model, f.imp_state, f.runtimes))


def test_nnunet_model_runtime_image_mismatch_is_rejected(tmp_path: Path):
    f = _fixture()
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="ascii")
    model = deepcopy(f.model_artifacts)
    model["nnunet"]["container_image_digest"] = "sha256:" + "9" * 64
    configs = {
        "imp": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "mit_b3"} for s in (206, 1206, 2206)],
        "nnunet": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "nnunetv2"} for s in (206, 1206, 2206)],
    }
    with pytest.raises(ValueError, match="model/runtime container image mismatch"):
        build_experiment_input_manifest(parent, _protocol(), {"status": "verified", "report_sha256": "f" * 64}, model, f.runtimes, configs)


def test_seed_neutral_config_family_must_match_within_arm(tmp_path: Path):
    f = _fixture()
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="ascii")
    configs = {
        "imp": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "mit_b3"} for s in (206, 1206, 2206)],
        "nnunet": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "nnunetv2"} for s in (206, 1206, 2206)],
    }
    configs["imp"][1]["arch"] = "drift"
    with pytest.raises(ValueError, match="family equality"):
        build_experiment_input_manifest(parent, _protocol(), {"status": "verified", "report_sha256": "f" * 64}, f.model_artifacts, f.runtimes, configs)


def test_nested_release_reference_is_rejected(tmp_path: Path):
    f = _fixture()
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="ascii")
    configs = {
        "imp": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "mit_b3"} for s in (206, 1206, 2206)],
        "nnunet": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "nnunetv2"} for s in (206, 1206, 2206)],
    }
    with pytest.raises(ValueError, match="release manifest reference"):
        build_experiment_input_manifest(
            parent, {"nested": {"future_release_manifest_sha256": "f" * 64}},
            {"status": "verified", "report_sha256": "f" * 64}, f.model_artifacts, f.runtimes, configs,
        )


def test_runtime_manifest_binds_arm_and_config_family():
    runtime = build_runtime_manifest("imp", "sha256:" + "a" * 64, "b" * 64,
                                     {"arch": "mit_b3", "checkpoint_selection": "final_epoch_only"}, _contracts(),
                                     {name: _fixture().model_artifacts["imp"][name] for name in
                                      ("imagenet_pretrained_state_sha256", "artifact_file_sha256")})
    payload = runtime.to_dict()
    assert payload["arm"] == "imp"
    assert payload["config_family_sha256"] == config_family_sha256({"arch": "mit_b3", "checkpoint_selection": "final_epoch_only"})


def test_canonical_writer_refuses_drift(tmp_path: Path):
    output = tmp_path / "manifest.json"
    payload = {"b": 2, "a": 1}
    assert canonical_json_bytes(payload) == b'{"a":1,"b":2}'
    write_immutable_json(output, payload)
    write_immutable_json(output, payload)
    output.write_bytes(b'{"a":9}')
    with pytest.raises(ValueError, match="immutable JSON drift"):
        write_immutable_json(output, payload)


def test_experiment_input_digest_is_external_and_stable(tmp_path: Path):
    f = _fixture()
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="ascii")
    configs = {
        "imp": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "mit_b3"} for s in (206, 1206, 2206)],
        "nnunet": [{"seed": s, "checkpoint_selection": "final_epoch_only", "arch": "nnunetv2"} for s in (206, 1206, 2206)],
    }
    manifest = build_experiment_input_manifest(parent, _protocol(), {"status": "verified", "report_sha256": "f" * 64}, f.model_artifacts, f.runtimes, configs)
    assert len(experiment_input_sha256(manifest)) == 64


@pytest.mark.parametrize(
    "private_path",
    [
        "E:" + chr(92) * 2 + "models" + chr(92) * 2 + "checkpoint.pt",
        chr(92) * 4 + "server" + chr(92) * 2 + "share" + chr(92) * 2 + "checkpoint.pt",
        "/" + "home/" + "user/private/checkpoint.pt",
        "/" + "mnt/data/checkpoint.pt",
    ],
)
def test_canonical_manifest_rejects_all_absolute_private_path_forms(private_path: str):
    with pytest.raises(ValueError, match="private paths"):
        canonical_json_bytes({"artifact": private_path})


def test_freezer_output_is_consumed_by_train_eval_and_safe_summarizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    data = tmp_path / "data-index.json"
    data.write_text('{"rows":[]}', encoding="ascii")
    imp_state = {"encoder.weight": [[1.0, 2.0], [3.0, 4.0]], "encoder.bias": [0.0, 1.0]}
    imp_input = tmp_path / "imp-init.json"
    imp_input.write_text(json.dumps(imp_state, sort_keys=True), encoding="ascii")
    nn_plans = tmp_path / "nn-plans.json"
    nn_plans.write_bytes(b"nn plans")
    nn_checkpoint = tmp_path / "checkpoint_final.pth"
    nn_checkpoint.write_bytes(b"nn checkpoint")
    identities = {
        "checkpoint_sha256": hashlib.sha256(nn_checkpoint.read_bytes()).hexdigest(),
        "plans_sha256": hashlib.sha256(nn_plans.read_bytes()).hexdigest(),
        "fingerprint_sha256": "9" * 64,
        "dataset_sha256": "8" * 64,
    }
    monkeypatch.setattr(freezer, "_NNUNET_IDENTITIES", identities)

    public_protocol = json.loads((root / "experiments/rq1_v2/protocol.json").read_text(encoding="ascii"))
    public_protocol["dataset_index_status"] = "verified"
    public_protocol["dataset_index_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    protocol = dict(public_protocol)
    protocol.update({
        "protocol_sha256": "1" * 64,
        "condition_contract_sha256": "2" * 64,
        "metric_contract_sha256": "3" * 64,
    })
    algorithms = {
        "ordered_identity": "sample_id|group_key|sha256_raw|sha256_rgb\\n sorted by sample_id,group_key; ASCII; SHA-256",
        "group": "exact group_key crossing",
        "exact_rgb": "decoded RGB SHA-256 crossing",
        "near_candidate": "63-bit luminance pHash; Lanczos 32x32; DCT 8x8 excluding DC; Hamming <=4",
        "near_confirmation": "luminance SSIM at Lanczos 256x256; Gaussian 11x11 sigma 1.5; threshold >=0.98",
    }
    report = {
        "schema_version": "imp.rq1_v2.data_integrity_report.v1",
        "audit_id": "RQ1v2-data-integrity",
        "train_count": 2008,
        "validation_count": 431,
        "train_ordered_identity_sha256": "4" * 64,
        "validation_ordered_identity_sha256": "5" * 64,
        "cross_split_groups": 0,
        "cross_split_exact_rgb": 0,
        "near_duplicate_candidate_count": 0,
        "cross_split_near_rgb": 0,
        "clean_v3_manifest_sha256": public_protocol["clean_v3_manifest_sha256"],
        "dataset_index_status": "verified",
        "dataset_index_sha256": public_protocol["dataset_index_sha256"],
        "test_v3_access": False,
        "test_v3_open_count": 0,
        "algorithms": algorithms,
    }
    report["canonical_report_sha256"] = canonical_report_sha256(report)
    model = {
        "schema_version": "imp.rq1_v2.model_artifacts.v1",
        "imp": {
            "imagenet_pretrained_state_sha256": tensor_state_sha256(imp_state),
            "artifact_file_sha256": hashlib.sha256(imp_input.read_bytes()).hexdigest(),
        },
        "nnunet": {**identities, "container_image_digest": "sha256:" + "d" * 64},
    }
    configs = {
        arm: [
            yaml.safe_load((root / "experiments/rq1_v2/configs" / f"{arm}_seed{seed}.yaml").read_text(encoding="ascii"))
            for seed in (206, 1206, 2206)
        ]
        for arm in ("imp", "nnunet")
    }
    contracts = {
        "protocol_sha256": protocol["protocol_sha256"],
        "condition_contract_sha256": protocol["condition_contract_sha256"],
        "metric_contract_sha256": protocol["metric_contract_sha256"],
        "data_report_sha256": report["canonical_report_sha256"],
    }
    runtimes = {
        "imp": build_runtime_manifest(
            "imp", "sha256:" + "b" * 64, "c" * 64, configs["imp"][0], contracts,
            {name: model["imp"][name] for name in
             ("imagenet_pretrained_state_sha256", "artifact_file_sha256")},
        ),
        "nnunet": build_runtime_manifest(
            "nnunet", "sha256:" + "d" * 64, "e" * 64, configs["nnunet"][0], contracts,
            identities,
        ),
    }
    validate_artifacts(FrozenArtifacts(model, imp_state, runtimes))
    parent = tmp_path / "parent-release.json"
    parent.write_text('{"schema_version":"imp.release.manifest.v1"}', encoding="ascii")
    frozen = build_experiment_input_manifest(parent, protocol, report, model, runtimes, configs)
    experiment = tmp_path / "experiment.json"
    experiment.write_text(json.dumps(frozen.to_dict(), sort_keys=True), encoding="ascii")

    trusted, reason = validate_frozen_experiment_trust(
        experiment, parent_release=parent, data_manifest=data, imp_input_artifact=imp_input,
        nnunet_input_artifact=nn_plans, nnunet_checkpoint=nn_checkpoint,
        protocol=public_protocol,
    )
    assert trusted is not None, reason

    forged_family = deepcopy(frozen.to_dict())
    forged_family["runtimes"]["imp"]["config_family_sha256"] = "7" * 64
    forged_family_path = tmp_path / "forged-config-family.json"
    forged_family_path.write_text(json.dumps(forged_family, sort_keys=True), encoding="ascii")
    family_trusted, family_reason = validate_frozen_experiment_trust(
        forged_family_path, parent_release=parent, data_manifest=data,
        imp_input_artifact=imp_input, nnunet_input_artifact=nn_plans,
        nnunet_checkpoint=nn_checkpoint, protocol=public_protocol,
    )

    forged_image = deepcopy(frozen.to_dict())
    forged_image["model_artifacts"]["nnunet"]["container_image_digest"] = "sha256:" + "9" * 64
    forged_image_path = tmp_path / "forged-runtime-image.json"
    forged_image_path.write_text(json.dumps(forged_image, sort_keys=True), encoding="ascii")
    image_trusted, image_reason = validate_frozen_experiment_trust(
        forged_image_path, parent_release=parent, data_manifest=data,
        imp_input_artifact=imp_input, nnunet_input_artifact=nn_plans,
        nnunet_checkpoint=nn_checkpoint, protocol=public_protocol,
    )
    assert image_trusted is None
    assert family_trusted is None
    assert "config family" in str(family_reason).lower()
    assert "container image" in str(image_reason).lower()

    common = dict(
        protocol=tmp_path / "public-protocol.json", config=root / "experiments/rq1_v2/configs/imp_seed206.yaml",
        data_manifest=data, experiment_manifest=experiment, input_artifact=imp_input,
        input_artifact_sha256=hashlib.sha256(imp_input.read_bytes()).hexdigest(), parent_release=parent,
        imp_input_artifact=imp_input, nnunet_input_artifact=nn_plans, nnunet_checkpoint=nn_checkpoint,
        preflight_only=True, dry_run=False,
    )
    common["protocol"].write_text(json.dumps(public_protocol), encoding="ascii")
    train_checkpoint = tmp_path / "runner-output" / "imp_seed206" / "final.pt"
    assert run_contract(SimpleNamespace(**common, output_checkpoint=train_checkpoint), operation="train") == 0
    eval_checkpoint = tmp_path / "eval-input" / "imp_seed206" / "final.pt"
    eval_checkpoint.parent.mkdir(parents=True)
    eval_checkpoint.write_bytes(b"prospective output checkpoint")
    eval_args = dict(common)
    eval_args["input_artifact"] = eval_checkpoint
    eval_args["input_artifact_sha256"] = hashlib.sha256(eval_checkpoint.read_bytes()).hexdigest()
    assert run_contract(SimpleNamespace(**eval_args), operation="evaluate") == 2

    receipts = tmp_path / "receipts"
    receipts.mkdir()
    checkpoints = tmp_path / "outputs"
    checkpoints.mkdir()
    exp_sha = hashlib.sha256(experiment.read_bytes()).hexdigest()
    for arm in ("imp", "nnunet"):
        for seed in (206, 1206, 2206):
            checkpoint = checkpoints / f"{arm}_seed{seed}" / "final.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{arm}:{seed}".encode("ascii"))
            config_path = root / "experiments/rq1_v2/configs" / f"{arm}_seed{seed}.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="ascii"))
            receipt = {
                "schema_version": "imp.rq1_v2.job_receipt.v1", "job_id": f"{arm}_seed{seed}",
                "arm": arm, "seed": seed, "model_id": config["model_id"], "status": "validated",
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "protocol_sha256": protocol["protocol_sha256"],
                "experiment_manifest_sha256": exp_sha,
                "data_manifest_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "dependency_lock_sha256": runtimes[arm].dependency_lock_sha256,
                "input_artifact_sha256": model[arm][config["private_inputs"]["training_input_artifact_key"]],
                "output_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "metric_source_sha256": config["metric_contract"]["source_sha256"],
                "metrics": {"dice": 0.5, "iou": 0.4, "precision": 0.6, "recall": 0.7,
                            "boundary_f1": 0.4, "hd95": 4.0, "assd": 1.0},
            }
            (receipts / f"{arm}-{seed}.json").write_text(json.dumps(receipt), encoding="ascii")
    summary, status = summarize(
        receipts, data_manifest=data, experiment_manifest=experiment, parent_release=parent,
        imp_input_artifact=imp_input, nnunet_input_artifact=nn_plans,
        nnunet_checkpoint=nn_checkpoint, checkpoint_dir=checkpoints,
    )
    assert status == 2 and summary["status"] == "pending/unverified" and summary["metrics"] == []
    assert summary["trust_chain_status"] == "validated_inputs_analysis_deferred"

    swapped = tmp_path / "swapped-checkpoint.pth"
    swapped.write_bytes(b"different checkpoint")
    trusted, reason = validate_frozen_experiment_trust(
        experiment, parent_release=parent, data_manifest=data, imp_input_artifact=imp_input,
        nnunet_input_artifact=nn_plans, nnunet_checkpoint=swapped,
        protocol=public_protocol,
    )
    assert trusted is None and "checkpoint bytes drift" in str(reason)

    for field, value in (
        ("audit_id", "forged-audit"),
        (
            "algorithms",
            {
                "forged": "E:"
                + chr(92) * 2
                + "models"
                + chr(92) * 2
                + "private.bin"
            },
        ),
    ):
        malicious_payload = deepcopy(frozen.to_dict())
        malicious_payload["data_report"][field] = value
        malicious_payload["data_report"]["canonical_report_sha256"] = canonical_report_sha256(
            malicious_payload["data_report"]
        )
        malicious = tmp_path / f"malicious-{field}.json"
        malicious.write_text(json.dumps(malicious_payload, sort_keys=True), encoding="ascii")
        trusted, reason = validate_frozen_experiment_trust(
            malicious, parent_release=parent, data_manifest=data, imp_input_artifact=imp_input,
            nnunet_input_artifact=nn_plans, nnunet_checkpoint=nn_checkpoint,
            protocol=public_protocol,
        )
        assert trusted is None, field
