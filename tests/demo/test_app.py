from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import inspect
import re
from types import SimpleNamespace
import warnings

import numpy as np
from PIL import Image as PILImage
import pytest

import lesion_robustness.demo.app as app_module

from lesion_robustness.demo.app import (
    CONTROL_LOCKED_HTML,
    RequestGenerationGuard,
    create_app,
    dual_component_values,
    sample_source_change_values,
    run_guarded_dual,
    run_dual_live,
    run_comparison,
    run_control_preview,
    run_fixed_comparison,
    upload_source_change_values,
)
from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MODEL_ID,
    PROTOCOL_ID,
    rgb_sha256,
)
from lesion_robustness.demo.live_inputs import (
    LiveInputEvidence,
    LiveSample,
    synthetic_evidence,
    upload_evidence,
)
from lesion_robustness.demo.model_service import CandidateUnavailableError
from lesion_robustness.demo.presentation import NO_GT_MESSAGE
from lesion_robustness.demo.immutable_io import ImmutableSnapshot
from lesion_robustness.release_manifest import runtime_projection


ROOT = Path(__file__).resolve().parents[2]


def _registry() -> dict:
    value = json.loads(
        (ROOT / "demo/data/evidence_registry.json").read_text(encoding="ascii")
    )
    value["_demo_runtime"] = {
        "fixed_choices": [("ISIC_verified / component:verified", "component:verified")],
        "corruptions": ["clean", "gaussian_noise", "illumination_shift"],
    }
    return value


def _fixed_result() -> SimpleNamespace:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    return SimpleNamespace(
        original_rgb=image,
        control_mask=mask,
        candidate_mask=mask.copy(),
        control_overlay=image,
        candidate_overlay=image.copy(),
        control_latency_ms=12.5,
        candidate_latency_ms=14.5,
        total_latency_ms=27.0,
        device="cpu",
        control_model_id="control",
        candidate_model_id="candidate",
        control_checkpoint_sha256="a" * 64,
        candidate_checkpoint_sha256="b" * 64,
        prior_receipt_sha256=None,
        metadata={
            "comparison_source": "exact_fixed_cache",
            "group_key": "component:verified",
            "sample_id": "ISIC_verified",
            "corruption": "clean",
            "candidate_manifest_sha256": "c" * 64,
            "candidate_data_sha256": "d" * 64,
            "zero_manifest_sha256": "e" * 64,
            "zero_data_sha256": "f" * 64,
            "mask_sha256_raw": "1" * 64,
            "mask_sha256_binary": "2" * 64,
            "mask_sha256_runtime": ImmutableSnapshot.decoded_binary_mask_sha256(mask),
            "historical_cache_provenance_drift": True,
        },
    )


def _control_result() -> SimpleNamespace:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    return SimpleNamespace(
        mode="control_only",
        original_rgb=image,
        control_mask=mask,
        control_overlay=image,
        control_latency_ms=9.0,
        device="cpu",
        control_model_id="control",
        control_checkpoint_sha256="a" * 64,
        metadata={"result_type": "arbitrary_image_control_only"},
    )


class FakeService:
    def __init__(self) -> None:
        self.fixed_calls: list[tuple[str, str]] = []
        self.control_calls = 0

    def compare_fixed(self, identifier: str, *, corruption: str):
        self.fixed_calls.append((identifier, corruption))
        return _fixed_result()

    def preview_control(self, image: np.ndarray):
        self.control_calls += 1
        return _control_result()

    def compare(self, _image: np.ndarray):
        raise AssertionError("arbitrary candidate comparison must not be called")


def _dual_result(
    *, complete: bool, image: np.ndarray | None = None
) -> SimpleNamespace:
    image = (
        np.zeros((16, 24, 3), dtype=np.uint8)
        if image is None
        else np.asarray(image).copy()
    )
    imp_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    nnunet_mask = np.ones(image.shape[:2], dtype=np.uint8)
    return SimpleNamespace(
        request_id="a" * 32,
        input_sha256=rgb_sha256(image),
        original_rgb=image,
        receipt_eligible=complete,
        total_latency_ms=18.5 if complete else None,
        total_latency_scope="dual_service_run_end_to_end_wall" if complete else None,
        imp=SimpleNamespace(
            status="completed",
            mask=imp_mask,
            overlay=image.copy(),
            failure_code=None,
            model_id="L206-control-s206",
            checkpoint_sha256=runtime_projection()["imp"]["checkpoint_sha256"],
            preprocessing="imp_runtime_control",
            reported_latency_ms=7.0,
            coordinator_latency_ms=8.0,
            reported_latency_scope="imp_model_forward_cuda_sync",
            coordinator_latency_scope="imp_preview_control_end_to_end_wall",
            device="cuda:0",
        ),
        nnunet=SimpleNamespace(
            status="completed" if complete else "failed",
            mask=nnunet_mask if complete else None,
            overlay=image.copy() if complete else None,
            failure_code=None if complete else "timeout",
            model_id=MODEL_ID if complete else None,
            checkpoint_sha256=CHECKPOINT_SHA256 if complete else None,
            preprocessing="nnunet_natural_image_2d_czyx" if complete else None,
            reported_latency_ms=11.5 if complete else None,
            coordinator_latency_ms=17.5 if complete else None,
            reported_latency_scope="nnunet_predict_single_npy_array_cuda_sync" if complete else None,
            coordinator_latency_scope="nnunet_localhost_client_end_to_end_wall" if complete else None,
            device="cuda:0" if complete else None,
            protocol=PROTOCOL_ID if complete else None,
        ),
    )


class FakeDualService:
    def __init__(self, *, complete: bool = True) -> None:
        self.complete = complete
        self.images: list[np.ndarray] = []

    def run(self, image: np.ndarray) -> SimpleNamespace:
        self.images.append(np.asarray(image).copy())
        return _dual_result(complete=self.complete, image=image)


def test_fixed_prediction_without_gt_has_no_accuracy_payload() -> None:
    service = FakeService()

    response = run_fixed_comparison(
        service, _registry(), "component:verified", "clean", None
    )

    assert response.ok
    assert response.metrics_markdown == NO_GT_MESSAGE
    assert service.fixed_calls == [("component:verified", "clean")]
    assert response.control_checkpoint_sha256 == "a" * 64
    assert response.candidate_checkpoint_sha256 == "b" * 64
    assert "train_screen / exact_fixed_cache / historical_cache_provenance_drift" in response.status_html


def test_fixed_prediction_with_gt_has_both_arms() -> None:
    gt = _fixed_result().control_mask.copy()

    response = run_fixed_comparison(
        FakeService(), _registry(), "component:verified", "clean", gt
    )

    assert response.ok
    assert "Control" in response.metrics_markdown
    assert "Candidate" in response.metrics_markdown
    assert response.receipt["metrics"]["control"]["dice"] == 1.0
    assert response.receipt["metrics"]["candidate"]["dice"] == 1.0
    assert response.receipt["ground_truth_binding"] == {
        "mask_sha256_raw": "1" * 64,
        "mask_sha256_binary": "2" * 64,
        "mask_sha256_runtime": ImmutableSnapshot.decoded_binary_mask_sha256(gt),
    }


def test_fixed_prediction_rejects_ground_truth_outside_verified_mask_binding() -> None:
    gt = _fixed_result().control_mask.copy()
    gt[0, 0] = 1

    response = run_fixed_comparison(
        FakeService(), _registry(), "component:verified", "clean", gt
    )

    assert not response.ok
    assert "ground truth" in response.error_html.lower()
    assert response.receipt is None


def test_control_only_preview_never_calls_candidate_compare() -> None:
    service = FakeService()
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    response = run_control_preview(service, _registry(), image)

    assert response.ok and response.mode == "control_only"
    assert service.control_calls == 1
    assert response.candidate_overlay is None
    assert response.candidate_mask is None
    assert response.metrics_markdown == ""
    assert response.candidate_state_html == CONTROL_LOCKED_HTML
    assert "candidate" not in json.dumps(response.receipt).lower()


def test_legacy_run_comparison_alias_is_control_only() -> None:
    response = run_comparison(
        FakeService(), _registry(), np.zeros((32, 32, 3), dtype=np.uint8), None
    )

    assert response.mode == "control_only"
    assert response.metrics_markdown == NO_GT_MESSAGE


def test_candidate_unavailable_returns_one_error_and_clears_outputs() -> None:
    class Unavailable(FakeService):
        def compare_fixed(self, identifier: str, *, corruption: str):
            raise CandidateUnavailableError("E:" + "/private/cache unavailable")

    response = run_fixed_comparison(
        Unavailable(), _registry(), "component:verified", "clean", None
    )

    assert not response.ok
    assert "error-card" in response.error_html
    assert ("E:" + "/private") not in response.error_html
    assert response.original_rgb is None
    assert response.control_overlay is None and response.candidate_overlay is None
    assert response.control_mask is None and response.candidate_mask is None
    assert response.receipt is None


def test_invalid_gt_returns_one_error_and_clears_outputs() -> None:
    invalid_gt = np.zeros((31, 32), dtype=np.uint8)

    response = run_fixed_comparison(
        FakeService(), _registry(), "component:verified", "clean", invalid_gt
    )

    assert not response.ok
    assert "ground truth" in response.error_html.lower()
    assert response.original_rgb is None
    assert response.receipt is None


def test_service_error_clears_every_result_slot_without_path_leak() -> None:
    class Broken(FakeService):
        def compare_fixed(self, identifier: str, *, corruption: str):
            raise RuntimeError("C:" + "\u005csecret\u005ccheckpoint.pt")

    response = run_fixed_comparison(
        Broken(), _registry(), "component:verified", "clean", None
    )

    assert not response.ok
    assert all(
        value is None
        for value in (
            response.original_rgb,
            response.control_overlay,
            response.control_mask,
            response.candidate_overlay,
            response.candidate_mask,
            response.receipt,
        )
    )
    assert "secret" not in response.error_html.lower()


def test_app_has_one_worker_queue_and_no_forbidden_claims() -> None:
    demo = create_app(FakeService(), _registry())
    config = json.dumps(demo.get_config_file(), sort_keys=True)

    assert demo._queue is not None
    assert demo._queue.default_concurrency_limit == 1
    assert demo.api_open is False
    assert "arbitrary-upload candidate is disabled" in config
    assert "0/76" in config
    assert "L206-control-s206" in config
    assert "L192-nnUNet-v2-raw-100ep" in config
    assert "be606b0a0940" in config
    assert "3814716033af" in config
    assert "be606b0a0940839b019ea60117dda4b27f9b8f04d54306b5b676f2c29516fcef" not in config
    assert "3814716033afd464dacc573f92a5a44ff20eb7f2163d99b4f16ecff8aa278ea2" not in config
    assert "Live demo only; paper RQ1 uses Loop191 versus Loop192" in config
    assert "afb86b2a5161189369dbc3c985e78f214c305470661048c6643726612f57638b" not in config
    assert "Legacy Audit" in config and "Clean-v3 Evidence" in config
    assert "clinical grid" not in config.lower()
    assert "state-of-the-art" not in config.lower()
    assert "protected test result" not in config.lower()


def test_primary_tab_is_live_dual_model_compare() -> None:
    registry = _registry()
    registry["_demo_runtime"]["sidecar_ready"] = True
    demo = create_app(FakeService(), registry, dual_service=FakeDualService())
    config = json.dumps(demo.get_config_file())

    assert config.index("Live Dual-Model Compare") < config.index(
        "Exact Fixed-Cache Compare"
    )
    assert "Run both models" in config
    assert "IMP mask" in config and "nnU-Net mask" in config
    assert "Exploratory \\u2014 no ground truth" in config
    assert demo._queue.default_concurrency_limit == 1
    assert demo.api_open is False


def test_incomplete_callback_keeps_current_imp_and_clears_nnunet_receipt() -> None:
    values = dual_component_values(_dual_result(complete=False), _registry())

    assert values[3] is not None
    assert values[5] is None
    assert values[-1] is None
    assert "unavailable" in values[0].lower()
    assert "accuracy" not in values[6].lower()


def test_complete_callback_emits_current_result_and_receipt() -> None:
    result = _dual_result(complete=True)
    values = dual_component_values(
        result, _registry(), input_evidence=synthetic_evidence(result.original_rgb)
    )

    assert values[1] is not None
    assert values[2] is not None and values[4] is not None
    assert "reconstructed runtime" in values[6]
    assert values[-1] is not None


def test_dual_run_uses_selected_current_image_and_fails_closed_without_input() -> None:
    service = FakeDualService()
    image = np.full((9, 13, 3), 17, dtype=np.uint8)

    values = run_dual_live(service, _registry(), image, upload_evidence(image))

    assert len(service.images) == 1
    np.testing.assert_array_equal(service.images[0], image)
    assert values[1] is not None

    cleared = run_dual_live(service, _registry(), None)
    assert cleared[1:6] == (None, None, None, None, None)
    assert cleared[-1] is None


def test_dual_run_rejects_mismatched_input_evidence_without_stale_receipt() -> None:
    image = np.full((9, 13, 3), 17, dtype=np.uint8)
    mismatched = synthetic_evidence(np.zeros((9, 13, 3), dtype=np.uint8))

    values = run_dual_live(FakeDualService(), _registry(), image, mismatched)

    assert values[1:6] == (None, None, None, None, None)
    assert values[-1] is None


def test_dual_run_rejects_result_bound_to_a_different_current_input() -> None:
    class StaleDualService:
        def run(self, _image: np.ndarray) -> SimpleNamespace:
            return _dual_result(complete=True)

    values = run_dual_live(
        StaleDualService(),
        _registry(),
        np.full((16, 24, 3), 17, dtype=np.uint8),
    )

    assert values[1:6] == (None, None, None, None, None)
    assert values[-1] is None


@pytest.mark.parametrize("superseding_event", ["input_change", "second_run"])
def test_inflight_old_result_is_discarded_after_generation_changes(
    superseding_event: str,
) -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    generation = guard.begin(session)
    image = np.full((12, 18, 3), 29, dtype=np.uint8)

    class SupersedingService(FakeDualService):
        def run(self, current: np.ndarray) -> SimpleNamespace:
            result = super().run(current)
            if superseding_event == "input_change":
                guard.invalidate(session)
            else:
                guard.begin(session)
            return result

    values = run_guarded_dual(
        guard,
        session,
        generation,
        SupersedingService(),
        _registry(),
        None,
        image,
        {},
    )

    assert "superseded" in values[0].lower()
    assert values[1:6] == (None, None, None, None, None)
    assert values[-1] is None


def test_superseded_inflight_failure_still_returns_superseded_state() -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    generation = guard.begin(session)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    class SupersededFailure:
        def run(self, _image: np.ndarray) -> SimpleNamespace:
            guard.invalidate(session)
            raise RuntimeError("old request failed")

    values = run_guarded_dual(
        guard,
        session,
        generation,
        SupersededFailure(),
        _registry(),
        None,
        image,
        {},
    )

    assert "superseded" in values[0].lower()
    assert values[1:6] == (None, None, None, None, None)
    assert values[-1] is None


def test_guarded_callback_rejects_two_active_sources() -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    service = FakeDualService()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    values = run_guarded_dual(
        guard,
        session,
        guard.begin(session),
        service,
        _registry(),
        "sample",
        image,
        {"sample": image},
    )

    assert service.images == []
    assert "one input source" in values[0].lower()
    assert values[-1] is None


def test_guarded_callback_rejects_bundled_array_without_evidence() -> None:
    guard = RequestGenerationGuard()
    service = FakeDualService()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    values = run_guarded_dual(
        guard,
        "session-missing-evidence",
        guard.begin("session-missing-evidence"),
        service,
        _registry(),
        "sample",
        None,
        {"sample": image},
    )

    assert service.images == []
    assert "invalid" in values[0].lower()
    assert values[-1] is None


def test_upload_and_sample_switches_clear_the_opposite_source() -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    service = FakeDualService()
    upload = np.full((10, 14, 3), 13, dtype=np.uint8)
    sample = np.full((10, 14, 3), 31, dtype=np.uint8)
    first = guard.begin(session)

    upload_change = upload_source_change_values(guard, session)
    assert upload_change[0] is None
    assert not guard.is_current(session, first)

    upload_values = run_guarded_dual(
        guard,
        session,
        guard.begin(session),
        service,
        _registry(),
        None,
        upload,
        {},
    )
    assert upload_values[1] is not None

    second = guard.begin(session)
    sample_change = sample_source_change_values(guard, session)
    assert sample_change[0] is None
    assert not guard.is_current(session, second)

    sample_values = run_guarded_dual(
        guard,
        session,
        guard.begin(session),
        service,
        _registry(),
        "sample",
        sample_change[0],
        {
            "sample": LiveSample(
                "Synthetic calibration field - no ground truth",
                sample,
                synthetic_evidence(sample),
            )
        },
    )
    assert sample_values[1] is not None
    np.testing.assert_array_equal(service.images[-1], sample)


def test_guarded_dual_receipt_binds_selected_public_sample_evidence() -> None:
    image = np.full((10, 14, 3), 31, dtype=np.uint8)
    evidence = synthetic_evidence(image)
    sample = LiveSample("Synthetic calibration field - no ground truth", image, evidence)
    guard = RequestGenerationGuard()
    values = run_guarded_dual(
        guard, "session-a", guard.begin("session-a"),
        FakeDualService(), _registry(), "synthetic", None, {"synthetic": sample},
    )

    assert values[-1] is not None
    receipt = json.loads(Path(values[-1]).read_text(encoding="ascii"))
    assert receipt["input"]["evidence_kind"] == "synthetic"
    assert receipt["evidence_class"] == "illustrative_synthetic_no_ground_truth"


def _public_live_sample(sample_id: str, image: np.ndarray) -> LiveSample:
    evidence = LiveInputEvidence(
        "public_sample",
        "illustrative_public_sample_no_ground_truth",
        rgb_sha256(image),
        sample_id,
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
    return LiveSample(f"{sample_id} public", image, evidence)


def test_guarded_dual_receipt_preserves_public_evidence_and_gt_flags() -> None:
    image = np.full((10, 14, 3), 41, dtype=np.uint8)
    sample = _public_live_sample("ISIC_0000050", image)
    guard = RequestGenerationGuard()

    values = run_guarded_dual(
        guard,
        "session-public",
        guard.begin("session-public"),
        FakeDualService(),
        _registry(),
        "ISIC_0000050",
        None,
        {"ISIC_0000050": sample},
    )

    receipt = json.loads(Path(values[-1]).read_text(encoding="ascii"))
    assert receipt["evidence_class"] == "illustrative_public_sample_no_ground_truth"
    assert receipt["input"]["evidence_kind"] == "public_sample"
    assert receipt["input"]["sample_id"] == "ISIC_0000050"
    assert receipt["input"]["ground_truth_used"] is False
    assert receipt["input"]["ground_truth_not_loaded"] is True


def test_guarded_dual_receipt_marks_upload_without_public_metadata() -> None:
    image = np.full((10, 14, 3), 43, dtype=np.uint8)
    guard = RequestGenerationGuard()

    values = run_guarded_dual(
        guard,
        "session-upload",
        guard.begin("session-upload"),
        FakeDualService(),
        _registry(),
        None,
        image,
        {},
    )

    receipt = json.loads(Path(values[-1]).read_text(encoding="ascii"))
    evidence = receipt["input"]
    assert evidence["evidence_kind"] == "arbitrary_upload"
    assert set(evidence) == {"rgb_sha256", "dimensions", "evidence_kind"}


def test_guarded_dual_fails_closed_on_invalid_upload_schema() -> None:
    guard = RequestGenerationGuard()
    values = run_guarded_dual(
        guard,
        "session-invalid-upload",
        guard.begin("session-invalid-upload"),
        FakeDualService(),
        _registry(),
        None,
        np.zeros((10, 14), dtype=np.uint8),
        {},
    )

    assert "invalid" in values[0].lower()
    assert values[-1] is None


def test_guarded_dual_fails_closed_on_evidence_rgb_mismatch() -> None:
    image = np.full((10, 14, 3), 47, dtype=np.uint8)
    sample = _public_live_sample("ISIC_0000050", image)
    forged = LiveSample(
        sample.label,
        sample.image,
        LiveInputEvidence(
            sample.evidence.kind,
            sample.evidence.evidence_class,
            "0" * 64,
            sample.evidence.sample_id,
            sample.evidence.source_dataset,
            sample.evidence.source_page,
            sample.evidence.image_license,
            sample.evidence.training_exposure,
            False,
            True,
        ),
    )
    guard = RequestGenerationGuard()

    values = run_guarded_dual(
        guard,
        "session-mismatch",
        guard.begin("session-mismatch"),
        FakeDualService(),
        _registry(),
        "ISIC_0000050",
        None,
        {"ISIC_0000050": forged},
    )

    assert "invalid" in values[0].lower()
    assert values[1:6] == (None, None, None, None, None)
    assert values[-1] is None


def test_live_dropdown_contains_synthetic_plus_exactly_two_public_samples() -> None:
    registry = _registry()
    registry["_demo_runtime"]["sidecar_ready"] = True
    first = _public_live_sample(
        "ISIC_0000050", np.full((5, 7, 3), 1, dtype=np.uint8)
    )
    last = _public_live_sample(
        "ISIC_0016069", np.full((5, 7, 3), 2, dtype=np.uint8)
    )
    registry["_demo_runtime"]["dual_live_samples"] = {
        "ISIC_0000050": first,
        "ISIC_0016069": last,
    }
    registry["_demo_runtime"]["dual_live_choices"] = [
        (first.label, "ISIC_0000050"),
        (last.label, "ISIC_0016069"),
    ]

    config = create_app(
        FakeService(), registry, dual_service=FakeDualService()
    ).get_config_file()
    dropdown = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label")
        == "Bundled public / synthetic sample"
    )
    choices = dropdown["props"]["choices"]

    assert [choice[1] for choice in choices] == [
        "ISIC_0000050",
        "ISIC_0016069",
        "synthetic-calibration",
    ]


def _ready_app(*, public_tunnel_mode: bool):
    registry = _registry()
    registry["_demo_runtime"]["sidecar_ready"] = True
    first = _public_live_sample(
        "ISIC_0000050", np.full((5, 7, 3), 1, dtype=np.uint8)
    )
    second = _public_live_sample(
        "ISIC_0016069", np.full((5, 7, 3), 2, dtype=np.uint8)
    )
    registry["_demo_runtime"]["dual_live_samples"] = {
        "ISIC_0000050": first,
        "ISIC_0016069": second,
    }
    registry["_demo_runtime"]["dual_live_choices"] = [
        (first.label, "ISIC_0000050"),
        (second.label, "ISIC_0016069"),
    ]
    return create_app(
        FakeService(),
        registry,
        dual_service=FakeDualService(),
        public_tunnel_mode=public_tunnel_mode,
    )


def _ready_config(*, public_tunnel_mode: bool) -> str:
    return json.dumps(
        _ready_app(public_tunnel_mode=public_tunnel_mode).get_config_file(),
        sort_keys=True,
    )


def _sanitized_config_summary(app, config: dict) -> dict:
    components = {
        component["id"]: component
        for component in config["components"]
    }
    return {
        "queue_workers": app._queue.default_concurrency_limit,
        "components": [
            {
                "id": component["id"],
                "type": component["type"],
                "label": component.get("props", {}).get("label"),
                "sources": component.get("props", {}).get("sources"),
                "visible": component.get("props", {}).get("visible"),
            }
            for component in config["components"]
            if component["type"] in {"file", "image", "number", "dropdown"}
        ],
        "dependencies": [
            {
                "api_name": dependency["api_name"],
                "input_types": [components[item]["type"] for item in dependency["inputs"]],
                "targets": dependency["targets"],
                "queue": dependency["queue"],
                "visibility": dependency["api_visibility"],
            }
            for dependency in config["dependencies"]
        ],
    }


def test_create_app_defaults_to_preserved_public_safe_config_graph() -> None:
    assert inspect.signature(create_app).parameters["preserve_mode"].default is True

    local = _ready_app(public_tunnel_mode=False)
    public = _ready_app(public_tunnel_mode=True)
    local_summary = _sanitized_config_summary(local, local.get_config_file())
    public_summary = _sanitized_config_summary(public, public.get_config_file())

    assert local.delete_cache is None and public.delete_cache is None
    assert local_summary["queue_workers"] == public_summary["queue_workers"] == 1
    assert any(
        component["type"] == "image" and component["sources"] == ["upload"]
        for component in local_summary["components"]
    )
    assert all(
        not (
            component["type"] in {"file", "image"}
            and "upload" in (component["sources"] or [])
        )
        for component in public_summary["components"]
    )
    local_dual = next(
        dependency
        for dependency in local_summary["dependencies"]
        if dependency["api_name"] == "dual_live_compare"
    )
    public_dual = next(
        dependency
        for dependency in public_summary["dependencies"]
        if dependency["api_name"] == "dual_live_compare"
    )
    assert local_dual["input_types"] == ["number", "dropdown", "image"]
    assert public_dual == {
        "api_name": "dual_live_compare",
        "input_types": ["number", "dropdown"],
        "targets": [(None, "then")],
        "queue": True,
        "visibility": "public",
    }


@pytest.mark.parametrize("public_tunnel_mode", [False, True])
def test_live_sample_change_invalidates_stale_dual_outputs(
    public_tunnel_mode: bool,
) -> None:
    config = _ready_app(public_tunnel_mode=public_tunnel_mode).get_config_file()
    components = {component["id"]: component for component in config["components"]}
    sample_id = next(
        component_id
        for component_id, component in components.items()
        if component["props"].get("label") == "Bundled public / synthetic sample"
    )
    output_ids = {
        component_id
        for component_id, component in components.items()
        if component["props"].get("elem_id")
        in {"dual-live-state", "dual-receipt"}
    }

    invalidation = next(
        dependency
        for dependency in config["dependencies"]
        if dependency["targets"] == [(sample_id, "change")]
    )

    assert invalidation["queue"] is False
    assert output_ids <= set(invalidation["outputs"])


@pytest.mark.parametrize("public_tunnel_mode", [False, True])
def test_live_sample_change_callback_clears_outputs_in_declared_order(
    public_tunnel_mode: bool,
) -> None:
    app = _ready_app(public_tunnel_mode=public_tunnel_mode)
    config = app.get_config_file()
    components = {component["id"]: component for component in config["components"]}
    sample_id = next(
        component_id
        for component_id, component in components.items()
        if component["props"].get("label") == "Bundled public / synthetic sample"
    )
    invalidation = next(
        dependency
        for dependency in config["dependencies"]
        if dependency["targets"] == [(sample_id, "change")]
    )
    callback = app.fns[invalidation["id"]].fn

    values = asyncio.run(
        callback(SimpleNamespace(session_hash=f"sample-change-{public_tunnel_mode}"))
    )
    cleared_dual = (
        app_module.DUAL_IDLE_HTML,
        None,
        None,
        None,
        None,
        None,
        "",
        None,
    )
    expected = cleared_dual if public_tunnel_mode else (None, *cleared_dual)

    assert len(values) == len(invalidation["outputs"])
    assert values == expected


def test_ready_ui_discloses_live_only_model_identity_and_three_evidence_classes() -> None:
    config = _ready_config(public_tunnel_mode=False)

    assert "L206 zero-channel control / seed 206" in config
    assert "Loop192 reconstructed runtime" in config
    assert "ISIC_0000050" in config and "ISIC_0016069" in config
    assert "Synthetic" in config and "Public sample" in config and "Exploratory" in config
    assert "This live comparison is not paper RQ1; paper RQ1 compares Loop191 with Loop192." in config


def test_public_tunnel_mode_has_no_upload_component_or_upload_api() -> None:
    config = _ready_config(public_tunnel_mode=True)

    assert '"sources": ["upload"]' not in config
    assert "Exploratory \\u2014 no ground truth" not in config
    assert "Public tunnel: bundled public/synthetic inputs only" in config


def test_public_mode_callback_rejects_forged_upload_server_side() -> None:
    service = FakeDualService()
    guard = RequestGenerationGuard()
    values = run_guarded_dual(
        guard,
        "session-a",
        guard.begin("session-a"),
        service,
        _registry(),
        None,
        np.zeros((8, 8, 3), dtype=np.uint8),
        {},
        public_tunnel_mode=True,
    )

    assert "source not allowed" in values[0].lower()
    assert values[1:6] == (None, None, None, None, None)
    assert values[-1] is None
    assert service.images == []


def test_request_generations_are_isolated_by_session() -> None:
    guard = RequestGenerationGuard()
    token_a = guard.begin("session-a")
    token_b = guard.begin("session-b")

    guard.invalidate("session-a")

    assert not guard.is_current("session-a", token_a)
    assert guard.is_current("session-b", token_b)
    replacement_b = guard.begin("session-b")
    assert not guard.is_current("session-b", token_b)
    assert guard.is_current("session-b", replacement_b)


@pytest.mark.parametrize("session_id", ["", "   ", None])
def test_request_generation_rejects_empty_session_id(session_id: str | None) -> None:
    with pytest.raises(ValueError, match="session"):
        RequestGenerationGuard().begin(session_id)


def test_gradio_request_requires_nonempty_session_hash() -> None:
    assert (
        app_module._request_session_id(SimpleNamespace(session_hash="session-a"))
        == "session-a"
    )
    with pytest.raises(ValueError, match="session"):
        app_module._request_session_id(SimpleNamespace(session_hash=""))


def test_stale_generation_does_not_materialize_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    token = guard.begin(session)
    guard.invalidate(session)
    called = False

    def forbidden_receipt(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("stale receipt materialized")

    monkeypatch.setattr(app_module, "_receipt_file", forbidden_receipt)

    assert guard.publish_receipt(session, token, {"ok": True}) is None
    assert not called


def test_stale_after_result_formatting_does_not_materialize_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    generation = guard.begin(session)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    materialized = False

    class FormattingInvalidation:
        def __init__(self, result: SimpleNamespace) -> None:
            self._result = result

        @property
        def receipt_eligible(self) -> bool:
            guard.invalidate(session)
            return self._result.receipt_eligible

        def __getattr__(self, name: str):
            return getattr(self._result, name)

    class Service(FakeDualService):
        def run(self, current: np.ndarray) -> FormattingInvalidation:
            return FormattingInvalidation(super().run(current))

    def forbidden_receipt(*_args, **_kwargs):
        nonlocal materialized
        materialized = True
        raise AssertionError("stale formatted receipt materialized")

    monkeypatch.setattr(app_module, "_receipt_file", forbidden_receipt)
    values = run_guarded_dual(
        guard,
        session,
        generation,
        Service(),
        _registry(),
        None,
        image,
        {},
    )

    assert "superseded" in values[0].lower()
    assert values[-1] is None
    assert not materialized


@pytest.mark.parametrize("clear_action", ["begin", "invalidate"])
def test_new_generation_deletes_session_owned_receipt(clear_action: str) -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    token = guard.begin(session)
    receipt = guard.publish_receipt(session, token, {"ok": True})
    assert receipt is not None
    path = Path(receipt)
    assert path.is_file()

    if clear_action == "begin":
        guard.begin(session)
    else:
        guard.invalidate(session)

    assert not path.exists()


@pytest.mark.parametrize("clear_action", ["begin", "invalidate"])
def test_receipt_cleanup_failure_blocks_session_without_advancing(
    clear_action: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    token = guard.begin(session)
    receipt = guard.publish_receipt(session, token, {"ok": True})
    assert receipt is not None
    path = Path(receipt)
    monkeypatch.setattr(app_module, "_delete_owned_receipt", lambda *_args: False)

    with pytest.raises(RuntimeError, match="receipt cleanup"):
        getattr(guard, clear_action)(session)

    assert path.is_file()
    assert not guard.is_current(session, token)
    path.unlink()


@pytest.mark.parametrize(
    "change_handler",
    [upload_source_change_values, sample_source_change_values],
)
def test_source_change_reports_sanitized_receipt_cleanup_failure(
    change_handler, monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    token = guard.begin(session)
    receipt = guard.publish_receipt(session, token, {"ok": True})
    assert receipt is not None
    path = Path(receipt)
    monkeypatch.setattr(app_module, "_delete_owned_receipt", lambda *_args: False)

    values = change_handler(guard, session)

    assert values[0] is None
    assert "cleanup failed" in values[1].lower()
    assert values[2:] == (None, None, None, None, None, "", None)
    assert path.is_file()
    assert not guard.is_current(session, token)
    path.unlink()


def test_generation_guard_evicts_oldest_session_at_capacity() -> None:
    guard = RequestGenerationGuard(max_sessions=2)
    token_a = guard.begin("session-a")
    guard.complete("session-a", token_a)
    token_b = guard.begin("session-b")
    guard.complete("session-b", token_b)
    token_c = guard.begin("session-c")

    assert not guard.is_current("session-a", token_a)
    assert guard.is_current("session-b", token_b)
    assert guard.is_current("session-c", token_c)


def test_generation_guard_fails_closed_when_capacity_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = RequestGenerationGuard(max_sessions=1)
    token = guard.begin("session-a")
    receipt = guard.publish_receipt("session-a", token, {"ok": True})
    assert receipt is not None
    path = Path(receipt)
    guard.complete("session-a", token)
    monkeypatch.setattr(app_module, "_delete_owned_receipt", lambda *_args: False)

    with pytest.raises(RuntimeError, match="session capacity"):
        guard.begin("session-b")

    assert path.is_file()
    assert not guard.is_current("session-a", token)
    path.unlink()


def test_generation_guard_prunes_expired_session() -> None:
    now = [100.0]
    guard = RequestGenerationGuard(
        max_sessions=2, ttl_seconds=10.0, clock=lambda: now[0]
    )
    token_a = guard.begin("session-a")
    guard.complete("session-a", token_a)
    now[0] = 111.0
    token_b = guard.begin("session-b")

    assert not guard.is_current("session-a", token_a)
    assert guard.is_current("session-b", token_b)


def test_generation_guard_discard_removes_receipt_and_state() -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    token = guard.begin(session)
    receipt = guard.publish_receipt(session, token, {"ok": True})
    assert receipt is not None
    path = Path(receipt)

    assert guard.discard(session) is True
    assert not path.exists()
    assert not guard.is_current(session, token)


@pytest.mark.parametrize("lifecycle", ["discard", "evict"])
def test_generation_tokens_are_not_reused_after_session_recreation(
    lifecycle: str,
) -> None:
    guard = RequestGenerationGuard(max_sessions=1)
    old_token = guard.begin("session-a")
    if lifecycle == "discard":
        assert guard.discard("session-a") is True
    else:
        guard.complete("session-a", old_token)
        token_b = guard.begin("session-b")
        guard.complete("session-b", token_b)

    new_token = guard.begin("session-a")

    assert new_token > old_token
    assert not guard.is_current("session-a", old_token)
    assert guard.is_current("session-a", new_token)


def test_capacity_pressure_never_evicts_active_session() -> None:
    guard = RequestGenerationGuard(max_sessions=1)
    token_a = guard.begin("session-a")

    with pytest.raises(RuntimeError, match="session capacity"):
        guard.begin("session-b")

    assert guard.is_current("session-a", token_a)
    guard.complete("session-a", token_a)
    token_b = guard.begin("session-b")
    assert not guard.is_current("session-a", token_a)
    assert guard.is_current("session-b", token_b)


def test_abandoned_active_session_expires_at_hard_deadline() -> None:
    now = [100.0]
    guard = RequestGenerationGuard(
        max_sessions=1,
        ttl_seconds=60.0,
        active_ttl_seconds=10.0,
        clock=lambda: now[0],
    )
    token_a = guard.begin("session-a")
    now[0] = 111.0

    token_b = guard.begin("session-b")

    assert not guard.is_current("session-a", token_a)
    assert guard.is_current("session-b", token_b)


def test_stale_completion_does_not_unpin_newer_generation() -> None:
    guard = RequestGenerationGuard(max_sessions=1)
    old_token = guard.begin("session-a")
    current_token = guard.begin("session-a")

    guard.complete("session-a", old_token)
    with pytest.raises(RuntimeError, match="session capacity"):
        guard.begin("session-b")

    assert guard.is_current("session-a", current_token)
    guard.complete("session-a", current_token)
    assert guard.begin("session-b") > current_token


def test_guarded_callback_releases_active_generation_after_failure() -> None:
    guard = RequestGenerationGuard(max_sessions=1)
    token = guard.begin("session-a")

    class FailingService:
        def run(self, _image: np.ndarray) -> SimpleNamespace:
            with pytest.raises(RuntimeError, match="session capacity"):
                guard.begin("session-b")
            raise RuntimeError("failed")

    values = run_guarded_dual(
        guard,
        "session-a",
        token,
        FailingService(),
        _registry(),
        "sample",
        None,
        {
            "sample": LiveSample(
                "Synthetic calibration field - no ground truth",
                np.zeros((8, 8, 3), dtype=np.uint8),
                synthetic_evidence(np.zeros((8, 8, 3), dtype=np.uint8)),
            )
        },
    )

    assert "failed closed" in values[0].lower()
    assert guard.begin("session-b") > token


def test_cleanup_failure_sentinel_remains_visible_after_chained_callback() -> None:
    values = run_guarded_dual(
        RequestGenerationGuard(),
        "session-a",
        -1,
        FakeDualService(),
        _registry(),
        "sample",
        None,
        {"sample": np.zeros((8, 8, 3), dtype=np.uint8)},
    )

    assert "cleanup failed" in values[0].lower()
    assert values[1:6] == (None, None, None, None, None)
    assert values[-1] is None


def test_app_registers_session_unload_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    discarded: list[str] = []
    callbacks: list[object] = []

    class FakeGuard:
        def discard(self, session_id: str) -> bool:
            discarded.append(session_id)
            return True

    monkeypatch.setattr(app_module, "RequestGenerationGuard", FakeGuard)
    monkeypatch.setattr(
        app_module.gr.Blocks,
        "unload",
        lambda _self, callback: callbacks.append(callback),
    )

    create_app(
        FakeService(), _registry(), dual_service=FakeDualService(), preserve_mode=False
    )
    assert len(callbacks) == 1
    callbacks[0](SimpleNamespace(session_hash="session-a"))
    assert discarded == ["session-a"]


def test_invalidating_one_session_preserves_other_session_receipt() -> None:
    guard = RequestGenerationGuard()
    token_a = guard.begin("session-a")
    token_b = guard.begin("session-b")
    receipt_a = Path(guard.publish_receipt("session-a", token_a, {"session": "a"}))
    receipt_b = Path(guard.publish_receipt("session-b", token_b, {"session": "b"}))

    guard.invalidate("session-a")

    assert not receipt_a.exists()
    assert receipt_b.is_file()
    guard.invalidate("session-b")
    assert not receipt_b.exists()


def test_generation_invalidation_is_logical_and_never_unlinks_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    guard = RequestGenerationGuard(preserve_mode=True, preserve_run_id="run-a")
    old = guard.publish_receipt("session-a", guard.begin("session-a"), {"ok": True})
    assert old is not None
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("delete")),
    )

    guard.invalidate("session-a")

    assert Path(old).is_file()
    assert guard.current_receipt("session-a") is None


def test_nonpreserve_cleanup_does_not_poison_preserve_receipt_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = RequestGenerationGuard()
    token_a = first.begin("session-a")
    token_b = first.begin("session-b")
    assert first.publish_receipt("session-a", token_a, {"session": "a"})
    assert first.publish_receipt("session-b", token_b, {"session": "b"})
    first.invalidate("session-a")
    first.invalidate("session-b")

    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    guard = RequestGenerationGuard(preserve_mode=True, preserve_run_id="run-a")
    old = guard.publish_receipt("session-a", guard.begin("session-a"), {"ok": True})

    assert old is not None


def test_preserve_mode_disables_gradio_delete_cache() -> None:
    app = create_app(
        FakeService(), _registry(), dual_service=FakeDualService(), preserve_mode=True
    )

    assert app.delete_cache is None


def test_preserve_guard_journals_under_launcher_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    guard = RequestGenerationGuard(preserve_mode=True, preserve_run_id="shared-run")

    path = guard.publish_receipt(
        "session-a", guard.begin("session-a"), {"ok": True}
    )

    assert path is not None
    assert Path(path).is_relative_to(
        tmp_path / "demo_runtime/preserved/shared-run"
    )


def test_preserve_guard_joins_precreated_shared_launcher_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = (
        tmp_path
        / "demo_runtime/preserved/shared-run/gradio/owner-0123456789abcdef.json"
    )
    owner.parent.mkdir(parents=True)
    owner.write_text("historical", encoding="ascii")
    monkeypatch.setattr(app_module, "ROOT", tmp_path)

    guard = RequestGenerationGuard(
        preserve_mode=True, preserve_run_id="shared-run"
    )
    receipt = guard.publish_receipt(
        "session-a", guard.begin("session-a"), {"ok": True}
    )

    assert receipt is not None
    assert owner.read_text(encoding="ascii") == "historical"
    assert Path(receipt).is_relative_to(
        tmp_path / "demo_runtime/preserved/shared-run/receipt"
    )


def test_cli_preserve_run_id_mismatch_fails_before_runtime_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMP_LOOP206_PRESERVE_RUN_ID", "shared-run")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("runtime loading started before run ID rejection")
        ),
    )

    with pytest.raises(SystemExit):
        app_module.main(["--preserve-mode", "--run-id", "different-run"])


def test_failed_receipt_serialization_leaves_no_owned_file() -> None:
    guard = RequestGenerationGuard()
    session = "session-a"
    token = guard.begin(session)
    root = Path(app_module.tempfile.gettempdir())
    pattern = f"{app_module._session_receipt_prefix(session)}*.json"
    before = set(root.glob(pattern))
    created: set[Path] = set()

    try:
        receipt = guard.publish_receipt(session, token, {"bad": float("nan")})
    finally:
        created = set(root.glob(pattern)) - before
        for path in created:
            path.unlink(missing_ok=True)

    assert receipt is None
    assert created == set()


def test_receipt_deletion_rejects_unowned_path(tmp_path: Path) -> None:
    unowned = tmp_path / "loop206-public-receipt-unowned.json"
    unowned.write_text("{}", encoding="ascii")

    assert app_module._delete_owned_receipt(unowned, "session-a") is False
    assert unowned.is_file()


def test_false_readiness_disables_dual_run_button() -> None:
    registry = _registry()
    registry["_demo_runtime"]["sidecar_ready"] = False

    config = create_app(
        FakeService(), registry, dual_service=FakeDualService()
    ).get_config_file()
    run_button = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("value") == "Run both models"
    )

    assert run_button["props"]["interactive"] is False


def test_service_presence_without_explicit_readiness_disables_dual_run() -> None:
    config = create_app(
        FakeService(), _registry(), dual_service=FakeDualService()
    ).get_config_file()
    run_button = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("value") == "Run both models"
    )

    assert run_button["props"]["interactive"] is False


def test_pillow_rejects_images_above_the_16_megapixel_contract_before_decode(
    tmp_path: Path,
) -> None:
    assert app_module.MAX_UPLOAD_PIXELS == 16_000_000
    assert PILImage.MAX_IMAGE_PIXELS == app_module.MAX_UPLOAD_PIXELS
    compressed = tmp_path / "compressed-bomb.png"
    PILImage.new("1", (4001, 4000), color=0).save(compressed, optimize=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", PILImage.DecompressionBombWarning)
        with pytest.raises(PILImage.DecompressionBombWarning):
            PILImage.open(compressed)


def _guarded_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    session = root / "demo_runtime/sessions/demo-00000000000000000000000000000000"
    session.mkdir(parents=True)
    monkeypatch.setattr(app_module, "ROOT", root)
    for name in ("IMP_LOOP206_DEMO_SESSION", "GRADIO_TEMP_DIR", "TMP", "TEMP"):
        monkeypatch.setenv(name, str(session))
    return session


def test_application_requires_and_verifies_launcher_owned_temp_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _guarded_session(tmp_path, monkeypatch)

    assert app_module._verified_launcher_session() == session.resolve()
    monkeypatch.delenv("IMP_LOOP206_DEMO_SESSION")
    with pytest.raises(RuntimeError, match="guarded launcher"):
        app_module._verified_launcher_session()


def test_direct_share_fails_before_runtime_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("runtime loading started before direct-share rejection")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(SystemExit):
        app_module.main(["--share"])


def test_non_loopback_host_fails_before_runtime_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("runtime loading started before host rejection")

    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(SystemExit):
        app_module.main(["--host", "0.0.0.0"])


def test_guarded_launch_wires_server_side_upload_byte_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _guarded_session(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    class FakeDemo:
        def launch(self, **kwargs):
            captured.update(kwargs)

    class FakeLoaded:
        def build_fixed_provider(self, **kwargs):
            return object()

        def build_service(self, **kwargs):
            return object()

    monkeypatch.setattr(app_module, "validate_registry", lambda _value: None)
    monkeypatch.setattr(app_module, "load_model_registry", lambda *_args, **_kwargs: FakeLoaded())
    monkeypatch.setattr(app_module, "_official_roots", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(app_module, "_build_runtime_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        app_module,
        "NnUNetClient",
        lambda: SimpleNamespace(
            health=lambda: SimpleNamespace(
                protocol=PROTOCOL_ID,
                model_id=MODEL_ID,
                checkpoint_sha256=CHECKPOINT_SHA256,
                device="cuda:0",
                ready=True,
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "DualLiveService",
        lambda _imp, _client: object(),
    )
    monkeypatch.setattr(app_module, "create_app", lambda *_args, **_kwargs: FakeDemo())

    app_module.main([])

    assert captured["share"] is False
    assert captured["max_file_size"] == app_module.MAX_UPLOAD_BYTES


@pytest.mark.parametrize(
    ("argv", "expected_public", "expected_preserve"),
    [
        ([], False, False),
        (["--public-tunnel-mode", "--preserve-mode", "--run-id", "run-a"], True, True),
    ],
)
def test_main_preflights_pinned_sidecar_and_wires_one_imp_service(
    argv: list[str],
    expected_public: bool,
    expected_preserve: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guarded_session(tmp_path, monkeypatch)
    if expected_preserve:
        monkeypatch.setenv("IMP_LOOP206_PRESERVE_RUN_ID", "run-a")
    events: list[str] = []
    captured: dict[str, object] = {}
    imp_service = object()
    client = SimpleNamespace()
    dual_service = object()

    class FakeLoaded:
        service_calls = 0

        def build_fixed_provider(self, **_kwargs):
            return object()

        def build_service(self, **_kwargs):
            self.service_calls += 1
            events.append("build_service")
            return imp_service

    loaded = FakeLoaded()

    class FakeClient:
        def __new__(cls):
            events.append("client")
            captured["client"] = client
            return client

    def health() -> SimpleNamespace:
        events.append("health")
        return SimpleNamespace(
            protocol=PROTOCOL_ID,
            model_id=MODEL_ID,
            checkpoint_sha256=CHECKPOINT_SHA256,
            device="cuda:0",
            ready=True,
        )

    client.health = health

    def build_dual(actual_imp, actual_client):
        events.append("dual")
        assert actual_imp is imp_service and actual_client is client
        return dual_service

    class FakeDemo:
        def launch(self, **_kwargs):
            events.append("launch")

    def build_app(actual_imp, registry, *, dual_service=None, **kwargs):
        events.append("create_app")
        captured["imp_service"] = actual_imp
        captured["dual_service"] = dual_service
        captured["registry"] = registry
        captured["create_kwargs"] = kwargs
        return FakeDemo()

    monkeypatch.setattr(app_module, "validate_registry", lambda _value: None)
    monkeypatch.setattr(app_module, "load_model_registry", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(app_module, "_official_roots", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(app_module, "_build_runtime_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(app_module, "NnUNetClient", FakeClient, raising=False)
    monkeypatch.setattr(app_module, "DualLiveService", build_dual)
    monkeypatch.setattr(app_module, "create_app", build_app)

    app_module.main(argv)

    assert loaded.service_calls == 1
    assert events == ["build_service", "client", "health", "dual", "create_app", "launch"]
    assert captured["imp_service"] is imp_service
    assert captured["dual_service"] is dual_service
    assert captured["registry"]["_demo_runtime"]["sidecar_ready"] is True
    assert captured["create_kwargs"]["public_tunnel_mode"] is expected_public
    assert captured["create_kwargs"]["preserve_mode"] is expected_preserve
    if expected_preserve:
        assert captured["create_kwargs"]["preserve_run_id"] == "run-a"


def test_main_sidecar_preflight_failure_stops_before_gradio_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _guarded_session(tmp_path, monkeypatch)

    class FakeLoaded:
        def build_fixed_provider(self, **_kwargs):
            return object()

        def build_service(self, **_kwargs):
            return object()

    class BrokenClient:
        def health(self):
            raise RuntimeError("sidecar unavailable")

    def forbidden_create_app(*_args, **_kwargs):
        raise AssertionError("Gradio bound before sidecar preflight")

    monkeypatch.setattr(app_module, "validate_registry", lambda _value: None)
    monkeypatch.setattr(app_module, "load_model_registry", lambda *_args, **_kwargs: FakeLoaded())
    monkeypatch.setattr(app_module, "_official_roots", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(app_module, "_build_runtime_context", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(app_module, "NnUNetClient", BrokenClient, raising=False)
    monkeypatch.setattr(app_module, "create_app", forbidden_create_app)

    with pytest.raises(RuntimeError, match="sidecar unavailable"):
        app_module.main([])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol", "wrong.protocol"),
        ("model_id", "wrong-model"),
        ("checkpoint_sha256", "0" * 64),
        ("device", "cpu"),
        ("ready", False),
        ("ready", 1),
    ],
)
def test_sidecar_health_requires_exact_pinned_ready_identity(
    field: str, value: object
) -> None:
    health = SimpleNamespace(
        protocol=PROTOCOL_ID,
        model_id=MODEL_ID,
        checkpoint_sha256=CHECKPOINT_SHA256,
        device="cuda:0",
        ready=True,
    )
    setattr(health, field, value)

    with pytest.raises(RuntimeError, match="identity"):
        app_module._require_pinned_sidecar_health(health)


def test_runtime_ground_truth_uses_verified_index_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lesion_robustness.demo.loop206_prior as prior_module

    mask = np.zeros((384, 384), dtype=np.uint8)
    mask[100:200, 120:240] = 1
    holdout = SimpleNamespace(
        sample_id="sample",
        group_key="group",
        mask=mask,
    )
    payload = {
        "rows": [
            {
                "sample_id": "sample",
                "group_key": "group",
                "role": "holdout",
                "mask_root": 999,
                "mask_relative": "must-not-be-reopened.png",
            }
        ]
    }
    monkeypatch.setattr(
        prior_module,
        "load_dataset_index",
        lambda *_args, **_kwargs: ([], [holdout], payload),
    )
    monkeypatch.setattr(app_module, "load_public_live_samples", lambda *_args, **_kwargs: {})
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"rows": []}', encoding="ascii")

    runtime = app_module._build_runtime_context(tmp_path / "index.json", [], candidate)

    np.testing.assert_array_equal(runtime["fixed_ground_truth"]["group"], mask)


def test_theme_declares_required_tokens_and_mobile_motion_contract() -> None:
    css = (ROOT / "src/lesion_robustness/demo/theme.css").read_text(
        encoding="ascii"
    )

    for token in ("--paper", "--ink", "--teal", "--rust", "--line", "--warning"):
        assert token in css
    assert "@media (max-width: 760px)" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "linear-gradient" in css
    assert "purple" not in css.lower()


def test_theme_covers_dual_live_layout_and_operational_states() -> None:
    css = (ROOT / "src/lesion_robustness/demo/theme.css").read_text(
        encoding="ascii"
    )

    for selector in (
        ".dual-result-grid",
        ".live-state--loading",
        ".live-state--completed",
        ".live-state--unavailable",
        ".live-state--invalid",
        ".live-state--superseded",
        ":focus-visible",
    ):
        assert selector in css
    assert "grid-template-columns: repeat(3" in css
    assert "dual-reveal" in css


def test_theme_uses_dark_ink_for_text_on_cream_surfaces() -> None:
    css = (ROOT / "src/lesion_robustness/demo/theme.css").read_text(
        encoding="ascii"
    )

    for selector in (
        ".workbench-header h1",
        ".header-note",
        ".hash-strip code",
        ".tab-nav button:not(.selected)",
    ):
        assert f"{selector} {{\n  color: var(--ink) !important;" in css

    colors = dict(re.findall(r"--([a-z-]+): (#[0-9a-f]{6});", css))

    def relative_luminance(value: str) -> float:
        channels = [int(value[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    for surface in ("paper", "paper-light", "paper-deep"):
        contrast = (relative_luminance(colors[surface]) + 0.05) / (
            relative_luminance(colors["ink"]) + 0.05
        )
        assert contrast >= 4.5


def test_editorial_tabs_and_section_headings_use_ink_on_cream() -> None:
    config = _ready_app(public_tunnel_mode=True).get_config_file()
    tabs = next(component for component in config["components"] if component["type"] == "tabs")
    css = (ROOT / "src/lesion_robustness/demo/theme.css").read_text(
        encoding="ascii"
    )

    assert tabs["props"]["elem_classes"] == ["tab-nav"]
    assert ".tab-nav button:not(.selected) {\n  color: var(--ink) !important;" in css
    assert ".section-heading h2 {\n  color: var(--ink) !important;" in css


def test_mobile_hero_title_is_bounded_for_a_390px_viewport() -> None:
    css = (ROOT / "src/lesion_robustness/demo/theme.css").read_text(
        encoding="ascii"
    )
    mobile_rule = re.search(
        r"@media \(max-width: 760px\) \{(?P<content>.*)\n\}",
        css,
        flags=re.DOTALL,
    )

    assert mobile_rule is not None
    assert (
        ".workbench-header h1 {\n"
        "    font-size: clamp(42px, 12vw, 52px);"
    ) in mobile_rule.group("content")

    # 390px screenshot measurement: 408.1px title at a 58.56px glyph size.
    viewport_width_px = 390.0
    content_width_px = 375.0
    measured_title_width_px = 408.1
    measured_glyph_size_px = 58.56
    computed_glyph_size_px = min(max(42.0, viewport_width_px * 0.12), 52.0)
    projected_title_width_px = (
        measured_title_width_px / measured_glyph_size_px * computed_glyph_size_px
    )

    assert computed_glyph_size_px == pytest.approx(46.8)
    assert projected_title_width_px <= content_width_px
