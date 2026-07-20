from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lesion_robustness.demo.app import (
    CONTROL_LOCKED_HTML,
    create_app,
    run_comparison,
    run_control_preview,
    run_fixed_comparison,
)
from lesion_robustness.demo.model_service import CandidateUnavailableError
from lesion_robustness.demo.presentation import NO_GT_MESSAGE


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
            raise CandidateUnavailableError("E:/private/cache unavailable")

    response = run_fixed_comparison(
        Unavailable(), _registry(), "component:verified", "clean", None
    )

    assert not response.ok
    assert "error-card" in response.error_html
    assert "E:/private" not in response.error_html
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
            raise RuntimeError("C:\\secret\\checkpoint.pt")

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
    assert "be606b0a0940839b019ea60117dda4b27f9b8f04d54306b5b676f2c29516fcef" in config
    assert "afb86b2a5161189369dbc3c985e78f214c305470661048c6643726612f57638b" in config
    assert "Legacy Audit" in config and "Clean-v3 Evidence" in config
    assert "clinical grid" not in config.lower()
    assert "state-of-the-art" not in config.lower()
    assert "protected test result" not in config.lower()


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
