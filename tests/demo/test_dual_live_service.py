from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import lesion_robustness.demo.dual_live_service as dual_live_service
from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MODEL_ID,
    PROTOCOL_ID,
    SidecarResult,
    mask_sha256,
    rgb_sha256,
)
from lesion_robustness.demo.dual_live_service import DualLiveArm, DualLiveResult, DualLiveService
from lesion_robustness.demo.nnunet_client import SidecarUnavailable


def rgb(height: int, width: int) -> np.ndarray:
    return np.full((height, width, 3), 127, dtype=np.uint8)


@dataclass(frozen=True)
class FakeImpResult:
    control_mask: np.ndarray
    control_overlay: np.ndarray
    control_latency_ms: float
    device: str
    control_model_id: str
    control_checkpoint_sha256: str


class FakeImp:
    def __init__(
        self, events: list[str], *, device: str = "cuda:0", latency_ms: float = 2.0
    ) -> None:
        self.events = events
        self.device = device
        self.latency_ms = latency_ms
        self.seen: np.ndarray | None = None

    def preview_control(self, image: np.ndarray) -> FakeImpResult:
        self.events.append("imp")
        self.seen = image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return FakeImpResult(
            mask,
            image.copy(),
            self.latency_ms,
            self.device,
            "L206-control-s206",
            "be606b0a0940839b019ea60117dda4b27f9b8f04d54306b5b676f2c29516fcef",
        )


class FakeNnUNet:
    def __init__(self, events: list[str], *, latency_ms: float = 1.0) -> None:
        self.events = events
        self.latency_ms = latency_ms
        self.seen: np.ndarray | None = None

    def predict(self, request_id: str, image: np.ndarray) -> SidecarResult:
        self.events.append("nnunet")
        self.seen = image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        return SidecarResult(
            request_id=request_id,
            input_sha256=rgb_sha256(image),
            mask=mask,
            mask_sha256=mask_sha256(mask),
            model_id=MODEL_ID,
            checkpoint_sha256=CHECKPOINT_SHA256,
            latency_ms=self.latency_ms,
            execution="live",
        )


class BrokenNnUNet:
    def predict(self, _request_id: str, _image: np.ndarray) -> SidecarResult:
        raise SidecarUnavailable("timeout")


class BrokenImp:
    def preview_control(self, _image: np.ndarray) -> FakeImpResult:
        raise RuntimeError("unavailable")


class MalformedImp:
    def __init__(self, kind: str, events: list[str]) -> None:
        self.kind = kind
        self.events = events

    def preview_control(self, image: np.ndarray) -> SimpleNamespace:
        self.events.append("imp")
        payload = SimpleNamespace(
            control_mask=np.zeros(image.shape[:2], dtype=np.uint8),
            control_overlay=image.copy(),
            control_latency_ms=2.0,
            device="cuda:0",
            control_model_id="L206-control-s206",
            control_checkpoint_sha256="b" * 64,
        )
        if self.kind == "model":
            payload.control_model_id = ""
        elif self.kind == "checkpoint":
            payload.control_checkpoint_sha256 = "B" * 64
        elif self.kind == "latency":
            payload.control_latency_ms = float("nan")
        elif self.kind == "device":
            payload.device = "gpu:0"
        elif self.kind == "claim_model":
            payload.control_model_id = "SOTA-model"
        elif self.kind == "claim_device":
            payload.device = "Clinical-device"
        return payload


class MalformedNnUNet:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def predict(self, request_id: str, image: np.ndarray) -> SidecarResult:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        model_id = MODEL_ID
        checkpoint_sha256 = CHECKPOINT_SHA256
        execution = "live"
        protocol = PROTOCOL_ID
        mask_sha = mask_sha256(mask)
        if self.kind == "binding":
            request_id = "b" * 32
        elif self.kind == "mask":
            mask.fill(2)
        elif self.kind == "geometry":
            mask = np.zeros((image.shape[0] + 1, image.shape[1]), dtype=np.uint8)
        elif self.kind == "model":
            model_id = "untrusted-model"
        elif self.kind == "checkpoint":
            checkpoint_sha256 = "c" * 64
        elif self.kind == "execution":
            execution = "cached"
        elif self.kind == "protocol":
            protocol = "wrong.protocol.v1"
        elif self.kind == "mask_sha256":
            mask_sha = "a" * 64
        return SidecarResult(
            request_id=request_id,
            input_sha256=rgb_sha256(image),
            mask=mask,
            mask_sha256=mask_sha,
            model_id=model_id,
            checkpoint_sha256=checkpoint_sha256,
            latency_ms=1.0,
            execution=execution,
            protocol=protocol,
        )


def test_dual_service_runs_imp_then_nnunet_on_same_rgb() -> None:
    events: list[str] = []
    imp = FakeImp(events)
    nnunet = FakeNnUNet(events)
    image = rgb(24, 31)

    result = DualLiveService(imp, nnunet).run(image)

    assert events == ["imp", "nnunet"]
    assert result.input_sha256 == rgb_sha256(image)
    assert imp.seen is not None and nnunet.seen is not None
    np.testing.assert_array_equal(imp.seen, nnunet.seen)
    assert result.receipt_eligible
    assert result.imp is not None and result.nnunet is not None
    assert result.imp.model_id == "L206-control-s206"
    assert result.imp.preprocessing == "imp_runtime_control"
    assert result.nnunet.model_id == MODEL_ID
    assert result.nnunet.checkpoint_sha256 == CHECKPOINT_SHA256
    assert result.nnunet.preprocessing == "nnunet_natural_image_2d_czyx"
    assert result.nnunet.protocol == PROTOCOL_ID
    assert result.total_latency_ms is not None
    assert result.imp.coordinator_latency_ms is not None
    assert result.nnunet.coordinator_latency_ms is not None
    assert result.total_latency_ms >= result.imp.coordinator_latency_ms
    assert result.total_latency_ms >= result.nnunet.coordinator_latency_ms


def test_total_latency_is_full_coordinator_wall_not_reported_arm_sum() -> None:
    clock_values = iter((0.0, 0.001, 0.012, 0.013, 0.030, 0.030))
    result = DualLiveService(
        FakeImp([], latency_ms=4.0),
        FakeNnUNet([], latency_ms=5.0),
        clock=lambda: next(clock_values),
    ).run(rgb(8, 8))

    assert result.imp is not None and result.nnunet is not None
    assert result.imp.reported_latency_ms == 4.0
    assert result.nnunet.reported_latency_ms == 5.0
    assert result.imp.coordinator_latency_ms == 11.0
    assert result.nnunet.coordinator_latency_ms == 17.0
    assert result.total_latency_ms == 30.0
    assert result.imp.coordinator_latency_ms + result.nnunet.coordinator_latency_ms != result.total_latency_ms


def test_total_latency_samples_after_complete_result_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    clock_values = iter((0.0, 0.001, 0.012, 0.013, 0.030, 0.031))
    original_complete = DualLiveResult.complete.__func__

    def trace_complete(
        cls: type[DualLiveResult], *args: object, **kwargs: object
    ) -> DualLiveResult:
        events.append("complete")
        return original_complete(cls, *args, **kwargs)

    def clock() -> float:
        events.append("clock")
        return next(clock_values)

    monkeypatch.setattr(DualLiveResult, "complete", classmethod(trace_complete))

    result = DualLiveService(
        FakeImp(events), FakeNnUNet(events), clock=clock
    ).run(rgb(8, 8))

    assert result.total_latency_ms == 31.0
    assert events[-1] == "clock"
    assert events[-2] == "complete"


def _completed_arm(*, reported: float = 1.0, coordinator: float = 2.0) -> DualLiveArm:
    return DualLiveArm(
        "completed",
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        reported_latency_ms=reported,
        coordinator_latency_ms=coordinator,
    )


@pytest.mark.parametrize(
    ("arm", "latency", "value"),
    [
        ("imp", "reported", float("nan")),
        ("imp", "reported", -1.0),
        ("imp", "coordinator", float("nan")),
        ("imp", "coordinator", -1.0),
        ("nnunet", "reported", float("nan")),
        ("nnunet", "reported", -1.0),
        ("nnunet", "coordinator", float("nan")),
        ("nnunet", "coordinator", -1.0),
    ],
)
def test_complete_rejects_nonfinite_or_negative_arm_latency(
    arm: str, latency: str, value: float
) -> None:
    imp = _completed_arm(
        **{f"{latency}": value} if arm == "imp" else {}
    )
    nnunet = _completed_arm(
        **{f"{latency}": value} if arm == "nnunet" else {}
    )

    with pytest.raises(ValueError, match="latency"):
        DualLiveResult.complete(
            "a" * 32,
            "b" * 64,
            rgb(2, 2),
            imp,
            nnunet,
            3.0,
        )


def test_nnunet_failure_keeps_current_imp_but_forbids_receipt() -> None:
    result = DualLiveService(FakeImp([]), BrokenNnUNet()).run(rgb(8, 8))

    assert result.imp.status == "completed"
    assert result.nnunet.status == "failed"
    assert result.nnunet.mask is None and result.nnunet.overlay is None
    assert not result.receipt_eligible
    assert result.total_latency_ms is None


def test_imp_failure_has_no_latency_total() -> None:
    result = DualLiveService(BrokenImp(), FakeNnUNet([])).run(rgb(8, 8))

    assert result.imp is None and result.nnunet is None
    assert not result.receipt_eligible
    assert result.total_latency_ms is None


def test_live_service_rejects_a_valid_but_unpinned_imp_identity() -> None:
    class UnpinnedImp(FakeImp):
        def preview_control(self, image: np.ndarray) -> FakeImpResult:
            value = super().preview_control(image)
            return FakeImpResult(
                value.control_mask,
                value.control_overlay,
                value.control_latency_ms,
                value.device,
                "L191-C0-clean-v3-IMP-control",
                "unverified",
            )

    result = DualLiveService(UnpinnedImp([]), FakeNnUNet([])).run(rgb(8, 8))

    assert result.imp is None
    assert result.receipt_eligible is False


def test_live_service_rejects_correct_imp_id_with_forged_checkpoint() -> None:
    class ForgedCheckpointImp(FakeImp):
        def preview_control(self, image: np.ndarray) -> FakeImpResult:
            value = super().preview_control(image)
            return FakeImpResult(
                value.control_mask,
                value.control_overlay,
                value.control_latency_ms,
                value.device,
                value.control_model_id,
                "b" * 64,
            )

    result = DualLiveService(ForgedCheckpointImp([]), FakeNnUNet([])).run(rgb(8, 8))

    assert result.imp is None
    assert result.receipt_eligible is False


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:3"])
def test_honest_imp_device_keeps_complete_dual_result(device: str) -> None:
    result = DualLiveService(FakeImp([], device=device), FakeNnUNet([])).run(rgb(8, 8))

    assert result.receipt_eligible
    assert result.imp is not None and result.imp.device == device


@pytest.mark.parametrize(
    "kind", ["model", "checkpoint", "latency", "device", "claim_model", "claim_device"]
)
def test_malformed_imp_metadata_forbids_dual_receipt(kind: str) -> None:
    events: list[str] = []
    result = DualLiveService(MalformedImp(kind, events), FakeNnUNet(events)).run(rgb(8, 8))

    assert events == ["imp"]
    assert result.imp is None and result.nnunet is None
    assert not result.receipt_eligible


def test_prohibited_imp_preprocessing_forbids_dual_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        dual_live_service, "_IMP_PREPROCESSING", "SOTA-preprocessing", raising=False
    )

    result = DualLiveService(FakeImp(events), FakeNnUNet(events)).run(rgb(8, 8))

    assert events == ["imp"]
    assert result.imp is None and result.nnunet is None
    assert not result.receipt_eligible


@pytest.mark.parametrize(
    "kind", [
        "binding",
        "mask",
        "geometry",
        "model",
        "checkpoint",
        "execution",
        "protocol",
        "mask_sha256",
    ]
)
def test_malformed_nnunet_result_keeps_imp_but_forbids_receipt(kind: str) -> None:
    result = DualLiveService(FakeImp([]), MalformedNnUNet(kind)).run(rgb(8, 8))

    assert result.imp is not None and result.imp.status == "completed"
    assert result.nnunet is not None and result.nnunet.status == "failed"
    assert result.nnunet.failure_code == "binding_mismatch"
    assert result.nnunet.mask is None and result.nnunet.overlay is None
    assert not result.receipt_eligible
