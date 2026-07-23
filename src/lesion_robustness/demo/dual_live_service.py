"""Sequential, receipt-gated live inference for the two independent arms."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hmac
import math
import re
import secrets
from time import perf_counter
from typing import Any, Callable, Protocol

import numpy as np

from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MODEL_ID,
    PROTOCOL_ID,
    SidecarResult,
    is_public_metadata_text,
    mask_sha256,
    rgb_sha256,
    validate_binary_mask,
    validate_rgb,
)
from lesion_robustness.demo.geometry import overlay_mask
from lesion_robustness.demo.nnunet_client import SidecarUnavailable
from lesion_robustness.demo.runtime_identity import IMP


_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_PUBLIC_MODEL_ID = re.compile(r"[A-Za-z0-9._:-]+\Z")
_IMP_DEVICE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)\Z")
_IMP_PREPROCESSING = "imp_runtime_control"
IMP_REPORTED_SCOPE = "imp_model_forward_cuda_sync"
IMP_COORDINATOR_SCOPE = "imp_preview_control_end_to_end_wall"
NNUNET_REPORTED_SCOPE = "nnunet_predict_single_npy_array_cuda_sync"
NNUNET_COORDINATOR_SCOPE = "nnunet_localhost_client_end_to_end_wall"
TOTAL_SCOPE = "dual_service_run_end_to_end_wall"


class ImpPreview(Protocol):
    def preview_control(self, image: np.ndarray) -> Any: ...


class NnUNetPredictor(Protocol):
    def predict(self, request_id: str, image: np.ndarray) -> SidecarResult: ...


@dataclass(frozen=True)
class DualLiveArm:
    status: str
    mask: np.ndarray | None
    overlay: np.ndarray | None
    failure_code: str | None = None
    model_id: str | None = None
    checkpoint_sha256: str | None = None
    preprocessing: str | None = None
    reported_latency_ms: float | None = None
    coordinator_latency_ms: float | None = None
    reported_latency_scope: str | None = None
    coordinator_latency_scope: str | None = None
    device: str | None = None
    protocol: str | None = None


@dataclass(frozen=True)
class DualLiveResult:
    request_id: str
    input_sha256: str
    original_rgb: np.ndarray
    imp: DualLiveArm | None
    nnunet: DualLiveArm | None
    receipt_eligible: bool
    total_latency_ms: float | None
    total_latency_scope: str | None = None

    @classmethod
    def complete(
        cls,
        request_id: str,
        digest: str,
        rgb: np.ndarray,
        imp: DualLiveArm,
        nnunet: DualLiveArm,
        total_latency_ms: float | None,
    ) -> "DualLiveResult":
        _complete_arm_latencies(imp, nnunet)
        result = cls(
            request_id,
            digest,
            rgb.copy(),
            imp,
            nnunet,
            False,
            None,
            None,
        )
        if total_latency_ms is None:
            return result
        return result.with_total_latency(total_latency_ms)

    def with_total_latency(self, total_latency_ms: float) -> "DualLiveResult":
        if self.imp is None or self.nnunet is None:
            raise ValueError("complete dual-live result is missing arms")
        _, imp_coordinator, _, nnunet_coordinator = _complete_arm_latencies(
            self.imp, self.nnunet
        )
        total = _finite_latency(total_latency_ms)
        if total < imp_coordinator or total < nnunet_coordinator:
            raise ValueError("complete dual-live total latency is invalid")
        return replace(
            self,
            receipt_eligible=True,
            total_latency_ms=total,
            total_latency_scope=TOTAL_SCOPE,
        )

    @classmethod
    def incomplete(
        cls,
        request_id: str,
        digest: str,
        rgb: np.ndarray,
        imp: DualLiveArm,
        failure_code: str,
    ) -> "DualLiveResult":
        return cls(
            request_id,
            digest,
            rgb.copy(),
            imp,
            DualLiveArm("failed", None, None, failure_code),
            False,
            None,
            None,
        )

    @classmethod
    def without_imp(
        cls, request_id: str, digest: str, rgb: np.ndarray
    ) -> "DualLiveResult":
        return cls(request_id, digest, rgb.copy(), None, None, False, None, None)


class DualLiveService:
    def __init__(
        self,
        imp: ImpPreview,
        nnunet: NnUNetPredictor,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.imp = imp
        self.nnunet = nnunet
        self.clock = clock

    def run(self, image: np.ndarray) -> DualLiveResult:
        total_start = self.clock()
        rgb = validate_rgb(image).copy(order="C")
        request_id = secrets.token_hex(16)
        digest = rgb_sha256(rgb)
        try:
            imp_start = self.clock()
            imp = self.imp.preview_control(rgb.copy())
            imp_arm = _imp_arm(rgb, imp)
            imp_arm = _with_coordinator_latency(
                imp_arm, self.clock() - imp_start, IMP_COORDINATOR_SCOPE
            )
        except Exception:
            return DualLiveResult.without_imp(request_id, digest, rgb)
        try:
            nnunet_start = self.clock()
            nnunet = self.nnunet.predict(request_id, rgb.copy())
            nnunet_arm = _nnunet_arm(rgb, request_id, digest, nnunet)
            nnunet_arm = _with_coordinator_latency(
                nnunet_arm, self.clock() - nnunet_start, NNUNET_COORDINATOR_SCOPE
            )
        except SidecarUnavailable as exc:
            return DualLiveResult.incomplete(
                request_id, digest, rgb, imp_arm, exc.public_code
            )
        except (AttributeError, TypeError, ValueError):
            return DualLiveResult.incomplete(
                request_id, digest, rgb, imp_arm, "binding_mismatch"
            )
        finalized = DualLiveResult.complete(
            request_id,
            digest,
            rgb,
            imp_arm,
            nnunet_arm,
            None,
        )
        return finalized.with_total_latency((self.clock() - total_start) * 1000.0)


def _mask_and_overlay(
    rgb: np.ndarray, mask: np.ndarray, overlay: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    binary = validate_binary_mask(mask)
    if binary.shape != rgb.shape[:2]:
        raise ValueError("dual-live output geometry mismatch")
    rendered = validate_rgb(overlay)
    if rendered.shape != rgb.shape:
        raise ValueError("dual-live output geometry mismatch")
    return binary.copy(), rendered.copy()


def _imp_arm(rgb: np.ndarray, result: Any) -> DualLiveArm:
    try:
        mask, overlay = _mask_and_overlay(rgb, result.control_mask, result.control_overlay)
        latency = _finite_latency(result.control_latency_ms)
        model_id = _public_model_id(result.control_model_id)
        checkpoint = _checkpoint_identity(result.control_checkpoint_sha256)
        device = _cuda_device(result.device)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("dual-live IMP result is invalid") from None
    if not hmac.compare_digest(model_id, str(IMP["model_id"])):
        raise ValueError("IMP model is not the release identity")
    return DualLiveArm(
        "completed",
        mask,
        overlay,
        model_id=model_id,
        checkpoint_sha256=checkpoint,
        preprocessing=_trusted_imp_preprocessing(),
        reported_latency_ms=latency,
        reported_latency_scope=IMP_REPORTED_SCOPE,
        device=device,
    )


def _nnunet_arm(
    rgb: np.ndarray, request_id: str, digest: str, result: SidecarResult
) -> DualLiveArm:
    if (
        result.request_id != request_id
        or result.input_sha256 != digest
        or result.model_id != MODEL_ID
        or result.checkpoint_sha256 != CHECKPOINT_SHA256
        or result.execution != "live"
        or result.protocol != PROTOCOL_ID
    ):
        raise ValueError("dual-live sidecar binding mismatch")
    mask = validate_binary_mask(result.mask)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("dual-live output geometry mismatch")
    if not hmac.compare_digest(result.mask_sha256, mask_sha256(mask)):
        raise ValueError("dual-live sidecar binding mismatch")
    return DualLiveArm(
        "completed",
        mask.copy(),
        overlay_mask(rgb, mask),
        model_id=MODEL_ID,
        checkpoint_sha256=CHECKPOINT_SHA256,
        preprocessing="nnunet_natural_image_2d_czyx",
        reported_latency_ms=_finite_latency(result.latency_ms),
        reported_latency_scope=NNUNET_REPORTED_SCOPE,
        device="cuda:0",
        protocol=PROTOCOL_ID,
    )


def _with_coordinator_latency(
    arm: DualLiveArm, elapsed_seconds: float, scope: str
) -> DualLiveArm:
    return DualLiveArm(
        arm.status,
        arm.mask,
        arm.overlay,
        arm.failure_code,
        arm.model_id,
        arm.checkpoint_sha256,
        arm.preprocessing,
        arm.reported_latency_ms,
        _finite_latency(elapsed_seconds * 1000.0),
        arm.reported_latency_scope,
        scope,
        arm.device,
        arm.protocol,
    )


def _complete_arm_latencies(
    imp: DualLiveArm, nnunet: DualLiveArm
) -> tuple[float, float, float, float]:
    values = (
        imp.reported_latency_ms,
        imp.coordinator_latency_ms,
        nnunet.reported_latency_ms,
        nnunet.coordinator_latency_ms,
    )
    if any(value is None for value in values):
        raise ValueError("complete dual-live result is missing latency")
    return (
        _finite_latency(imp.reported_latency_ms),
        _finite_latency(imp.coordinator_latency_ms),
        _finite_latency(nnunet.reported_latency_ms),
        _finite_latency(nnunet.coordinator_latency_ms),
    )


def _finite_latency(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("dual-live latency is invalid")
    latency = float(value)
    if not math.isfinite(latency) or latency < 0.0:
        raise ValueError("dual-live latency is invalid")
    return latency


def _public_model_id(value: Any) -> str:
    if (
        not is_public_metadata_text(value)
        or _PUBLIC_MODEL_ID.fullmatch(value) is None
    ):
        raise ValueError("IMP model ID is invalid")
    return value


def _checkpoint_sha256(value: Any) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError("IMP checkpoint is invalid")
    return value


def _checkpoint_identity(value: Any) -> str:
    expected = str(IMP["checkpoint_sha256"])
    if not isinstance(value, str) or not hmac.compare_digest(value, expected):
        raise ValueError("IMP checkpoint is not the release identity")
    return value


def _cuda_device(value: Any) -> str:
    if not is_public_metadata_text(value) or _IMP_DEVICE.fullmatch(value) is None:
        raise ValueError("IMP device is invalid")
    return value


def _trusted_imp_preprocessing() -> str:
    if not is_public_metadata_text(_IMP_PREPROCESSING):
        raise ValueError("IMP preprocessing is invalid")
    return _IMP_PREPROCESSING
