"""Gradio entry point for the evidence-first degraded Loop206 workbench."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import warnings

from PIL import Image as PILImage

MAX_UPLOAD_PIXELS = 16_000_000
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
PILImage.MAX_IMAGE_PIXELS = MAX_UPLOAD_PIXELS
warnings.filterwarnings("error", category=PILImage.DecompressionBombWarning)

import gradio as gr
import numpy as np

from lesion_robustness.demo.dual_live_service import DualLiveService
from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256 as NNUNET_CHECKPOINT_SHA256,
    MODEL_ID as NNUNET_MODEL_ID,
    PROTOCOL_ID as NNUNET_PROTOCOL_ID,
    rgb_sha256,
)
from lesion_robustness.demo.live_inputs import (
    LiveInputEvidence,
    LiveSample,
    load_public_live_samples,
    synthetic_evidence,
    upload_evidence,
    validate_live_input_evidence,
)
from lesion_robustness.demo.metrics_service import evaluate_optional_ground_truth
from lesion_robustness.demo.model_service import (
    PINNED_REGISTRY,
    Loop206ComparisonService,
    load_model_registry,
)
from lesion_robustness.demo.presentation import (
    NO_GT_MESSAGE,
    build_control_receipt,
    build_dual_live_receipt,
    build_fixed_receipt,
    render_clean_evidence,
    render_dual_live_ledger,
    render_legacy_table,
    render_metrics,
)
from lesion_robustness.demo.nnunet_client import NnUNetClient
from lesion_robustness.demo.preserve_runtime import PreserveJournal
from lesion_robustness.evidence_registry import validate_registry
from lesion_robustness.release_manifest import load_release_manifest, runtime_projection


ROOT = Path(__file__).resolve().parents[3]
THEME_PATH = Path(__file__).with_name("theme.css")
DEFAULT_MODEL_REGISTRY = ROOT / "demo/model_registry.example.json"
DEFAULT_EVIDENCE_REGISTRY = ROOT / "demo/data/evidence_registry.json"
DEFAULT_DATASET_INDEX = ROOT / "demo_runtime/loop206_dataset_index.json"
DEFAULT_CANDIDATE_MANIFEST = ROOT / ".artifacts/preprocessing_search/loop206_leac_drlse/pilot_cache_v2_candidate/manifest.json"
DEFAULT_ZERO_MANIFEST = ROOT / ".artifacts/preprocessing_search/loop206_leac_drlse/pilot_cache_v2_zero_control/manifest.json"
DEFAULT_LIVE_CONFIG = ROOT / "configs/demo/loop206_live.yaml"
DEFAULT_CONTROL_CHECKPOINT = ROOT / "runs/loop206-control-train-screen-pilot20-checkpoints/best.pt"
DEFAULT_CANDIDATE_CHECKPOINT = ROOT / "runs/loop206-contour-channel-train-screen-pilot20-checkpoints/best.pt"

HEADER_STATUS = (
    "arbitrary-upload candidate is disabled: exact Loop206 prior parity 0/76; "
    "no approximation used."
)
CONTROL_LOCKED_HTML = (
    '<div class="locked-card" role="status"><span>Candidate locked</span>'
    '<strong>Arbitrary upload authorization unavailable.</strong>'
    '<p>Use Exact Fixed-Cache Compare for the audited two-arm result.</p></div>'
)
FIXED_BADGE_HTML = (
    '<div class="result-scope"><span>train_screen / exact_fixed_cache / '
    'historical_cache_provenance_drift</span>'
    '<p>Illustrative train-screen output. Not validation, test, or clinical evidence.</p></div>'
)
DUAL_LOADING_HTML = (
    '<div class="live-state live-state--loading" role="status" aria-live="polite">'
    '<span>Sequential run</span><strong>Running IMP, then nnU-Net</strong>'
    '<p>Prior outputs cleared. Both arms receive the same current RGB input.</p></div>'
)
DUAL_IDLE_HTML = (
    '<div class="live-state live-state--idle" role="status">'
    '<span>Ready</span><strong>Awaiting live request</strong>'
    '<p>Exploratory only. Ground truth and accuracy metrics are unavailable.</p></div>'
)
DUAL_SUPERSEDED_HTML = (
    '<div class="live-state live-state--superseded" role="status">'
    '<span>Superseded</span><strong>Newer input or run selected</strong>'
    '<p>Old result discarded. No new output or receipt was published.</p></div>'
)
CLEANUP_FAILED_GENERATION = -1


class RequestGenerationGuard:
    """Serialize current-request identity across queued Gradio callbacks."""

    def __init__(
        self,
        *,
        max_sessions: int = 256,
        ttl_seconds: float = 3600.0,
        active_ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        preserve_mode: bool = False,
        preserve_run_id: str | None = None,
    ) -> None:
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if active_ttl_seconds <= 0:
            raise ValueError("active_ttl_seconds must be positive")
        self._generations: dict[str, int] = {}
        self._receipts: dict[str, str] = {}
        self._last_seen: dict[str, float] = {}
        self._blocked: set[str] = set()
        self._active: dict[str, int] = {}
        self._max_sessions = max_sessions
        self._ttl_seconds = float(ttl_seconds)
        self._active_ttl_seconds = float(active_ttl_seconds)
        self._clock = clock
        self._preserve_mode = preserve_mode
        self._journal = (
            PreserveJournal(ROOT / "demo_runtime", run_id=preserve_run_id)
            if preserve_mode
            else None
        )
        self._next_generation = 0
        self._lock = threading.Lock()

    def _delete_receipt_locked(self, session_id: str) -> bool:
        receipt = self._receipts.get(session_id)
        if receipt is None:
            return True
        if self._preserve_mode:
            self._receipts.pop(session_id, None)
            return True
        if not _delete_owned_receipt(Path(receipt), session_id):
            return False
        self._receipts.pop(session_id, None)
        return True

    def _drop_session_locked(self, session_id: str) -> bool:
        if not self._delete_receipt_locked(session_id):
            self._blocked.add(session_id)
            return False
        self._generations.pop(session_id, None)
        self._last_seen.pop(session_id, None)
        self._blocked.discard(session_id)
        self._active.pop(session_id, None)
        return True

    def _prune_expired_locked(self, now: float, protected: str | None = None) -> None:
        expired = sorted(
            (
                (last_seen, session)
                for session, last_seen in self._last_seen.items()
                if (
                    session != protected
                    and now - last_seen
                    >= (
                        self._active_ttl_seconds
                        if session in self._active
                        else self._ttl_seconds
                    )
                )
            )
        )
        for _, session in expired:
            self._drop_session_locked(session)

    def _ensure_capacity_locked(self, session_id: str) -> None:
        if session_id in self._generations:
            return
        for _, candidate in sorted(
            (last_seen, session)
            for session, last_seen in self._last_seen.items()
            if session not in self._active
        ):
            if len(self._generations) < self._max_sessions:
                break
            self._drop_session_locked(candidate)
        if len(self._generations) >= self._max_sessions:
            raise RuntimeError("session capacity cleanup failed")

    def _next_generation_locked(self) -> int:
        self._next_generation += 1
        return self._next_generation

    def begin(self, session_id: str | None) -> int:
        session = _validated_session_id(session_id)
        with self._lock:
            now = self._clock()
            self._prune_expired_locked(now, protected=session)
            self._ensure_capacity_locked(session)
            if not self._delete_receipt_locked(session):
                self._blocked.add(session)
                self._last_seen[session] = now
                raise RuntimeError("receipt cleanup failed")
            self._blocked.discard(session)
            generation = self._next_generation_locked()
            self._generations[session] = generation
            self._active[session] = generation
            self._last_seen[session] = now
            return generation

    def invalidate(self, session_id: str | None) -> None:
        session = _validated_session_id(session_id)
        with self._lock:
            now = self._clock()
            self._prune_expired_locked(now, protected=session)
            self._ensure_capacity_locked(session)
            if not self._delete_receipt_locked(session):
                self._blocked.add(session)
                self._last_seen[session] = now
                raise RuntimeError("receipt cleanup failed")
            self._blocked.discard(session)
            self._generations[session] = self._next_generation_locked()
            self._active.pop(session, None)
            self._last_seen[session] = now

    def is_current(self, session_id: str | None, generation: int) -> bool:
        session = _validated_session_id(session_id)
        with self._lock:
            now = self._clock()
            self._prune_expired_locked(now, protected=session)
            if session in self._generations:
                self._last_seen[session] = now
            return (
                session not in self._blocked
                and generation == self._generations.get(session)
            )

    def discard(self, session_id: str | None) -> bool:
        session = _validated_session_id(session_id)
        with self._lock:
            return self._drop_session_locked(session)

    def current_receipt(self, session_id: str | None) -> str | None:
        session = _validated_session_id(session_id)
        with self._lock:
            return self._receipts.get(session)

    def complete(self, session_id: str | None, generation: int) -> None:
        session = _validated_session_id(session_id)
        with self._lock:
            if self._active.get(session) == generation:
                self._active.pop(session, None)
                if session in self._generations:
                    self._last_seen[session] = self._clock()

    def publish_receipt(
        self,
        session_id: str | None,
        generation: int,
        receipt: Mapping[str, Any],
    ) -> str | None:
        session = _validated_session_id(session_id)
        with self._lock:
            now = self._clock()
            self._prune_expired_locked(now, protected=session)
            if (
                session in self._blocked
                or generation != self._generations.get(session)
            ):
                return None
            if not self._delete_receipt_locked(session):
                self._blocked.add(session)
                return None
            try:
                path = _receipt_file(
                    receipt, session_id=session, preserve_journal=self._journal
                )
            except (OSError, TypeError, ValueError):
                return None
            if path is None:
                return None
            if not self._preserve_mode and not _owned_receipt_path(Path(path), session):
                return None
            self._receipts[session] = path
            self._last_seen[session] = now
            return path


@dataclass(frozen=True)
class WorkbenchResponse:
    ok: bool
    mode: str
    error_html: str
    status_html: str
    original_rgb: np.ndarray | None
    control_overlay: np.ndarray | None
    control_mask: np.ndarray | None
    candidate_overlay: np.ndarray | None
    candidate_mask: np.ndarray | None
    candidate_state_html: str
    metrics_markdown: str
    latency_markdown: str
    hashes_markdown: str
    control_checkpoint_sha256: str | None
    candidate_checkpoint_sha256: str | None
    receipt: dict[str, Any] | None


def _error_response(*, ground_truth: bool = False) -> WorkbenchResponse:
    detail = (
        "Ground truth could not be authorized for this result. Check mask geometry."
        if ground_truth
        else "Request failed closed. No prediction arm or prior output was retained."
    )
    return WorkbenchResponse(
        ok=False,
        mode="error",
        error_html=(
            '<div class="error-card" role="alert"><span>Workbench error</span>'
            f"<strong>{detail}</strong><p>Review the input and retry.</p></div>"
        ),
        status_html="",
        original_rgb=None,
        control_overlay=None,
        control_mask=None,
        candidate_overlay=None,
        candidate_mask=None,
        candidate_state_html="",
        metrics_markdown="",
        latency_markdown="",
        hashes_markdown="",
        control_checkpoint_sha256=None,
        candidate_checkpoint_sha256=None,
        receipt=None,
    )


def run_fixed_comparison(
    service: Loop206ComparisonService,
    registry: Mapping[str, Any],
    identifier: str,
    corruption: str,
    ground_truth: np.ndarray | None,
) -> WorkbenchResponse:
    try:
        result = service.compare_fixed(str(identifier), corruption=str(corruption))
    except Exception:
        return _error_response()
    try:
        if ground_truth is not None:
            from lesion_robustness.demo.immutable_io import ImmutableSnapshot

            actual_mask_sha256 = ImmutableSnapshot.decoded_binary_mask_sha256(
                np.asarray(ground_truth)
            )
            if actual_mask_sha256 != str(
                result.metadata.get("mask_sha256_runtime", "")
            ):
                raise ValueError("ground truth hash binding mismatch")
        metrics = evaluate_optional_ground_truth(
            result.control_mask, result.candidate_mask, ground_truth
        )
    except Exception:
        return _error_response(ground_truth=True)
    receipt = build_fixed_receipt(result, metrics=metrics, registry=registry)
    return WorkbenchResponse(
        ok=True,
        mode="exact_fixed_cache",
        error_html="",
        status_html=FIXED_BADGE_HTML,
        original_rgb=result.original_rgb,
        control_overlay=result.control_overlay,
        control_mask=result.control_mask,
        candidate_overlay=result.candidate_overlay,
        candidate_mask=result.candidate_mask,
        candidate_state_html="",
        metrics_markdown=render_metrics(metrics),
        latency_markdown=(
            "### Sequential latency\n"
            f"Control `{result.control_latency_ms:.1f} ms`  |  "
            f"Candidate `{result.candidate_latency_ms:.1f} ms`  |  "
            f"Total `{result.total_latency_ms:.1f} ms`  |  Device `{result.device}`"
        ),
        hashes_markdown=(
            "### Immutable checkpoints\n"
            f"Control `{result.control_checkpoint_sha256}`\n\n"
            f"Candidate `{result.candidate_checkpoint_sha256}`"
        ),
        control_checkpoint_sha256=str(result.control_checkpoint_sha256),
        candidate_checkpoint_sha256=str(result.candidate_checkpoint_sha256),
        receipt=receipt,
    )


def run_control_preview(
    service: Loop206ComparisonService,
    registry: Mapping[str, Any],
    image: np.ndarray | None,
) -> WorkbenchResponse:
    if image is None:
        return _error_response()
    try:
        result = service.preview_control(image)
    except Exception:
        return _error_response()
    receipt = build_control_receipt(result, registry=registry)
    return WorkbenchResponse(
        ok=True,
        mode="control_only",
        error_html="",
        status_html=(
            '<div class="result-scope result-scope--control"><span>control_only</span>'
            '<p>Illustrative arbitrary-image preview. Accuracy unavailable.</p></div>'
        ),
        original_rgb=result.original_rgb,
        control_overlay=result.control_overlay,
        control_mask=result.control_mask,
        candidate_overlay=None,
        candidate_mask=None,
        candidate_state_html=CONTROL_LOCKED_HTML,
        metrics_markdown="",
        latency_markdown=(
            "### Control latency\n"
            f"Control `{result.control_latency_ms:.1f} ms`  |  Device `{result.device}`"
        ),
        hashes_markdown=(
            "### Immutable control checkpoint\n"
            f"`{result.control_checkpoint_sha256}`"
        ),
        control_checkpoint_sha256=str(result.control_checkpoint_sha256),
        candidate_checkpoint_sha256=None,
        receipt=receipt,
    )


def run_comparison(
    service: Loop206ComparisonService,
    registry: Mapping[str, Any],
    image: np.ndarray | None,
    ground_truth: np.ndarray | None,
) -> WorkbenchResponse:
    """Compatibility shim: arbitrary images remain control-only in degraded mode."""
    response = run_control_preview(service, registry, image)
    if response.ok and ground_truth is None:
        return replace(response, metrics_markdown=NO_GT_MESSAGE)
    if response.ok:
        return _error_response(ground_truth=True)
    return response


def _validated_session_id(session_id: str | None) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("nonempty Gradio session ID required")
    if len(session_id) > 512:
        raise ValueError("Gradio session ID is too long")
    return session_id.strip()


def _request_session_id(request: gr.Request) -> str:
    return _validated_session_id(getattr(request, "session_hash", None))


def _session_receipt_prefix(session_id: str) -> str:
    session = _validated_session_id(session_id)
    digest = hashlib.sha256(session.encode("utf-8")).hexdigest()[:16]
    return f"loop206-public-receipt-{digest}-"


def _owned_receipt_path(path: Path, session_id: str) -> bool:
    try:
        candidate = path.resolve(strict=False)
        temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
    except OSError:
        return False
    return (
        not path.is_symlink()
        and candidate.parent == temp_root
        and candidate.name.startswith(_session_receipt_prefix(session_id))
        and candidate.suffix == ".json"
    )


def _delete_owned_receipt(path: Path, session_id: str) -> bool:
    if not _owned_receipt_path(path, session_id):
        return False
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _receipt_file(
    receipt: Mapping[str, Any] | None,
    *,
    session_id: str | None = None,
    preserve_journal: PreserveJournal | None = None,
) -> str | None:
    if receipt is None:
        return None
    if preserve_journal is not None:
        return str(
            preserve_journal.start(
                "receipt",
                {
                    "receipt": dict(receipt),
                    "session_id": _validated_session_id(session_id),
                },
            )
        )
    prefix = (
        "loop206-public-receipt-"
        if session_id is None
        else _session_receipt_prefix(session_id)
    )
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=prefix, delete=False,
        encoding="ascii",
    )
    try:
        with handle:
            json.dump(
                receipt,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
    except (OSError, TypeError, ValueError):
        try:
            Path(handle.name).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return handle.name


def _mask_for_display(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    return np.asarray(mask, dtype=np.uint8) * 255


def _component_values(response: WorkbenchResponse) -> tuple[Any, ...]:
    candidate_visible = response.ok and response.mode == "exact_fixed_cache"
    locked_visible = response.ok and response.mode == "control_only"
    return (
        response.error_html,
        response.status_html,
        response.original_rgb,
        response.control_overlay,
        _mask_for_display(response.control_mask),
        gr.update(value=response.candidate_overlay, visible=candidate_visible),
        gr.update(
            value=_mask_for_display(response.candidate_mask), visible=candidate_visible
        ),
        gr.update(value=response.candidate_state_html, visible=locked_visible),
        response.metrics_markdown,
        response.latency_markdown,
        response.hashes_markdown,
        _receipt_file(response.receipt),
    )


def _dual_cleared_values(status_html: str) -> tuple[Any, ...]:
    return (status_html, None, None, None, None, None, "", None)


def _dual_superseded_values() -> tuple[Any, ...]:
    return _dual_cleared_values(DUAL_SUPERSEDED_HTML)


def _dual_cleanup_failed_values() -> tuple[Any, ...]:
    return _dual_cleared_values(
        '<div class="live-state live-state--invalid" role="alert">'
        '<span>Cleanup failed</span><strong>Prior receipt could not be removed</strong>'
        '<p>Session blocked. Stop the demo before continuing.</p></div>'
    )


def upload_source_change_values(
    guard: RequestGenerationGuard, session_id: str,
) -> tuple[Any, ...]:
    """Invalidate in-flight work and clear the sample selector plus outputs."""
    try:
        guard.invalidate(session_id)
    except RuntimeError:
        return (None, *_dual_cleanup_failed_values())
    return (None, *_dual_cleared_values(DUAL_IDLE_HTML))


def sample_source_change_values(
    guard: RequestGenerationGuard, session_id: str,
) -> tuple[Any, ...]:
    """Invalidate in-flight work and clear the upload plus outputs."""
    try:
        guard.invalidate(session_id)
    except RuntimeError:
        return (None, *_dual_cleanup_failed_values())
    return (None, *_dual_cleared_values(DUAL_IDLE_HTML))


def public_sample_source_change_values(
    guard: RequestGenerationGuard, session_id: str,
) -> tuple[Any, ...]:
    """Invalidate and clear only public-mode dual outputs."""
    try:
        guard.invalidate(session_id)
    except RuntimeError:
        return _dual_cleanup_failed_values()
    return _dual_cleared_values(DUAL_IDLE_HTML)


def _dual_component_payload(
    result: Any,
    registry: Mapping[str, Any] | None = None,
    input_evidence: LiveInputEvidence | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    """Format display values and receipt data without creating a file."""
    imp = getattr(result, "imp", None)
    nnunet = getattr(result, "nnunet", None)
    imp_complete = getattr(imp, "status", None) == "completed"
    nnunet_complete = getattr(nnunet, "status", None) == "completed"
    complete = (
        getattr(result, "receipt_eligible", False) is True
        and imp_complete
        and nnunet_complete
    )
    if complete:
        if registry is None:
            return _dual_cleared_values(
                '<div class="live-state live-state--invalid" role="alert">'
                '<span>Invalid result</span><strong>Receipt binding unavailable</strong>'
                '<p>All outputs cleared. Retry through the guarded application.</p></div>'
            )[:-1], None
        try:
            evidence = input_evidence
            if evidence is None:
                raise ValueError("live input evidence is required")
            if evidence.rgb_sha256 != getattr(result, "input_sha256", None):
                raise ValueError("live input evidence hash mismatch")
            ledger = render_dual_live_ledger(result)
            receipt = build_dual_live_receipt(
                result, load_release_manifest(), evidence
            )
        except Exception:
            return _dual_cleared_values(
                '<div class="live-state live-state--invalid" role="alert">'
                '<span>Invalid result</span><strong>Evidence binding failed</strong>'
                '<p>All outputs cleared. No receipt was issued.</p></div>'
            )[:-1], None
        status = (
            '<div class="live-state live-state--completed" role="status">'
            '<span>Complete</span><strong>Both live arms completed</strong>'
            '<p>Receipt bound to this request. Ground truth was not supplied.</p></div>'
        )
        return (
            (
                status,
                getattr(result, "original_rgb", None),
                getattr(imp, "overlay", None),
                _mask_for_display(getattr(imp, "mask", None)),
                getattr(nnunet, "overlay", None),
                _mask_for_display(getattr(nnunet, "mask", None)),
                ledger,
            ),
            receipt,
        )
    if imp_complete:
        status = (
            '<div class="live-state live-state--unavailable" role="alert">'
            '<span>Incomplete</span><strong>nnU-Net unavailable</strong>'
            '<p>Current IMP output retained. nnU-Net output cleared; no receipt issued.</p></div>'
        )
        ledger = (
            "Current request incomplete.\n\n"
            "IMP: completed.\n\n"
            "nnU-Net: unavailable. No receipt issued."
        )
        return (
            status,
            getattr(result, "original_rgb", None),
            getattr(imp, "overlay", None),
            _mask_for_display(getattr(imp, "mask", None)),
            None,
            None,
            ledger,
        ), None
    return _dual_cleared_values(
        '<div class="live-state live-state--invalid" role="alert">'
        '<span>Failed closed</span><strong>IMP result unavailable</strong>'
        '<p>All outputs cleared. No receipt was issued.</p></div>'
    )[:-1], None


def dual_component_values(
    result: Any,
    registry: Mapping[str, Any] | None = None,
    *,
    input_evidence: LiveInputEvidence | None = None,
) -> tuple[Any, ...]:
    """Map one current dual-live result to fail-closed Gradio outputs."""
    values, receipt = _dual_component_payload(result, registry, input_evidence)
    return (*values, _receipt_file(receipt))


def _run_current_dual_result(
    dual_service: DualLiveService, image: np.ndarray
) -> Any:
    current = np.asarray(image)
    result = dual_service.run(current)
    if (
        getattr(result, "input_sha256", None) != rgb_sha256(current)
        or not np.array_equal(getattr(result, "original_rgb", None), current)
    ):
        raise ValueError("dual-live result does not bind the current input")
    return result


def _invalid_dual_source_values() -> tuple[Any, ...]:
    return _dual_cleared_values(
        '<div class="live-state live-state--invalid" role="alert">'
        '<span>Invalid request</span><strong>Select exactly one input source</strong>'
        '<p>All outputs cleared. Choose one bundled sample or one upload.</p></div>'
    )


def _public_tunnel_upload_values() -> tuple[Any, ...]:
    return _dual_cleared_values(
        '<div class="live-state live-state--invalid" role="alert">'
        '<span>Source not allowed</span><strong>Public tunnel rejects uploads</strong>'
        '<p>Choose a bundled public or synthetic input.</p></div>'
    )


def run_guarded_dual(
    guard: RequestGenerationGuard,
    session_id: str,
    generation: int,
    dual_service: DualLiveService | None,
    registry: Mapping[str, Any],
    sample_key: str | None,
    uploaded: np.ndarray | None,
    samples: Mapping[str, LiveSample],
    *,
    public_tunnel_mode: bool = False,
) -> tuple[Any, ...]:
    session = _validated_session_id(session_id)
    try:
        return _run_guarded_dual_once(
            guard,
            session,
            generation,
            dual_service,
            registry,
            sample_key,
            uploaded,
            samples,
            public_tunnel_mode=public_tunnel_mode,
        )
    finally:
        guard.complete(session, generation)


def _run_guarded_dual_once(
    guard: RequestGenerationGuard,
    session: str,
    generation: int,
    dual_service: DualLiveService | None,
    registry: Mapping[str, Any],
    sample_key: str | None,
    uploaded: np.ndarray | None,
    samples: Mapping[str, LiveSample],
    *,
    public_tunnel_mode: bool = False,
) -> tuple[Any, ...]:
    if generation == CLEANUP_FAILED_GENERATION:
        return _dual_cleanup_failed_values()
    if not guard.is_current(session, generation):
        return _dual_superseded_values()
    if public_tunnel_mode and uploaded is not None:
        return _public_tunnel_upload_values()
    has_sample = sample_key not in (None, "")
    has_upload = uploaded is not None
    if has_sample == has_upload:
        return _invalid_dual_source_values()
    selected = samples.get(str(sample_key)) if has_sample else None
    sample = selected if isinstance(selected, LiveSample) else None
    image = np.asarray(uploaded) if has_upload else (sample.image if sample is not None else None)
    if dual_service is None or image is None:
        return _invalid_dual_source_values()
    try:
        evidence = (
            upload_evidence(image)
            if has_upload
            else sample.evidence
            if sample is not None
            else None
        )
        validate_live_input_evidence(evidence)
        if evidence.rgb_sha256 != rgb_sha256(image):
            raise ValueError("live input evidence hash mismatch")
    except (AttributeError, TypeError, ValueError):
        return _invalid_dual_source_values()
    try:
        result = _run_current_dual_result(dual_service, image)
    except Exception:
        if not guard.is_current(session, generation):
            return _dual_superseded_values()
        return _dual_cleared_values(
            '<div class="live-state live-state--invalid" role="alert">'
            '<span>Failed closed</span><strong>Live request rejected</strong>'
            '<p>All outputs cleared. No receipt was issued.</p></div>'
        )
    if not guard.is_current(session, generation):
        return _dual_superseded_values()
    values, receipt = _dual_component_payload(result, registry, evidence)
    if not guard.is_current(session, generation):
        return _dual_superseded_values()
    if receipt is None:
        return (
            (*values, None)
            if guard.is_current(session, generation)
            else _dual_superseded_values()
        )
    receipt_path = guard.publish_receipt(session, generation, receipt)
    if not guard.is_current(session, generation):
        return _dual_superseded_values()
    if receipt_path is None:
        return _dual_cleared_values(
            '<div class="live-state live-state--invalid" role="alert">'
            '<span>Failed closed</span><strong>Receipt publication failed</strong>'
            '<p>All outputs cleared. No receipt was issued.</p></div>'
        )
    return (*values, receipt_path)


def run_dual_live(
    dual_service: DualLiveService | None,
    registry: Mapping[str, Any],
    image: np.ndarray | None,
    input_evidence: LiveInputEvidence | None = None,
) -> tuple[Any, ...]:
    if dual_service is None or image is None:
        return _invalid_dual_source_values()
    try:
        return dual_component_values(
            _run_current_dual_result(dual_service, image),
            registry,
            input_evidence=input_evidence,
        )
    except Exception:
        return _dual_cleared_values(
            '<div class="live-state live-state--invalid" role="alert">'
            '<span>Failed closed</span><strong>Live request rejected</strong>'
            '<p>All outputs cleared. No receipt was issued.</p></div>'
        )


def _synthetic_live_sample() -> np.ndarray:
    height, width = 256, 320
    y, x = np.ogrid[:height, :width]
    radial = ((x - 166.0) / 88.0) ** 2 + ((y - 124.0) / 64.0) ** 2
    image = np.empty((height, width, 3), dtype=np.float64)
    image[:] = (223.0, 189.0, 150.0)
    vignette = np.clip(((x - 160.0) ** 2 + (y - 128.0) ** 2) / 90000.0, 0, 1)
    image -= vignette[..., None] * np.array((35.0, 29.0, 23.0))
    lesion = np.clip(1.0 - radial, 0.0, 1.0)[..., None]
    image = image * (1.0 - lesion) + np.array((83.0, 48.0, 35.0)) * lesion
    return np.ascontiguousarray(np.clip(image, 0, 255), dtype=np.uint8)


def _dual_ready(
    runtime: Mapping[str, Any], dual_service: DualLiveService | None
) -> bool:
    return dual_service is not None and runtime.get("sidecar_ready") is True


def _runtime(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    value = registry.get("_demo_runtime", {})
    return value if isinstance(value, Mapping) else {}


def _has_reparse_point(path: Path) -> bool:
    attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    is_junction = bool(getattr(os.path, "isjunction", lambda _path: False)(path))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & reparse_flag) or path.is_symlink() or is_junction


def _verified_launcher_session() -> Path:
    configured = os.environ.get("IMP_LOOP206_DEMO_SESSION", "").strip()
    if not configured:
        raise RuntimeError("Loop206 application requires the guarded launcher")
    root = ROOT.resolve()
    runtime_root = (root / "demo_runtime").resolve()
    sessions = (runtime_root / "sessions").resolve()
    session = Path(configured).expanduser().resolve()
    try:
        session.relative_to(sessions)
    except ValueError as exc:
        raise RuntimeError("Loop206 application requires the guarded launcher") from exc
    if (
        not session.is_dir()
        or session.parent != sessions
        or re.fullmatch(r"demo-[0-9a-f]{32}", session.name) is None
    ):
        raise RuntimeError("Loop206 application requires the guarded launcher")
    for path in (runtime_root, sessions, session):
        if not path.is_dir() or _has_reparse_point(path):
            raise RuntimeError("Loop206 application requires the guarded launcher")
    for name in ("GRADIO_TEMP_DIR", "TMP", "TEMP"):
        value = os.environ.get(name, "").strip()
        if not value or Path(value).expanduser().resolve() != session:
            raise RuntimeError("Loop206 application requires the guarded launcher")
    return session


def create_app(
    service: Loop206ComparisonService,
    registry: Mapping[str, Any],
    *,
    dual_service: DualLiveService | None = None,
    public_tunnel_mode: bool = False,
    preserve_mode: bool = True,
    preserve_run_id: str | None = None,
) -> gr.Blocks:
    runtime = _runtime(registry)
    choices = list(runtime.get("fixed_choices", []))
    corruptions = list(runtime.get("corruptions", ["clean"]))
    ground_truths = runtime.get("fixed_ground_truth", {})
    default_choice = choices[0][1] if choices else None
    dual_ready = _dual_ready(runtime, dual_service)
    synthetic = _synthetic_live_sample()
    synthetic_samples: dict[str, LiveSample] = {
        "synthetic-calibration": LiveSample(
            "Synthetic calibration field - no ground truth", synthetic, synthetic_evidence(synthetic)
        )
    }
    runtime_samples = runtime.get("dual_live_samples", {})
    if isinstance(runtime_samples, Mapping):
        synthetic_samples.update(
            {
                str(key): value
                for key, value in runtime_samples.items()
                if isinstance(value, LiveSample)
            }
        )
    sample_choices: list[tuple[str, str]] = []
    configured_choices = runtime.get("dual_live_choices", [])
    if isinstance(configured_choices, Sequence) and not isinstance(
        configured_choices, (str, bytes)
    ):
        for value in configured_choices:
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and str(value[1]) in synthetic_samples
            ):
                sample_choices.append((f"Public sample: {value[0]}", str(value[1])))
    sample_choices.append(
        ("Synthetic calibration field - no ground truth", "synthetic-calibration")
    )
    request_guard = (
        RequestGenerationGuard(
            preserve_mode=True, preserve_run_id=preserve_run_id
        )
        if preserve_mode
        else RequestGenerationGuard()
    )

    # Gradio 6 defers CSS to launch; the compatibility argument keeps imported apps styled.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The parameters have been moved")
        demo = gr.Blocks(
            title="Audited Dermoscopy Workbench",
            fill_width=True,
            delete_cache=None if preserve_mode else (3600, 3600),
            css_paths=THEME_PATH,
        )
    with demo:
        gr.HTML(
            '<header class="workbench-header"><div class="plate">IMP / LOOP206 / AUDIT 07</div>'
            '<div><p class="eyebrow">Evidence-first segmentation instrument</p>'
            '<h1>Audited Dermoscopy<br>Workbench</h1></div>'
            '<p class="header-note">Research review surface. Outputs are non-clinical and scoped by provenance.</p>'
            "</header>"
        )
        gr.HTML(
            (
                '<div class="status-band status-band--ready" role="status">'
                '<strong>Dual-live ready</strong><span>Sequential IMP then nnU-Net; '
                'same RGB input, one queued run.</span></div>'
                if dual_ready
                else f'<div class="status-band" role="status"><strong>Degraded runtime</strong><span>{HEADER_STATUS}</span></div>'
            )
        )
        live_identity = runtime_projection()
        gr.HTML(
            '<div class="hash-strip"><div><span>Live IMP identity</span>'
            f'<code>{live_identity["imp"]["model_id"]} / '
            f'{live_identity["imp"]["checkpoint_sha256"][:12]}</code></div>'
            '<div><span>Live nnU-Net identity</span>'
            f'<code>{live_identity["nnunet"]["model_id"]} / '
            f'{live_identity["nnunet"]["checkpoint_sha256"][:12]}</code></div>'
            f'<p>{live_identity["paper_rq1_notice"]}</p></div>'
        )
        with gr.Tabs(selected="dual", elem_classes="tab-nav"):
            with gr.Tab("Live Dual-Model Compare", id="dual"):
                gr.HTML(
                    '<div class="section-heading"><span>01</span><div><p>Primary live mode</p>'
                    '<h2>Live Dual-Model Compare</h2></div></div>'
                )
                with gr.Row(elem_classes="dual-inputs"):
                    live_sample = gr.Dropdown(
                        choices=sample_choices,
                        value=sample_choices[0][1],
                        label="Bundled public / synthetic sample",
                        info=(
                            "Public tunnel: bundled public/synthetic inputs only"
                            if public_tunnel_mode
                            else "Illustrative input only; no ground truth or accuracy metrics."
                        ),
                        allow_custom_value=False,
                    )
                    live_upload = None
                    if not public_tunnel_mode:
                        live_upload = gr.Image(
                            label="Exploratory \u2014 no ground truth",
                            type="numpy",
                            sources=["upload"],
                            image_mode="RGB",
                        )
                    with gr.Column(elem_classes="live-command"):
                        gr.HTML(
                            (
                                '<div class="readiness readiness--ready"><span>Sidecar</span>'
                                '<strong>Ready / pinned</strong></div>'
                                if dual_ready
                                else '<div class="readiness readiness--unavailable"><span>Sidecar</span>'
                                '<strong>Unavailable</strong></div>'
                            )
                        )
                        run_dual = gr.Button(
                            "Run both models",
                            variant="primary",
                            elem_id="run-dual",
                            interactive=dual_ready,
                        )

                dual_status = gr.HTML(DUAL_IDLE_HTML, elem_id="dual-live-state")
                with gr.Row(elem_classes="dual-result-grid"):
                    with gr.Column(elem_classes="output-bay output-bay--source"):
                        gr.HTML(
                            '<div class="arm-label"><span>0</span><strong>Current RGB input</strong></div>'
                        )
                        dual_original = gr.Image(
                            label="Original RGB",
                            interactive=False,
                            sources=[],
                            buttons=["fullscreen"],
                        )
                    with gr.Column(elem_classes="output-bay output-bay--imp"):
                        gr.HTML(
                            '<div class="arm-label"><span>A</span><strong>IMP: L206 zero-channel control / seed 206 / live-demo-only</strong></div>'
                        )
                        dual_imp_overlay = gr.Image(
                            label="IMP overlay",
                            interactive=False,
                            sources=[],
                            buttons=["fullscreen"],
                        )
                        dual_imp_mask = gr.Image(
                            label="IMP mask",
                            interactive=False,
                            image_mode="L",
                            sources=[],
                            buttons=["fullscreen"],
                        )
                    with gr.Column(elem_classes="output-bay output-bay--nnunet"):
                        gr.HTML(
                            '<div class="arm-label"><span>B</span><strong aria-label="Loop192 reconstructed runtime">nnU-Net: Loop192 / reconstructed runtime / not original-equivalent</strong></div>'
                        )
                        dual_nnunet_overlay = gr.Image(
                            label="nnU-Net overlay",
                            interactive=False,
                            sources=[],
                            buttons=["fullscreen"],
                        )
                        dual_nnunet_mask = gr.Image(
                            label="nnU-Net mask",
                            interactive=False,
                            image_mode="L",
                            sources=[],
                            buttons=["fullscreen"],
                        )
                dual_ledger = gr.Markdown(
                    "Ground truth not supplied; no accuracy metrics are available.",
                    elem_classes="dual-ledger",
                )
                gr.HTML(
                    '<p class="live-scope">This live comparison is not paper RQ1; '
                    'paper RQ1 compares Loop191 with Loop192.</p>'
                )
                dual_receipt = gr.DownloadButton(
                    "Download current live receipt", value=None, elem_id="dual-receipt"
                )
                request_generation = gr.Number(
                    value=0, precision=0, visible=False, elem_id="dual-generation"
                )
                dual_outputs = [
                    dual_status,
                    dual_original,
                    dual_imp_overlay,
                    dual_imp_mask,
                    dual_nnunet_overlay,
                    dual_nnunet_mask,
                    dual_ledger,
                    dual_receipt,
                ]

                def dual_callback(
                    generation: int,
                    sample_key: str | None,
                    uploaded: np.ndarray | None,
                    request: gr.Request,
                ) -> tuple[Any, ...]:
                    session = _request_session_id(request)
                    return run_guarded_dual(
                        request_guard,
                        session,
                        int(generation),
                        dual_service,
                        registry,
                        sample_key,
                        uploaded,
                        synthetic_samples,
                        public_tunnel_mode=public_tunnel_mode,
                    )

                def public_dual_callback(
                    generation: int,
                    sample_key: str | None,
                    request: gr.Request,
                ) -> tuple[Any, ...]:
                    session = _request_session_id(request)
                    return run_guarded_dual(
                        request_guard,
                        session,
                        int(generation),
                        dual_service,
                        registry,
                        sample_key,
                        None,
                        synthetic_samples,
                        public_tunnel_mode=True,
                    )

                async def begin_dual_request(request: gr.Request) -> tuple[Any, ...]:
                    session = _request_session_id(request)
                    try:
                        generation = request_guard.begin(session)
                    except RuntimeError:
                        return (
                            CLEANUP_FAILED_GENERATION,
                            *_dual_cleanup_failed_values(),
                        )
                    return (generation, *_dual_cleared_values(DUAL_LOADING_HTML))

                async def upload_changed(request: gr.Request) -> tuple[Any, ...]:
                    return upload_source_change_values(
                        request_guard, _request_session_id(request)
                    )

                async def sample_changed(request: gr.Request) -> tuple[Any, ...]:
                    return sample_source_change_values(
                        request_guard, _request_session_id(request)
                    )

                async def public_sample_changed(
                    request: gr.Request,
                ) -> tuple[Any, ...]:
                    return public_sample_source_change_values(
                        request_guard, _request_session_id(request)
                    )

                clear_event = run_dual.click(
                    begin_dual_request,
                    inputs=[],
                    outputs=[request_generation, *dual_outputs],
                    queue=False,
                    api_name=False,
                )
                if public_tunnel_mode:
                    clear_event.then(
                        public_dual_callback,
                        inputs=[request_generation, live_sample],
                        outputs=dual_outputs,
                        concurrency_limit=1,
                        concurrency_id="loop206-inference",
                        api_name="dual_live_compare",
                    )
                    live_sample.change(
                        public_sample_changed,
                        inputs=[],
                        outputs=dual_outputs,
                        queue=False,
                        api_name=False,
                    )
                else:
                    clear_event.then(
                        dual_callback,
                        inputs=[request_generation, live_sample, live_upload],
                        outputs=dual_outputs,
                        concurrency_limit=1,
                        concurrency_id="loop206-inference",
                        api_name="dual_live_compare",
                    )
                    live_upload.input(
                        upload_changed,
                        inputs=[],
                        outputs=[live_sample, *dual_outputs],
                        queue=False,
                        api_name=False,
                    )
                    live_sample.change(
                        sample_changed,
                        inputs=[],
                        outputs=[live_upload, *dual_outputs],
                        queue=False,
                        api_name=False,
                    )

            with gr.Tab("Audited Fixed Samples", id="fixed"):
                gr.HTML(
                    '<div class="section-heading"><span>02</span><div><p>Separated audited mode</p>'
                    '<h2>Exact Fixed-Cache Compare</h2></div></div>'
                )
                with gr.Row(elem_classes="fixed-controls"):
                    fixed_sample = gr.Dropdown(
                        choices=choices,
                        value=default_choice,
                        label="Allowlisted train-screen sample",
                        info="76 leakage-safe holdout groups; source and channels are provider-owned.",
                        allow_custom_value=False,
                        elem_id="fixed-sample",
                    )
                    corruption = gr.Dropdown(
                        choices=corruptions,
                        value=corruptions[0] if corruptions else None,
                        label="Locked corruption",
                        allow_custom_value=False,
                    )
                    use_gt = gr.Checkbox(
                        value=False,
                        label="Use provider-bound train-screen GT",
                        info="Enables metrics for this fixed sample only.",
                    )
                    run_fixed = gr.Button(
                        "Run exact fixed compare", variant="primary", elem_id="run-fixed"
                    )

                error_html = gr.HTML(elem_id="workbench-error")
                result_scope = gr.HTML(
                    '<div class="empty-scope"><span>Awaiting audited run</span>'
                    '<p>Select a fixed sample and corruption. No output is precomputed.</p></div>',
                    elem_id="result-scope",
                )
                with gr.Row(elem_classes="result-grid"):
                    original = gr.Image(
                        label="Verified source",
                        interactive=False,
                        sources=[],
                        buttons=["fullscreen"],
                    )
                    with gr.Column(elem_classes="arm-panel arm-panel--control"):
                        gr.HTML('<div class="arm-label"><span>A</span><strong>Zero-channel control</strong></div>')
                        control_overlay = gr.Image(
                            label="Control overlay",
                            interactive=False,
                            sources=[],
                            buttons=["fullscreen"],
                        )
                        control_mask = gr.Image(
                            label="Control mask",
                            interactive=False,
                            image_mode="L",
                            sources=[],
                            buttons=["fullscreen"],
                        )
                    with gr.Column(elem_classes="arm-panel arm-panel--candidate"):
                        gr.HTML('<div class="arm-label"><span>B</span><strong>Contour-channel candidate</strong></div>')
                        candidate_overlay = gr.Image(
                            label="Candidate overlay",
                            interactive=False,
                            sources=[],
                            buttons=["fullscreen"],
                        )
                        candidate_mask = gr.Image(
                            label="Candidate mask",
                            interactive=False,
                            image_mode="L",
                            sources=[],
                            buttons=["fullscreen"],
                        )
                        candidate_state = gr.HTML()
                with gr.Row(elem_classes="result-meta"):
                    metrics = gr.Markdown(NO_GT_MESSAGE, elem_classes="metric-panel")
                    latency = gr.Markdown("### Sequential latency\nAwaiting run.")
                    hashes = gr.Markdown("### Immutable checkpoints\nVisible after an authorized run.")
                receipt = gr.DownloadButton("Download safe JSON receipt", value=None)

                outputs = [
                    error_html,
                    result_scope,
                    original,
                    control_overlay,
                    control_mask,
                    candidate_overlay,
                    candidate_mask,
                    candidate_state,
                    metrics,
                    latency,
                    hashes,
                    receipt,
                ]

                def fixed_callback(identifier: str, view: str, include_gt: bool):
                    gt = None
                    if include_gt and isinstance(ground_truths, Mapping):
                        gt = ground_truths.get(str(identifier))
                    if include_gt and gt is None:
                        return _component_values(_error_response(ground_truth=True))
                    return _component_values(
                        run_fixed_comparison(service, registry, identifier, view, gt)
                    )

                run_fixed.click(
                    fixed_callback,
                    inputs=[fixed_sample, corruption, use_gt],
                    outputs=outputs,
                    concurrency_limit=1,
                    concurrency_id="loop206-inference",
                    api_name="fixed_compare",
                )

            with gr.Tab("Clean-v3 Evidence", id="evidence"):
                gr.HTML(
                    '<div class="section-heading"><span>03</span><div><p>Scoped benchmark registry</p>'
                    '<h2>Clean-v3 Evidence</h2></div></div>'
                )
                gr.HTML(render_clean_evidence(registry))

            with gr.Tab("Legacy Audit", id="legacy"):
                gr.HTML(
                    '<div class="section-heading"><span>04</span><div><p>Separated historical record</p>'
                    '<h2>Legacy Audit</h2></div></div>'
                )
                gr.HTML(render_legacy_table(registry))

        gr.HTML(
            '<footer class="workbench-footer"><span>IMP / AUDITED DERMOSCOPY WORKBENCH</span>'
            '<span>Non-clinical research instrument</span></footer>'
        )

    def unload_session(request: gr.Request) -> None:
        try:
            request_guard.discard(_request_session_id(request))
        except ValueError:
            return

    if not preserve_mode:
        demo.unload(unload_session)
    demo.queue(api_open=False, max_size=8, default_concurrency_limit=1)
    return demo


def _official_roots(dataset_index: Path, configured: Sequence[str]) -> list[Path]:
    if configured:
        return [Path(value).expanduser().resolve() for value in configured]
    env_value = os.environ.get("IMP_LOOP206_DATA_ROOT", "").strip()
    if env_value:
        return [Path(value).expanduser().resolve() for value in env_value.split(os.pathsep) if value]
    candidates = (
        ROOT / "demo_runtime/dataset",
        ROOT.parent / "datasets/loop206",
        ROOT.parent / "datasets",
    )
    return [path.resolve() for path in candidates if path.is_dir()]


def _safe_index_path(root: Path, relative: object) -> Path:
    path = (root / str(relative)).resolve()
    path.relative_to(root.resolve())
    return path


def _build_runtime_context(
    dataset_index: Path,
    roots: Sequence[Path],
    candidate_manifest: Path,
) -> dict[str, Any]:
    from lesion_robustness.demo.loop206_prior import load_dataset_index

    _, holdout, _ = load_dataset_index(dataset_index, dataset_roots=roots)
    choices = [
        (f"{row.sample_id} / {row.group_key}", row.group_key)
        for row in sorted(holdout, key=lambda value: (value.sample_id, value.group_key))
    ]
    masks: dict[str, np.ndarray] = {}
    for row in holdout:
        if row.mask is None:
            raise ValueError("Loop206 verified holdout mask is missing")
        masks[row.group_key] = np.ascontiguousarray(row.mask, dtype=np.uint8).copy()
    manifest = json.loads(candidate_manifest.read_text(encoding="ascii"))
    corruptions = sorted(
        {
            str(row["corruption"])
            for row in manifest.get("rows", [])
            if row.get("runtime_split") == "train_screen_holdout"
        }
    )
    if "clean" in corruptions:
        corruptions.remove("clean")
        corruptions.insert(0, "clean")
    public_samples = load_public_live_samples(
        load_release_manifest(), dataset_index, roots
    )
    return {
        "fixed_choices": choices,
        "corruptions": corruptions,
        "fixed_ground_truth": masks,
        "dual_live_samples": public_samples,
        "dual_live_choices": [
            (sample.label, sample_id)
            for sample_id, sample in public_samples.items()
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the audited Loop206 workbench")
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--public-tunnel-mode", action="store_true")
    parser.add_argument("--preserve-mode", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--model-registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--evidence-registry", type=Path, default=DEFAULT_EVIDENCE_REGISTRY)
    parser.add_argument("--dataset-index", type=Path, default=DEFAULT_DATASET_INDEX)
    parser.add_argument("--dataset-root", action="append", default=[])
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--zero-manifest", type=Path, default=DEFAULT_ZERO_MANIFEST)
    parser.add_argument("--live-config", type=Path, default=DEFAULT_LIVE_CONFIG)
    return parser


def _require_pinned_sidecar_health(health: Any) -> None:
    identity = (
        getattr(health, "protocol", None),
        getattr(health, "model_id", None),
        getattr(health, "checkpoint_sha256", None),
        getattr(health, "device", None),
    )
    expected = (
        NNUNET_PROTOCOL_ID,
        NNUNET_MODEL_ID,
        NNUNET_CHECKPOINT_SHA256,
        "cuda:0",
    )
    if identity != expected or getattr(health, "ready", None) is not True:
        raise RuntimeError("nnU-Net sidecar identity is not pinned and ready")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.share:
        parser.error("direct --share is disabled; use the guarded launcher and tunnel")
    if args.public_tunnel_mode and not args.preserve_mode:
        parser.error("--public-tunnel-mode requires --preserve-mode")
    if args.preserve_mode:
        try:
            run_id = PreserveJournal.validate_run_id(args.run_id)
        except ValueError as exc:
            parser.error(str(exc))
        launcher_run_id = os.environ.get("IMP_LOOP206_PRESERVE_RUN_ID", "")
        if launcher_run_id != run_id:
            parser.error("preserve run ID does not match the guarded launcher")
    elif args.run_id:
        parser.error("--run-id requires --preserve-mode")
    _verified_launcher_session()
    evidence = json.loads(args.evidence_registry.read_text(encoding="ascii"))
    validate_registry(evidence)
    roots = _official_roots(args.dataset_index, args.dataset_root)
    model_environment = {
        "IMP_LOOP206_CONTROL_CHECKPOINT": os.environ.get(
            "IMP_LOOP206_CONTROL_CHECKPOINT", ""
        ).strip()
        or str(DEFAULT_CONTROL_CHECKPOINT),
        "IMP_LOOP206_CANDIDATE_CHECKPOINT": os.environ.get(
            "IMP_LOOP206_CANDIDATE_CHECKPOINT", ""
        ).strip()
        or str(DEFAULT_CANDIDATE_CHECKPOINT),
    }
    loaded = load_model_registry(
        args.model_registry, environ=model_environment, device=args.device
    )
    provider = loaded.build_fixed_provider(
        dataset_index=args.dataset_index,
        dataset_roots=roots,
        candidate_manifest=args.candidate_manifest,
        zero_manifest=args.zero_manifest,
        live_config=args.live_config,
    )
    imp_service = loaded.build_service(fixed_provider=provider)
    nnunet_client = NnUNetClient()
    _require_pinned_sidecar_health(nnunet_client.health())
    dual_service = DualLiveService(imp_service, nnunet_client)
    runtime_registry = deepcopy(evidence)
    runtime = _build_runtime_context(
        args.dataset_index, roots, args.candidate_manifest
    )
    runtime["sidecar_ready"] = True
    runtime_registry["_demo_runtime"] = runtime
    create_kwargs: dict[str, Any] = {
        "dual_service": dual_service,
        "public_tunnel_mode": args.public_tunnel_mode,
        "preserve_mode": args.preserve_mode,
    }
    if args.preserve_mode:
        create_kwargs["preserve_run_id"] = run_id
    demo = create_app(imp_service, runtime_registry, **create_kwargs)
    launch_kwargs: dict[str, Any] = {
        "server_name": args.host,
        "server_port": args.port,
        "share": False,
        "show_error": False,
        "max_threads": 1,
        "num_workers": 1,
        "css_paths": THEME_PATH,
    }
    if not args.public_tunnel_mode:
        launch_kwargs["max_file_size"] = MAX_UPLOAD_BYTES
    demo.launch(
        **launch_kwargs,
    )


if __name__ == "__main__":
    main()
