"""Evidence-scoped HTML and public receipt formatting for the demo."""

from __future__ import annotations

from html import escape
import hmac
import json
import math
import re
from typing import Any, Mapping

from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MODEL_ID,
    PROTOCOL_ID,
    is_public_metadata_text,
    mask_sha256,
    rgb_sha256,
)
from lesion_robustness.demo.dual_live_service import (
    IMP_COORDINATOR_SCOPE,
    IMP_REPORTED_SCOPE,
    NNUNET_COORDINATOR_SCOPE,
    NNUNET_REPORTED_SCOPE,
    TOTAL_SCOPE,
)
from lesion_robustness.demo.live_inputs import LiveInputEvidence, validate_live_input_evidence
from lesion_robustness.release_manifest import ReleaseManifest, load_release_manifest


NO_GT_MESSAGE = "Ground truth not supplied; accuracy metrics are unavailable."
FIXED_EVIDENCE_BADGE = (
    "train_screen",
    "exact_fixed_cache",
    "historical_cache_provenance_drift",
)
METRIC_KEYS = ("dice", "iou", "boundary_f1", "hd95", "assd")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")
_PUBLIC_TEXT = re.compile(r"[A-Za-z0-9 .:_-]+\Z")


def _number(value: Any, *, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not computable"
    if not math.isfinite(numeric):
        return "not computable"
    return f"{numeric:.{digits}f}"


def render_metrics(metrics: Mapping[str, Mapping[str, float]] | None) -> str:
    if metrics is None:
        return NO_GT_MESSAGE
    headings = ("Arm", "Dice", "IoU", "BF1", "HD95", "ASSD")
    lines = ["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
    for key, label in (("control", "Control"), ("candidate", "Candidate")):
        row = metrics.get(key, {})
        values = [_number(row.get(metric)) for metric in METRIC_KEYS]
        lines.append("| " + " | ".join((label, *values)) + " |")
    return "\n".join(lines)


def _limitations(row: Mapping[str, Any]) -> str:
    items = row.get("limitations", [])
    if not isinstance(items, list):
        return ""
    return "; ".join(escape(str(value)) for value in items)


def render_clean_evidence(registry: Mapping[str, Any]) -> str:
    observations = list(registry.get("observations", []))
    validation = [
        row for row in observations if row.get("evidence_class") == "protected_validation"
    ]
    screen = next(
        (row for row in observations if row.get("model_id") == "L206-contour-vs-control"),
        None,
    )
    rows = []
    for row in validation:
        rows.append(
            "<tr>"
            f"<td><strong>{escape(str(row.get('display_name', '')))}</strong>"
            f"<span>{escape(str(row.get('model_id', '')))}</span></td>"
            f"<td>{escape(str(row.get('partition', '')))}</td>"
            f"<td>{_number(row.get('robust_dice'))}</td>"
            f"<td>{_number(row.get('robust_bf1'))}</td>"
            f"<td>{int(row.get('seed_count', 0))}</td>"
            f"<td>{_limitations(row)}</td>"
            "</tr>"
        )
    screen_html = ""
    if screen is not None:
        dice_ci = screen.get("robust_dice_ci95", [None, None])
        bf1_ci = screen.get("robust_bf1_ci95", [None, None])
        seed_count = int(screen.get("seed_count", 0))
        seed_label = "three" if seed_count == 3 else str(seed_count)
        group_count = int(screen.get("group_count", 0))
        bootstrap_resamples = int(screen.get("bootstrap_resamples", 0))
        screen_html = (
            '<article class="evidence-card evidence-card--negative">'
            '<div class="evidence-card__index">02 / CONTROLLED NEGATIVE ABLATION</div>'
            '<div class="badge badge--train">train_screen</div>'
            '<h3>Contour channel minus zero-channel control</h3>'
            '<div class="ablation-grid">'
            f'<div><span>Robust Dice delta</span><strong>{_number(screen.get("robust_dice_delta"))}</strong>'
            f'<small>95% CI [{_number(dice_ci[0])}, {_number(dice_ci[1])}]</small></div>'
            f'<div><span>BF1 delta</span><strong>{_number(screen.get("robust_bf1_delta"))}</strong>'
            f'<small>95% CI [{_number(bf1_ci[0])}, {_number(bf1_ci[1])}]</small></div>'
            "</div>"
            '<p class="scope-note">'
            f"After averaging {seed_label} selected seeds and three views, "
            f"{bootstrap_resamples:,} bootstrap resamples draw {group_count} groups as "
            "whole split-group clusters. The interval is conditional on those selected "
            "seeds and does not estimate variability over seed selection.</p>"
            f'<p>{_limitations(screen)}</p>'
            "</article>"
        )
    return (
        '<section class="evidence-deck">'
        '<article class="evidence-card evidence-card--validation">'
        '<div class="evidence-card__index">01 / ARCHITECTURE POINT ESTIMATES</div>'
        '<div class="badge badge--verified">protected_validation</div>'
        '<h3>Clean-v3 validation record</h3>'
        '<p class="scope-note">Adaptive development and checkpoint-selection validation; '
        'single-run, selection-optimistic point estimates. Protected test-v3 remained sealed.</p>'
        '<div class="table-scroll"><table class="audit-table"><thead><tr>'
        '<th>Model</th><th>Partition</th><th>Dice</th><th>BF1</th><th>Seeds</th><th>Limitations</th>'
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
        + screen_html
        + "</section>"
    )


def render_legacy_table(registry: Mapping[str, Any]) -> str:
    observations = [
        row
        for row in registry.get("observations", [])
        if row.get("evidence_class") == "legacy_patient_contaminated"
    ]
    dataset = registry.get("dataset", {})
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.get('display_name', '')))}</td>"
        f"<td>{_number(row.get('clean_dice'))}</td>"
        f"<td>{_number(row.get('robust_dice'))}</td>"
        f"<td>{_number(row.get('robust_iou'))}</td>"
        f"<td>{_number(row.get('robust_bf1'))}</td>"
        "</tr>"
        for row in observations
    )
    return (
        '<section class="legacy-audit">'
        '<div class="contamination-banner" role="alert">'
        '<strong>legacy_patient_contaminated</strong>'
        f'<span>{int(dataset.get("clean_v2_cross_split_patient_ids", 0))} patient IDs / '
        f'{int(dataset.get("clean_v2_cross_split_rows", 0))} rows cross splits</span>'
        '<p>Historical operational evidence only. Excluded from Clean-v3 ranking and main conclusions.</p>'
        "</div>"
        '<div class="table-scroll"><table class="audit-table audit-table--legacy">'
        '<thead><tr><th>Historical model</th><th>Clean Dice</th><th>Robust Dice</th><th>IoU</th><th>BF1</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def _public_metrics(
    metrics: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]] | None:
    if metrics is None:
        return None
    return {
        arm: {metric: float(metrics[arm][metric]) for metric in METRIC_KEYS}
        for arm in ("control", "candidate")
    }

def build_fixed_receipt(
    result: Any,
    *,
    metrics: Mapping[str, Mapping[str, float]] | None,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = result.metadata
    ground_truth_binding = None
    if metrics is not None:
        ground_truth_binding = {
            "mask_sha256_raw": str(metadata["mask_sha256_raw"]),
            "mask_sha256_binary": str(metadata["mask_sha256_binary"]),
            "mask_sha256_runtime": str(metadata["mask_sha256_runtime"]),
        }
    return {
        "schema_version": "loop206.demo.receipt.v1",
        "mode": "exact_fixed_cache",
        "evidence_badge": list(FIXED_EVIDENCE_BADGE),
        "sample": {
            "group_key": str(metadata["group_key"]),
            "sample_id": str(metadata["sample_id"]),
            "corruption": str(metadata["corruption"]),
        },
        "models": {
            "control": {
                "model_id": str(result.control_model_id),
                "checkpoint_sha256": str(result.control_checkpoint_sha256),
            },
            "candidate": {
                "model_id": str(result.candidate_model_id),
                "checkpoint_sha256": str(result.candidate_checkpoint_sha256),
            },
        },
        "latency_ms": {
            "control": float(result.control_latency_ms),
            "candidate": float(result.candidate_latency_ms),
            "total": float(result.total_latency_ms),
        },
        "device": str(result.device),
        "cache_sha256": {
            "candidate_manifest": str(metadata["candidate_manifest_sha256"]),
            "candidate_data": str(metadata["candidate_data_sha256"]),
            "zero_manifest": str(metadata["zero_manifest_sha256"]),
            "zero_data": str(metadata["zero_data_sha256"]),
        },
        "historical_cache_provenance_drift": bool(
            metadata["historical_cache_provenance_drift"]
        ),
        "metrics": _public_metrics(metrics),
        "ground_truth_binding": ground_truth_binding,
        "evidence_registry_sha256": str(registry.get("registry_sha256", "")),
    }


def build_control_receipt(result: Any, *, registry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "loop206.demo.receipt.v1",
        "mode": "control_only",
        "evidence_class": "illustrative_control_preview",
        "model": {
            "model_id": str(result.control_model_id),
            "checkpoint_sha256": str(result.control_checkpoint_sha256),
        },
        "latency_ms": float(result.control_latency_ms),
        "device": str(result.device),
        "evidence_registry_sha256": str(registry.get("registry_sha256", "")),
    }


def _receipt_attr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError:
        raise ValueError(f"complete live result is missing {name}") from None


def _public_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"complete live result has invalid {name}")
    if not is_public_metadata_text(value):
        raise ValueError(f"complete live result has unsafe {name}")
    if _PUBLIC_TEXT.fullmatch(value) is None:
        raise ValueError(f"complete live result has invalid {name}")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"complete live result has invalid {name}")
    return value


def _latency(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    latency = float(value)
    if not math.isfinite(latency) or latency < 0.0:
        raise ValueError(f"{name} must be finite")
    return latency


def _scope(value: Any, name: str, expected: str) -> str:
    if value != expected:
        raise ValueError(f"complete live result has invalid {name}")
    return expected


def _dual_live_model(
    arm: Any,
    *,
    name: str,
    reported_scope: str,
    coordinator_scope: str,
    expected_model: str | None = None,
    expected_checkpoint: str | None = None,
) -> dict[str, Any]:
    if arm is None or _receipt_attr(arm, "status") != "completed":
        raise ValueError("receipt requires a complete live result")
    model_id = _public_text(_receipt_attr(arm, "model_id"), f"{name}.model_id")
    checkpoint = _sha256(
        _receipt_attr(arm, "checkpoint_sha256"), f"{name}.checkpoint_sha256"
    )
    if expected_model is not None and model_id != expected_model:
        raise ValueError(f"complete live result has invalid {name}.model_id")
    if expected_checkpoint is not None and not hmac.compare_digest(
        checkpoint, expected_checkpoint
    ):
        raise ValueError(f"complete live result has invalid {name}.checkpoint_sha256")
    return {
        "model_id": model_id,
        "checkpoint_sha256": checkpoint,
        "preprocessing": _public_text(
            _receipt_attr(arm, "preprocessing"), f"{name}.preprocessing"
        ),
        "segmentation_sha256": mask_sha256(_receipt_attr(arm, "mask")),
        "latency": {
            "reported_ms": _latency(
                _receipt_attr(arm, "reported_latency_ms"),
                f"{name}.reported_latency_ms",
            ),
            "reported_scope": _scope(
                _receipt_attr(arm, "reported_latency_scope"),
                f"{name}.reported_latency_scope",
                reported_scope,
            ),
            "coordinator_ms": _latency(
                _receipt_attr(arm, "coordinator_latency_ms"),
                f"{name}.coordinator_latency_ms",
            ),
            "coordinator_scope": _scope(
                _receipt_attr(arm, "coordinator_latency_scope"),
                f"{name}.coordinator_latency_scope",
                coordinator_scope,
            ),
        },
        "device": _public_text(_receipt_attr(arm, "device"), f"{name}.device"),
        "status": "completed",
    }


def _dual_live_payload(result: Any, release_manifest: ReleaseManifest) -> dict[str, Any]:
    if _receipt_attr(result, "receipt_eligible") is not True:
        raise ValueError("receipt requires a complete live result")
    request_id = _receipt_attr(result, "request_id")
    if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
        raise ValueError("complete live result has invalid request_id")
    rgb = _receipt_attr(result, "original_rgb")
    input_digest = _sha256(_receipt_attr(result, "input_sha256"), "input_sha256")
    if not hmac.compare_digest(input_digest, rgb_sha256(rgb)):
        raise ValueError("complete live result input binding mismatch")
    imp = _dual_live_model(
        _receipt_attr(result, "imp"),
        name="imp",
        reported_scope=IMP_REPORTED_SCOPE,
        coordinator_scope=IMP_COORDINATOR_SCOPE,
        expected_model="L206-control-s206",
        expected_checkpoint=release_manifest.model("L206-control-s206").checkpoint_sha256,
    )
    nnunet = _dual_live_model(
        _receipt_attr(result, "nnunet"),
        name="nnunet",
        reported_scope=NNUNET_REPORTED_SCOPE,
        coordinator_scope=NNUNET_COORDINATOR_SCOPE,
        expected_model=MODEL_ID,
        expected_checkpoint=CHECKPOINT_SHA256,
    )
    protocol = _receipt_attr(_receipt_attr(result, "nnunet"), "protocol")
    if protocol != PROTOCOL_ID:
        raise ValueError("complete live result has invalid nnunet.protocol")
    total_latency = _latency(
        _receipt_attr(result, "total_latency_ms"), "total_latency_ms"
    )
    total_scope = _scope(
        _receipt_attr(result, "total_latency_scope"), "total_latency_scope", TOTAL_SCOPE
    )
    if (
        total_latency < imp["latency"]["coordinator_ms"]
        or total_latency < nnunet["latency"]["coordinator_ms"]
    ):
        raise ValueError("complete live result total_latency is below coordinator latency")
    height, width, channels = rgb.shape
    return {
        "schema_version": "imp.dual_live.receipt.v2",
        "request_id": request_id,
        "execution": "live_sequential",
        "protocol_id": PROTOCOL_ID,
        "input": {
            "rgb_sha256": input_digest,
            "dimensions": {
                "height": int(height),
                "width": int(width),
                "channels": int(channels),
            },
        },
        "models": {
            "imp": imp,
            "nnunet": {
                **nnunet,
                "runtime_status": "reconstructed_not_original_equivalent",
            },
        },
        "total_latency": {
            "coordinator_ms": total_latency,
            "coordinator_scope": total_scope,
        },
        "loop192_validation_status": "val_gate_failed_no_test",
        "claim_policy": {"clinical_use": False},
    }


def _validated_input_evidence_hash(result: Any, input_evidence: Any) -> str:
    if not isinstance(input_evidence, LiveInputEvidence):
        raise ValueError("input evidence is required")
    try:
        validate_live_input_evidence(input_evidence)
    except ValueError as exc:
        raise ValueError("input evidence is invalid") from exc
    evidence_hash = _sha256(
        input_evidence.rgb_sha256, "input evidence rgb_sha256"
    )
    result_hash = _sha256(_receipt_attr(result, "input_sha256"), "input_sha256")
    if not hmac.compare_digest(result_hash, rgb_sha256(_receipt_attr(result, "original_rgb"))):
        raise ValueError("complete live result input binding mismatch")
    if not hmac.compare_digest(evidence_hash, result_hash):
        raise ValueError("input evidence binding mismatch")
    return evidence_hash


def build_dual_live_receipt(
    result: Any, release_manifest: ReleaseManifest, input_evidence: Any
) -> dict[str, Any]:
    """Build the allowlisted, current-request-only dual-live receipt."""
    if not isinstance(release_manifest, ReleaseManifest):
        raise ValueError("release manifest is invalid")
    input_hash = _validated_input_evidence_hash(result, input_evidence)
    receipt = _dual_live_payload(result, release_manifest)
    if not hmac.compare_digest(input_hash, receipt["input"]["rgb_sha256"]):
        raise ValueError("input evidence binding mismatch")
    receipt["evidence_class"] = _public_text(getattr(input_evidence, "evidence_class", None), "input evidence class")
    evidence_kind = _public_text(getattr(input_evidence, "kind", None), "input evidence kind")
    input_projection: dict[str, Any] = {
        **receipt["input"],
        "evidence_kind": evidence_kind,
    }
    if evidence_kind == "public_sample":
        input_projection.update(
            {
                "sample_id": _public_text(input_evidence.sample_id, "input evidence sample_id"),
                "source_dataset": _public_text(input_evidence.source_dataset, "input evidence source_dataset"),
                "source_page": input_evidence.source_page,
                "license": _public_text(input_evidence.image_license, "input evidence license"),
                "training_exposure": dict(input_evidence.training_exposure),
                "ground_truth_used": False,
                "ground_truth_not_loaded": True,
            }
        )
    receipt["input"] = input_projection
    receipt["release_manifest_sha256"] = _sha256(
        release_manifest.digest, "release_manifest_sha256"
    )
    # Validate canonical JSON now; callers may persist only this safe payload.
    json_payload = json.dumps(
        receipt, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    if not json_payload:
        raise ValueError("complete live result receipt is empty")
    return receipt


def render_dual_live_ledger(result: Any) -> str:
    """Render public, no-ground-truth facts for one completed live request."""
    receipt = _dual_live_payload(result, load_release_manifest())
    models = receipt["models"]
    return "\n".join(
        (
            "Live sequential result",
            "Ground truth not supplied; accuracy metrics are unavailable.",
            f"IMP: {models['imp']['status']} | {models['imp']['latency']['coordinator_ms']:.1f} ms coordinator wall | {models['imp']['device']}",
            f"nnU-Net: {models['nnunet']['status']} | {models['nnunet']['latency']['coordinator_ms']:.1f} ms coordinator wall | {models['nnunet']['device']}",
            f"Total: {receipt['total_latency']['coordinator_ms']:.1f} ms coordinator wall",
            "Reported model timings use distinct internal scopes; not a controlled efficiency benchmark.",
            "nnU-Net uses a reconstructed runtime; original-runtime equivalence is not established.",
            "Clinical use: false.",
        )
    )
