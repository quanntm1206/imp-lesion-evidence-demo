"""Evidence-scoped HTML and public receipt formatting for the demo."""

from __future__ import annotations

from html import escape
import math
from typing import Any, Mapping


NO_GT_MESSAGE = "Ground truth not supplied; accuracy metrics are unavailable."
FIXED_EVIDENCE_BADGE = (
    "train_screen",
    "exact_fixed_cache",
    "historical_cache_provenance_drift",
)
METRIC_KEYS = ("dice", "iou", "boundary_f1", "hd95", "assd")


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
            f'<p>{_limitations(screen)}</p>'
            "</article>"
        )
    return (
        '<section class="evidence-deck">'
        '<article class="evidence-card evidence-card--validation">'
        '<div class="evidence-card__index">01 / ARCHITECTURE POINT ESTIMATES</div>'
        '<div class="badge badge--verified">protected_validation</div>'
        '<h3>Clean-v3 validation record</h3>'
        '<p class="scope-note">Single-run point estimates. Protected test-v3 remained sealed.</p>'
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
