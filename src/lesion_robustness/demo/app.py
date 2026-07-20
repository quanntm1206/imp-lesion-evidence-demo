"""Gradio entry point for the evidence-first degraded Loop206 workbench."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import gradio as gr
import numpy as np

from lesion_robustness.demo.metrics_service import evaluate_optional_ground_truth
from lesion_robustness.demo.model_service import (
    PINNED_REGISTRY,
    Loop206ComparisonService,
    load_model_registry,
)
from lesion_robustness.demo.presentation import (
    NO_GT_MESSAGE,
    build_control_receipt,
    build_fixed_receipt,
    render_clean_evidence,
    render_legacy_table,
    render_metrics,
)
from lesion_robustness.evidence_registry import validate_registry


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


def _receipt_file(receipt: Mapping[str, Any] | None) -> str | None:
    if receipt is None:
        return None
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="loop206-public-receipt-", delete=False,
        encoding="ascii",
    )
    with handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
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


def _runtime(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    value = registry.get("_demo_runtime", {})
    return value if isinstance(value, Mapping) else {}


def create_app(
    service: Loop206ComparisonService, registry: Mapping[str, Any]
) -> gr.Blocks:
    runtime = _runtime(registry)
    choices = list(runtime.get("fixed_choices", []))
    corruptions = list(runtime.get("corruptions", ["clean"]))
    ground_truths = runtime.get("fixed_ground_truth", {})
    default_choice = choices[0][1] if choices else None

    # Gradio 6 defers CSS to launch; the compatibility argument keeps imported apps styled.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The parameters have been moved")
        demo = gr.Blocks(
            title="Audited Dermoscopy Workbench",
            fill_width=True,
            delete_cache=(3600, 3600),
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
            f'<div class="status-band" role="status"><strong>Degraded runtime</strong><span>{HEADER_STATUS}</span></div>'
        )
        gr.HTML(
            '<div class="hash-strip"><div><span>Control checkpoint</span>'
            f'<code>{PINNED_REGISTRY["control"]["checkpoint_sha256"]}</code></div>'
            '<div><span>Candidate checkpoint</span>'
            f'<code>{PINNED_REGISTRY["candidate"]["checkpoint_sha256"]}</code></div></div>'
        )
        with gr.Tabs(selected="live"):
            with gr.Tab("Live Workbench", id="live"):
                gr.HTML(
                    '<div class="section-heading"><span>01</span><div><p>Primary audited mode</p>'
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
                    original = gr.Image(label="Verified source", interactive=False, buttons=["fullscreen"])
                    with gr.Column(elem_classes="arm-panel arm-panel--control"):
                        gr.HTML('<div class="arm-label"><span>A</span><strong>Zero-channel control</strong></div>')
                        control_overlay = gr.Image(label="Control overlay", interactive=False, buttons=["fullscreen"])
                        control_mask = gr.Image(label="Control mask", interactive=False, image_mode="L", buttons=["fullscreen"])
                    with gr.Column(elem_classes="arm-panel arm-panel--candidate"):
                        gr.HTML('<div class="arm-label"><span>B</span><strong>Contour-channel candidate</strong></div>')
                        candidate_overlay = gr.Image(label="Candidate overlay", interactive=False, buttons=["fullscreen"])
                        candidate_mask = gr.Image(label="Candidate mask", interactive=False, image_mode="L", buttons=["fullscreen"])
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

                gr.HTML(
                    '<div class="section-heading section-heading--secondary"><span>02</span><div>'
                    '<p>Secondary illustrative mode</p><h2>Arbitrary Image &mdash; Control Preview</h2></div></div>'
                )
                with gr.Row(elem_classes="control-preview"):
                    upload = gr.Image(
                        label="Upload dermoscopic image",
                        type="numpy",
                        sources=["upload"],
                        image_mode="RGB",
                    )
                    with gr.Column():
                        gr.HTML(CONTROL_LOCKED_HTML)
                        run_control = gr.Button("Run control preview", variant="secondary")

                run_control.click(
                    lambda image: _component_values(run_control_preview(service, registry, image)),
                    inputs=[upload],
                    outputs=outputs,
                    concurrency_limit=1,
                    concurrency_id="loop206-inference",
                    api_name="control_preview",
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
    from lesion_robustness.image_utils import read_mask, resize_image_and_mask

    _, holdout, payload = load_dataset_index(dataset_index, dataset_roots=roots)
    choices = [
        (f"{row.sample_id} / {row.group_key}", row.group_key)
        for row in sorted(holdout, key=lambda value: (value.sample_id, value.group_key))
    ]
    rows = {
        str(row["group_key"]): row
        for row in payload["rows"]
        if row.get("role") == "holdout"
    }
    masks: dict[str, np.ndarray] = {}
    for identifier, row in rows.items():
        mask_path = _safe_index_path(roots[int(row["mask_root"])], row["mask_relative"])
        mask = read_mask(mask_path)
        _, resized = resize_image_and_mask(np.zeros((*mask.shape, 3), dtype=np.uint8), mask, (384, 384))
        assert resized is not None
        masks[identifier] = np.asarray(resized, dtype=np.uint8)
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
    return {
        "fixed_choices": choices,
        "corruptions": corruptions,
        "fixed_ground_truth": masks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the audited Loop206 workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--model-registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--evidence-registry", type=Path, default=DEFAULT_EVIDENCE_REGISTRY)
    parser.add_argument("--dataset-index", type=Path, default=DEFAULT_DATASET_INDEX)
    parser.add_argument("--dataset-root", action="append", default=[])
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--zero-manifest", type=Path, default=DEFAULT_ZERO_MANIFEST)
    parser.add_argument("--live-config", type=Path, default=DEFAULT_LIVE_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    evidence = json.loads(args.evidence_registry.read_text(encoding="ascii"))
    validate_registry(evidence)
    roots = _official_roots(args.dataset_index, args.dataset_root)
    model_environment = {
        "IMP_LOOP206_CONTROL_CHECKPOINT": str(DEFAULT_CONTROL_CHECKPOINT),
        "IMP_LOOP206_CANDIDATE_CHECKPOINT": str(DEFAULT_CANDIDATE_CHECKPOINT),
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
    runtime_registry = deepcopy(evidence)
    runtime_registry["_demo_runtime"] = _build_runtime_context(
        args.dataset_index, roots, args.candidate_manifest
    )
    demo = create_app(loaded.build_service(fixed_provider=provider), runtime_registry)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=False,
        max_threads=1,
        num_workers=1,
        css_paths=THEME_PATH,
    )


if __name__ == "__main__":
    main()
