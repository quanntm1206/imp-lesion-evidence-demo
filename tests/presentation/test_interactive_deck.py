from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "presentation" / "interactive"
CONTENT = DECK / "content.json"
ASSET_MANIFEST = DECK / "assets" / "asset-manifest.json"
INDEX = DECK / "index.html"
CSS = DECK / "deck.css"
JS = DECK / "deck.js"
PPTX = ROOT / "outputs" / "imp-lesion-evidence-defense.pptx"
PDF = ROOT / "outputs" / "imp-lesion-evidence-defense.pdf"
PORTABLE_HTML = ROOT / "outputs" / "imp-lesion-evidence-defense.html"
DELIVERY_MANIFEST = ROOT / "outputs" / "imp-lesion-evidence-defense-manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_content_contract_has_exact_slide_and_pipeline_targets() -> None:
    content = json.loads(CONTENT.read_text(encoding="utf-8"))
    assert len(content["slides"]) == 12
    assert [slide["id"] for slide in content["slides"]] == [
        f"s{number:02d}-{slug}"
        for number, slug in enumerate(
            [
                "title",
                "leakage",
                "questions",
                "pipeline",
                "data",
                "models",
                "validation",
                "ablation-design",
                "negative-result",
                "demo",
                "reproducibility",
                "conclusion",
            ],
            start=1,
        )
    ]
    slide_ids = {slide["id"] for slide in content["slides"]}
    assert len(content["pipeline"]) == 6
    assert {node["target"] for node in content["pipeline"]} <= slide_ids


def test_scientific_claims_are_bounded() -> None:
    content = json.loads(CONTENT.read_text(encoding="utf-8"))
    text = json.dumps(content, ensure_ascii=True).lower()
    for value in [
        "2,869",
        "2,008",
        "0.8959",
        "0.9019",
        "0.4145",
        "0.4369",
        "-0.0313",
        "-0.0491",
        "-0.0156",
        "-0.0147",
        "-0.0308",
        "0.0010",
    ]:
        assert value in text
    assert "protected test remains sealed" in text
    assert "not state of the art" in text
    assert "adaptive development-validation" in text
    assert "conditional on the selected seeds" in text
    assert "non-clinical" in text


def test_each_slide_has_claim_evidence_label_and_notes() -> None:
    content = json.loads(CONTENT.read_text(encoding="utf-8"))
    for slide in content["slides"]:
        assert slide["title"].strip()
        assert slide["claim"].strip()
        assert slide["evidence_label"].strip()
        assert len(slide["notes"]) >= 2


def test_asset_manifest_binds_source_and_output_bytes() -> None:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == "imp.presentation.assets/v1"
    assert len(manifest["assets"]) == 2
    for entry in manifest["assets"]:
        source = ROOT / entry["source"]
        output = ROOT / entry["output"]
        assert not Path(entry["source"]).is_absolute()
        assert not Path(entry["output"]).is_absolute()
        assert _sha256(source) == entry["source_sha256"]
        assert _sha256(output) == entry["output_sha256"]
        assert output.suffix == ".png"
        assert output.stat().st_size > 20_000


def test_html_exposes_accessible_navigation_and_reduced_motion() -> None:
    html = INDEX.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    assert 'aria-live="polite"' in html
    assert 'aria-label="Presentation controls"' in html
    assert "prefers-reduced-motion" in css
    assert "[hidden]" in css
    assert "hashchange" in js
    assert 'event.key === "Escape"' in js
    assert "Back to Pipeline" in js
    assert "default_demo_url" in js


def test_html_has_no_network_runtime_dependencies() -> None:
    html = INDEX.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    combined = html + css + js
    assert "https://cdn" not in combined
    assert "fonts.googleapis.com" not in combined
    assert "unpkg.com" not in combined
    assert "node_modules" not in combined


def test_final_pptx_exists_and_is_nontrivial() -> None:
    assert PPTX.exists()
    assert PPTX.stat().st_size > 100_000


def test_portable_html_is_self_contained() -> None:
    html = PORTABLE_HTML.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert 'data:application/json;base64,' in html
    assert 'data:image/png;base64,' in html
    assert not re.search(r'(?:src|href)="(?:presentation/|assets/)', html)


def test_delivery_manifest_binds_outputs() -> None:
    receipt = json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8"))
    assert receipt["schema"] == "imp.presentation.delivery/v1"
    assert receipt["slide_count"] == 12
    assert {entry["path"] for entry in receipt["files"]} == {
        "outputs/imp-lesion-evidence-defense.html",
        "outputs/imp-lesion-evidence-defense.pdf",
        "outputs/imp-lesion-evidence-defense.pptx",
    }
    for entry in receipt["files"]:
        output = ROOT / entry["path"]
        assert output.stat().st_size == entry["bytes"]
        assert _sha256(output) == entry["sha256"]
