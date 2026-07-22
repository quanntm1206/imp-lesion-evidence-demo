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
    assert len(content["slides"]) == 17
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
                "challenge-leakage",
                "challenge-fairness",
                "challenge-uncertainty",
                "challenge-demo",
                "challenge-repro",
                "reproducibility",
                "conclusion",
            ],
            start=1,
        )
    ]
    slide_ids = {slide["id"] for slide in content["slides"]}
    assert len(content["pipeline"]) == 6
    assert {node["target"] for node in content["pipeline"]} <= slide_ids


def test_challenge_slides_have_bounded_three_part_contract() -> None:
    content = json.loads(CONTENT.read_text(encoding="utf-8"))
    challenges = content["slides"][10:15]

    assert [slide["id"] for slide in challenges] == [
        "s11-challenge-leakage",
        "s12-challenge-fairness",
        "s13-challenge-uncertainty",
        "s14-challenge-demo",
        "s15-challenge-repro",
    ]
    topic_terms = {
        "s11-challenge-leakage": ("leakage", "split"),
        "s12-challenge-fairness": ("comparison", "geometry"),
        "s13-challenge-uncertainty": ("uncertainty", "adaptive"),
        "s14-challenge-demo": ("demo", "trust"),
        "s15-challenge-repro": ("reproducibility", "deployment"),
    }
    for slide in challenges:
        assert slide["visual"] == "challenge"
        assert set(slide["challenge"]) == {"problem", "response", "limitation"}
        assert all(slide["challenge"][field].strip() for field in slide["challenge"])
        text = json.dumps(slide).lower()
        assert all(term in text for term in topic_terms[slide["id"]])


def test_challenge_renderer_uses_dom_sections_without_pipeline_navigation() -> None:
    script = (ROOT / "presentation" / "interactive" / "deck.js").read_text(
        encoding="utf-8"
    )
    css = (ROOT / "presentation" / "interactive" / "deck.css").read_text(
        encoding="utf-8"
    )

    marker = "function createChallengeStage(slideData)"
    assert marker in script
    start = script.find(marker)
    end = script.find("function createReproducibilityStage(slideData)")
    challenge_stage = script[start:end] if start >= 0 and end >= 0 else ""
    assert '"challenge": createChallengeStage' in script
    assert "Problem" in challenge_stage
    assert "Response" in challenge_stage
    assert "Remaining limitation" in challenge_stage
    assert "slideData.challenge.problem" in challenge_stage
    assert "slideData.challenge.response" in challenge_stage
    assert "slideData.challenge.limitation" in challenge_stage
    assert "createBreadcrumb" not in challenge_stage
    assert "createBackButton" not in challenge_stage
    assert (
        '["s05-data", "s06-models", "s07-validation", "s08-ablation-design", '
        '"s09-negative-result", "s10-demo"].includes(slideData.id)'
    ) in script
    assert ".challenge-stage" in css
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in css
    assert re.search(
        r"@media \(max-width: 860px\)[\s\S]*?\.challenge-stage[\s\S]*?"
        r"grid-template-columns: 1fr;",
        css,
    )


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

def test_pptx_builder_renders_challenges_in_source_order() -> None:
    build = (ROOT / "scripts/presentation/build_pptx.mjs").read_text(encoding="utf-8")

    assert "function addChallengeSlide(data, index)" in build
    assert "`challenge-${key}-label`" in build
    assert "`challenge-${key}-text`" in build
    assert "[\"problem\", \"PROBLEM\"" in build
    assert "[\"response\", \"RESPONSE\"" in build
    assert "[\"limitation\", \"REMAINING LIMITATION\"" in build
    assert "function addBackToPipeline(slide)" in build
    assert "presentation.slides.items.slice(4, 10)" in build
    assert build.index("await addDemoSlide(content.slides[9], 10);") < build.index(
        "addChallengeSlide(content.slides[10], 11);"
    )
    assert build.index("addChallengeSlide(content.slides[14], 15);") < build.index(
        "addReproSlide(content.slides[15], 16);"
    ) < build.index("addConclusionSlide(content.slides[16], 17);")
