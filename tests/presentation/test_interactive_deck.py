from __future__ import annotations

import hashlib
import json
from pathlib import Path


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

