from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from xml.etree import ElementTree as ET
import zipfile

import pytest


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
MANIFEST = ROOT / "release" / "imp_release_manifest.json"
PUBLISH_TRANSACTION = ROOT / "scripts" / "presentation" / "publish_deck_transaction.ps1"
PRIOR_CURRENT_RELEASE = "a" * 64
CURRENT_RELEASE = "b" * 64
PRIOR_CONTENT = "c" * 64
CURRENT_CONTENT = "d" * 64

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


class PptxPackage:
    """Small OOXML reader for the final navigation contract."""

    def __init__(self, path: Path) -> None:
        self.package = zipfile.ZipFile(path)
        self.slide_count = len(
            [
                name
                for name in self.package.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ]
        )

    def __enter__(self) -> "PptxPackage":
        return self

    def __exit__(self, *_: object) -> None:
        self.package.close()

    def internal_jump(self, slide: str, shape_name: str) -> str:
        number = int(slide.removeprefix("slide"))
        slide_xml = ET.fromstring(self.package.read(f"ppt/slides/slide{number}.xml"))
        shape = next(
            item
            for item in slide_xml.iter(f"{{{P}}}cNvPr")
            if item.attrib.get("name") == shape_name
        )
        click = shape.find(f"{{{A}}}hlinkClick")
        assert click is not None
        assert click.attrib.get("action") == "ppaction://hlinksldjump"
        relationship_id = click.attrib[f"{{{R}}}id"]
        relationships = ET.fromstring(
            self.package.read(f"ppt/slides/_rels/slide{number}.xml.rels")
        )
        relationship = next(
            item
            for item in relationships.findall(f"{{{PACKAGE_REL}}}Relationship")
            if item.attrib.get("Id") == relationship_id
        )
        assert relationship.attrib.get("Type") == (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
        )
        assert relationship.attrib.get("TargetMode") is None
        return Path(relationship.attrib["Target"]).stem

    def has_fade_transition(self, number: int) -> bool:
        slide_xml = ET.fromstring(self.package.read(f"ppt/slides/slide{number}.xml"))
        transition = slide_xml.find(f"{{{P}}}transition")
        return (
            transition is not None
            and transition.attrib.get("spd") == "med"
            and transition.find(f"{{{P}}}fade") is not None
        )

    def has_external_actions(self) -> bool:
        for number in range(1, self.slide_count + 1):
            relationships = ET.fromstring(
                self.package.read(f"ppt/slides/_rels/slide{number}.xml.rels")
            )
            if any(
                relationship.attrib.get("TargetMode") == "External"
                for relationship in relationships.findall(f"{{{PACKAGE_REL}}}Relationship")
            ):
                return True
        return False

    def shape_names(self, number: int) -> set[str]:
        slide_xml = ET.fromstring(self.package.read(f"ppt/slides/slide{number}.xml"))
        return {
            item.attrib["name"]
            for item in slide_xml.iter(f"{{{P}}}cNvPr")
            if "name" in item.attrib
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _powershell_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _prior_delivery_receipt(html: Path, pdf: Path, pptx: Path) -> dict[str, object]:
    records = (
        (html, "current", "1" * 64, PRIOR_CONTENT),
        (pdf, "stale_rebuild_blocked", "2" * 64, "4" * 64),
        (pptx, "stale_unregenerated", "3" * 64, "5" * 64),
    )
    return {
        "schema": "imp.presentation.delivery/v2",
        "package_state": "incomplete_blocked",
        "current_release_manifest_sha256": PRIOR_CURRENT_RELEASE,
        "slide_count": 17,
        "content_sha256": PRIOR_CONTENT,
        "generated_utc": "2026-07-22T00:00:00Z",
        "files": [
            {
                "path": f"outputs/{path.name}",
                "status": status,
                "built_release_manifest_sha256": built_digest,
                "current_release_manifest_sha256": PRIOR_CURRENT_RELEASE,
                "content_sha256": content_digest,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path, status, built_digest, content_digest in records
        ],
    }


def _run_blocked_manifest_projection(
    tmp_path: Path, mutation: str | None = None
) -> dict[str, object]:
    html = tmp_path / "deck.html"
    pdf = tmp_path / "deck.pdf"
    pptx = tmp_path / "deck.pptx"
    receipt_path = tmp_path / "manifest.json"
    html.write_bytes(b"old html")
    pdf.write_bytes(b"old pdf")
    pptx.write_bytes(b"old pptx")
    receipt = _prior_delivery_receipt(html, pdf, pptx)
    if mutation == "schema":
        receipt["schema"] = "imp.presentation.delivery/wrong"
    elif mutation == "path":
        receipt["files"][0]["path"] = "outputs/wrong.html"
    elif mutation == "hash":
        receipt["files"][0]["sha256"] = "0" * 64
    elif mutation == "provenance":
        receipt["files"][0]["built_release_manifest_sha256"] = "invalid"
    receipt_path.write_text(json.dumps(receipt), encoding="ascii")
    command = f"""
$ErrorActionPreference = 'Stop'
. {_powershell_literal(PUBLISH_TRANSACTION)}
function Get-FileHash {{ throw 'Get-FileHash unavailable in subprocess' }}
$thrown = ''
$blocked = $null
try {{
    $blocked = New-BlockedDeckManifest `
        -PriorManifestPath {_powershell_literal(receipt_path)} `
        -HtmlPath {_powershell_literal(html)} `
        -PdfPath {_powershell_literal(pdf)} `
        -PptxPath {_powershell_literal(pptx)} `
        -CurrentReleaseManifestSha256 '{CURRENT_RELEASE}' `
        -CurrentContentSha256 '{CURRENT_CONTENT}' `
        -SlideCount 17
}}
catch {{ $thrown = $_.Exception.Message }}
[ordered]@{{ thrown = $thrown; manifest = $blocked }} |
    ConvertTo-Json -Depth 8 -Compress
"""
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    assert powershell is not None, "PowerShell is required for receipt tests"
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _run_transaction_fault(
    tmp_path: Path, *, rollback_restore_fails: bool = False
) -> tuple[dict[str, object], Path, Path]:
    stage = tmp_path / "stage"
    rollback = tmp_path / "rollback"
    stage.mkdir()
    html = tmp_path / "deck.html"
    pdf = tmp_path / "deck.pdf"
    pptx = tmp_path / "deck.pptx"
    manifest = tmp_path / "manifest.json"
    marker = tmp_path / "commit.json"
    staged_html = stage / "deck.html"
    staged_pdf = stage / "deck.pdf"
    staged_manifest = stage / "manifest.json"
    old_html = b"old html"
    old_pdf = b"old pdf"
    html.write_bytes(old_html)
    pdf.write_bytes(old_pdf)
    pptx.write_bytes(b"old pptx")
    manifest.write_text(
        json.dumps(_prior_delivery_receipt(html, pdf, pptx)), encoding="ascii"
    )
    staged_html.write_bytes(b"new html")
    staged_pdf.write_bytes(b"new pdf")
    staged_manifest.write_text('{"package_state":"complete"}\n', encoding="ascii")

    rollback_html = rollback / html.name
    rollback_pdf = rollback / pdf.name
    rollback_guard = (
        f"if ($Source -eq {_powershell_literal(rollback_pdf)}) {{ "
        "Move-Item -LiteralPath $Source -Destination $Destination -Force; "
        "throw 'injected rollback restore failure' }"
        if rollback_restore_fails
        else ""
    )
    command = f"""
$ErrorActionPreference = 'Stop'
. {_powershell_literal(PUBLISH_TRANSACTION)}
function Get-FileHash {{ throw 'Get-FileHash unavailable in subprocess' }}
$labels = @{{
    {_powershell_literal(html)} = 'live-html'
    {_powershell_literal(pdf)} = 'live-pdf'
    {_powershell_literal(staged_html)} = 'staged-html'
    {_powershell_literal(staged_pdf)} = 'staged-pdf'
    {_powershell_literal(staged_manifest)} = 'staged-manifest'
    {_powershell_literal(rollback_html)} = 'rollback-html'
    {_powershell_literal(rollback_pdf)} = 'rollback-pdf'
}}
$traceState = [ordered]@{{
    move_count = 0
    entries = New-Object 'System.Collections.Generic.List[string]'
}}
$move = {{
    param([string]$Source, [string]$Destination)
    $traceState.move_count++
    $traceState.entries.Add(
        "move:$($traceState.move_count):$($labels[$Source])->$($labels[$Destination])"
    )
    if ($Source -eq {_powershell_literal(staged_pdf)}) {{ throw 'injected publish failure' }}
    {rollback_guard}
    Move-Item -LiteralPath $Source -Destination $Destination -Force
}}
$remove = {{
    param([string]$Path, [bool]$Recurse)
    $traceState.entries.Add("remove:$($labels[$Path])")
    if ($Recurse) {{ Remove-Item -LiteralPath $Path -Recurse -Force }}
    else {{ Remove-Item -LiteralPath $Path -Force }}
}}
$blocked = New-BlockedDeckManifest `
    -PriorManifestPath {_powershell_literal(manifest)} `
    -HtmlPath {_powershell_literal(html)} `
    -PdfPath {_powershell_literal(pdf)} `
    -PptxPath {_powershell_literal(pptx)} `
    -CurrentReleaseManifestSha256 '{CURRENT_RELEASE}' `
    -CurrentContentSha256 '{CURRENT_CONTENT}' `
    -SlideCount 17
$thrown = ''
try {{
    Invoke-DeckPublishTransaction `
        -StagedHtmlPath {_powershell_literal(staged_html)} `
        -StagedPdfPath {_powershell_literal(staged_pdf)} `
        -StagedManifestPath {_powershell_literal(staged_manifest)} `
        -HtmlPath {_powershell_literal(html)} `
        -PdfPath {_powershell_literal(pdf)} `
        -ManifestPath {_powershell_literal(manifest)} `
        -CommitMarkerPath {_powershell_literal(marker)} `
        -RollbackDir {_powershell_literal(rollback)} `
        -BlockedManifest $blocked `
        -MoveOperation $move `
        -RemoveOperation $remove
}}
catch {{ $thrown = $_.Exception.Message }}
[ordered]@{{
    thrown = $thrown
    trace = @($traceState.entries)
    manifest = [IO.File]::ReadAllText({_powershell_literal(manifest)}) | ConvertFrom-Json
    marker = [IO.File]::ReadAllText({_powershell_literal(marker)}) | ConvertFrom-Json
}} | ConvertTo-Json -Depth 6 -Compress
"""
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    assert powershell is not None, "PowerShell is required for transaction fault tests"
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), html, pdf


def _assert_blocked_manifest_binds_live_artifacts(
    result: dict[str, object], html: Path, pdf: Path
) -> None:
    manifest = result["manifest"]
    entries = {Path(entry["path"]).suffix: entry for entry in manifest["files"]}
    assert entries[".html"]["bytes"] == html.stat().st_size
    assert entries[".html"]["sha256"] == _sha256(html)
    assert entries[".pdf"]["bytes"] == pdf.stat().st_size
    assert entries[".pdf"]["sha256"] == _sha256(pdf)
    assert entries[".html"]["built_release_manifest_sha256"] == "1" * 64
    assert entries[".pdf"]["built_release_manifest_sha256"] == "2" * 64
    assert entries[".html"]["content_sha256"] == PRIOR_CONTENT
    assert entries[".pdf"]["content_sha256"] == "4" * 64
    assert all(
        entry["current_release_manifest_sha256"] == CURRENT_RELEASE
        for entry in entries.values()
    )
    assert manifest["current_release_manifest_sha256"] == CURRENT_RELEASE
    assert manifest["content_sha256"] == CURRENT_CONTENT


def _assert_valid_delivery_v2(receipt: dict) -> None:
    current_digest = _sha256(MANIFEST)
    assert receipt["schema"] == "imp.presentation.delivery/v2"
    assert receipt["package_state"] in {"complete", "incomplete_blocked"}
    assert receipt["current_release_manifest_sha256"] == current_digest
    pptx_status = next(
        entry["status"] for entry in receipt["files"] if entry["path"].endswith(".pptx")
    )
    assert receipt["slide_count"] == 17
    assert receipt["content_sha256"] == _sha256(CONTENT)
    assert {entry["path"] for entry in receipt["files"]} == {
        "outputs/imp-lesion-evidence-defense.html",
        "outputs/imp-lesion-evidence-defense.pdf",
        "outputs/imp-lesion-evidence-defense.pptx",
    }
    for entry in receipt["files"]:
        output = ROOT / entry["path"]
        assert output.stat().st_size == entry["bytes"]
        assert _sha256(output) == entry["sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["built_release_manifest_sha256"])
        assert entry["current_release_manifest_sha256"] == current_digest
        if entry["status"] == "current":
            assert entry["content_sha256"] == _sha256(CONTENT)
        else:
            assert re.fullmatch(r"[0-9a-f]{64}", entry["content_sha256"])
        assert entry["status"] in {
            "current",
            "stale_unregenerated",
            "stale_rebuild_blocked",
        }
    if receipt["package_state"] == "complete":
        assert all(entry["status"] == "current" for entry in receipt["files"])
        assert all(
            entry["built_release_manifest_sha256"] == current_digest
            for entry in receipt["files"]
        )
    else:
        assert any(entry["status"] != "current" for entry in receipt["files"])


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


def test_demo_slide_names_both_demo_lanes_and_exact_live_models() -> None:
    content = json.loads(CONTENT.read_text(encoding="utf-8"))
    slide = next(slide for slide in content["slides"] if slide["id"] == "s10-demo")
    text = json.dumps(slide)
    assert "L206-control-s206" in text
    assert "L192-nnUNet-v2-raw-100ep" in text
    assert "not paper RQ1" in text
    assert "public/synthetic" in text
    assert "fixed" in text.lower()
    assert "eligible only after a successful dual run" in text
    assert "no canonical live receipt is included" in text
    assert "Clean-v3 training" in text
    assert "excluded from the L206 308-group fit" in text
    assert "76-group train-screen holdout" in text
    assert "generalization" in text
    assert "masks plus hash receipts" not in text


def test_demo_slide_keeps_authorized_row_boundary_and_live_ground_truth_semantics() -> None:
    content = json.loads(CONTENT.read_text(encoding="utf-8"))
    slide = next(slide for slide in content["slides"] if slide["id"] == "s10-demo")
    text = json.dumps(slide)
    assert "illustrative fixed-cache examples; not protected-test evidence" in text
    assert "ground_truth_not_loaded" in text


def test_core_presenter_route_skips_challenge_appendix_by_default() -> None:
    content = json.loads(CONTENT.read_text(encoding="utf-8"))
    expected = [
        "s01-title", "s02-leakage", "s03-questions", "s04-pipeline",
        "s05-data", "s06-models", "s07-validation", "s08-ablation-design",
        "s09-negative-result", "s10-demo", "s16-reproducibility", "s17-conclusion",
    ]
    assert content["meta"]["presenter_route"] == expected
    script = JS.read_text(encoding="utf-8")
    assert "presenterRoute" in script
    assert "stepPresenter" in script
    assert "s11-challenge-leakage" not in script.split("DEFAULT_PRESENTER_ROUTE", 1)[-1].split("]", 1)[0]
    html = INDEX.read_text(encoding="utf-8")
    assert 'data-route="core-defense"' in html


def test_repro_slide_bounds_hash_verification_and_pending_rq1_v2() -> None:
    content = json.loads(CONTENT.read_text(encoding="utf-8"))
    slide = next(
        slide for slide in content["slides"] if slide["id"] == "s16-reproducibility"
    )
    assert slide["title"] == "Artifacts are hash-bound; full experiments are not clone-runnable"
    text = json.dumps(slide).lower()
    assert "digest-recorded" in text
    assert "source bytes are verified only in the strict local audit" in text
    assert "hash-verifiable" not in text
    assert "rq1-v2" in text
    assert "pending/unverified until p1" in text


def test_asset_manifest_binds_source_and_output_bytes() -> None:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == "imp.presentation.assets/v1"
    assert len(manifest["assets"]) == 3
    assert {entry["name"] for entry in manifest["assets"]} == {
        "loop206-delta", "qualitative-demo", "qualitative-demo-middle"
    }
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


def test_slide_10_ooxml_contains_exact_live_model_ids() -> None:
    receipt = json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8"))
    pptx_state = next(
        entry["status"] for entry in receipt["files"] if entry["path"].endswith(".pptx")
    )
    if pptx_state != "current":
        assert receipt["package_state"] == "incomplete_blocked"
        pytest.skip(f"PPTX declared {pptx_state}")
    with zipfile.ZipFile(PPTX) as package:
        slide = package.read("ppt/slides/slide10.xml").decode("utf-8")

    assert "L206-control-s206" in slide
    assert "L192-nnUNet-v2-raw-100ep" in slide


def test_pptx_contains_pipeline_jumps_back_links_and_fade_transitions() -> None:
    receipt = json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8"))
    pptx_state = next(
        entry["status"] for entry in receipt["files"] if entry["path"].endswith(".pptx")
    )
    if pptx_state != "current":
        assert receipt["package_state"] == "incomplete_blocked"
        pytest.skip(f"PPTX declared {pptx_state}")
    with PptxPackage(PPTX) as package:
        assert [
            package.internal_jump("slide4", f"pipeline-node-{index}")
            for index in range(6)
        ] == ["slide5", "slide6", "slide6", "slide7", "slide8", "slide10"]
        for number in range(5, 11):
            assert package.internal_jump(f"slide{number}", "back-to-pipeline") == "slide4"
        for number in range(1, 5):
            assert "back-to-pipeline" not in package.shape_names(number)
        for number in range(11, 18):
            assert "back-to-pipeline" not in package.shape_names(number)
        assert all(package.has_fade_transition(number) for number in range(1, 18))
        assert not package.has_external_actions()


def test_external_relationship_scan_includes_slide17(tmp_path: Path) -> None:
    mutated = tmp_path / "slide17-external.pptx"
    relationship_path = "ppt/slides/_rels/slide17.xml.rels"
    with zipfile.ZipFile(PPTX) as source, zipfile.ZipFile(mutated, "w") as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == relationship_path:
                relationships = ET.fromstring(payload)
                relationship = relationships.findall(f"{{{PACKAGE_REL}}}Relationship")[0]
                relationship.set("TargetMode", "External")
                payload = ET.tostring(relationships, encoding="utf-8", xml_declaration=True)
            target.writestr(info, payload)

    with PptxPackage(mutated) as package:
        assert package.slide_count == 17
        assert package.has_external_actions()


def test_package_gate_binds_exact_navigation_relationship_targets() -> None:
    package = (ROOT / "scripts/presentation/package_deck.ps1").read_text(
        encoding="utf-8"
    )
    assert "Resolve-InternalSlideJumpTarget" in package
    assert "$expectedPipelineTargets = @(5, 6, 6, 7, 8, 10)" in package
    assert "Unexpected pipeline target" in package
    assert "Unexpected Back to Pipeline" in package
    assert "for ($number = 1; $number -le 4; $number++)" in package
    assert "for ($number = 11; $number -le $ExpectedSlideCount; $number++)" in package


def test_pptx_builder_renders_challenges_in_source_order() -> None:
    build = (ROOT / "scripts/presentation/build_pptx.mjs").read_text(encoding="utf-8")

    assert "function addChallengeSlide(data, index)" in build
    assert "`challenge-${key}-label`" in build
    assert "`challenge-${key}-text`" in build
    assert "[\"problem\", \"PROBLEM\"" in build
    assert "[\"response\", \"RESPONSE\"" in build
    assert "[\"limitation\", \"REMAINING LIMITATION\"" in build
    assert build.index("await addDemoSlide(content.slides[9], 10);") < build.index(
        "addChallengeSlide(content.slides[10], 11);"
    )
    assert build.index("addChallengeSlide(content.slides[14], 15);") < build.index(
        "addReproSlide(content.slides[15], 16);"
    ) < build.index("addConclusionSlide(content.slides[16], 17);")


def test_negative_slide_caption_clears_back_navigation() -> None:
    build = (ROOT / "scripts/presentation/build_pptx.mjs").read_text(encoding="utf-8")
    assert (
        'addText(slide, "negative-boundary", "Candidate rejected before protected '
        'evaluation.", 850, 572, 360, 24'
    ) in build


def test_demo_slide_uses_legible_native_labels_and_enlarged_crops() -> None:
    build = (ROOT / "scripts/presentation/build_pptx.mjs").read_text(encoding="utf-8")
    script = JS.read_text(encoding="utf-8")
    assert 'name: "demo-a-original"' not in build
    assert 'name: "demo-a-control"' not in build
    assert 'name: "demo-a-candidate"' not in build
    assert 'name: "demo-b-original"' not in build
    assert 'name: "demo-b-control"' not in build
    assert 'name: "demo-b-candidate"' not in build
    assert "ISIC_0000050" in build
    assert "ISIC_0012690" in build
    assert "Fixed-cache sample: original + control mask + candidate mask" in build
    assert "Ground truth is authorized only in the audited fixed lane" in build
    assert "qualitative-demo-middle.png" in build
    assert "rows y=390..660" in build
    assert 'image.src = "assets/qualitative-demo-middle.png"' in script
    assert "position: { left: 68, top: 312, width: 612, height: 294 }" not in build


def test_portable_html_is_self_contained() -> None:
    html = PORTABLE_HTML.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert '<script id="deck-content" type="application/json">' in html
    assert 'fetch("data:application/json;base64,' not in html
    assert 'data:image/png;base64,' in html
    assert not re.search(r'(?:src|href)="(?:presentation/|assets/)', html)


def test_portable_html_startup_avoids_file_fetch_and_has_visible_fallback() -> None:
    html = PORTABLE_HTML.read_text(encoding="utf-8")
    package = (ROOT / "scripts/presentation/package_deck.ps1").read_text(
        encoding="utf-8"
    )
    script = JS.read_text(encoding="utf-8")
    source_html = INDEX.read_text(encoding="utf-8")

    assert "deck-content" in package
    assert "application/json" in package
    assert "ConvertTo-SafeInlineJson" in package
    assert 'document.getElementById("deck-content")' in script
    assert "JSON.parse(embeddedContent.textContent)" in script
    assert "<noscript>" in html
    assert "JavaScript is required" in html
    assert "Portable file: verify that the transfer completed and reopen it." in source_html
    assert "Development source: serve this directory over HTTP." in source_html
    assert "Serve this directory over HTTP so the evidence contract can be read." not in source_html


def test_delivery_manifest_binds_outputs() -> None:
    receipt = json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8"))
    _assert_valid_delivery_v2(receipt)
    assert receipt["package_state"] == "complete"
    assert all(entry["status"] == "current" for entry in receipt["files"])


def test_delivery_manifest_accepts_complete_current_v2_branch() -> None:
    receipt = deepcopy(json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8")))
    current_digest = _sha256(MANIFEST)
    receipt["package_state"] = "complete"
    receipt["slide_count"] = 17
    for entry in receipt["files"]:
        entry["status"] = "current"
        entry["built_release_manifest_sha256"] = current_digest

    _assert_valid_delivery_v2(receipt)


def test_delivery_manifest_accepts_valid_prior_content_for_stale_entries() -> None:
    receipt = deepcopy(json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8")))
    receipt["package_state"] = "incomplete_blocked"
    stale = next(entry for entry in receipt["files"] if entry["path"].endswith(".html"))
    stale["status"] = "stale_rebuild_blocked"
    stale["content_sha256"] = "1" * 64

    _assert_valid_delivery_v2(receipt)

    stale["content_sha256"] = "A" * 64
    with pytest.raises(AssertionError):
        _assert_valid_delivery_v2(receipt)


def test_package_contract_is_17_slide_and_emits_content_hash() -> None:
    package = (ROOT / "scripts/presentation/package_deck.ps1").read_text(
        encoding="utf-8"
    )
    assert "expectedSlideCount = 17" in package
    assert "Expected a $expectedSlideCount-page PDF" in package
    assert "slide_count = $expectedSlideCount" in package
    assert "content_sha256" in package
    assert "for ($number = 1; $number -le 12; $number++)" not in package
    assert "Expected a 12-page PDF" not in package


def test_manifest_validator_rejects_one_byte_artifact_drift() -> None:
    receipt = deepcopy(json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8")))
    entry = next(item for item in receipt["files"] if item["path"].endswith(".html"))
    entry["bytes"] += 1
    with pytest.raises(AssertionError):
        _assert_valid_delivery_v2(receipt)


def test_delivery_archive_is_byte_identical_to_head_manifest() -> None:
    receipt = json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8"))
    if "prior_manifest_record" not in receipt:
        return
    archive = ROOT / receipt["prior_manifest_record"]["path"]

    assert receipt["prior_manifest_record"]["status"] == "byte_identical_archive"
    assert _sha256(archive) == receipt["prior_manifest_record"]["sha256"]
    assert receipt["prior_manifest_record"]["sha256"] == (
        "faafda7aa684417ecbe5e90a650474ea262e3fa7496d422a9057ab3f9aa15554"
    )


def test_presentation_build_and_package_fail_closed_on_release_projection_drift() -> None:
    build = (ROOT / "scripts/presentation/build_pptx.mjs").read_text(encoding="utf-8")
    package = (ROOT / "scripts/presentation/package_deck.ps1").read_text(encoding="utf-8")

    digest = _sha256(MANIFEST)
    content = json.loads(CONTENT.read_text(encoding="utf-8"))

    assert content["release_manifest_sha256"] == digest
    assert "release_manifest_sha256" in build
    assert "release manifest projection mismatch" in build
    assert "release_manifest_sha256" in package
    assert "release manifest projection mismatch" in package
    assert "imp.presentation.delivery/v2" in package
    assert "package_state = 'complete'" in package
    assert "built_release_manifest_sha256" in package
    assert "current_release_manifest_sha256" in package
    assert "Get-Command pdfinfo" not in package


def test_package_preflights_exact_ooxml_and_all_dependencies_before_mutation() -> None:
    package = (ROOT / "scripts/presentation/package_deck.ps1").read_text(
        encoding="utf-8"
    )
    first_mutation = package.index("New-Item -ItemType Directory")

    assert package.index("Assert-PresentationReleaseContract") < first_mutation
    assert package.index("Assert-PresentationSourcesUnchanged") < first_mutation
    assert package.index("$pdfInfoExe = Resolve-TrustedPdfInfo") < first_mutation
    assert package.index("New-Object -ComObject PowerPoint.Application") < first_mutation
    assert "ppt/slides/slide10.xml" in package
    assert "ppt/notesSlides/notesSlide10.xml" in package
    assert "L206-control-s206" in package
    assert "L192-nnUNet-v2-raw-100ep" in package
    assert "presentation source provenance mismatch" in package
    assert "$releaseDigest = Get-Sha256Hex -Bytes $releaseBytes" in package
    assert "PPTX Slide 10 identity mismatch" in package


def test_package_dot_sources_publish_transaction_helper() -> None:
    package = (ROOT / "scripts/presentation/package_deck.ps1").read_text(
        encoding="utf-8"
    )
    transaction = PUBLISH_TRANSACTION.read_text(encoding="utf-8")

    assert "$stageDir" in package
    assert ". (Join-Path $PSScriptRoot 'publish_deck_transaction.ps1')" in package
    assert "Invoke-DeckPublishTransaction" in package
    assert "function Invoke-DeckPublishTransaction" in transaction
    assert "[scriptblock]$MoveOperation" in transaction
    assert "[scriptblock]$RemoveOperation" in transaction


def test_package_builds_blocked_manifest_from_live_artifact_bytes() -> None:
    package = (ROOT / "scripts/presentation/package_deck.ps1").read_text(
        encoding="utf-8"
    )

    assert "$blockedManifest = New-BlockedDeckManifest" in package
    assert package.index("New-BlockedDeckManifest") < package.index(
        "Invoke-DeckPublishTransaction"
    )


def test_blocked_manifest_preserves_validated_prior_provenance(tmp_path: Path) -> None:
    result = _run_blocked_manifest_projection(tmp_path)

    assert result["thrown"] == ""
    manifest = result["manifest"]
    entries = {Path(entry["path"]).suffix: entry for entry in manifest["files"]}
    assert manifest["package_state"] == "incomplete_blocked"
    assert manifest["current_release_manifest_sha256"] == CURRENT_RELEASE
    assert manifest["content_sha256"] == CURRENT_CONTENT
    assert [entries[suffix]["built_release_manifest_sha256"] for suffix in (
        ".html", ".pdf", ".pptx"
    )] == ["1" * 64, "2" * 64, "3" * 64]
    assert [entries[suffix]["content_sha256"] for suffix in (
        ".html", ".pdf", ".pptx"
    )] == [PRIOR_CONTENT, "4" * 64, "5" * 64]
    assert all(
        entry["current_release_manifest_sha256"] == CURRENT_RELEASE
        and entry["status"] == "stale_rebuild_blocked"
        for entry in entries.values()
    )
    for suffix in (".html", ".pdf", ".pptx"):
        artifact = tmp_path / f"deck{suffix}"
        assert entries[suffix]["bytes"] == artifact.stat().st_size
        assert entries[suffix]["sha256"] == _sha256(artifact)


@pytest.mark.parametrize("mutation", ["schema", "path", "hash", "provenance"])
def test_blocked_manifest_rejects_untrusted_prior_receipt(
    tmp_path: Path, mutation: str
) -> None:
    result = _run_blocked_manifest_projection(tmp_path, mutation)

    assert result["manifest"] is None
    assert result["thrown"] == "prior delivery receipt invalid"


def test_publish_failure_restores_old_artifacts_and_blocks_manifest(tmp_path: Path) -> None:
    result, html, pdf = _run_transaction_fault(tmp_path)

    assert result["thrown"] == "deck publish failed: injected publish failure"
    assert result["trace"] == [
        "move:1:live-html->rollback-html",
        "move:2:live-pdf->rollback-pdf",
        "move:3:staged-html->live-html",
        "move:4:staged-pdf->live-pdf",
        "remove:live-html",
        "move:5:rollback-html->live-html",
        "move:6:rollback-pdf->live-pdf",
    ]
    assert html.read_bytes() == b"old html"
    assert pdf.read_bytes() == b"old pdf"
    _assert_blocked_manifest_binds_live_artifacts(result, html, pdf)
    assert result["manifest"]["package_state"] == "incomplete_blocked"
    assert result["marker"]["package_state"] == "incomplete_blocked"
    assert result["marker"]["rollback_status"] == "restored"
    assert result["marker"]["rollback_error_count"] == 0


def test_publish_and_restore_failures_keep_blocked_manifest_and_surface_error(
    tmp_path: Path,
) -> None:
    result, html, pdf = _run_transaction_fault(tmp_path, rollback_restore_fails=True)

    assert result["thrown"] == (
        "deck publish failed: injected publish failure; rollback: "
        "artifact restore failed: deck.pdf"
    )
    assert result["trace"] == [
        "move:1:live-html->rollback-html",
        "move:2:live-pdf->rollback-pdf",
        "move:3:staged-html->live-html",
        "move:4:staged-pdf->live-pdf",
        "remove:live-html",
        "move:5:rollback-html->live-html",
        "move:6:rollback-pdf->live-pdf",
    ]
    assert html.read_bytes() == b"old html"
    assert pdf.read_bytes() == b"old pdf"
    _assert_blocked_manifest_binds_live_artifacts(result, html, pdf)
    assert result["manifest"]["package_state"] == "incomplete_blocked"
    assert result["marker"]["package_state"] == "incomplete_blocked"
    assert result["marker"]["rollback_status"] == "failed"
    assert result["marker"]["rollback_error_count"] == 1
