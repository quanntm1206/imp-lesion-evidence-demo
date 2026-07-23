from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from scripts.paper.audit_clean_v3_paper import audit_paper, main
import scripts.paper.audit_clean_v3_paper as audit_module
import scripts.paper.build_clean_v3_tables as table_builder


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "demo/data/evidence_registry.json"
PAPER = ROOT / "paper/clean_v3_loop206"
FIGURE_SOURCE = PAPER / "figures/evidence_pipeline.drawio"


def read_paper_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(PAPER.rglob("*.tex"))
    )


def _registry_hash(payload: dict) -> str:
    unsigned = deepcopy(payload)
    unsigned.pop("registry_sha256", None)
    unsigned.pop("release_manifest_sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256((encoded + "\n").encode("ascii")).hexdigest()


def _write_registry(path: Path, payload: dict) -> Path:
    payload["registry_sha256"] = _registry_hash(payload)
    path.write_text(json.dumps(payload), encoding="ascii")
    return path


def make_minimal_paper(tmp_path: Path, body: str) -> Path:
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        f"{body}\n"
        "\\end{document}\n",
        encoding="ascii",
    )
    (paper / "references.bib").write_text("", encoding="ascii")
    return paper


def _copy_paper(tmp_path: Path) -> Path:
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper)
    return paper


def _copy_portable_project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    paper = root / "paper/clean_v3_loop206"
    registry = root / "demo/data/evidence_registry.json"
    generator = root / "scripts/paper/generate_evidence_pipeline.py"
    shutil.copytree(PAPER, paper)
    registry.parent.mkdir(parents=True)
    shutil.copy2(REGISTRY, registry)
    generator.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/paper/generate_evidence_pipeline.py", generator)
    return paper, registry


@pytest.mark.parametrize(
    "forbidden",
    [
        "state-of-the-art",
        "statistically superior",
        "clinical-grade",
        "diagnostic accuracy",
    ],
)
def test_audit_rejects_affirmative_forbidden_claims(
    tmp_path: Path, forbidden: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, forbidden), REGISTRY)

    assert not result.passed
    assert any("affirmative protected claim" in error for error in result.errors)


def test_current_paper_audit_accepts_current_pdf() -> None:
    result = audit_paper(PAPER, REGISTRY, source_verification="registry-only")

    assert result.passed
    assert not result.errors
    assert not result.blockers
    assert result.source_verification == "registry-only"


def test_paper_separates_historical_rq1_from_live_demo_models() -> None:
    paper = read_paper_text()
    assert "L191-C0-clean-v3-IMP-control versus L192-nnUNet-v2-raw-100ep" in paper
    assert "L206-control-s206 versus a reconstructed Loop192 runtime" in paper
    assert "does not reproduce the Loop191-versus-Loop192 RQ1 comparison" in paper


def test_paper_keeps_historical_audit_separate_from_prospective_admission() -> None:
    data_protocol = (PAPER / "sections/03_data_protocol.tex").read_text(
        encoding="utf-8"
    )
    experiments = (PAPER / "sections/05_experiments.tex").read_text(
        encoding="utf-8"
    )
    results = (PAPER / "sections/06_results.tex").read_text(encoding="utf-8")
    reproducibility = (PAPER / "sections/09_reproducibility.tex").read_text(
        encoding="utf-8"
    )

    assert "recorded source-report audit" in data_protocol
    for section in (experiments, reproducibility):
        assert r"Prospective RQ1-v2 index/integrity admission remains \texttt{blocked}" in section
    assert r"\input{tables/loop206_gate_audit}" in results
    assert "Every interval is conditional on the selected seeds" in results


def test_results_report_only_the_global_loop206_gate_decision() -> None:
    results = (PAPER / "sections/06_results.tex").read_text(encoding="utf-8")

    assert r"\texttt{gate\_passed=false}" in results
    assert "classifies the candidate as failing the primary-improvement" not in results


def test_manuscript_classifies_bounded_live_and_blocked_p1_evidence() -> None:
    results = (PAPER / "sections/06_results.tex").read_text(encoding="utf-8")
    limitations = (PAPER / "sections/08_limitations_ethics.tex").read_text(
        encoding="utf-8"
    )
    reproducibility = (PAPER / "sections/09_reproducibility.tex").read_text(
        encoding="utf-8"
    )

    assert "Observed:" in results
    assert "Established:" in results
    assert "Assumption:" in results
    assert "Speculation:" in results
    assert "Current-release browser, mobile/desktop visual, and tunnel status is" in results
    assert r"\texttt{unverified/blocked}" in results
    assert "83-mask-pixel" in results
    assert "one bundled public sample exceeds the pinned 16 MiB request contract" in limitations
    assert r"P1 remains \texttt{BLOCKED}" in reproducibility
    assert "Clean-v3 index" in reproducibility
    assert "six locked configs/runtime manifests/job receipts" in reproducibility


def test_figure_source_names_all_three_comparison_lanes() -> None:
    xml = FIGURE_SOURCE.read_text(encoding="utf-8")
    for label in ("Paper RQ1", "Fixed-cache demo", "Live dual demo"):
        assert label in xml


@pytest.mark.parametrize(
    "claim",
    [
        "Live dual demo reports accuracy.",
        "Live dual demo reports a metric.",
        "Live dual demo uses ground truth.",
        "Live dual demo is equivalent to Paper RQ1.",
        "Live dual demo reproduces Paper RQ1.",
    ],
)
def test_audit_rejects_unbounded_live_demo_claim(
    tmp_path: Path, claim: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert "unbounded live demo claim" in result.errors


@pytest.mark.parametrize(
    "claim",
    [
        "Live dual demo improves Dice.",
        "Live comparison reports accuracy.",
        "Live dual demo does not reproduce Paper RQ1 because accuracy is 0.99.",
    ],
)
def test_audit_rejects_adversarial_live_claims(
    tmp_path: Path, claim: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert "unbounded live demo claim" in result.errors


@pytest.mark.parametrize(
    "claim",
    [
        "The Live dual demo is operational. Ground truth is loaded.",
        "The Live dual demo is operational. Accuracy is 0.99.",
        "The Live dual demo is operational. It reproduces Paper RQ1.",
        "The Live dual demo is operational. It reports Dice.",
        "The Live dual demo is operational. This comparison reports accuracy.",
        "The Live dual demo is operational. Robust Dice improves.",
    ],
)
def test_audit_retains_live_scope_across_sentences(
    tmp_path: Path, claim: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert "unbounded live demo claim" in result.errors


def test_audit_allows_negated_live_anaphora(tmp_path: Path) -> None:
    result = audit_paper(
        make_minimal_paper(
            tmp_path,
            "The Live dual demo is operational. Ground truth is not loaded. "
            "It reports no accuracy or metric. Neither original-runtime equivalence "
            "nor a replay is claimed.",
        ),
        REGISTRY,
    )

    assert "unbounded live demo claim" not in result.errors


@pytest.mark.parametrize(
    "claim",
    [
        "The Live dual demo reports no accuracy, metric, or ground truth.",
        "The Live dual demo does not reproduce Paper RQ1.",
        "The Live dual demo documents non-equivalence to Paper RQ1.",
    ],
)
def test_audit_allows_local_live_negation(tmp_path: Path, claim: str) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert "unbounded live demo claim" not in result.errors


def test_audit_does_not_scope_paper_rq1_metric_to_prior_live_sentence(
    tmp_path: Path,
) -> None:
    paper = make_minimal_paper(
        tmp_path,
        "The Live dual demo reports no metric. "
        "Paper RQ1 uses Loop191 IMP-SegFormer-B3 protected-validation robust Dice "
        "of "
        "0.895870479294128.",
    )

    result = audit_paper(paper, REGISTRY)

    assert "unbounded live demo claim" not in result.errors


def test_audit_resets_live_scope_for_fixed_cache_demo(tmp_path: Path) -> None:
    result = audit_paper(
        make_minimal_paper(
            tmp_path,
            "The Live dual demo reports no metric. "
            "Fixed-cache demo reports audited metrics and authorized ground truth.",
        ),
        REGISTRY,
    )

    assert "unbounded live demo claim" not in result.errors


def test_evidence_pipeline_source_fonts_survive_latex_scaling() -> None:
    root = ET.parse(FIGURE_SOURCE).getroot()
    model = root.find(".//mxGraphModel")
    assert model is not None
    page_width = float(model.attrib["pageWidth"])
    label_sizes = [
        float(cell.attrib["style"].split("fontSize=")[1].split(";")[0])
        for cell in root.iter("mxCell")
        if cell.attrib.get("vertex") == "1"
        and int(cell.attrib.get("id", "0")) >= 4
        and "fontSize=" in cell.attrib.get("style", "")
    ]

    assert min(label_sizes) * 459.0 / page_width >= 8.0


def test_evidence_pipeline_generator_is_deterministic_vector(tmp_path: Path) -> None:
    from pypdf import PdfReader
    from scripts.paper.generate_evidence_pipeline import (
        LATEX_TARGET_WIDTH,
        MIN_LABEL_FONT_SIZE,
        PAGE,
        render,
    )

    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    render(FIGURE_SOURCE, first)
    render(FIGURE_SOURCE, second)

    assert first.read_bytes() == second.read_bytes()
    assert MIN_LABEL_FONT_SIZE * LATEX_TARGET_WIDTH / PAGE[0] >= 8.0
    assert len(PdfReader(first).pages[0].images) == 0


def test_registry_only_audit_reports_missing_sources_without_false_strict_claim(
    tmp_path: Path,
) -> None:
    paper, registry = _copy_portable_project(tmp_path)

    strict = audit_paper(paper, registry)
    portable = audit_paper(paper, registry, source_verification="registry-only")
    receipt = portable.receipt(paper)

    assert not strict.passed
    assert any("source hash drift" in error for error in strict.errors)
    assert portable.passed
    assert not portable.errors
    assert not portable.blockers
    assert receipt["source_verification"] == "registry-only"
    assert receipt["missing_source_ids"] == sorted(receipt["missing_source_ids"])
    assert receipt["missing_source_ids"] == [
        "loop170_bootstrap",
        "loop170_locked_panel",
        "loop191_report",
        "loop192_report",
        "loop206_report",
    ]
    assert receipt["warnings"] == [
        "source bytes unavailable; strict local release audit required"
    ]


def test_audit_requires_registry_evidence_mapping(tmp_path: Path) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, "No numeric evidence."), REGISTRY)

    assert "missing evidence mapping" in result.errors


@pytest.mark.parametrize("value", (None, "0" * 64))
def test_audit_rejects_missing_or_stale_release_manifest_projection(
    tmp_path: Path, value: str | None
) -> None:
    paper, registry = _copy_portable_project(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if value is None:
        manifest.pop("release_manifest_sha256", None)
    else:
        manifest["release_manifest_sha256"] = value
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, registry, source_verification="registry-only")

    assert "release manifest projection mismatch" in result.errors


def test_audit_keeps_derived_demo_summary_independent_of_release_projection(
    tmp_path: Path,
) -> None:
    paper = _copy_paper(tmp_path)
    receipt_path = paper / "figures/qualitative_demo_receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    receipt.pop("release_manifest_sha256", None)
    receipt_path.write_text(json.dumps(receipt), encoding="ascii")

    result = audit_paper(paper, REGISTRY, source_verification="registry-only")

    assert "demo release manifest projection mismatch" not in result.errors


def test_qualitative_public_summary_is_compact_and_manifest_bound() -> None:
    summary_path = PAPER / "figures/qualitative_demo_receipts.json"
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    manifest = json.loads((PAPER / "artifact_manifest.json").read_text(encoding="ascii"))
    qualitative = manifest["figures"]["qualitative_demo"]

    assert summary["schema_version"] == "loop206.qualitative_public_summary.v1"
    assert summary["artifact_role"] == "derived_public_aggregate_provenance"
    assert "receipts" not in summary
    assert "metrics" not in summary
    assert qualitative["public_summary_sha256"] == hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()


def test_audit_rejects_undefined_citation(tmp_path: Path) -> None:
    paper = make_minimal_paper(tmp_path, "See \\citep{missing_key}.")

    result = audit_paper(paper, REGISTRY)

    assert not result.passed
    assert any("undefined citation key" in error for error in result.errors)


def test_audit_rejects_unsupported_result_number(tmp_path: Path) -> None:
    paper = make_minimal_paper(tmp_path, "Robust Dice was 0.7777.")

    result = audit_paper(paper, REGISTRY)

    assert not result.passed
    assert any("unsupported numeric result" in error for error in result.errors)


def test_audit_rejects_dataset_count_as_a_metric_value(tmp_path: Path) -> None:
    paper = make_minimal_paper(tmp_path, "Robust Dice was 430.")

    result = audit_paper(paper, REGISTRY)

    assert any("unsupported numeric result" in error for error in result.errors)


@pytest.mark.parametrize("value", ["0.0046", "-0.0313"])
def test_audit_rejects_undeclared_dice_point_estimates(
    tmp_path: Path, value: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, f"Robust Dice was {value}."), REGISTRY)

    assert any("unsupported numeric result" in error for error in result.errors)


def test_audit_allows_declared_dice_delta_in_delta_context(tmp_path: Path) -> None:
    result = audit_paper(
        make_minimal_paper(tmp_path, "Robust Dice delta was -0.0313."), REGISTRY
    )

    assert not any("unsupported numeric result" in error for error in result.errors)


@pytest.mark.parametrize(
    "claim",
    [
        "Loop191 IMP-SegFormer-B3 protected-validation robust Dice was 0.9019177076063616.",
        "Loop192 nnU-Net v2 protected-validation robust Dice was 0.895870479294128.",
        "ResNet protected-validation robust Dice was 0.895870479294128.",
        "ResNet50 protected-validation robust Dice was 0.895870479294128.",
        "resnet protected-validation robust Dice was 0.895870479294128.",
        "resnet robust Dice was 0.895870479294128.",
        "Loop191 IMP-SegFormer-B3 clean Dice was 0.895870479294128.",
        "Loop206 train-screen robust Dice was 0.895870479294128.",
        "Under protected-validation evidence, robust Dice was 0.8913.",
        "Under metric contract legacy_nearest_384_t2, robust Dice was 0.8913.",
    ],
)
def test_audit_rejects_cross_identity_or_evidence_number_swaps(
    tmp_path: Path, claim: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert any("unsupported numeric result" in error for error in result.errors)


@pytest.mark.parametrize(
    "claim",
    [
        "Loop191 IMP-SegFormer-B3 protected-validation robust Dice was 0.895870479294128.",
        "Loop192 nnU-Net v2 protected-validation robust Dice was 0.9019177076063616.",
        "Loop206 train-screen robust Dice delta was -0.03129624395473221.",
        "Loop206 train-screen robust Dice 95% CI was [-0.049121296024302145, -0.015627817085354864].",
    ],
)
def test_audit_allows_identity_bound_declared_values(
    tmp_path: Path, claim: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert not any("unsupported numeric result" in error for error in result.errors)


def test_audit_rejects_registry_source_hash_drift(tmp_path: Path) -> None:
    payload = json.loads(REGISTRY.read_text(encoding="ascii"))
    payload["sources"][0]["sha256"] = "0" * 64
    drifted = _write_registry(tmp_path / "registry.json", payload)

    result = audit_paper(PAPER, drifted)

    assert not result.passed
    assert any("source hash drift" in error for error in result.errors)


def test_audit_rejects_manifest_figure_hash_drift(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["figures"]["loop206_delta"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert not result.passed
    assert any("figure hash drift" in error for error in result.errors)


def test_audit_requires_committed_paper_pdf_binding(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest.pop("paper_pdf", None)
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert "missing paper PDF binding" in result.errors


def test_audit_rejects_committed_paper_pdf_hash_drift(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    (paper / "main.pdf").write_bytes(b"replaced PDF")

    result = audit_paper(paper, REGISTRY)

    assert "paper PDF hash drift" in result.errors


def test_audit_rejects_committed_paper_pdf_page_drift(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    pdf = paper / "main.pdf"
    manifest["paper_pdf"] = {
        "path": "main.pdf",
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "pages": 999,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert "paper PDF page drift" in result.errors


@pytest.mark.parametrize(
    ("status", "built_digest"),
    [
        (None, "c09a735c8147049623ffb39d8b86bc83d3edd21b2ba63b6e205a0661380322fa"),
        ("unknown", "c09a735c8147049623ffb39d8b86bc83d3edd21b2ba63b6e205a0661380322fa"),
        ("stale_uncompiled", "malformed"),
        ("current", "c09a735c8147049623ffb39d8b86bc83d3edd21b2ba63b6e205a0661380322fa"),
    ],
)
def test_audit_rejects_invalid_paper_pdf_release_binding(
    tmp_path: Path, status: str | None, built_digest: str
) -> None:
    paper = _copy_paper(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if status is None:
        manifest["paper_pdf"].pop("status", None)
    else:
        manifest["paper_pdf"]["status"] = status
    manifest["paper_pdf"]["built_release_manifest_sha256"] = built_digest
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, REGISTRY, source_verification="registry-only")

    assert "invalid paper PDF release binding" in result.errors


def test_stale_paper_pdf_is_a_release_blocker_not_an_audit_error(
    tmp_path: Path,
) -> None:
    paper, registry = _copy_portable_project(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["paper_pdf"]["status"] = "stale_uncompiled"
    manifest["paper_pdf"]["built_release_manifest_sha256"] = (
        "c09a735c8147049623ffb39d8b86bc83d3edd21b2ba63b6e205a0661380322fa"
    )
    manifest["paper_input_sha256"] = table_builder.paper_input_sha256(paper)
    manifest["paper_pdf"]["built_paper_input_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, registry, source_verification="registry-only")
    receipt = result.receipt(paper)

    assert not result.passed
    assert not result.errors
    assert result.blockers == ("paper PDF is stale for current paper inputs",)
    assert receipt["blockers"] == ["paper PDF is stale for current paper inputs"]
    assert receipt["passed"] is False


def test_stale_paper_pdf_cannot_pass_strict_audit(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["paper_input_sha256"] = table_builder.paper_input_sha256(paper)
    manifest["paper_pdf"]["built_paper_input_sha256"] = "0" * 64
    manifest["paper_pdf"]["status"] = "stale_uncompiled"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert not result.passed
    assert "paper PDF is stale for current paper inputs" in result.blockers


def test_audit_accepts_current_pdf_input_binding(tmp_path: Path) -> None:
    paper, registry = _copy_portable_project(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    digest = table_builder.paper_input_sha256(paper)
    manifest["paper_input_sha256"] = digest
    manifest["paper_pdf"]["built_paper_input_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, registry, source_verification="registry-only")

    assert "invalid paper input binding" not in result.errors
    assert "paper input hash drift" not in result.errors


def test_audit_blocks_stale_pdf_input_binding(tmp_path: Path) -> None:
    paper, registry = _copy_portable_project(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["paper_input_sha256"] = table_builder.paper_input_sha256(paper)
    manifest["paper_pdf"]["built_paper_input_sha256"] = "0" * 64
    manifest["paper_pdf"]["status"] = "stale_uncompiled"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, registry, source_verification="registry-only")

    assert not result.errors
    assert result.blockers == ("paper PDF is stale for current paper inputs",)


@pytest.mark.parametrize(
    ("container", "value"),
    [
        ("manifest", None),
        ("manifest", "malformed"),
        ("paper_pdf", None),
        ("paper_pdf", "malformed"),
    ],
)
def test_audit_rejects_missing_or_malformed_paper_input_digest(
    tmp_path: Path, container: str, value: str | None
) -> None:
    paper, registry = _copy_portable_project(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    digest = table_builder.paper_input_sha256(paper)
    manifest["paper_input_sha256"] = digest
    manifest["paper_pdf"]["built_paper_input_sha256"] = digest
    target = manifest if container == "manifest" else manifest["paper_pdf"]
    key = "paper_input_sha256" if container == "manifest" else "built_paper_input_sha256"
    if value is None:
        target.pop(key, None)
    else:
        target[key] = value
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, registry, source_verification="registry-only")

    assert "invalid paper input binding" in result.errors


def test_audit_requires_editable_source_hash(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    del manifest["figures"]["evidence_pipeline"]["editable_source_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert "missing source hash" in result.errors


def test_audit_rejects_orphan_editable_source_hash(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    del manifest["figures"]["evidence_pipeline"]["editable_source_path"]
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert "missing source hash" in result.errors


def test_audit_rejects_manifest_path_outside_paper(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    entry = manifest["figures"]["loop206_delta"]
    entry["path"] = "../outside.pdf"
    entry["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert "unsafe manifest path" in result.errors


def test_audit_reconciles_declared_figure_input(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    main = paper / "main.tex"
    main.write_text(
        main.read_text(encoding="utf-8").replace(
            "\\end{document}", "\\includegraphics{figures/unmapped.pdf}\n\\end{document}"
        ),
        encoding="utf-8",
    )

    result = audit_paper(paper, REGISTRY)

    assert any("unmapped figure input" in error for error in result.errors)


def test_audit_reconciles_declared_table_input(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    main = paper / "main.tex"
    main.write_text(
        main.read_text(encoding="utf-8").replace(
            "\\end{document}", "\\input{tables/unmapped}\n\\end{document}"
        ),
        encoding="utf-8",
    )

    result = audit_paper(paper, REGISTRY)

    assert any("unmapped table input" in error for error in result.errors)


def test_audit_scans_nested_tex_claims(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    nested = paper / "sections/nested_claim.tex"
    nested.write_text("Robust Dice was 0.7777.\n", encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert any("unsupported numeric result" in error for error in result.errors)


def test_audit_rejects_unlabeled_loop170_values(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    table = paper / "tables/legacy_loop170.tex"
    table.write_text(
        table.read_text(encoding="utf-8").replace(
            "legacy\\_patient\\_contaminated", "legacy evidence"
        ),
        encoding="utf-8",
    )

    result = audit_paper(paper, REGISTRY)

    assert not result.passed
    assert any("unlabeled Loop170 values" in error for error in result.errors)


def test_audit_rejects_raw_entries_in_qualitative_public_summary(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    bundle = paper / "figures/qualitative_demo_receipts.json"
    payload = json.loads(bundle.read_text(encoding="ascii"))
    payload["receipts"] = []
    bundle.write_text(json.dumps(payload), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert not result.passed
    assert "invalid demo public summary" in result.errors


def test_audit_rejects_qualitative_public_summary_schema_drift(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    bundle = paper / "figures/qualitative_demo_receipts.json"
    payload = json.loads(bundle.read_text(encoding="ascii"))
    payload["schema_version"] = "loop206.qualitative_public_summary.v2"
    bundle.write_text(json.dumps(payload), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert not result.passed
    assert "invalid demo public summary" in result.errors


def test_audit_rejects_qualitative_authorization_count_mismatch(
    tmp_path: Path,
) -> None:
    paper = _copy_paper(tmp_path)
    bundle = paper / "figures/qualitative_demo_receipts.json"
    payload = json.loads(bundle.read_text(encoding="ascii"))
    payload["authorized_sample_count"] = 4
    bundle.write_text(json.dumps(payload), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert "demo authorization count mismatch" in result.errors


def test_audit_rejects_missing_aggregate_mask_binding(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    bundle = paper / "figures/qualitative_demo_receipts.json"
    payload = json.loads(bundle.read_text(encoding="ascii"))
    payload.pop("aggregate_mask_bindings_sha256")
    bundle.write_text(json.dumps(payload), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert not result.passed
    assert "unbound demo mask authorization" in result.errors


def test_manuscript_discloses_statistical_and_reproducibility_limits() -> None:
    manuscript = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PAPER.rglob("*.tex"))
    )
    reproducibility = (PAPER / "sections/09_reproducibility.tex").read_text(
        encoding="utf-8"
    )
    normalized = manuscript.lower()

    assert "after averaging the three selected seeds and three views" in normalized
    assert "does not estimate variability over seed selection" in normalized
    assert "per-seed directions are unavailable in the authorized paper evidence" in normalized
    assert "one slightly positive and two negative" not in normalized
    assert "adaptive development and checkpoint-selection validation" in normalized
    assert "across three paired seeds" not in normalized
    assert "not tracked in this release" in reproducibility
    assert "no tracked, commit-bound receipt" in reproducibility


def test_qualitative_ground_truth_claim_names_each_hash_authority() -> None:
    results = (PAPER / "sections/06_results.tex").read_text(encoding="utf-8").lower()

    assert "provenance manifest verifies source identity" in results
    assert "raw and decoded image hashes" in results
    assert "mask bytes are separately bound to the pinned dataset index" in results
    assert "image and mask hashes" not in results


def test_audit_rejects_affirmative_clause_after_negated_clause(tmp_path: Path) -> None:
    paper = make_minimal_paper(
        tmp_path, "The baseline is not SOTA, but ours is state-of-the-art."
    )

    result = audit_paper(paper, REGISTRY)

    assert any("affirmative protected claim" in error for error in result.errors)


def test_audit_binds_negation_to_the_matched_claim_predicate(tmp_path: Path) -> None:
    for index, claim in enumerate((
        "No test was opened and our model is state of the art.",
        "No test was opened and we are state of the art.",
        "No test was opened and our model remains state of the art.",
        "Our method is not diagnostic and the model is state of the art.",
    )):
        case = tmp_path / f"case-{index}"
        case.mkdir()
        result = audit_paper(make_minimal_paper(case, claim), REGISTRY)

        assert any("affirmative protected claim" in error for error in result.errors)


def test_audit_rejects_affirmative_modal_predicate_after_negated_claim(
    tmp_path: Path,
) -> None:
    result = audit_paper(
        make_minimal_paper(
            tmp_path,
            "The report does not claim diagnostic use and is state of the art.",
        ),
        REGISTRY,
    )

    assert any("affirmative protected claim" in error for error in result.errors)


def test_audit_rejects_affirmative_claim_by_possessive_pronoun_subject(
    tmp_path: Path,
) -> None:
    result = audit_paper(
        make_minimal_paper(
            tmp_path,
            "No method is diagnostic and ours is state of the art.",
        ),
        REGISTRY,
    )

    assert any("affirmative protected claim" in error for error in result.errors)


def test_pdf_page_count_uses_pdfinfo_not_raw_pdf_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF-1.7\n/Type /Page\n/Type /Page\n")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "Pages:           12\n", "")

    pdfinfo = tmp_path / "pdfinfo.exe"
    pdfinfo.write_bytes(b"trusted")
    monkeypatch.setenv("IMP_PDFINFO_EXE", str(pdfinfo.resolve()))
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert audit_module._pdf_page_count(pdf) == 12
    assert calls == [["pdfinfo", str(pdf)]]


def test_pdf_page_count_executes_only_trusted_absolute_pdfinfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF")
    pdfinfo = tmp_path / "trusted-pdfinfo.exe"
    pdfinfo.write_bytes(b"trusted")
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "Pages: 1\n", "")

    monkeypatch.setenv("IMP_PDFINFO_EXE", str(pdfinfo.resolve()))
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert audit_module._pdf_page_count(pdf) == 1
    assert seen["executable"] == str(pdfinfo.resolve())


def test_pdf_page_count_rejects_path_or_cwd_pdfinfo_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF")
    malicious = tmp_path / "pdfinfo.exe"
    malicious.write_bytes(b"malicious")
    runtime = tmp_path / "runtime" / "python.exe"
    runtime.parent.mkdir()
    runtime.write_bytes(b"python")
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("untrusted executable invoked")

    monkeypatch.delenv("IMP_PDFINFO_EXE", raising=False)
    monkeypatch.setattr(audit_module.sys, "executable", str(runtime))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="trusted pdfinfo executable unavailable"):
        audit_module._pdf_page_count(pdf)
    assert not called


@pytest.mark.parametrize(
    "claim",
    [
        "Although the baseline is not SOTA, ours is state-of-the-art.",
        "While the baseline is not SOTA, ours is state-of-the-art.",
    ],
)
def test_audit_rejects_affirmative_claim_after_subordinate_negation(
    tmp_path: Path, claim: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert any("affirmative protected claim" in error for error in result.errors)


@pytest.mark.parametrize(
    "claim",
    [
        "No method, including ours, is state-of-the-art.",
        "No method is diagnostic or state-of-the-art.",
        "No contribution is a clinical system or a state-of-the-art claim.",
        "No contribution is a clinical system or the state-of-the-art system.",
        "No baseline is diagnostic and ResNet is not state-of-the-art.",
        "The baseline is not diagnostic and ours does not claim state-of-the-art performance.",
        "This is not clinical-grade, is not diagnostic, and is not intended for clinical use.",
    ],
)
def test_audit_allows_coordinated_negated_claims(tmp_path: Path, claim: str) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert not any("affirmative protected claim" in error for error in result.errors)


@pytest.mark.parametrize(
    "identity",
    [
        "ResNet",
        "resnet",
        "the ResNet model",
        r"\impmodel{}",
        "ours",
        "an-unregistered-system",
    ],
)
def test_coordination_claim_scope_is_identity_agnostic(
    tmp_path: Path, identity: str
) -> None:
    affirmative_root = tmp_path / "affirmative"
    negated_root = tmp_path / "negated"
    affirmative_root.mkdir()
    negated_root.mkdir()
    affirmative = audit_paper(
        make_minimal_paper(
            affirmative_root,
            f"No baseline is diagnostic and {identity} is state of the art.",
        ),
        REGISTRY,
    )
    locally_negated = audit_paper(
        make_minimal_paper(
            negated_root,
            f"No baseline is diagnostic and {identity} is not state of the art.",
        ),
        REGISTRY,
    )

    assert any("affirmative protected claim" in error for error in affirmative.errors)
    assert not any(
        "affirmative protected claim" in error for error in locally_negated.errors
    )


@pytest.mark.parametrize(
    "claim",
    [
        "No baseline is diagnostic and Model-Z qualifies as state of the art.",
        "No baseline is diagnostic and ResNet is considered state of the art.",
        "No baseline is diagnostic and Model-Z unexpectedly became state of the art.",
    ],
)
def test_coordination_claim_scope_fails_closed_for_unknown_predicates(
    tmp_path: Path, claim: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert any("affirmative protected claim" in error for error in result.errors)


@pytest.mark.parametrize(
    "claim",
    [
        "No baseline is diagnostic and family model is state of the art.",
        "No baseline is diagnostic and family state of the art.",
        "No baseline is diagnostic and ally state of the art.",
        "No baseline is diagnostic and supply state of the art.",
    ],
)
def test_coordination_claim_scope_treats_ly_nouns_as_substantive(
    tmp_path: Path, claim: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert any("affirmative protected claim" in error for error in result.errors)


@pytest.mark.parametrize(
    "claim",
    [
        "The protected-test Dice was 0.9019.",
        "The protected-test accuracy was 0.9019.",
        "The protected-test metric was 0.9019.",
        "The candidate significantly outperforms the baseline.",
        "The candidate significantly outperformed the baseline.",
        "The candidate is significantly outperforming the baseline.",
    ],
)
def test_audit_rejects_additional_affirmative_claims(
    tmp_path: Path, claim: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert any("affirmative protected claim" in error for error in result.errors)


@pytest.mark.parametrize(
    "claim",
    [
        "The protected-test recall was 0.9019.",
        "The protected-test HD95 was 0.9019.",
        "The protected-test precision was 0.9019.",
        "The protected-test ASSD was 0.9019.",
        "The protected-test boundary F1 was 0.9019.",
    ],
)
def test_audit_rejects_all_protected_test_metric_claims(
    tmp_path: Path, claim: str
) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert any("affirmative protected claim" in error for error in result.errors)


def test_audit_allows_sealed_protected_test_metric(tmp_path: Path) -> None:
    result = audit_paper(
        make_minimal_paper(tmp_path, "The protected-test recall remains sealed."),
        REGISTRY,
    )

    assert not any("affirmative protected claim" in error for error in result.errors)


@pytest.mark.parametrize("macro", ["textcite", "parencite", "autocite", "Citep"])
def test_audit_rejects_undefined_generic_cite_macro(tmp_path: Path, macro: str) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, f"\\{macro}{{missing_key}}"), REGISTRY)

    assert any("undefined citation key" in error for error in result.errors)


def test_audit_ignores_citation_style_commands(tmp_path: Path) -> None:
    result = audit_paper(
        make_minimal_paper(tmp_path, "\\setcitestyle{round}\\citestyle{authoryear}"),
        REGISTRY,
    )

    assert not any("undefined citation key" in error for error in result.errors)


def test_cli_returns_nonzero_and_writes_path_free_failure_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "paper_audit.json"
    paper = make_minimal_paper(tmp_path, "state-of-the-art")

    assert main(["--paper", str(paper), "--registry", str(REGISTRY), "--receipt", str(receipt)]) == 1
    payload = receipt.read_text(encoding="ascii")
    assert '"passed": false' in payload
    assert str(tmp_path) not in payload


def test_task6_readiness_audit_keeps_p1_blocked() -> None:
    audit = ROOT / "reports/paper_revision/manuscript_readiness_audit.md"

    assert audit.is_file()
    text = audit.read_text(encoding="utf-8")

    assert "16/33" in text
    assert "Hard blockers" in text
    assert "Venue/template status: unverified" in text
    assert "P1 scientific rerun: BLOCKED" in text


def test_task6_live_evidence_classification_satisfies_claim_audit() -> None:
    errors: list[str] = []

    audit_module._check_live_demo_claims(
        (PAPER / "sections/06_results.tex").read_text(encoding="utf-8"), errors
    )

    assert errors == []


def test_task4_browser_and_tunnel_observations_are_historical_not_current_release_evidence() -> None:
    results = (PAPER / "sections/06_results.tex").read_text(encoding="utf-8")
    reproducibility = (PAPER / "sections/09_reproducibility.tex").read_text(
        encoding="utf-8"
    )

    for text in (results, reproducibility):
        assert "historical Task 4" in text
        assert "not current-release acceptance evidence" in text
        assert r"\texttt{unverified/blocked}" in text
    assert "ephemeral Cloudflare public GET" in results
    assert "No canonical live runtime receipt is present" in reproducibility
    assert "Neither original-runtime equivalence" in reproducibility


def test_limitations_and_readiness_audit_mark_task4_browser_evidence_superseded() -> None:
    limitations = (PAPER / "sections/08_limitations_ethics.tex").read_text(encoding="utf-8")
    readiness = (ROOT / "reports/paper_revision/manuscript_readiness_audit.md").read_text(
        encoding="utf-8"
    )
    for text in (limitations, readiness):
        assert "historical/superseded" in text
        assert "Task 4" in text
        assert "unverified/blocked" in text
