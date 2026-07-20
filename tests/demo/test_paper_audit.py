from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.paper.audit_clean_v3_paper import audit_paper, main


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "demo/data/evidence_registry.json"
PAPER = ROOT / "paper/clean_v3_loop206"


def _registry_hash(payload: dict) -> str:
    unsigned = deepcopy(payload)
    unsigned.pop("registry_sha256", None)
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
    shutil.copytree(PAPER, paper)
    registry.parent.mkdir(parents=True)
    shutil.copy2(REGISTRY, registry)
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


def test_audit_allows_explicit_evidence_bounded_negation() -> None:
    result = audit_paper(PAPER, REGISTRY, source_verification="registry-only")

    assert result.passed, result.errors
    assert result.source_verification == "registry-only"


def test_registry_only_audit_reports_missing_sources_without_false_strict_claim(
    tmp_path: Path,
) -> None:
    paper, registry = _copy_portable_project(tmp_path)

    strict = audit_paper(paper, registry)
    portable = audit_paper(paper, registry, source_verification="registry-only")
    receipt = portable.receipt(paper)

    assert not strict.passed
    assert any("source hash drift" in error for error in strict.errors)
    assert portable.passed, portable.errors
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


def test_audit_rejects_metrics_without_ground_truth_authorization(
    tmp_path: Path,
) -> None:
    paper = _copy_paper(tmp_path)
    bundle = paper / "figures/qualitative_demo_receipts.json"
    payload = json.loads(bundle.read_text(encoding="ascii"))
    payload["receipts"][0]["display_authorization"]["mask_variant"] = "none"
    bundle.write_text(json.dumps(payload), encoding="ascii")

    result = audit_paper(paper, REGISTRY)

    assert not result.passed
    assert "hidden no-GT metrics" in result.errors


def test_audit_rejects_affirmative_clause_after_negated_clause(tmp_path: Path) -> None:
    paper = make_minimal_paper(
        tmp_path, "The baseline is not SOTA, but ours is state-of-the-art."
    )

    result = audit_paper(paper, REGISTRY)

    assert any("affirmative protected claim" in error for error in result.errors)


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
        "This is not clinical-grade, diagnostic, or intended for clinical use.",
    ],
)
def test_audit_allows_coordinated_negated_claims(tmp_path: Path, claim: str) -> None:
    result = audit_paper(make_minimal_paper(tmp_path, claim), REGISTRY)

    assert not any("affirmative protected claim" in error for error in result.errors)


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
