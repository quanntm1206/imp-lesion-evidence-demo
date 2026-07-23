from pathlib import Path
from copy import deepcopy
import hashlib
import json
import shutil

import pytest

import scripts.paper.build_clean_v3_tables as table_builder
from scripts.paper.build_clean_v3_tables import build_tables
from lesion_robustness.release_manifest import paper_projection


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "demo/data/evidence_registry.json"
PAPER = ROOT / "paper/clean_v3_loop206"


def _copy_paper(tmp_path: Path) -> Path:
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper)
    return paper


def _gate_rows(text: str) -> list[list[str]]:
    body = text.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    rows = []
    for index, line in enumerate(lines[:-1]):
        if line.endswith("&"):
            normalized = f"{line} {lines[index + 1]}".removesuffix(r" \\")
            rows.append([cell.strip() for cell in normalized.split("&")])
    return rows


def _rehash_registry(payload: dict) -> dict:
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
    payload["registry_sha256"] = hashlib.sha256(
        (encoded + "\n").encode("ascii")
    ).hexdigest()
    return payload


def _mutated_registry_project(tmp_path: Path, receipt: dict | bytes) -> Path:
    root = tmp_path / "project"
    registry_path = root / "demo/data/evidence_registry.json"
    source_path = root / ".artifacts/preprocessing_search/current_bdou_loop206_final_closure_report.json"
    registry_path.parent.mkdir(parents=True)
    source_path.parent.mkdir(parents=True)
    if isinstance(receipt, bytes):
        source_path.write_bytes(receipt)
    else:
        source_path.write_text(json.dumps(receipt, allow_nan=False), encoding="utf-8")
    registry = json.loads(REGISTRY.read_text(encoding="ascii"))
    loop206_source = next(
        source for source in registry["sources"] if source["source_id"] == "loop206_report"
    )
    loop206_source["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    registry_path.write_text(
        json.dumps(_rehash_registry(registry), ensure_ascii=True), encoding="ascii"
    )
    return registry_path


def _synthetic_closure_receipt() -> dict:
    return {
        "schema_version": "loop206.final_closure.v1",
        "loop": 206,
        "evidence_validation": {"passed": True},
        "bootstrap": {
            "dice": {
                "point_delta": -0.03129624395473221,
                "ci95_lower": -0.049121296024302145,
                "ci95_upper": -0.015627817085354864,
            },
            "boundary_f1": {
                "point_delta": -0.01465831334754726,
                "ci95_lower": -0.030758654691150956,
                "ci95_upper": 0.0010438469457382654,
            },
        },
        "robust_deltas": {
            "precision": -0.01,
            "recall": -0.02,
            "hd95": 0.03,
            "assd": 0.04,
        },
    }


def test_clean_v3_table_contains_scoped_point_estimates(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    text = outputs["clean_v3_validation"].read_text(encoding="utf-8")
    assert "0.8959" in text
    assert "0.9019" in text
    assert "validation" in text.lower()
    assert "SOTA" not in text


def test_loop206_table_discloses_group_bootstrap_with_fixed_seeds(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    text = outputs["loop206_ablation"].read_text(encoding="utf-8")
    assert "-0.0313" in text
    assert "[-0.0491, -0.0156]" in text
    assert "averaging three selected seeds and three views" in text
    assert "76 groups as whole split-group clusters" in text
    assert "conditional on those seeds" in text


def test_loop206_gate_audit_has_fixed_rows_and_auditable_columns(
    tmp_path: Path,
) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    text = outputs["loop206_gate_audit"].read_text(encoding="utf-8")
    gate_ids = (
        "primary_improvement",
        "dice_noninferiority",
        "boundary_noninferiority",
        "clean_dice",
        "precision",
        "recall",
        "distance",
        "per_corruption",
    )

    assert outputs["loop206_gate_audit"].name == "loop206_gate_audit.tex"
    assert [text.index(gate_id.replace("_", r"\_")) for gate_id in gate_ids] == sorted(
        text.index(gate_id.replace("_", r"\_")) for gate_id in gate_ids
    )
    for column in (
        "threshold",
        "observed",
        "interval",
        "status",
    ):
        assert column in text
    assert r"\def\sidprefix{seed-}" in text
    for seed in (206, 1206, 2206):
        assert rf"\sidprefix{seed}" in text
    assert "unavailable" in text
    assert "blocked" in text
    assert r"loop206\_report@7d6fcd61259a" in text

    rows = _gate_rows(text)
    assert len(rows) == 8
    for row in rows:
        assert len(row) == 10
        assert row[2] == "unavailable"
        assert row[5:8] == ["unavailable", "unavailable", "unavailable"]
        assert row[8] == "blocked"


def test_gate_table_fails_closed_on_source_hash_drift(tmp_path: Path) -> None:
    receipt = _synthetic_closure_receipt()
    registry_path = _mutated_registry_project(tmp_path, receipt)
    source_path = registry_path.parents[2] / ".artifacts/preprocessing_search/current_bdou_loop206_final_closure_report.json"
    source_path.write_bytes(b"{}")

    text = build_tables(registry_path, tmp_path / "paper")[
        "loop206_gate_audit"
    ].read_text(encoding="utf-8")

    assert len(_gate_rows(text)) == 8
    assert "-0.0313" not in text
    assert "-0.0147" not in text
    assert all(row[3] == "unavailable" and row[8] == "blocked" for row in _gate_rows(text))


def test_table_write_failure_leaves_pdf_non_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = _copy_paper(tmp_path)
    calls = 0
    original_write = table_builder._write

    def fail_mid_write(path: Path, text: str) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected table write failure")
        return original_write(path, text)

    monkeypatch.setattr(table_builder, "_write", fail_mid_write)

    with pytest.raises(OSError, match="injected table write failure"):
        build_tables(REGISTRY, paper)

    manifest = json.loads((paper / "artifact_manifest.json").read_text(encoding="ascii"))
    assert manifest["paper_pdf"]["status"] != "current"
    assert manifest["paper_build"]["status"] == "building"


@pytest.mark.parametrize("mutation", ["missing", "malformed"])
def test_gate_table_fails_closed_on_missing_or_malformed_receipt_fields(
    tmp_path: Path, mutation: str
) -> None:
    receipt = _synthetic_closure_receipt()
    if mutation == "missing":
        receipt["bootstrap"]["dice"].pop("point_delta")
        receipt["robust_deltas"].pop("precision")
    else:
        receipt["bootstrap"]["dice"]["point_delta"] = "not-a-number"
        receipt["robust_deltas"]["precision"] = None
    registry_path = _mutated_registry_project(tmp_path, receipt)

    rows = _gate_rows(
        build_tables(registry_path, tmp_path / "paper")["loop206_gate_audit"].read_text(
            encoding="utf-8"
        )
    )

    by_gate = {row[0].replace(r"\_", "_"): row for row in rows}
    assert by_gate["primary_improvement"][3] == "unavailable"
    assert by_gate["dice_noninferiority"][3] == "unavailable"
    assert by_gate["precision"][3] == "unavailable"
    assert all(row[8] == "blocked" for row in rows)


def test_two_table_builds_are_byte_identical(tmp_path: Path) -> None:
    paper = tmp_path / "paper"

    first = build_tables(REGISTRY, paper)
    first_bytes = {name: path.read_bytes() for name, path in first.items()}
    second = build_tables(REGISTRY, paper)

    assert set(second) == set(first_bytes)
    assert all(path.read_bytes() == first_bytes[name] for name, path in second.items())


def test_legacy_table_carries_contamination_label(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    text = outputs["legacy_loop170"].read_text(encoding="utf-8")
    assert "legacy\\_patient\\_contaminated" in text
    assert "cross-split" in text


def test_table_manifest_binds_registry_hash(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    manifest = outputs["artifact_manifest"].read_text(encoding="ascii")
    assert "f6ed2eace90c49ee1b9f0c122e736920791b6301035bf8905c6a0ce27b755f32" in manifest


def test_table_manifest_binds_current_release_projection(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)

    manifest = json.loads(outputs["artifact_manifest"].read_text(encoding="ascii"))

    assert manifest["release_manifest_sha256"] == paper_projection()[
        "release_manifest_sha256"
    ]


def test_table_rebuild_rejects_stale_release_projection(tmp_path: Path) -> None:
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "imp.paper_artifacts.v1",
                "release_manifest_sha256": "0" * 64,
            }
        ),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="release manifest projection"):
        build_tables(REGISTRY, tmp_path)


def test_table_rebuild_preserves_non_table_manifest_evidence(tmp_path: Path) -> None:
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "imp.paper_artifacts.v1",
                "release_manifest_sha256": paper_projection()[
                    "release_manifest_sha256"
                ],
                "figures": {"evidence_pipeline": {"path": "figures/evidence_pipeline.pdf"}},
                "review_status": "approved",
                "tables": {"obsolete": {"path": "tables/obsolete.tex"}},
            }
        ),
        encoding="ascii",
    )

    build_tables(REGISTRY, tmp_path)

    rebuilt = json.loads(manifest_path.read_text(encoding="ascii"))
    assert rebuilt["figures"] == {
        "evidence_pipeline": {"path": "figures/evidence_pipeline.pdf"}
    }
    assert rebuilt["review_status"] == "approved"
    assert "obsolete" not in rebuilt["tables"]
    assert set(rebuilt["tables"]) == {
        "evidence_scope",
        "clean_v3_validation",
        "loop206_ablation",
        "loop206_gate_audit",
        "legacy_loop170",
    }


def test_table_rebuild_marks_pdf_stale_when_paper_inputs_change(
    tmp_path: Path,
) -> None:
    paper = _copy_paper(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    built_digest = table_builder.paper_input_sha256(paper)
    manifest["paper_input_sha256"] = built_digest
    manifest["paper_pdf"]["built_paper_input_sha256"] = built_digest
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    section = paper / "sections/06_results.tex"
    section.write_text(section.read_text(encoding="utf-8") + "\n% input drift\n", encoding="utf-8")

    build_tables(REGISTRY, paper)

    rebuilt = json.loads(manifest_path.read_text(encoding="ascii"))
    assert rebuilt["paper_pdf"]["status"] == "stale_uncompiled"
    assert rebuilt["paper_input_sha256"] != rebuilt["paper_pdf"][
        "built_paper_input_sha256"
    ]


def test_table_rebuild_rejects_missing_pdf_input_binding_before_writes(
    tmp_path: Path,
) -> None:
    paper = _copy_paper(tmp_path)
    manifest_path = paper / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["paper_pdf"].pop("built_paper_input_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    sentinel = paper / "tables/loop206_gate_audit.tex"
    sentinel.write_text("sentinel", encoding="ascii")
    before = {
        path.relative_to(paper).as_posix(): path.read_bytes()
        for path in (paper / "tables").glob("*.tex")
    }

    with pytest.raises(ValueError, match="input binding unavailable"):
        build_tables(REGISTRY, paper)

    after = {
        path.relative_to(paper).as_posix(): path.read_bytes()
        for path in (paper / "tables").glob("*.tex")
    }
    assert after == before


def test_paper_pdf_promotion_binds_current_inputs_after_inspection(
    tmp_path: Path,
) -> None:
    paper = _copy_paper(tmp_path)
    pdf = paper / "main.pdf"
    pdf.write_bytes(pdf.read_bytes())
    expected_digest = table_builder.paper_input_sha256(paper)

    table_builder.promote_paper_pdf(
        paper,
        expected_paper_input_sha256=expected_digest,
        visual_review_passed=True,
    )

    manifest = json.loads((paper / "artifact_manifest.json").read_text(encoding="ascii"))
    assert manifest["paper_input_sha256"] == expected_digest
    assert manifest["paper_pdf"]["built_paper_input_sha256"] == expected_digest
    assert manifest["paper_pdf"]["status"] == "current"


def test_paper_pdf_promotion_rejects_uncompiled_input_drift(tmp_path: Path) -> None:
    paper = _copy_paper(tmp_path)
    section = paper / "sections/06_results.tex"
    section.write_text(section.read_text(encoding="utf-8") + "\n% input drift\n", encoding="utf-8")
    expected_digest = table_builder.paper_input_sha256(paper)

    with pytest.raises(ValueError, match="older than paper inputs"):
        table_builder.promote_paper_pdf(
            paper,
            expected_paper_input_sha256=expected_digest,
            visual_review_passed=True,
        )
