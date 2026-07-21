from pathlib import Path
import json

from scripts.paper.build_clean_v3_tables import build_tables


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "demo/data/evidence_registry.json"


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


def test_legacy_table_carries_contamination_label(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    text = outputs["legacy_loop170"].read_text(encoding="utf-8")
    assert "legacy\\_patient\\_contaminated" in text
    assert "cross-split" in text


def test_table_manifest_binds_registry_hash(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    manifest = outputs["artifact_manifest"].read_text(encoding="ascii")
    assert "f6ed2eace90c49ee1b9f0c122e736920791b6301035bf8905c6a0ce27b755f32" in manifest


def test_table_rebuild_preserves_non_table_manifest_evidence(tmp_path: Path) -> None:
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "imp.paper_artifacts.v1",
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
        "legacy_loop170",
    }
