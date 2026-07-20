from pathlib import Path

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


def test_loop206_table_contains_three_seed_confidence_intervals(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    text = outputs["loop206_ablation"].read_text(encoding="utf-8")
    assert "-0.0313" in text
    assert "[-0.0491, -0.0156]" in text
    assert "three paired seeds" in text


def test_legacy_table_carries_contamination_label(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    text = outputs["legacy_loop170"].read_text(encoding="utf-8")
    assert "legacy\\_patient\\_contaminated" in text
    assert "cross-split" in text


def test_table_manifest_binds_registry_hash(tmp_path: Path) -> None:
    outputs = build_tables(REGISTRY, tmp_path)
    manifest = outputs["artifact_manifest"].read_text(encoding="ascii")
    assert "f6ed2eace90c49ee1b9f0c122e736920791b6301035bf8905c6a0ce27b755f32" in manifest
