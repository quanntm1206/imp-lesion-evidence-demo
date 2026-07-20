from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_demo_dependency_and_entry_points_are_declared() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert set(config["project"]["optional-dependencies"]["demo"]) >= {
        "gradio>=5,<7",
        "joblib>=1.3",
    }
    scripts = config["project"]["scripts"]
    assert scripts["lesion-demo"] == "lesion_robustness.demo.app:main"
    assert scripts["lesion-build-evidence"] == "lesion_robustness.evidence_registry:main"


def test_gitignore_blocks_private_runtime_assets() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="ascii")
    for pattern in (
        ".artifacts/",
        "runs/",
        "data/",
        "*.pt",
        "*.pth",
        ".env",
        ".venv-win/",
    ):
        assert pattern in text
