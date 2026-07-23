from pathlib import Path
import re
import subprocess
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
        "/.artifacts/",
        "/runs/",
        "/data/",
        "*.pt",
        "*.pth",
        ".env",
        ".venv-win/",
    ):
        assert pattern in text


def test_compact_demo_evidence_is_not_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "demo/data/evidence_registry.json"],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 1


def test_tracked_runtime_files_have_no_machine_specific_absolute_paths() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    excluded = ("tests/",)
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/home/[^/\s]+(?:[/\\]|$)|/mnt/[A-Za-z](?:[/\\]|$))"
    )
    violations: list[str] = []
    for relative in tracked:
        if not relative or relative.startswith(excluded):
            continue
        try:
            text = (ROOT / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                violations.append(f"{relative}:{line_number}")

    assert violations == []
