from pathlib import Path
import re
import shutil
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_windows_bootstrap_preserves_cuda_overlay_and_runs_delivery_checks() -> None:
    script = _read("scripts/bootstrap_windows.ps1")

    assert "[ValidateSet('cpu', 'cu130')]" in script
    assert "$Venv = '.venv-win'" in script
    assert "$Compute = 'cpu'" in script
    assert "UV_PROJECT_ENVIRONMENT" in script
    assert "--python 3.12" in script
    for extra in ("--extra dev", "--extra analysis", "--extra demo"):
        assert extra in script
    assert "--extra train" in script
    assert "torch==2.12.0+cu130" in script
    assert "torchvision==0.27.0+cu130" in script
    assert "https://download.pytorch.org/whl/cu130/" in script
    assert "torch.cuda.is_available()" in script
    assert "uv run --no-sync" in script
    assert "python -m pytest tests/demo" in script
    assert "build_clean_v3_tables.py" in script
    assert "$GeneratedPaperDir" in script
    assert "--paper-dir $GeneratedPaperDir" in script
    assert "Get-FileHash" in script
    assert "audit_clean_v3_paper.py" in script
    assert "--source-verification registry-only" in script
    assert "latexmk" in script
    assert "pdflatex" in script and "bibtex" in script
    assert "$PaperBuilt = $false" in script
    assert "latexmk failed; trying pdflatex/bibtex fallback" in script
    assert "No TeX toolchain found" in script
    sync_invocations = re.findall(r"^\s*uv sync\b.*$", script, flags=re.MULTILINE)
    assert len(sync_invocations) == 2
    assert all("--locked" in invocation for invocation in sync_invocations)


def test_ci_is_cpu_only_reproducible_and_uploads_receipts() -> None:
    workflow_text = _read(".github/workflows/ci.yml")
    workflow = yaml.safe_load(workflow_text)
    job = workflow["jobs"]["delivery"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["CUDA_VISIBLE_DEVICES"] == ""

    uses = [step["uses"] for step in job["steps"] if "uses" in step]
    assert uses
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", value) for value in uses)

    commands = "\n".join(
        str(step.get("run", "")) for step in job["steps"] if "run" in step
    )
    assert "--python 3.12" in commands
    assert "python -m pytest tests/demo" in commands
    assert "build_clean_v3_tables.py" in commands
    assert "--paper-dir ci-receipts/generated-paper" in commands
    assert "cmp" in commands
    assert "audit_clean_v3_paper.py" in commands
    assert "--source-verification registry-only" in commands
    assert "latexmk" in commands
    assert "apt-get install" in commands
    assert "texlive" in commands
    assert "latexmk -C" in commands
    assert "latexmk -pdf" in commands
    assert "pdfinfo" in commands
    assert "TeX runner unavailable; paper compilation skipped" not in commands
    for receipt_field in (
        "GITHUB_SHA",
        "built_pdf_sha256",
        "pages",
        "bytes",
        "build_command",
        "status",
    ):
        assert receipt_field in commands
    assert "git diff --exit-code" in commands
    assert "torch.cuda.is_available()" not in commands
    upload_paths = "\n".join(
        str(step.get("with", {}).get("path", "")) for step in job["steps"]
    )
    assert "ci-receipts/paper/main.pdf" in upload_paths
    assert "paper/clean_v3_loop206/main.pdf" not in upload_paths
    assert any("receipt" in str(step.get("with", {}).get("path", "")) for step in job["steps"])


def test_runbook_and_readme_define_private_two_machine_handoff() -> None:
    runbook = _read("docs/runbooks/two-machine-delivery.md")
    readme = _read("README.md")

    for token in (
        "paper-review",
        "demo-runtime",
        "RTX 4060",
        "RTX 5060 Ti",
        "LAN/USB",
        "SHA-256",
        "never through GitHub",
        "uv run --no-sync",
        "-Compute cu130",
        "isPrivate",
    ):
        assert token in runbook
    assert "physical laptop" in runbook.lower()
    assert "unverified" in runbook.lower()
    assert "portable verification" in runbook.lower()
    assert "strict local release audit" in runbook.lower()
    assert "source_verification=registry-only" in runbook
    assert "two-machine-delivery.md" in readme
    assert "bootstrap_windows.ps1" in readme
    assert "/paper/clean_v3_loop206/main.pdf" in _read(".gitignore")
    attributes = _read(".gitattributes")
    for pattern in (
        "demo/data/evidence_registry.json text eol=lf",
        "paper/clean_v3_loop206/**/*.py text eol=lf",
        "paper/clean_v3_loop206/**/*.json text eol=lf",
        "paper/clean_v3_loop206/**/*.tex text eol=lf",
        "paper/clean_v3_loop206/figures/qualitative_demo_receipts.json text eol=crlf",
    ):
        assert pattern in attributes


def test_delivery_files_ignore_private_assets() -> None:
    ignore = _read(".gitignore")
    for pattern in (
        "/weights/",
        "/secrets/",
        "*.safetensors",
        "*.engine",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.onnx",
    ):
        assert pattern in ignore
    assert "*.bin" not in ignore


def test_delivery_files_avoid_machine_local_paths() -> None:
    for relative in (
        "README.md",
        "docs/runbooks/two-machine-delivery.md",
        "scripts/bootstrap_windows.ps1",
        ".github/workflows/ci.yml",
    ):
        text = _read(relative)
        assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text), relative
        assert not re.search(r"/(?:mnt|home|Users)/", text), relative


def test_laptop_handoff_uses_external_venv_and_propagates_native_failures() -> None:
    runbook = _read("docs/runbooks/two-machine-delivery.md")
    bootstrap_section = runbook.split("## Windows Bootstrap", 1)[1].split(
        "## Private GitHub Provisioning", 1
    )[0]
    handoff = runbook.split("## Laptop Handoff", 1)[1].split(
        "## Artifact Transfer", 1
    )[0]
    guard = "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"

    assert "$laptopVenv = Join-Path $env:LOCALAPPDATA" in bootstrap_section
    assert "-Venv $laptopVenv" in bootstrap_section
    assert ".venvs/imp-paper" not in bootstrap_section
    for invocation in (
        r"gh repo view [^;]+",
        r"git clone [^;]+",
        r"powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows\.ps1",
        r"git status --short",
    ):
        assert re.search(rf"{invocation}; {re.escape(guard)}", handoff)

    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell unavailable; static handoff guards verified")
    simulation = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-Command",
            f"$global:LASTEXITCODE = 37; {guard}; Write-Output 'masked'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert simulation.returncode == 37
    assert "masked" not in simulation.stdout
