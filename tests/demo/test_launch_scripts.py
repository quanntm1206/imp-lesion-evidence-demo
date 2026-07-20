from __future__ import annotations

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import re
import shutil
import subprocess
from threading import Thread

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="ascii")


def _powershell() -> str:
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell unavailable")
    return shell


def _run_script(relative: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(relative), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_demo_is_fail_closed_and_uses_the_cuda_overlay_directly() -> None:
    script = _read("scripts/demo/run_demo.ps1")

    assert ".venv-win\\Scripts\\python.exe" in script
    assert "uv sync" not in script
    assert "--host" in script and "127.0.0.1" in script
    assert "--port" in script and "7860" in script
    assert "--share" not in script
    assert "IMP_LOOP206_PRIOR" in script
    assert "IMP_LOOP206_PRIOR_RECEIPT" in script
    assert "Remove-Item Env:" in script
    assert "& $PythonExe -m lesion_robustness.demo.app" in script
    assert "$LASTEXITCODE" in script


def test_run_demo_preflight_covers_every_release_binding() -> None:
    script = _read("scripts/demo/run_demo.ps1")

    for token in (
        "validate_registry",
        "PINNED_REGISTRY",
        "FixedCacheExpectations.loop206",
        "DATASET_INDEX_SHA256",
        "LIVE_CONFIG_SHA256",
        "candidate_manifest_sha256",
        "candidate_data_sha256",
        "zero_manifest_sha256",
        "zero_data_sha256",
        "checkpoint_sha256",
        "historical_cache_provenance_drift",
        "train_screen / exact_fixed_cache / historical_cache_provenance_drift",
        "candidate_upload_authorized=false",
        "parity=0/76",
        "f6ed2eace90c49ee1b9f0c122e736920791b6301035bf8905c6a0ce27b755f32",
        'evidence["sources"]',
    ):
        assert token in script
    assert re.search(r"evidence_registry\.json", script)
    assert re.search(r"loop206_dataset_index\.json", script)
    assert "Write-Error $_" not in script


def test_tunnel_checks_health_resolves_application_and_preserves_exit_code() -> None:
    script = _read("scripts/demo/run_tunnel.ps1")

    assert "http://127.0.0.1:7860" in script
    assert "Invoke-WebRequest" in script
    assert "Get-Command" in script
    assert "-CommandType Application" in script
    assert re.search(r"tunnel\s+--url", script)
    assert "$LASTEXITCODE" in script
    assert "exit $exitCode" in script
    assert "Start-Process" not in script
    assert "--token" not in script
    assert "cloudflared.log" not in script
    assert "[string]$LocalUrl" not in script


def test_demo_runbook_documents_locked_and_fixed_cache_modes() -> None:
    readme = _read("demo/README.md")

    for token in (
        "127.0.0.1:7860",
        "run_demo.ps1",
        "run_tunnel.ps1",
        "0/76",
        "train_screen / exact_fixed_cache / historical_cache_provenance_drift",
        "arbitrary upload",
        "control-only",
        "temporary",
        "Ctrl+C",
        "non-clinical",
        "synthetic",
        "temporary upload",
        "purge",
    ):
        assert token.lower() in readme.lower()
    assert "cloudflared tunnel --url http://127.0.0.1:7860" in readme
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", readme)


def test_powershell_scripts_parse() -> None:
    shell = _powershell()
    for relative in ("scripts/demo/run_demo.ps1", "scripts/demo/run_tunnel.ps1"):
        environment = dict(os.environ)
        environment["IMP_PARSE_FILE"] = str(ROOT / relative)
        result = subprocess.run(
            [
                shell,
                "-NoProfile",
                "-Command",
                "$errors=$null; [void][System.Management.Automation.Language.Parser]::"
                "ParseFile($env:IMP_PARSE_FILE,[ref]$null,[ref]$errors); if($errors){"
                "$errors | ForEach-Object { Write-Error $_ }; exit 1}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert result.returncode == 0, f"{relative}: {result.stderr}"


def test_run_demo_preserves_missing_python_exit_code(tmp_path: Path) -> None:
    script = tmp_path / "scripts/demo/run_demo.ps1"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/demo/run_demo.ps1", script)

    result = _run_script(script, "-CheckOnly")

    assert result.returncode == 2


def test_tunnel_preserves_health_failure_exit_code() -> None:
    result = _run_script(ROOT / "scripts/demo/run_tunnel.ps1")

    assert result.returncode == 3


def test_tunnel_preserves_resolver_failure_exit_code(tmp_path: Path) -> None:
    class Healthy(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Healthy)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        script = tmp_path / "run_tunnel.ps1"
        source = _read("scripts/demo/run_tunnel.ps1").replace(
            "http://127.0.0.1:7860", f"http://127.0.0.1:{server.server_port}"
        )
        script.write_text(source, encoding="ascii")
        result = _run_script(
            script,
            "-CloudflaredPath",
            str(tmp_path / "missing/cloudflared.exe"),
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)

    assert result.returncode == 4
