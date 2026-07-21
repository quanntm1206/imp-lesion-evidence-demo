from __future__ import annotations

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
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


def _powershell_literal(value: Path | str) -> str:
    return str(value).replace("'", "''")


def _fake_python(
    tmp_path: Path, *, capture_environment: bool = False
) -> Path:
    if os.name == "nt":
        path = tmp_path / "python.cmd"
        lines = [
            "@echo off",
            'if "%1"=="-" echo preflight=passed & exit /b 0',
        ]
        if capture_environment:
            lines.extend(
                [
                    '> "%IMP_TEST_OBSERVATION%" echo GRADIO_TEMP_DIR=%GRADIO_TEMP_DIR%',
                    '>> "%IMP_TEST_OBSERVATION%" echo IMP_LOOP206_DEMO_SESSION=%IMP_LOOP206_DEMO_SESSION%',
                    '>> "%IMP_TEST_OBSERVATION%" echo TMP=%TMP%',
                    '>> "%IMP_TEST_OBSERVATION%" echo TEMP=%TEMP%',
                    '> "%GRADIO_TEMP_DIR%\\owned.tmp" echo temporary',
                ]
            )
        lines.append("exit /b %IMP_TEST_APP_EXIT%")
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path

    path = tmp_path / "python-fake"
    body = [
        "#!/usr/bin/env python3",
        "import os",
        "from pathlib import Path",
        "import sys",
        "if sys.argv[1:2] == ['-']:",
        "    print('preflight=passed')",
        "    raise SystemExit(0)",
    ]
    if capture_environment:
        body.extend(
            [
                "names = ('GRADIO_TEMP_DIR', 'IMP_LOOP206_DEMO_SESSION', 'TMP', 'TEMP')",
                "Path(os.environ['IMP_TEST_OBSERVATION']).write_text(",
                "    ''.join(f'{name}={os.environ[name]}\\n' for name in names), encoding='ascii'",
                ")",
                "(Path(os.environ['GRADIO_TEMP_DIR']) / 'owned.tmp').write_text('temporary', encoding='ascii')",
            ]
        )
    body.append("raise SystemExit(int(os.environ.get('IMP_TEST_APP_EXIT', '0')))")
    path.write_text("\n".join(body) + "\n", encoding="ascii")
    path.chmod(0o755)
    return path


def _cloudflared_pids() -> set[int]:
    result = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-Command",
            "Get-Process cloudflared -ErrorAction SilentlyContinue | "
            "Select-Object -ExpandProperty Id",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in {0, 1}, result.stderr
    if result.returncode == 1:
        assert not result.stdout.strip() and not result.stderr.strip()
    return {int(line) for line in result.stdout.splitlines() if line.strip().isdigit()}


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
    for token in (
        "Invoke-DemoLaunch",
        "Assert-OwnedSessionPath",
        "GRADIO_TEMP_DIR",
        "IMP_LOOP206_DEMO_SESSION",
        "$env:TMP",
        "$env:TEMP",
        "demo_runtime",
        "sessions",
        "[guid]::NewGuid()",
        "Remove-Item -LiteralPath $sessionPath -Recurse -Force",
    ):
        assert token in script


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
    assert "[switch]$CheckOnly" in script
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
        "launcher-owned",
        "removed automatically",
        "sibling",
    ):
        assert token.lower() in readme.lower()
    assert "cloudflared tunnel --url http://127.0.0.1:7860" in readme
    assert "provider-bound train-screen gt checkbox enables metrics" in readme.lower()
    assert "ground-truth upload enables metrics" not in readme.lower()
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


def test_run_demo_removes_only_owned_session_and_restores_environment(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    script = root / "scripts/demo/run_demo.ps1"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/demo/run_demo.ps1", script)
    runtime = root / "demo_runtime"
    runtime.mkdir()
    sentinel = runtime / "sibling-sentinel.txt"
    sentinel.write_text("preserve", encoding="ascii")
    fake_python = _fake_python(tmp_path, capture_environment=True)
    observation = tmp_path / "app-env.txt"
    command = (
        f". '{_powershell_literal(script)}'; "
        "$env:GRADIO_TEMP_DIR='before-gradio'; $env:TMP='before-tmp'; "
        "$env:TEMP='before-temp'; "
        f"$env:IMP_TEST_OBSERVATION='{_powershell_literal(observation)}'; "
        f"$code=Invoke-DemoLaunch -Device cpu -Root '{_powershell_literal(root)}' "
        f"-PythonExe '{_powershell_literal(fake_python)}'; "
        "[pscustomobject]@{code=$code;gradio=$env:GRADIO_TEMP_DIR;tmp=$env:TMP;"
        "temp=$env:TEMP} | ConvertTo-Json -Compress"
    )

    result = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "preflight=passed" in result.stdout
    state = json.loads(result.stdout.strip().splitlines()[-1])
    assert state == {
        "code": 0,
        "gradio": "before-gradio",
        "tmp": "before-tmp",
        "temp": "before-temp",
    }
    observed = dict(
        line.split("=", 1)
        for line in observation.read_text(encoding="ascii").splitlines()
    )
    assert observed["GRADIO_TEMP_DIR"] == observed["TMP"] == observed["TEMP"]
    assert observed["IMP_LOOP206_DEMO_SESSION"] == observed["GRADIO_TEMP_DIR"]
    session = Path(observed["GRADIO_TEMP_DIR"])
    assert session.parent == runtime / "sessions"
    assert session.name.startswith("demo-")
    assert not session.exists()
    assert sentinel.read_text(encoding="ascii") == "preserve"


def test_owned_session_guard_rejects_equal_outside_and_traversal_paths(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "demo_runtime"
    valid = runtime / "sessions" / "demo-00000000000000000000000000000000"
    session_parent = runtime / "sessions"
    sibling = runtime / "sibling"
    malformed = session_parent / "demo-not-a-guid"
    outside = tmp_path / "outside"
    valid.mkdir(parents=True)
    sibling.mkdir()
    malformed.mkdir()
    outside.mkdir()
    junction_runtime = tmp_path / "junction_runtime"
    junction_target = tmp_path / "junction_target"
    junction_runtime.mkdir()
    junction_target.mkdir()
    junction = junction_runtime / "sessions"
    junction_result = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-Command",
            f"New-Item -ItemType Junction -Path '{_powershell_literal(junction)}' "
            f"-Target '{_powershell_literal(junction_target)}' | Out-Null",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert junction_result.returncode == 0, junction_result.stderr
    junction_session = junction / "demo-11111111111111111111111111111111"
    (junction_target / junction_session.name).mkdir()
    command = (
        f". '{_powershell_literal(ROOT / 'scripts/demo/run_demo.ps1')}'; "
        f"$runtime='{_powershell_literal(runtime)}'; "
        f"$valid='{_powershell_literal(valid)}'; "
        f"$sessionParent='{_powershell_literal(session_parent)}'; "
        f"$sibling='{_powershell_literal(sibling)}'; "
        f"$malformed='{_powershell_literal(malformed)}'; "
        f"$outside='{_powershell_literal(outside)}'; "
        f"$junctionRuntime='{_powershell_literal(junction_runtime)}'; "
        f"$junctionSession='{_powershell_literal(junction_session)}'; "
        "$accepted=[bool](Assert-OwnedSessionPath -SessionPath $valid -RuntimeRoot $runtime); "
        "$cases=@(@($runtime,$runtime),@($outside,$runtime),"
        "@((Join-Path $valid '..\\..\\..\\outside'),$runtime),"
        "@($sessionParent,$runtime),@($sibling,$runtime),@($malformed,$runtime),"
        "@($junctionSession,$junctionRuntime)); $rejected=@(); foreach($case in $cases){"
        "try{Assert-OwnedSessionPath -SessionPath $case[0] -RuntimeRoot $case[1] | Out-Null;"
        "$rejected += $false}catch{$rejected += $true}}; "
        "[pscustomobject]@{accepted=$accepted;rejected=$rejected} | ConvertTo-Json -Compress"
    )

    result = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "accepted": True,
        "rejected": [True, True, True, True, True, True, True],
    }


@pytest.mark.parametrize(("app_exit", "expected"), [(0, 5), (7, 7)])
def test_cleanup_failure_is_nonzero_and_preserves_existing_app_failure(
    tmp_path: Path, app_exit: int, expected: int
) -> None:
    root = tmp_path / "repo"
    (root / "demo_runtime").mkdir(parents=True)
    fake_python = _fake_python(tmp_path)
    script = ROOT / "scripts/demo/run_demo.ps1"
    command = (
        f". '{_powershell_literal(script)}'; "
        f"$env:IMP_TEST_APP_EXIT='{app_exit}'; $script:guardCalls=0; "
        "function Assert-OwnedSessionPath { param($SessionPath,$RuntimeRoot) "
        "$script:guardCalls++; if($script:guardCalls -gt 1){throw 'simulated cleanup failure'} "
        "[pscustomobject]@{RuntimeRoot=$RuntimeRoot;SessionPath=$SessionPath} }; "
        f"Invoke-DemoLaunch -Device cpu -Root '{_powershell_literal(root)}' "
        f"-PythonExe '{_powershell_literal(fake_python)}'"
    )

    result = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip().splitlines()[-1]) == expected
    assert "cleanup failed closed" in result.stderr.lower()


def test_tunnel_preserves_health_failure_exit_code(tmp_path: Path) -> None:
    script = tmp_path / "run_tunnel.ps1"
    source = _read("scripts/demo/run_tunnel.ps1").replace(
        "http://127.0.0.1:7860", "http://127.0.0.1:1"
    )
    script.write_text(source, encoding="ascii")

    result = _run_script(script, "-CheckOnly")

    assert result.returncode == 3


def test_tunnel_check_only_never_invokes_cloudflared_when_health_is_live(
    tmp_path: Path,
) -> None:
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
        before = _cloudflared_pids()
        result = _run_script(
            script,
            "-CheckOnly",
            "-CloudflaredPath",
            str(tmp_path / "must-not-be-resolved/cloudflared.exe"),
        )
        after = _cloudflared_pids()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)

    assert result.returncode == 0, result.stderr
    assert "tunnel was not started" in result.stdout
    assert after == before


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
