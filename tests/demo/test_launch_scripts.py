from __future__ import annotations

from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import shutil
import subprocess
import sys
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


def _run_recovery_function_harness(
    function_names: tuple[str, ...], body: str
) -> subprocess.CompletedProcess[str]:
    names = ",".join(f"'{name}'" for name in function_names)
    script = _powershell_literal(ROOT / "scripts/demo/recover_nnunet_artifacts.ps1")
    command = (
        f"$tokens=$null; $errors=$null; $ast=[Management.Automation.Language.Parser]::"
        f"ParseFile('{script}',[ref]$tokens,[ref]$errors); if($errors){{throw $errors[0]}}; "
        f"foreach($name in @({names})){{$definition=$ast.Find({{param($node) "
        "$node -is [Management.Automation.Language.FunctionDefinitionAst] -and "
        "$node.Name -ceq $name},$true); if($null -eq $definition){throw "
        "\"missing function: $name\"}; Invoke-Expression $definition.Extent.Text}; "
        + body
    )
    return subprocess.run(
        [_powershell(), "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )


WSL_INSPECTION_TOOLS = (
    "sh",
    "lsblk",
    "mount",
    "umount",
    "cp",
    "awk",
    "wslpath",
    "mountpoint",
    "mkdir",
    "sed",
    "head",
    "tr",
    "dirname",
)


def _wsl_probe_lines(*, uid: int = 0, missing: tuple[str, ...] = ()) -> str:
    lines = [f"'uid={uid}'"]
    lines.extend(
        f"'{('missing' if tool in missing else 'tool')}={tool}'"
        for tool in WSL_INSPECTION_TOOLS
    )
    return "@(" + ",".join(lines) + ")"


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


def test_recovery_script_is_read_only_and_forbids_distro_mutation() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")

    for token in (
        "-ReadOnly",
        "--bare",
        "ro,noload",
        "verify_nnunet_bundle.py",
        "Dismount-VHD",
        "finally",
        "source_vhd_proof",
    ):
        assert token in script
    for forbidden in (
        "--unregister",
        "--import-in-place",
        "Resize-VHD",
        "Optimize-VHD",
        "Remove-Item",
    ):
        assert forbidden not in script


def test_recovery_container_backend_precedes_administrator_attach() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")
    workflow = script[script.index("function Invoke-AutomaticRecovery") :]

    assert workflow.index("Get-ContainerRecoveryContext") < workflow.index(
        "Assert-Administrator"
    )
    assert workflow.index("Invoke-ContainerRecovery") < workflow.index(
        "Assert-Administrator"
    )


def test_recovery_marks_original_environment_unavailable() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")

    for token in (
        "environment_status = 'reconstructed_required'",
        "original_transitive_package_lock = 'unavailable'",
        "2.8.1",
        "3e9fdc5fec7c8164f8fc2c6263af8be73278130e",
        "Task 4",
        "checkpoint load",
        "output replay",
    ):
        assert token in script


def test_recovery_container_contract_is_exact_and_readonly() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")

    for token in (
        "alpine:3.22@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce",
        "7zip=24.09-r0",
        "docker.exe",
        "--rm",
        "readonly",
        "Headers Error",
        "__IMP_7Z_BODY_BEGIN",
        "__IMP_7Z_BODY_END",
        "__IMP_7Z_EXIT=",
        "267947879",
        "home/admin_mugen/imp_cache/external_repos/loop170/nnUNet/pyproject.toml",
        "home/admin_mugen/imp_cache/external_repos/loop170/nnUNet/nnunetv2.egg-info/PKG-INFO",
        "home/admin_mugen/imp_cache/external_repos/loop170/nnUNet/.git/HEAD",
        "home/admin_mugen/imp_cache/external_repos/loop170/nnUNet/.git/refs/heads/master",
    ):
        assert token in script
    assert "--privileged" not in script
    assert "--device" not in script
    assert "Mount-VHD -Path $resolvedVhd -ReadOnly" in script
    assert "mount -t ext4 -o ro,noload" in script


def test_recovery_checks_installed_7zip_without_repository_indexes() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")

    assert (
        "apk info -v --installed 7zip | grep -Fx '7zip-24.09-r0'"
        in script
    )
    assert "apk info -v 7zip |" not in script


def test_recovery_isolates_7zip_from_shell_script_stdin() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")
    command = '7zz e -y -o/output /input/source.vhdx "$@"'

    assert (
        command
        + " </dev/null >/tmp/7zip.stdout 2>/tmp/7zip.stderr"
        in script
    )
    assert command + " >/tmp/7zip.stdout 2>/tmp/7zip.stderr" not in script


def _run_trusted_python(path: Path) -> subprocess.CompletedProcess[str]:
    return _run_recovery_function_harness(
        ("Resolve-TrustedPython",),
        f"Resolve-TrustedPython -ExplicitPath '{_powershell_literal(path)}'",
    )


def test_recovery_accepts_explicit_trusted_python(tmp_path: Path) -> None:
    python_exe = tmp_path / "python.exe"
    python_exe.write_text("trusted", encoding="ascii")

    result = _run_trusted_python(python_exe)

    assert result.returncode == 0, result.stderr
    assert os.path.normcase(result.stdout.strip()) == os.path.normcase(
        str(python_exe.resolve())
    )


def test_recovery_rejects_missing_trusted_python(tmp_path: Path) -> None:
    result = _run_trusted_python(tmp_path / "python.exe")

    assert result.returncode != 0
    assert "trusted Python executable unavailable" in result.stderr


def test_recovery_rejects_reparse_trusted_python(tmp_path: Path) -> None:
    python_exe = tmp_path / "python.exe"
    python_exe.write_text("reparse", encoding="ascii")
    path = _powershell_literal(python_exe)
    result = _run_recovery_function_harness(
        ("Resolve-TrustedPython",),
        "function Get-Item { [pscustomobject]@{PSIsContainer=$false;"
        "Attributes=[IO.FileAttributes]::ReparsePoint} }; "
        f"Resolve-TrustedPython -ExplicitPath '{path}'",
    )

    assert result.returncode != 0
    assert "must not be a reparse point" in result.stderr


def test_recovery_rejects_non_python_executable(tmp_path: Path) -> None:
    executable = tmp_path / "python3.exe"
    executable.write_text("wrong name", encoding="ascii")

    result = _run_trusted_python(executable)

    assert result.returncode != 0
    assert "filename must be exactly python.exe" in result.stderr


def test_recovery_python_parameter_wires_cli_after_safety_checks() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")
    container = script[
        script.index("function Invoke-ContainerRecovery") : script.index(
            "function Invoke-WindowsAttachRecovery"
        )
    ]
    windows = script[
        script.index("function Invoke-WindowsAttachRecovery") : script.index(
            "function Invoke-AutomaticRecovery"
        )
    ]

    assert script.index("[string]$PythonExe = ''") < script.index("Set-StrictMode")
    assert script.count("Resolve-TrustedPython -ExplicitPath $PythonExe") == 2
    assert "$verifierCode" not in script
    assert "-c $verifierCode" not in script
    assert script.count("Invoke-TrustedBundleVerifier") == 3
    assert container.index(
        "Assert-ExactRecoveryFiles -Root $resolvedOutput -Expected $rawNames"
    ) < container.index("Resolve-TrustedPython -ExplicitPath $PythonExe")
    assert windows.index(
        "Assert-SnapshotUnchanged -Before $before -After $after"
    ) < windows.index("Resolve-TrustedPython -ExplicitPath $PythonExe")
    resolver = script[
        script.index("function Resolve-TrustedPython") : script.index(
            "function Get-VhdSnapshot"
        )
    ]
    assert "Get-Command" not in resolver
    assert script.count(".venv-win\\Scripts\\python.exe") == 1
    helper = script[
        script.index("function Invoke-TrustedBundleVerifier") : script.index(
            "function Write-RuntimeIdentity"
        )
    ]
    for token in (
        "--bundle",
        "--report '-'",
        "--receipt",
        "ConvertTo-Json -Depth 8 -Compress",
        "| & $TrustedPython $VerifierPath",
    ):
        assert token in helper
    assert ".verification-report" not in helper
    assert "FileStream" not in helper
    assert "[IO.File]::Delete" not in helper


def test_recovery_verifier_cli_avoids_inline_argv_and_cleans_owned_report(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    result = _run_recovery_function_harness(
        ("Invoke-TrustedBundleVerifier",),
        "Invoke-TrustedBundleVerifier "
        f"-TrustedPython '{_powershell_literal(sys.executable)}' "
        f"-VerifierPath '{_powershell_literal(ROOT / 'scripts/demo/verify_nnunet_bundle.py')}' "
        f"-OutputRoot '{_powershell_literal(bundle)}' "
        "-VerificationReport ([ordered]@{})",
    )

    assert result.returncode != 0
    assert "Loop192 bundle verification failed with exit code" in result.stderr
    assert "SyntaxError" not in result.stderr
    assert "RuntimeError(unable" not in result.stderr
    assert not list(bundle.glob(".verification-report*.json"))


def test_recovery_plan_passes_explicit_trusted_python() -> None:
    plan = _read("docs/superpowers/plans/2026-07-21-dual-live-demo.md")
    argument = "-PythonExe 'E:\\0. IMP\\.venv-win\\Scripts\\python.exe'"

    assert plan.count(argument) >= 2


def test_recovery_container_arguments_lock_mounts_and_image() -> None:
    result = _run_recovery_function_harness(
        ("New-ContainerRecoveryArguments",),
        "$value=@(New-ContainerRecoveryArguments -VhdPath 'E:\\source.vhdx' "
        "-OutputRoot 'E:\\fresh output'); $value | ConvertTo-Json -Compress",
    )

    assert result.returncode == 0, result.stderr
    arguments = json.loads(result.stdout.strip())
    command = " ".join(arguments)
    assert arguments[:3] == ["run", "--rm", "-i"]
    assert "source=E:\\source.vhdx,target=/input/source.vhdx,readonly" in command
    assert "source=E:\\fresh output,target=/output" in command
    assert "--privileged" not in arguments and "--device" not in arguments
    assert arguments[-4:] == [
        "alpine:3.22@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce",
        "sh",
        "-s",
        "--",
    ]


def _run_container_parser_lines(lines: list[str]) -> subprocess.CompletedProcess[str]:
    values = ",".join("'" + line.replace("'", "''") + "'" for line in lines)
    return _run_recovery_function_harness(
        ("Assert-ContainerParserResult",),
        "$value=Assert-ContainerParserResult -ProcessResult ([pscustomobject]@{"
        f"ExitCode=0;Lines=@({values})}}); $value",
    )


def test_recovery_parser_accepts_exact_locked_diagnostics() -> None:
    result = _run_container_parser_lines(
        [
            "recovery_7zip=7zip-24.09-r0",
            "apk output outside parser body",
            "__IMP_7Z_BODY_BEGIN",
            "WARNINGS:",
            "Headers Error",
            "WARNINGS:",
            "Headers Error",
            "Archives with Warnings: 1",
            "Warnings: 1",
            "__IMP_7Z_BODY_END",
            "__IMP_7Z_EXIT=0",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Headers Error"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_marker", "exactly one __IMP_7Z_EXIT=0"),
        ("duplicate_marker", "exactly one __IMP_7Z_EXIT=0"),
        ("wrong_code", "exactly one __IMP_7Z_EXIT=0"),
        ("remove_duplicate", "diagnostic multiset"),
        ("add_third", "diagnostic multiset"),
        ("error_summary", "diagnostic multiset"),
        ("missing_summary", "diagnostic multiset"),
        ("extra_summary", "diagnostic multiset"),
    ],
)
def test_recovery_parser_rejects_diagnostic_drift(
    mutation: str, expected: str
) -> None:
    lines = [
        "recovery_7zip=7zip-24.09-r0",
        "__IMP_7Z_BODY_BEGIN",
        "WARNINGS:",
        "Headers Error",
        "WARNINGS:",
        "Headers Error",
        "Archives with Warnings: 1",
        "Warnings: 1",
        "__IMP_7Z_BODY_END",
        "__IMP_7Z_EXIT=0",
    ]
    if mutation == "missing_marker":
        lines.pop()
    elif mutation == "duplicate_marker":
        lines.append("__IMP_7Z_EXIT=0")
    elif mutation == "wrong_code":
        lines[-1] = "__IMP_7Z_EXIT=2"
    elif mutation == "remove_duplicate":
        lines.remove("Headers Error")
    elif mutation == "add_third":
        lines.insert(lines.index("__IMP_7Z_BODY_END"), "Headers Error")
    elif mutation == "error_summary":
        lines[lines.index("Archives with Warnings: 1")] = "Archives with Errors: 1"
    elif mutation == "missing_summary":
        lines.remove("Warnings: 1")
    else:
        lines.insert(lines.index("__IMP_7Z_BODY_END"), "Open Warnings: 1")

    result = _run_container_parser_lines(lines)

    assert result.returncode != 0
    assert expected in result.stderr


def test_recovery_container_context_uses_mocked_exact_docker() -> None:
    image = (
        "alpine:3.22@sha256:"
        "14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce"
    )
    repo_digest = image.replace(":3.22@", "@")
    result = _run_recovery_function_harness(
        ("Get-ContainerRecoveryContext",),
        "$script:calls=New-Object 'System.Collections.Generic.List[string]'; "
        "function Get-Command { [pscustomobject]@{Name='docker.exe';"
        "CommandType='Application';Source='C:\\Docker\\docker.exe'} }; "
        "function Test-Path { $true }; "
        "function Invoke-DockerCommand { param($DockerPath,$Arguments,$Label) "
        "$script:calls.Add(($Arguments -join ' ')); if($Arguments[0] -eq 'version'){"
        "[pscustomobject]@{ExitCode=0;Lines=@('27.0.0')}}else{"
        f"[pscustomobject]@{{ExitCode=0;Lines=@('{repo_digest}')}}}} }}; "
        "$context=Get-ContainerRecoveryContext; [pscustomobject]@{"
        "path=$context.DockerPath;image=$context.Image;calls=@($script:calls)} | "
        "ConvertTo-Json -Depth 4 -Compress",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["path"] == r"C:\Docker\docker.exe"
    assert payload["image"] == image
    assert payload["calls"][0].startswith("version")
    assert payload["calls"][1].startswith("image inspect")


def test_recovery_writes_honest_reconstructed_runtime_identity(tmp_path: Path) -> None:
    source = {
        "pyproject.toml": '[project]\nversion = "2.8.1"\n',
        "PKG-INFO": "Name: nnunetv2\nVersion: 2.8.1\n",
        "HEAD": "ref: refs/heads/master\n",
        "master": "3e9fdc5fec7c8164f8fc2c6263af8be73278130e\n",
    }
    for name, value in source.items():
        (tmp_path / name).write_text(value, encoding="ascii")
    root = _powershell_literal(tmp_path)
    result = _run_recovery_function_harness(
        ("Get-FileIdentity", "Write-Utf8NoBom", "Write-ReconstructedRuntimeIdentity"),
        f"Write-ReconstructedRuntimeIdentity -Root '{root}' -Backend "
        "'container-readonly-7zip' -ParserWarning 'Headers Error'",
    )

    assert result.returncode == 0, result.stderr
    identity = json.loads((tmp_path / "runtime_identity.json").read_text())
    lock = (tmp_path / "requirements.lock").read_text()
    assert identity["environment_status"] == "reconstructed_required"
    assert identity["source_identity"]["version"] == "2.8.1"
    assert identity["source_identity"]["git_commit"] == (
        "3e9fdc5fec7c8164f8fc2c6263af8be73278130e"
    )
    assert identity["original_transitive_package_lock"] == "unavailable"
    assert "Task 4" in lock and "full transitive lock" in lock
    assert "nnunetv2 @ git+https://github.com/MIC-DKFZ/nnUNet.git@3e9fdc5" in lock
    assert "torch==" not in lock and "numpy==" not in lock


def test_recovery_wsl_path_keeps_single_line_as_string() -> None:
    result = _run_recovery_function_harness(
        ("ConvertTo-WslPath",),
        "function Invoke-WslContext { '/mnt/e/path with spaces' }; "
        "$value=ConvertTo-WslPath -ContextPrefix @('--system','--') "
        "-WindowsPath 'E:\\path with spaces'; "
        "[pscustomobject]@{type=$value.GetType().FullName;value=$value} | "
        "ConvertTo-Json -Compress",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == {
        "type": "System.String",
        "value": "/mnt/e/path with spaces",
    }


def test_recovery_cleanup_attempts_every_layer_after_partial_failures() -> None:
    result = _run_recovery_function_harness(
        ("Invoke-RecoveryCleanup",),
        "$script:calls=New-Object 'System.Collections.Generic.List[string]'; "
        "function Invoke-WslContextScript { $script:calls.Add('filesystem'); "
        "throw 'filesystem cleanup failed' }; "
        "function global:wsl.exe { $script:calls.Add('wsl'); "
        "$global:LASTEXITCODE=9; 'not attached' }; "
        "function Get-DiskImage { $script:calls.Add('query'); "
        "[pscustomobject]@{Attached=$true} }; "
        "function Dismount-VHD { $script:calls.Add('vhd') }; "
        "$cleanup=@(Invoke-RecoveryCleanup -FilesystemMountAttempted $true "
        "-WslAttachAttempted $true -VhdMountAttempted $true "
        "-WslContextPrefix @('--system','--') "
        "-LinuxMount '/mnt/wsl/recovery' -PhysicalDrive '\\\\.\\PHYSICALDRIVE9' "
        "-ResolvedVhd 'E:\\source.vhdx'); "
        "[pscustomobject]@{calls=@($script:calls);errors=$cleanup} | "
        "ConvertTo-Json -Depth 4 -Compress",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["calls"] == ["filesystem", "wsl", "query", "vhd"]
    errors = "\n".join(payload["errors"])
    assert "filesystem cleanup failed" in errors
    assert "WSL disk detach failed" in errors


def test_recovery_preserves_operation_and_cleanup_failures() -> None:
    result = _run_recovery_function_harness(
        ("Assert-RecoveryCompleted",),
        "$message=$null; try { Assert-RecoveryCompleted "
        "-OperationError ([InvalidOperationException]::new('primary failure')) "
        "-CleanupErrors @('secondary failure') } catch { $message=$_.Exception.Message }; "
        "$message",
    )

    assert result.returncode == 0, result.stderr
    assert "primary failure" in result.stdout
    assert "secondary failure" in result.stdout


def test_recovery_prefers_available_system_context() -> None:
    result = _run_recovery_function_harness(
        ("Resolve-WslInspectionContext",),
        "$script:calls=New-Object 'System.Collections.Generic.List[string]'; "
        "function Invoke-WslContextScript { param($ContextPrefix,$Script,$Arguments,$Label) "
        "$script:calls.Add(($ContextPrefix -join ' ')); "
        + _wsl_probe_lines()
        + " }; $context=Resolve-WslInspectionContext; "
        "[pscustomobject]@{name=$context.Name;prefix=@($context.Prefix);"
        "calls=@($script:calls)} | ConvertTo-Json -Depth 4 -Compress",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == {
        "name": "system",
        "prefix": ["--system", "--"],
        "calls": ["--system --"],
    }


def test_recovery_uses_validated_docker_desktop_when_system_fails() -> None:
    result = _run_recovery_function_harness(
        ("Resolve-WslInspectionContext",),
        "$script:calls=New-Object 'System.Collections.Generic.List[string]'; "
        "function Invoke-WslContextScript { param($ContextPrefix,$Script,$Arguments,$Label) "
        "$key=$ContextPrefix -join ' '; $script:calls.Add($key); "
        "if($key -eq '--system --'){throw 'system unavailable'}; "
        + _wsl_probe_lines()
        + " }; $context=Resolve-WslInspectionContext; "
        "[pscustomobject]@{name=$context.Name;prefix=@($context.Prefix);"
        "calls=@($script:calls)} | ConvertTo-Json -Depth 4 -Compress",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == {
        "name": "docker-desktop",
        "prefix": ["-d", "docker-desktop", "-u", "root", "--"],
        "calls": ["--system --", "-d docker-desktop -u root --"],
    }


@pytest.mark.parametrize(
    ("fallback_lines", "expected"),
    [
        (_wsl_probe_lines(uid=1000), "uid 0"),
        (_wsl_probe_lines(missing=("awk",)), "required commands"),
    ],
)
def test_recovery_rejects_invalid_docker_desktop_fallback(
    fallback_lines: str, expected: str
) -> None:
    result = _run_recovery_function_harness(
        ("Resolve-WslInspectionContext",),
        "function Invoke-WslContextScript { param($ContextPrefix,$Script,$Arguments,$Label) "
        "if(($ContextPrefix -join ' ') -eq '--system --'){throw 'system unavailable'}; "
        + fallback_lines
        + " }; Resolve-WslInspectionContext",
    )

    assert result.returncode != 0
    assert expected in result.stderr


def test_recovery_rejects_unavailable_fallback() -> None:
    result = _run_recovery_function_harness(
        ("Resolve-WslInspectionContext",),
        "function Invoke-WslContextScript { throw (($ContextPrefix -join ' ') + "
        "' unavailable') }; Resolve-WslInspectionContext",
    )

    assert result.returncode != 0
    assert "system" in result.stderr
    assert "docker-desktop" in result.stderr


def test_recovery_routes_linux_commands_through_locked_context() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")

    system = "@('--system', '--')"
    fallback = "@('-d', 'docker-desktop', '-u', 'root', '--')"
    assert system in script and fallback in script
    assert script.index(system) < script.index(fallback)
    assert "Resolve-WslInspectionContext" in script
    assert "Invoke-WslSystem" not in script
    assert "Invoke-WslSystemScript" not in script
    assert "wsl.exe --shutdown" not in script
    for token in (
        "pre-attachment block inventory",
        "post-attachment block inventory",
        "read-only ext4 mount",
        "ext4 read-only mount proof",
        "artifact copy:",
        "package identity copy:",
        "output path conversion",
        "ext4 unmount",
    ):
        position = script.index(token)
        nearby = script[max(0, position - 220) : position + 220]
        assert "ContextPrefix" in nearby


def test_recovery_design_documents_locked_inspection_fallback() -> None:
    design = _read("docs/superpowers/specs/2026-07-21-dual-live-demo-design.md").lower()

    for token in (
        "system distribution",
        "preferred",
        "`docker-desktop`",
        "root-only",
        "fallback",
        "preflight",
    ):
        assert token in design


def test_recovery_design_documents_container_parser_proof_and_replay_gate() -> None:
    design = _read("docs/superpowers/specs/2026-07-21-dual-live-demo-design.md").lower()

    for token in (
        "container parser",
        "registered vhd",
        "read-only bind",
        "no block mount",
        "journal replay",
        "pinned artifact hashes",
        "reconstructed",
        "checkpoint load",
        "output replay",
    ):
        assert token in design


def test_recovery_marks_attempts_before_mutating_commands() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")

    assert script.index("$vhdMountAttempted = $true") < script.index(
        "$mountedVhd = Mount-VHD"
    )
    assert script.index("$wslAttachAttempted = $true") < script.index(
        "@('--mount', $physicalDrive, '--bare')"
    )
    assert script.index("$filesystemMountAttempted = $true") < script.index(
        "-Label 'read-only ext4 mount'"
    )
    assert 'if mountpoint -q -- "$1"; then' in script
    assert 'umount -- "$1"' in script


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
    for relative in (
        "scripts/demo/run_demo.ps1",
        "scripts/demo/run_tunnel.ps1",
        "scripts/demo/recover_nnunet_artifacts.ps1",
    ):
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
