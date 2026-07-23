from __future__ import annotations

from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace

import numpy as np

import pytest

from lesion_robustness.release_manifest import launcher_projection
from lesion_robustness.demo.app import create_app
from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    PROTOCOL_ID,
    rgb_sha256,
)
from lesion_robustness.demo.dual_live_service import DualLiveArm, DualLiveResult
from lesion_robustness.demo.live_inputs import synthetic_evidence
from lesion_robustness.demo.presentation import build_dual_live_receipt
from lesion_robustness.release_manifest import load_release_manifest, runtime_projection


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _copy_release_manifest(root: Path) -> None:
    release = root / "release"
    release.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "release/imp_release_manifest.json", release / "imp_release_manifest.json")


def _release_digest() -> str:
    return launcher_projection()["release_manifest_sha256"]


def test_preserve_mode_gates_launcher_deletion_primitives() -> None:
    for relative in (
        "scripts/demo/run_sidecar.ps1",
        "scripts/demo/run_demo.ps1",
        "scripts/demo/run_tunnel.ps1",
        "scripts/demo/stop_demo.ps1",
    ):
        assert "PreserveMode" in _read(relative)

    assert "if (-not $PreserveMode)" in _read("scripts/demo/run_sidecar.ps1")
    assert "if (-not $PreserveMode)" in _read("scripts/demo/run_demo.ps1")
    assert "if (-not $PreserveMode)" in _read("scripts/demo/run_tunnel.ps1")
    stop = _read("scripts/demo/stop_demo.ps1")
    assert "if (-not $PreserveMode)" in stop
    assert "stopped.json" in stop


def test_preserve_gradio_owner_restart_uses_unique_records_and_keeps_history(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    session = root / "demo_runtime/sessions/demo-0123456789abcdef0123456789abcdef"
    session.mkdir(parents=True)
    python_exe = tmp_path / "python.exe"
    python_exe.write_text("fake", encoding="ascii")
    stopped = root / "demo_runtime/preserved/run-a/gradio/old.stopped.json"
    stopped.parent.mkdir(parents=True)
    stopped.write_text("historical", encoding="ascii")
    body = (
        "function Remove-Item { throw 'delete forbidden' }; "
        f"$first=Write-GradioOwnerRecord -Root '{_powershell_literal(root)}' "
        f"-PythonExe '{_powershell_literal(python_exe)}' "
        f"-SessionPath '{_powershell_literal(session)}' -PreserveMode -RunId 'run-a'; "
        f"$second=Write-GradioOwnerRecord -Root '{_powershell_literal(root)}' "
        f"-PythonExe '{_powershell_literal(python_exe)}' "
        f"-SessionPath '{_powershell_literal(session)}' -PreserveMode -RunId 'run-a'; "
        "[pscustomobject]@{different=($first -cne $second);"
        "first=(Test-Path -LiteralPath $first)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "different": True,
        "first": True,
    }
    assert stopped.read_text(encoding="ascii") == "historical"


def test_preserve_tunnel_owner_restart_uses_unique_records(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "demo_runtime").mkdir(parents=True)
    body = (
        "function Remove-Item { throw 'delete forbidden' }; "
        f"$digest='{_release_digest()}'; "
        f"$first=Get-TunnelOwnerRecordPath -Root '{_powershell_literal(root)}' "
        "-PreserveMode -RunId 'run-a'; "
        f"$second=Get-TunnelOwnerRecordPath -Root '{_powershell_literal(root)}' "
        "-PreserveMode -RunId 'run-a'; "
        "Write-TunnelOwnerRecord -RecordPath $first -ProcessId 42 "
        "-ProcessStartTimeUtc '2026-07-22T01:02:03.1234567Z' "
        "-OwnerNonce '0123456789abcdef0123456789abcdef' -ExecutablePath 'E\u003a\u005ccloudflared.exe' "
        "-ReleaseManifestSha256 $digest; "
        "Write-TunnelOwnerRecord -RecordPath $second -ProcessId 43 "
        "-ProcessStartTimeUtc '2026-07-22T01:02:04.1234567Z' "
        "-OwnerNonce 'fedcba9876543210fedcba9876543210' -ExecutablePath 'E\u003a\u005ccloudflared.exe' "
        "-ReleaseManifestSha256 $digest; "
        "[pscustomobject]@{different=($first -cne $second);first=(Test-Path -LiteralPath $first);"
        "second=(Test-Path -LiteralPath $second)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "different": True,
        "first": True,
        "second": True,
    }


def test_preserve_sidecar_uses_unique_name_without_rm_and_journals_exact_name(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "demo_runtime/nnunet"
    runtime.mkdir(parents=True)
    body = (
        "$one=New-SidecarContainerName -RunId 'run-a' "
        "-OwnerToken '0123456789abcdef0123456789abcdef'; "
        "$two=New-SidecarContainerName -RunId 'run-a' "
        "-OwnerToken 'fedcba9876543210fedcba9876543210'; "
        "$arguments=@(New-SidecarRunArguments -BundlePath 'E\u003a\u005cbundle' "
        "-OwnerToken '0123456789abcdef0123456789abcdef' -ContainerName $one -PreserveMode); "
        f"$context=[pscustomobject]@{{Root='{_powershell_literal(tmp_path)}';RuntimeRoot='{_powershell_literal(runtime)}';"
        "PreserveMode=$true;PreserveRunId='run-a';ContainerName=$one;OwnerToken='0123456789abcdef0123456789abcdef';DockerPath='E\u003a\u005cdocker.exe'}; "
        "$record=Write-SidecarOwnerRecord -Context $context -ContainerId ('a' * 64) -PreserveMode; "
        "$payload=Get-Content -LiteralPath $record -Raw | ConvertFrom-Json; "
        "[pscustomobject]@{different=($one -cne $two);has_rm=($arguments -contains '--rm');"
        "bound=($payload.container_name -ceq $one);record=(Test-Path -LiteralPath $record)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "different": True,
        "has_rm": False,
        "bound": True,
        "record": True,
    }


def test_task7_uses_launcher_generated_owner_bound_container_names() -> None:
    script = _read("scripts/demo/run_sidecar.ps1")
    runbook = _read("docs/runbooks/demo-operations.md")
    deployment_guide = _read("DEMO_DEPLOYMENT_GUIDE.md")
    top_level_parameters = script[: script.index("$ErrorActionPreference")]

    assert "[string]$ContainerName" not in top_level_parameters
    for document in (runbook, deployment_guide):
        assert "-ContainerName" not in document
    assert "owner-bound" in runbook.lower()
    assert "New-SidecarContainerName -RunId $RunId -OwnerToken $ownerToken" in script


def test_task7_documented_run_id_expression_passes_exact_validators() -> None:
    documents = (
        _read("docs/runbooks/demo-operations.md"),
        _read("docs/runbooks/two-machine-delivery.md"),
        _read("DEMO_DEPLOYMENT_GUIDE.md"),
    )
    expected = (
        "$RunId = (Get-Date).ToUniversalTime()"
        ".ToString('yyyyMMddTHHmmssfffffffZ').ToLowerInvariant()"
    )
    for document in documents:
        assert expected in document

    for script in (
        "scripts/demo/run_sidecar.ps1",
        "scripts/demo/run_demo.ps1",
        "scripts/demo/run_tunnel.ps1",
        "scripts/demo/stop_demo.ps1",
    ):
        body = (
            f"{expected}; "
            "$validated=Assert-PreserveRunId -RunId $RunId; "
            "$uppercaseRejected=$false; try { "
            "Assert-PreserveRunId -RunId '20260722T1234561234567Z' | Out-Null "
            "} catch { $uppercaseRejected=$true }; "
            "[pscustomobject]@{run_id=$validated;uppercase_rejected=$uppercaseRejected} "
            "| ConvertTo-Json -Compress"
        )
        result = _run_launcher_function_harness(script, body)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert re.fullmatch(r"[0-9]{8}t[0-9]{13}z", payload["run_id"])
        assert payload["uppercase_rejected"] is True


def test_preserved_check_only_runbook_requires_retained_stopped_identity() -> None:
    runbook = _read("docs/runbooks/demo-operations.md").lower()
    normalized_runbook = re.sub(r"\s+", " ", runbook)

    assert "state.running=false" in runbook
    assert "absence is accepted only for non-preserve auto-remove" in normalized_runbook
    assert "exact container to be absent" not in runbook


def test_preserved_check_only_stops_proves_closed_identity_and_journals_stop() -> None:
    root = _powershell_literal(ROOT)
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        f"function Resolve-SidecarContext {{ [pscustomobject]@{{Root='{root}';RuntimeRoot='RR';"
        "BundlePath='B';DockerPath='D';OwnerToken='0123456789abcdef0123456789abcdef';"
        "ContainerName='imp-nnunet-sidecar-run-a';PreserveMode=$true;"
        "PreserveRunId='run-a';OwnerRecordPath='RR\u005cowner-current.json'} }; "
        "function Assert-VerifiedBundle {}; function Assert-PinnedDockerImage {}; "
        "function Start-OwnedSidecar { $script:events.Add('start'); 'a' * 64 }; "
        "function Wait-PinnedSidecarHealth { $script:events.Add('health') }; "
        "function Stop-OwnedSidecar { $script:events.Add('stop') }; "
        "function Test-OwnedSidecarStoppedOrAbsent { $script:events.Add('identity'); $true }; "
        "function Wait-SidecarPortClosed { $script:events.Add('port') }; "
        "function Write-PreserveSidecarLifecycleRecord { param($Event); "
        "$script:events.Add(('journal:' + $Event)); 'stop.stopped.json' }; "
        "$code=Invoke-SidecarLaunch -CheckOnly -PreserveMode -RunId 'run-a'; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "code": 0,
        "events": ["start", "health", "stop", "identity", "port", "journal:stopped"],
    }


def test_preserved_health_failure_stops_and_journals_before_failing_closed() -> None:
    root = _powershell_literal(ROOT)
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        f"function Resolve-SidecarContext {{ [pscustomobject]@{{Root='{root}';RuntimeRoot='RR';"
        "BundlePath='B';DockerPath='D';OwnerToken='0123456789abcdef0123456789abcdef';"
        "ContainerName='imp-nnunet-sidecar-run-a';PreserveMode=$true;"
        "PreserveRunId='run-a';OwnerRecordPath='RR\u005cowner-current.json'} }; "
        "function Assert-VerifiedBundle {}; function Assert-PinnedDockerImage {}; "
        "function Start-OwnedSidecar { $script:events.Add('start'); 'a' * 64 }; "
        "function Wait-PinnedSidecarHealth { $script:events.Add('health'); throw 'timeout' }; "
        "function Stop-OwnedSidecar { $script:events.Add('stop') }; "
        "function Test-OwnedSidecarStoppedOrAbsent { $script:events.Add('identity'); $true }; "
        "function Wait-SidecarPortClosed { $script:events.Add('port') }; "
        "function Write-PreserveSidecarLifecycleRecord { param($Event); "
        "$script:events.Add(('journal:' + $Event)); 'stop.stopped.json' }; "
        "$code=Invoke-SidecarLaunch -CheckOnly -PreserveMode -RunId 'run-a'; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "code": 5,
        "events": ["start", "health", "stop", "identity", "port", "journal:stopped"],
    }


def test_preserved_sidecar_closed_proof_accepts_exact_stopped_identity() -> None:
    body = (
        "function Invoke-DockerCommand { @((('a' * 64) + "
        "'|0123456789abcdef0123456789abcdef|/imp-nnunet-run-a|false')) }; "
        "$closed=Test-OwnedSidecarStoppedOrAbsent -DockerPath 'D' -ContainerId ('a' * 64) "
        "-OwnerToken '0123456789abcdef0123456789abcdef' "
        "-ContainerName 'imp-nnunet-run-a'; $closed"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True"


def test_preserved_sidecar_closed_proof_rejects_absent_container() -> None:
    body = (
        "function Invoke-DockerCommand { throw 'inspect missing' }; "
        "function Test-SidecarContainerAbsent { $true }; "
        "$rejected=$false; try{Test-OwnedSidecarStoppedOrAbsent -DockerPath 'D' "
        "-ContainerId ('a' * 64) -OwnerToken ('b' * 32) "
        "-ContainerName 'imp-nnunet-run-a' -PreserveMode | Out-Null}"
        "catch{$rejected=$true}; $rejected"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True"


def test_nonpreserved_sidecar_closed_proof_accepts_auto_remove_absence() -> None:
    body = (
        "function Invoke-DockerCommand { throw 'inspect missing' }; "
        "function Test-SidecarContainerAbsent { $true }; "
        "Test-OwnedSidecarStoppedOrAbsent -DockerPath 'D' -ContainerId ('a' * 64) "
        "-OwnerToken ('b' * 32) -ContainerName 'imp-nnunet-loop192'"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True"


def test_preserved_check_only_journal_failure_fails_closed() -> None:
    root = _powershell_literal(ROOT)
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        f"function Resolve-SidecarContext {{ [pscustomobject]@{{Root='{root}';RuntimeRoot='RR';"
        "BundlePath='B';DockerPath='D';OwnerToken=('b' * 32);ContainerName='imp-nnunet-run-a';"
        "PreserveMode=$true;PreserveRunId='run-a';OwnerRecordPath='RR\u005cowner.json'} }; "
        "function Assert-VerifiedBundle {}; function Assert-PinnedDockerImage {}; "
        "function Start-OwnedSidecar { 'a' * 64 }; function Wait-PinnedSidecarHealth {}; "
        "function Stop-OwnedSidecar { $script:events.Add('stop') }; "
        "function Test-OwnedSidecarStoppedOrAbsent { $script:events.Add('identity'); $true }; "
        "function Wait-SidecarPortClosed { $script:events.Add('port') }; "
        "function Write-PreserveSidecarLifecycleRecord { $script:events.Add('journal'); throw 'write failed' }; "
        "$code=Invoke-SidecarLaunch -CheckOnly -PreserveMode -RunId 'run-a'; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "code": 5,
        "events": ["stop", "identity", "port", "journal"],
    }


def test_run_demo_uses_authorized_checkpoint_environment_overrides() -> None:
    script = _read("scripts/demo/run_demo.ps1")
    app = _read("src/lesion_robustness/demo/app.py")
    deployment_guide = _read("DEMO_DEPLOYMENT_GUIDE.md")
    top_level_parameters = script[: script.index("$ErrorActionPreference")]

    for name in (
        "IMP_LOOP206_CONTROL_CHECKPOINT",
        "IMP_LOOP206_CANDIDATE_CHECKPOINT",
    ):
        assert f"Resolve-DemoRuntimeArtifactPath -EnvironmentName '{name}'" in script
        assert re.search(rf'os\.environ\.get\(\s*"{name}"', app)
    assert "load_model_registry(" in app
    assert "environ=model_environment" in app
    assert "[string]$PythonExe" in top_level_parameters
    assert "-PythonExe $resolvedPython" in script[script.rindex("if ($MyInvocation"):]
    assert deployment_guide.count("-PythonExe $PythonExe") >= 2
    assert "Resolve-DemoPythonApplication" in script
    assert "@preflightArguments" in script
    assert "@smokeArguments" in script
    assert "@appArguments" in script


def test_demo_python_resolver_accepts_literal_space_and_metachar_path(
    tmp_path: Path,
) -> None:
    python_exe = tmp_path / "space & [literal]" / "python.exe"
    python_exe.parent.mkdir()
    python_exe.write_text("fake", encoding="ascii")
    body = (
        f"Resolve-DemoPythonApplication -ExplicitPath '{_powershell_literal(python_exe)}' "
        f"-DefaultPath '{_powershell_literal(tmp_path / 'missing/python.exe')}'"
    )
    result = _run_launcher_function_harness("scripts/demo/run_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert os.path.normcase(result.stdout.strip().splitlines()[-1]) == os.path.normcase(
        str(python_exe.resolve())
    )


@pytest.mark.parametrize("name", ["python3.exe", "python.cmd"])
def test_demo_python_resolver_rejects_wrong_filename(
    tmp_path: Path, name: str
) -> None:
    executable = tmp_path / name
    executable.write_text("fake", encoding="ascii")
    body = (
        "$rejected=$false; try{Resolve-DemoPythonApplication "
        f"-ExplicitPath '{_powershell_literal(executable)}' "
        f"-DefaultPath '{_powershell_literal(tmp_path / 'missing/python.exe')}' "
        "| Out-Null}catch{$rejected=$true}; $rejected"
    )
    result = _run_launcher_function_harness("scripts/demo/run_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True"


def test_demo_python_resolver_rejects_missing_and_reparse_paths(tmp_path: Path) -> None:
    missing = tmp_path / "python.exe"
    missing_body = (
        "$rejected=$false; try{Resolve-DemoPythonApplication "
        f"-ExplicitPath '{_powershell_literal(missing)}' "
        f"-DefaultPath '{_powershell_literal(missing)}' | Out-Null}}"
        "catch{$rejected=$true}; $rejected"
    )
    missing_result = _run_launcher_function_harness(
        "scripts/demo/run_demo.ps1", missing_body
    )
    reparse_body = (
        "function Get-Item { [pscustomobject]@{PSIsContainer=$false;"
        "Attributes=[IO.FileAttributes]::ReparsePoint;Name='python.exe';FullName='E\u003a\u005cpython.exe'} }; "
        "$rejected=$false; try{Resolve-DemoPythonApplication -ExplicitPath 'E\u003a\u005cpython.exe' "
        "-DefaultPath 'E\u003a\u005cpython.exe' | Out-Null}catch{$rejected=$true}; $rejected"
    )
    reparse_result = _run_launcher_function_harness(
        "scripts/demo/run_demo.ps1", reparse_body
    )

    assert missing_result.returncode == 0, missing_result.stderr
    assert missing_result.stdout.strip().splitlines()[-1] == "True"
    assert reparse_result.returncode == 0, reparse_result.stderr
    assert reparse_result.stdout.strip().splitlines()[-1] == "True"


def test_tunnel_rejects_test_journal_owner_schema_and_extra_field() -> None:
    digest = _release_digest()
    for schema, extra in (
        ("imp.demo.test-journal.v1", ""),
        ("imp.demo.gradio-owner.v1", ";test_journal=$true"),
    ):
        body = (
            "$record=[pscustomobject]@{"
            f"schema_version='{schema}';launcher_pid=42;"
            "launcher_start_time_utc='2026-07-22T00:00:00.0000000Z';"
            "owner_nonce=('a' * 32);python_path='E\u003a\u005cpython.exe';"
            "session_path='E\u003a\u005csession';host='127.0.0.1';port=7860;"
            f"release_manifest_sha256='{digest}';public_tunnel_mode=$true;"
            f"preserve_mode=$true{extra}}}; $rejected=$false; "
            f"try{{Assert-GradioOwnerRecord -Record $record -ExpectedReleaseManifestSha256 '{digest}'}}"
            "catch{$rejected=$true}; $rejected"
        )
        result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == "True"


def test_preserve_stop_collision_never_overwrites_existing_record(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    stop_root = root / "demo_runtime/preserved/run-a/stop"
    stop_root.mkdir(parents=True)
    collision = stop_root / "stop-collision.stopped.json"
    collision.write_text("historical", encoding="ascii")
    body = (
        "$script:nonceIndex=0; "
        "$factory={ $value=@('collision','fresh')[$script:nonceIndex]; $script:nonceIndex++; $value }; "
        f"$path=Write-PreserveStopRecord -Root '{_powershell_literal(root)}' "
        "-RunId 'run-a' -NonceFactory $factory; "
        "[pscustomobject]@{different=($path -cne '" + _powershell_literal(collision) + "');"
        "new=(Test-Path -LiteralPath $path)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "different": True,
        "new": True,
    }
    assert collision.read_text(encoding="ascii") == "historical"


def test_preserve_stop_history_is_not_active_and_newer_restart_is_active(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    run_root = root / "demo_runtime/preserved/run-a"
    owner_root = run_root / "gradio"
    owner_root.mkdir(parents=True)
    old_owner = owner_root / "owner-old.json"
    old_owner.write_text("{}", encoding="ascii")
    stopped = owner_root / "stop-old.stopped.json"
    stopped.write_text('{"owner_record":"owner-old.json"}', encoding="ascii")
    body = (
        f"$inactive=Get-PreserveActiveOwnerRecordPath -Root '{_powershell_literal(root)}' "
        "-RunId 'run-a' -Component 'gradio'; "
        "if($null -eq $inactive){'inactive'}else{'active'}"
    )
    inactive = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    new_owner = owner_root / "owner-new.json"
    new_owner.write_text("{}", encoding="ascii")
    body = (
        f"$active=Get-PreserveActiveOwnerRecordPath -Root '{_powershell_literal(root)}' "
        "-RunId 'run-a' -Component 'gradio'; [IO.Path]::GetFileName($active)"
    )
    active = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert inactive.returncode == 0, inactive.stderr
    assert inactive.stdout.strip() == "inactive"
    assert active.returncode == 0, active.stderr
    assert active.stdout.strip() == "owner-new.json"


def test_newest_stopped_owner_never_resurrects_older_unstopped_owner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    owner_root = root / "demo_runtime/preserved/run-a/gradio"
    owner_root.mkdir(parents=True)
    (owner_root / "owner-000-old.json").write_text("{}", encoding="ascii")
    (owner_root / "owner-999-new.json").write_text("{}", encoding="ascii")
    (owner_root / "stop-new.stopped.json").write_text(
        '{"owner_record":"owner-999-new.json"}', encoding="ascii"
    )
    body = (
        f"$value=Get-PreserveActiveOwnerRecordPath -Root '{_powershell_literal(root)}' "
        "-RunId 'run-a' -Component 'gradio'; if($null -eq $value){'inactive'}else{$value}"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "inactive"


def test_preserve_workflow_requires_and_propagates_one_run_id() -> None:
    for relative in (
        "scripts/demo/run_sidecar.ps1",
        "scripts/demo/run_demo.ps1",
        "scripts/demo/run_tunnel.ps1",
        "scripts/demo/stop_demo.ps1",
    ):
        script = _read(relative)
        assert "[string]$RunId" in script
        assert "Assert-PreserveRunId -RunId $RunId" in script
        assert "if (-not $RunId) { $RunId = [guid]::NewGuid()" not in script

    demo = _read("scripts/demo/run_demo.ps1")
    assert "'--run-id', $RunId" in demo
    assert "IMP_LOOP206_PRESERVE_RUN_ID" in demo


def _powershell() -> str:
    if os.name != "nt":
        pytest.skip("Windows-only launcher execution; local release gate required")
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


def _run_launcher_function_harness(
    relative: str, body: str, *, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    script = _powershell_literal(ROOT / relative)
    root = _powershell_literal(ROOT)
    command = (
        f". '{script}'; "
        "if(Get-Command Initialize-ReleaseProjection -ErrorAction SilentlyContinue){"
        f"Initialize-ReleaseProjection -Root '{root}'}}; "
        + body
    )
    return subprocess.run(
        [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def _run_launcher_ast_harness(
    relative: str, function_names: tuple[str, ...], body: str
) -> subprocess.CompletedProcess[str]:
    names = ",".join(f"'{name}'" for name in function_names)
    script = _powershell_literal(ROOT / relative)
    command = (
        f"$tokens=$null; $errors=$null; $ast=[Management.Automation.Language.Parser]::"
        f"ParseFile('{script}',[ref]$tokens,[ref]$errors); if($errors){{throw $errors[0]}}; "
        f"foreach($name in @({names})){{$definition=$ast.Find({{param($node) "
        "$node -is [Management.Automation.Language.FunctionDefinitionAst] -and "
        "$node.Name -ceq $name},$true); if($null -eq $definition){throw "
        '"missing function: $name"}; Invoke-Expression $definition.Extent.Text}; '
        + body
    )
    return subprocess.run(
        [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
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
            'if "%1"=="-" echo preflight=passed & echo dual_smoke=passed & exit /b 0',
        ]
        if capture_environment:
            lines.extend(
                [
                    '> "%IMP_TEST_OBSERVATION%" echo GRADIO_TEMP_DIR=%GRADIO_TEMP_DIR%',
                    '>> "%IMP_TEST_OBSERVATION%" echo IMP_LOOP206_DEMO_SESSION=%IMP_LOOP206_DEMO_SESSION%',
                    '>> "%IMP_TEST_OBSERVATION%" echo TMP=%TMP%',
                    '>> "%IMP_TEST_OBSERVATION%" echo TEMP=%TEMP%',
                    '> "%GRADIO_TEMP_DIR%\u005cowned.tmp" echo temporary',
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
        "    print('dual_smoke=passed')",
        "    raise SystemExit(0)",
    ]
    if capture_environment:
        body.extend(
            [
                "names = ('GRADIO_TEMP_DIR', 'IMP_LOOP206_DEMO_SESSION', 'TMP', 'TEMP')",
                "Path(os.environ['IMP_TEST_OBSERVATION']).write_text(",
                "    ''.join(f'{name}={os.environ[name]}\u005cn' for name in names), encoding='ascii'",
                ")",
                "(Path(os.environ['GRADIO_TEMP_DIR']) / 'owned.tmp').write_text('temporary', encoding='ascii')",
            ]
        )
    body.append("raise SystemExit(int(os.environ.get('IMP_TEST_APP_EXIT', '0')))")
    path.write_text("\n".join(body) + "\n", encoding="ascii")
    path.chmod(0o755)
    return path


def _fake_checkpoint_environment(tmp_path: Path) -> tuple[Path, Path]:
    control = tmp_path / "control.pt"
    candidate = tmp_path / "candidate.pt"
    control.write_text("fake control", encoding="ascii")
    candidate.write_text("fake candidate", encoding="ascii")
    return control, candidate


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

    assert ".venv-win\u005cScripts\u005cpython.exe" in script
    assert "uv sync" not in script
    assert "--host" in script and "127.0.0.1" in script
    assert "--port" in script and "7860" in script
    assert "--share" not in script
    assert "IMP_LOOP206_PRIOR" in script
    assert "IMP_LOOP206_PRIOR_RECEIPT" in script
    assert "Remove-Item Env:" in script
    assert "& $PythonExe @appArguments" in script
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
        "live_demo_receipt_projection",
        "release_manifest_sha256",
        "evidence registry release manifest digest mismatch",
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
    assert "return $process.ExitCode" in script
    assert "exit (Invoke-TunnelLaunch" in script
    assert "[switch]$CheckOnly" in script
    assert "--token" not in script
    assert "cloudflared.log" not in script
    assert "[string]$LocalUrl" not in script


def test_dual_launcher_gates_on_pinned_local_sidecar_and_cuda_smoke() -> None:
    demo = _read("scripts/demo/run_demo.ps1")
    sidecar = _read("scripts/demo/run_sidecar.ps1")

    assert "http://127.0.0.1:7862/health" in demo
    for token in (
        "dual_smoke=passed",
        "DualLiveService",
        "NnUNetClient",
        "build_dual_live_receipt",
        "receipt_eligible",
        "cuda",
    ):
        assert token in demo
    for token in (
        "--gpus",
        "device=0",
        "--publish",
        "127.0.0.1:7862:7862",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=256m",
        "--memory",
        "--restart",
        "no",
        "--rm",
        "imp-nnunet-loop192",
        "recovery_receipt.json",
        "readonly",
        "/models/loop192",
        "sha256:86bd77c03c3918e3638565e29417cdf4360b499a0813fbc425dc36645f026f2d",
    ):
        assert token in sidecar
    assert "0.0.0.0" not in sidecar
    assert "docker pull" not in sidecar.lower()
    assert "Ubuntu-E" not in sidecar


def test_dual_smoke_binds_synthetic_input_evidence_to_receipt() -> None:
    demo = _read("scripts/demo/run_demo.ps1")

    assert "LiveInputEvidence" in demo
    assert "input_evidence.rgb_sha256 != rgb_sha256(image)" in demo
    assert "release_manifest = load_release_manifest()" in demo
    assert "build_dual_live_receipt(result, release_manifest, input_evidence)" in demo
    assert 'receipt.get("schema_version") != "imp.dual_live.receipt.v2"' in demo


def test_dual_smoke_receipt_binds_matching_synthetic_evidence_and_rejects_forgery() -> None:
    image = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
    imp_identity = runtime_projection()["imp"]
    imp = DualLiveArm(
        status="completed",
        mask=np.zeros((16, 24), dtype=np.uint8),
        overlay=image.copy(),
        model_id=imp_identity["model_id"],
        checkpoint_sha256=imp_identity["checkpoint_sha256"],
        preprocessing="imp_runtime_control",
        reported_latency_ms=7.0,
        coordinator_latency_ms=8.0,
        reported_latency_scope="imp_model_forward_cuda_sync",
        coordinator_latency_scope="imp_preview_control_end_to_end_wall",
        device="cuda:0",
    )
    nnunet = DualLiveArm(
        status="completed",
        mask=np.ones((16, 24), dtype=np.uint8),
        overlay=image.copy(),
        model_id="L192-nnUNet-v2-raw-100ep",
        checkpoint_sha256=CHECKPOINT_SHA256,
        preprocessing="raw_rgb_256",
        reported_latency_ms=11.5,
        coordinator_latency_ms=17.5,
        reported_latency_scope="nnunet_predict_single_npy_array_cuda_sync",
        coordinator_latency_scope="nnunet_localhost_client_end_to_end_wall",
        device="cuda:0",
        protocol=PROTOCOL_ID,
    )
    result = DualLiveResult.complete(
        "a" * 32, rgb_sha256(image), image, imp, nnunet, 18.5
    )
    evidence = synthetic_evidence(image)

    receipt = build_dual_live_receipt(result, load_release_manifest(), evidence)
    assert receipt["schema_version"] == "imp.dual_live.receipt.v2"
    assert receipt["input"]["rgb_sha256"] == rgb_sha256(image)

    forged = replace(evidence, rgb_sha256="b" * 64)
    with pytest.raises(ValueError, match="input evidence binding mismatch"):
        build_dual_live_receipt(result, load_release_manifest(), forged)


def test_demo_sidecar_health_requires_exact_pinned_ready_cuda_payload() -> None:
    body = (
        "$valid=[pscustomobject]@{protocol='imp.nnunet.sidecar.v1';"
        "model_id='L192-nnUNet-v2-raw-100ep';"
        "checkpoint_sha256='3814716033afd464dacc573f92a5a44ff20eb7f2163d99b4f16ecff8aa278ea2';"
        "device='cuda:0';ready=$true}; "
        "$accepted=Test-ExactDemoSidecarHealth -Payload $valid; "
        "$valid | Add-Member -NotePropertyName extra -NotePropertyValue 'forged'; "
        "$rejected=Test-ExactDemoSidecarHealth -Payload $valid; "
        "$integer=[pscustomobject]@{protocol='imp.nnunet.sidecar.v1';"
        "model_id='L192-nnUNet-v2-raw-100ep';"
        "checkpoint_sha256='3814716033afd464dacc573f92a5a44ff20eb7f2163d99b4f16ecff8aa278ea2';"
        "device='cuda:0';ready=1}; "
        "$integerRejected=Test-ExactDemoSidecarHealth -Payload $integer; "
        "[pscustomobject]@{accepted=$accepted;rejected=$rejected;"
        "integerRejected=$integerRejected} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "accepted": True,
        "rejected": False,
        "integerRejected": False,
    }


def test_sidecar_health_rejects_integer_ready() -> None:
    body = (
        "$payload=[pscustomobject]@{protocol='imp.nnunet.sidecar.v1';"
        "model_id='L192-nnUNet-v2-raw-100ep';"
        "checkpoint_sha256='3814716033afd464dacc573f92a5a44ff20eb7f2163d99b4f16ecff8aa278ea2';"
        "device='cuda:0';ready=1}; Test-PinnedHealthPayload -Payload $payload"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_gradio_owner_record_has_exact_incarnation_identity(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    session = root / "demo_runtime/sessions/demo-0123456789abcdef0123456789abcdef"
    session.mkdir(parents=True)
    python_exe = tmp_path / "python.exe"
    python_exe.write_text("fake", encoding="ascii")
    body = (
        f"$path=Write-GradioOwnerRecord -Root '{_powershell_literal(root)}' "
        f"-PythonExe '{_powershell_literal(python_exe)}' "
        f"-SessionPath '{_powershell_literal(session)}'; "
        "$record=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json; "
        "$record | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(record) == {
        "schema_version",
        "launcher_pid",
        "launcher_start_time_utc",
        "owner_nonce",
        "python_path",
        "session_path",
            "host",
            "port",
            "release_manifest_sha256",
            "public_tunnel_mode",
            "preserve_mode",
        }
    assert record["release_manifest_sha256"] == _release_digest()
    assert re.fullmatch(r"[0-9a-f]{32}", record["owner_nonce"])
    assert record["launcher_start_time_utc"].endswith("Z")
    assert record["public_tunnel_mode"] is False
    assert record["preserve_mode"] is False


def test_public_tunnel_journal_requires_public_preserve_mode_and_current_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    session = root / "demo_runtime/sessions/demo-0123456789abcdef0123456789abcdef"
    session.mkdir(parents=True)
    python_exe = tmp_path / "python.exe"
    python_exe.write_text("fake", encoding="ascii")
    body = (
        f"$path=Write-GradioOwnerRecord -Root '{_powershell_literal(root)}' "
        f"-PythonExe '{_powershell_literal(python_exe)}' "
        f"-SessionPath '{_powershell_literal(session)}' -PreserveMode -PublicTunnelMode -RunId 'run-a'; "
        "$record=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json; "
        "$record | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.strip().splitlines()[-1])
    assert record["public_tunnel_mode"] is True
    assert record["preserve_mode"] is True
    assert record["release_manifest_sha256"] == _release_digest()

    tunnel = _read("scripts/demo/run_tunnel.ps1")
    assert "public_tunnel_mode" in tunnel
    assert "preserve_mode" in tunnel
    assert "--public-tunnel-mode" in tunnel


def test_sidecar_receipt_is_file_pinned_and_rejects_boolean_or_metadata_drift() -> None:
    script = _read("scripts/demo/run_sidecar.ps1")
    projection = launcher_projection()
    nnunet = projection["nnunet"]
    recovery_digest = projection["sidecar"]["recovery_receipt_sha256"]
    assert recovery_digest not in script
    assert "$script:RecoveryReceiptSha256" in script
    body = (
        "$manifest=[pscustomobject]@{artifacts=[pscustomobject]@{"
        "'plans.json'=[pscustomobject]@{sha256='p'};"
        "'dataset_fingerprint.json'=[pscustomobject]@{sha256='f'}}}; "
        "$metadata=[pscustomobject]@{"
        "'dataset.json'=[pscustomobject]@{};"
        "'plans.json'=[pscustomobject]@{};"
        "'requirements.lock'=[pscustomobject]@{};"
        "'runtime_identity.json'=[pscustomobject]@{}}; "
        "$receipt=[pscustomobject]@{schema_version='loop192.recovery.receipt.v1';"
        f"model_id='{nnunet['model_id']}';"
        f"checkpoint_sha256='{nnunet['checkpoint_sha256']}';"
        "plans_sha256='p';fingerprint_sha256='f';source_vhd_unchanged=$true;"
        "runtime_status='reconstructed_required';metadata=$metadata}; "
        "$valid=Test-ExactRecoveryReceipt -Receipt $receipt -Manifest $manifest; "
        "$receipt.source_vhd_unchanged=1; "
        "$integer=Test-ExactRecoveryReceipt -Receipt $receipt -Manifest $manifest; "
        "$receipt.source_vhd_unchanged=$true; "
        "$receipt.metadata | Add-Member -NotePropertyName forged -NotePropertyValue ([pscustomobject]@{}); "
        "$extra=Test-ExactRecoveryReceipt -Receipt $receipt -Manifest $manifest; "
        "[pscustomobject]@{valid=$valid;integer=$integer;extra=$extra} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "valid": True,
        "integer": False,
        "extra": False,
    }


def test_sidecar_spacing_accepts_winps_decimal_json_numbers_and_rejects_drift() -> None:
    body = (
        "$decimal=ConvertFrom-Json '[999.0,1.0,1.0]'; "
        "$integral=@(999,1,1); "
        "$string_value=@('999',1,1); "
        "$extra=@(999,1,1,0); "
        "$nan=@([double]::NaN,1,1); "
        "$infinity=@([double]::PositiveInfinity,1,1); "
        "$fraction=@(999.5,1,1); "
        "[pscustomobject]@{"
        "decimal=(Test-ExactModelInputSpacing -Spacing $decimal);"
        "integral=(Test-ExactModelInputSpacing -Spacing $integral);"
        "string_value=(Test-ExactModelInputSpacing -Spacing $string_value);"
        "extra=(Test-ExactModelInputSpacing -Spacing $extra);"
        "nan=(Test-ExactModelInputSpacing -Spacing $nan);"
        "infinity=(Test-ExactModelInputSpacing -Spacing $infinity);"
        "fraction=(Test-ExactModelInputSpacing -Spacing $fraction)} "
        "| ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "decimal": True,
        "integral": True,
        "string_value": False,
        "extra": False,
        "nan": False,
        "infinity": False,
        "fraction": False,
    }


def test_sidecar_release_projection_uses_semantic_pins_without_consumer_hash() -> None:
    script = _read("scripts/demo/run_sidecar.ps1")
    sidecar = launcher_projection()["sidecar"]
    assert "model_manifest_sha256" not in script
    assert "$script:ManifestSha256" not in script
    body = (
        "[pscustomobject]@{"
        "checkpoint_size=$script:CheckpointSize;"
        "dataset_sha256=$script:DatasetSha256;dataset_size=$script:DatasetSize;"
        "fingerprint_sha256=$script:FingerprintSha256;fingerprint_size=$script:FingerprintSize;"
        "plans_sha256=$script:PlansSha256;plans_size=$script:PlansSize;"
        "runtime_git_commit=$script:RuntimeGitCommit;runtime_status=$script:RuntimeStatus;"
        "runtime_version=$script:RuntimeVersion;recovery=$script:RecoveryReceiptSha256} "
        "| ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "checkpoint_size": sidecar["checkpoint_size"],
        "dataset_sha256": sidecar["dataset_sha256"],
        "dataset_size": sidecar["dataset_size"],
        "fingerprint_sha256": sidecar["fingerprint_sha256"],
        "fingerprint_size": sidecar["fingerprint_size"],
        "plans_sha256": sidecar["plans_sha256"],
        "plans_size": sidecar["plans_size"],
        "runtime_git_commit": sidecar["runtime_git_commit"],
        "runtime_status": sidecar["runtime_status"],
        "runtime_version": sidecar["runtime_version"],
        "recovery": sidecar["recovery_receipt_sha256"],
    }


def test_tunnel_rejects_arbitrary_http_200_config() -> None:
    body = (
        "function Invoke-WebRequest { [pscustomobject]@{StatusCode=200;"
        "Content='{\"title\":\"arbitrary service\",\"api_open\":false,"
        "\"components\":[]}'} }; "
        "$rejected=$false; try{Assert-GradioConfigEndpoint}catch{$rejected=$true}; $rejected"
    )
    result = _run_launcher_ast_harness(
        "scripts/demo/run_tunnel.ps1", ("Assert-GradioConfigEndpoint",), body
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def _production_public_gradio_config() -> dict:
    placeholder = type("Placeholder", (), {})()
    return create_app(
        placeholder,
        {
            "_demo_runtime": {
                "fixed_choices": [],
                "corruptions": ["clean"],
                "fixed_ground_truth": {},
                "sidecar_ready": True,
            }
        },
        dual_service=placeholder,
        public_tunnel_mode=True,
    ).get_config_file()


def _tunnel_config_gate(
    config: dict,
    *,
    catch_rejection: bool = True,
    include_empty_page_property: bool = False,
    omit_api_open: bool = False,
) -> subprocess.CompletedProcess[str]:
    component_ids = {
        component_id
        for dependency in config["dependencies"]
        for component_id in [
            *dependency["inputs"],
            *dependency["outputs"],
            *(target[0] for target in dependency["targets"] if target[0] is not None),
        ]
    }
    payload_config = {
        "version": config.get("version"),
        "title": config["title"],
        "api_open": config.get("api_open", False),
        "components": [
            {
                "id": component["id"],
                "type": component["type"],
                "props": {
                    key: component.get("props", {}).get(key)
                    for key in (
                        "label",
                        "sources",
                        "visible",
                        "allow_custom_value",
                        "info",
                        "elem_id",
                        "elem_classes",
                    )
                    if key in component.get("props", {})
                },
            }
            for component in config["components"]
            if component["id"] in component_ids or component["type"] in {"file", "image"}
        ],
        "dependencies": [
            {
                key: dependency[key]
                for key in (
                    "id",
                    "targets",
                    "inputs",
                    "outputs",
                    "backend_fn",
                    "queue",
                    "api_name",
                    "api_visibility",
                    "trigger_after",
                )
                if key in dependency
            }
            for dependency in config["dependencies"]
        ],
    }
    if include_empty_page_property:
        payload_config["page"] = {"": {"server_fns": []}}
    if omit_api_open:
        payload_config.pop("api_open")
    body = "function Invoke-WebRequest { [pscustomobject]@{StatusCode=200;Content=$env:IMP_TEST_GRADIO_CONFIG} }; "
    if catch_rejection:
        body += "$accepted=$true; try{Assert-GradioConfigEndpoint -PublicTunnelMode}catch{$accepted=$false}; $accepted"
    else:
        body += "Assert-GradioConfigEndpoint -PublicTunnelMode; 'True'"
    environment = dict(os.environ)
    environment["IMP_TEST_GRADIO_CONFIG"] = json.dumps(
        payload_config, separators=(",", ":")
    )
    return _run_launcher_function_harness(
        "scripts/demo/run_tunnel.ps1", body, environment=environment
    )


def test_tunnel_accepts_production_public_gradio_config() -> None:
    result = _tunnel_config_gate(
        _production_public_gradio_config(), catch_rejection=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_tunnel_accepts_public_gradio_config_with_empty_property_name() -> None:
    result = _tunnel_config_gate(
        _production_public_gradio_config(),
        catch_rejection=False,
        include_empty_page_property=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_tunnel_accepts_version_6_config_without_api_open() -> None:
    result = _tunnel_config_gate(
        _production_public_gradio_config(),
        catch_rejection=False,
        omit_api_open=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_tunnel_rejects_true_api_open() -> None:
    config = json.loads(json.dumps(_production_public_gradio_config()))
    config["api_open"] = True

    result = _tunnel_config_gate(config)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_tunnel_rejects_version_5_config_without_api_open() -> None:
    config = json.loads(json.dumps(_production_public_gradio_config()))
    config["version"] = "5.49.1"

    result = _tunnel_config_gate(
        config,
        omit_api_open=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("version", ("4.44.1", "7.0.0", "not-a-version"))
def test_tunnel_rejects_unsupported_or_malformed_gradio_version(version: str) -> None:
    config = json.loads(json.dumps(_production_public_gradio_config()))
    config["version"] = version

    result = _tunnel_config_gate(config)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize(
    "forgery",
    (
        "spaced_source",
        "hidden_file",
        "upload_event",
        "wrong_dual_graph",
        "missing_backend",
        "missing_outputs",
        "alternate_then_dependency",
        "precursor_backend",
        "precursor_target",
        "precursor_outputs",
    ),
)
def test_tunnel_rejects_forged_public_gradio_upload_surfaces(forgery: str) -> None:
    config = json.loads(json.dumps(_production_public_gradio_config()))
    if forgery == "spaced_source":
        config["components"].append(
            {
                "id": 9001,
                "type": "image",
                "props": {"visible": False, "sources": ["clipboard", " upload "]},
            }
        )
    elif forgery == "hidden_file":
        config["components"].append(
            {"id": 9002, "type": "file", "props": {"visible": False}}
        )
    elif forgery == "upload_event":
        config["dependencies"].append(
            {
                "id": 9003,
                "targets": [[8, "upload"]],
                "inputs": [],
                "outputs": [],
                "backend_fn": True,
                "queue": False,
                "api_name": "upload_probe",
                "api_visibility": "public",
            }
        )
    else:
        dual = next(
            dependency
            for dependency in config["dependencies"]
            if dependency["api_name"] == "dual_live_compare"
        )
        if forgery == "wrong_dual_graph":
            dual["inputs"] = [dual["inputs"][0]]
        elif forgery == "missing_backend":
            dual["backend_fn"] = False
        elif forgery == "missing_outputs":
            dual["outputs"] = []
        elif forgery.startswith("precursor_"):
            precursor = next(
                dependency
                for dependency in config["dependencies"]
                if dependency["id"] == dual["trigger_after"]
            )
            if forgery == "precursor_backend":
                precursor["backend_fn"] = False
            elif forgery == "precursor_target":
                precursor["targets"] = [[precursor["targets"][0][0], "input"]]
            else:
                precursor["outputs"] = []
        else:
            config["dependencies"].append(
                {
                    "id": 9004,
                    "targets": [[None, "then"]],
                    "inputs": [dual["inputs"][1]],
                    "outputs": [],
                    "backend_fn": True,
                    "queue": True,
                    "api_name": "alternate_callback",
                    "api_visibility": "private",
                    "trigger_after": 0,
                }
            )

    result = _tunnel_config_gate(config)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_tunnel_requires_owned_gradio_record_listener_and_exact_config() -> None:
    script = _read("scripts/demo/run_tunnel.ps1")
    for token in (
        "gradio.json",
        "launcher_start_time_utc",
        "owner_nonce",
        "Get-NetTCPConnection",
        "Audited Dermoscopy Workbench",
            "Bundled public / synthetic sample",
        "/config",
        "api_open",
    ):
        assert token in script
    assert script.index("Assert-OwnedGradioRuntime") < script.index(
        "Assert-GradioConfigEndpoint"
    )


@pytest.mark.parametrize("release_digest", (None, "0" * 64))
def test_tunnel_rejects_missing_or_stale_gradio_release_digest(
    release_digest: str | None,
) -> None:
    digest = _release_digest()
    release_field = (
        "" if release_digest is None else f"release_manifest_sha256='{release_digest}';"
    )
    body = (
        "$record=[pscustomobject]@{schema_version='imp.demo.gradio-owner.v1';"
        "launcher_pid=42;launcher_start_time_utc='2026-07-22T01:02:03.1234567Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef';python_path='E\u003a\u005cpython.exe';"
        "session_path='E\u003a\u005csession';host='127.0.0.1';port=7860;"
        f"{release_field}}}; "
        "$rejected=$false; try{Assert-GradioOwnerRecord -Record $record "
        f"-ExpectedReleaseManifestSha256 '{digest}'}}catch{{$rejected=$true}}; $rejected"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_tunnel_accepts_gradio_owner_with_current_release_digest() -> None:
    digest = _release_digest()
    body = (
        "$record=[pscustomobject]@{schema_version='imp.demo.gradio-owner.v1';"
        "launcher_pid=42;launcher_start_time_utc='2026-07-22T01:02:03.1234567Z';"
            "owner_nonce='0123456789abcdef0123456789abcdef';python_path='E\u003a\u005cpython.exe';"
            "session_path='E\u003a\u005csession';host='127.0.0.1';port=7860;"
            "public_tunnel_mode=$true;preserve_mode=$true;"
            f"release_manifest_sha256='{digest}'}}; "
        f"Assert-GradioOwnerRecord -Record $record -ExpectedReleaseManifestSha256 '{digest}'; "
        "'accepted'"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "accepted"


def test_tunnel_accepts_json_deserialized_gradio_owner_identity() -> None:
    digest = _release_digest()
    owner_json = json.dumps(
        {
            "schema_version": "imp.demo.gradio-owner.v1",
            "release_manifest_sha256": digest,
            "public_tunnel_mode": True,
            "preserve_mode": True,
            "launcher_pid": 1752,
            "launcher_start_time_utc": "2026-07-22T01:02:03.1234567Z",
            "owner_nonce": "0123456789abcdef0123456789abcdef",
            "python_path": "E\u003a/repo/.venv-win/Scripts/python.exe",
            "session_path": "E\u003a/repo/demo_runtime/sessions/demo-0123456789abcdef0123456789abcdef",
            "host": "127.0.0.1",
            "port": 7860,
        }
    )
    body = (
        f"$record='{owner_json}' | ConvertFrom-Json; "
        "$record.launcher_pid=[long]$record.launcher_pid; $record.launcher_start_time_utc=[datetime]$record.launcher_start_time_utc; $record.port=[long]$record.port; "
        f"Assert-GradioOwnerRecord -Record $record -ExpectedReleaseManifestSha256 '{digest}'; "
        "'accepted'"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "accepted"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("launcher_pid", 1752.5),
        ("launcher_pid", "1752"),
        ("launcher_pid", 2**31),
        ("port", 7860.5),
        ("port", "7860"),
        ("port", 2**31),
    ],
)
def test_tunnel_rejects_nonintegral_or_out_of_range_gradio_owner_numbers(
    field: str, value: object
) -> None:
    digest = _release_digest()
    owner = {
        "schema_version": "imp.demo.gradio-owner.v1",
        "release_manifest_sha256": digest,
        "public_tunnel_mode": True,
        "preserve_mode": True,
        "launcher_pid": 1752,
        "launcher_start_time_utc": "2026-07-22T01:02:03.1234567Z",
        "owner_nonce": "0123456789abcdef0123456789abcdef",
        "python_path": "E\u003a/repo/.venv-win/Scripts/python.exe",
        "session_path": "E\u003a/repo/demo_runtime/sessions/demo-0123456789abcdef0123456789abcdef",
        "host": "127.0.0.1",
        "port": 7860,
    }
    owner[field] = value
    body = (
        f"$record='{json.dumps(owner)}' | ConvertFrom-Json; "
        "$record.launcher_start_time_utc=[datetime]$record.launcher_start_time_utc; "
        "$message=''; try{Assert-GradioOwnerRecord -Record $record "
        f"-ExpectedReleaseManifestSha256 '{digest}'}}catch{{$message=$_.Exception.Message}}; $message"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Gradio owner record identity mismatch."


def test_tunnel_preflight_returns_and_propagates_release_context() -> None:
    digest = _release_digest()
    body = (
        "$script:observed=$null; "
        "function Resolve-TunnelRoot { 'E\u003a\u005crepo' }; "
        f"function Assert-ReleaseManifest {{ '{digest}' }}; "
        "function Assert-OwnedGradioRuntime { param($Root,$PreserveMode,$RunId,"
        "$ExpectedReleaseManifestSha256); $script:observed=$ExpectedReleaseManifestSha256 }; "
        "function Assert-GradioConfigEndpoint {}; "
        "$context=Invoke-TunnelPreflight -PreserveMode -RunId 'run-a'; "
        "[pscustomobject]@{root=$context.Root;digest=$context.ReleaseManifestSha256;"
        "observed=$script:observed} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "root": "E\u003a\u005crepo",
        "digest": digest,
        "observed": digest,
    }


def test_tunnel_owned_runtime_accepts_real_wrapper_listener_worker_chain() -> None:
    digest = _release_digest()
    body = (
        "$record=[pscustomobject]@{launcher_pid=42;"
        "launcher_start_time_utc='2026-07-22T01:02:03.1234567Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef';"
        "python_path='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe'}; "
        "function Read-SafeGradioOwnerRecord { $record }; "
        "function Get-ExactProcess { param($ProcessId); "
        "if($ProcessId -eq 42) { [pscustomobject]@{ProcessId=42;ParentProcessId=1;"
        "CreationDate=[datetime]'2026-07-22T01:02:03.1234567Z';"
        "ExecutablePath='C\u003a\u005cWindows\u005cSystem32\u005cWindowsPowerShell\u005cv1.0\u005cpowershell.exe';"
        "CommandLine='powershell -File E\u003a\u005crepo\u005cscripts\u005cdemo\u005crun_demo.ps1'}; return }; "
        "if($ProcessId -eq 83) { [pscustomobject]@{ProcessId=83;ParentProcessId=42;"
        "ExecutablePath='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe';"
        "CommandLine='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe -m lesion_robustness.demo.app "
        "--public-tunnel-mode --preserve-mode --host 127.0.0.1 --port 7860'}; return }; "
        "if($ProcessId -eq 84) { [pscustomobject]@{ProcessId=84;ParentProcessId=83;"
        "ExecutablePath='C\u003a\u005cPython312\u005cpython.exe';"
        "CommandLine='C\u003a\u005cPython312\u005cpython.exe -m lesion_robustness.demo.app "
        "--public-tunnel-mode --preserve-mode --host 127.0.0.1 --port 7860'}; return }; "
        "if($ProcessId -eq 85) { [pscustomobject]@{ProcessId=85;ParentProcessId=84;"
        "ExecutablePath='C\u003a\u005cPython312\u005cpython.exe';CommandLine='C\u003a\u005cPython312\u005cpython.exe worker'}; return } }; "
        "function Get-NetTCPConnection { [pscustomobject]@{LocalAddress='127.0.0.1';"
        "LocalPort=7860;OwningProcess=84} }; "
        f"Assert-OwnedGradioRuntime -Root 'E\u003a\u005crepo' -ExpectedReleaseManifestSha256 '{digest}' "
        "-PreserveMode -RunId 'run-a'; 'accepted'"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "accepted"


@pytest.mark.parametrize("omitted_flag", ("--public-tunnel-mode", "--preserve-mode"))
def test_tunnel_rejects_wrapper_command_missing_required_mode_flag(
    omitted_flag: str,
) -> None:
    command = (
        "E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe -m lesion_robustness.demo.app "
        "--public-tunnel-mode --preserve-mode --host 127.0.0.1 --port 7860"
    ).replace(f"{omitted_flag} ", "")
    body = (
        "$record=[pscustomobject]@{launcher_pid=42;"
        "python_path='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe'}; "
        "$process=[pscustomobject]@{ProcessId=83;ParentProcessId=42;"
        "ExecutablePath='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe';"
        f"CommandLine='{command}'}}; "
        "$message=''; try{Assert-GradioPythonWrapperIdentity -Record $record "
        "-Process $process}catch{$message=$_.Exception.Message}; $message"
    )
    result = _run_launcher_ast_harness(
        "scripts/demo/run_tunnel.ps1", ("Assert-GradioPythonWrapperIdentity",), body
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Gradio Python wrapper identity mismatch."


@pytest.mark.parametrize(
    ("case", "wrapper_parent", "listener_id", "listener_parent", "wrapper_path", "listener_command", "message"),
    [
        ("wrong-parent", 42, 84, 99, "E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe", "C\u003a\u005cPython312\u005cpython.exe -m lesion_robustness.demo.app --public-tunnel-mode --preserve-mode --host 127.0.0.1 --port 7860", "Gradio Python wrapper identity mismatch."),
        ("foreign-pid", 42, 999, 99, "E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe", "C\u003a\u005cPython312\u005cpython.exe -m lesion_robustness.demo.app --public-tunnel-mode --preserve-mode --host 127.0.0.1 --port 7860", "Gradio Python wrapper identity mismatch."),
        ("wrong-command", 42, 84, 83, "E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe", "C\u003a\u005cPython312\u005cpython.exe -m unrelated.module --host 127.0.0.1 --port 7860", "Gradio listener process identity mismatch."),
        ("wrong-path", 42, 84, 83, "E\u003a\u005cforeign\u005cpython.exe", "C\u003a\u005cPython312\u005cpython.exe -m lesion_robustness.demo.app --public-tunnel-mode --preserve-mode --host 127.0.0.1 --port 7860", "Gradio Python wrapper identity mismatch."),
    ],
)
def test_tunnel_owned_runtime_rejects_unproven_wrapper_listener_chains(
    case: str,
    wrapper_parent: int,
    listener_id: int,
    listener_parent: int,
    wrapper_path: str,
    listener_command: str,
    message: str,
) -> None:
    digest = _release_digest()
    body = (
        "$record=[pscustomobject]@{launcher_pid=42;"
        "launcher_start_time_utc='2026-07-22T01:02:03.1234567Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef';"
        "python_path='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe'}; "
        "function Read-SafeGradioOwnerRecord { $record }; "
        "function Get-ExactProcess { param($ProcessId); "
        "if($ProcessId -eq 42) { [pscustomobject]@{ProcessId=42;ParentProcessId=1;"
        "CreationDate=[datetime]'2026-07-22T01:02:03.1234567Z';"
        "ExecutablePath='C\u003a\u005cWindows\u005cSystem32\u005cWindowsPowerShell\u005cv1.0\u005cpowershell.exe';"
        "CommandLine='powershell -File E\u003a\u005crepo\u005cscripts\u005cdemo\u005crun_demo.ps1'}; return }; "
        f"if($ProcessId -eq 83) "
        "{ [pscustomobject]@{ProcessId=83;ParentProcessId="
        f"{wrapper_parent};"
        f"ExecutablePath='{wrapper_path}';"
        "CommandLine='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe -m lesion_robustness.demo.app "
        "--public-tunnel-mode --preserve-mode --host 127.0.0.1 --port 7860'}; return }; "
        f"if($ProcessId -eq {listener_id}) "
        "{ [pscustomobject]@{ProcessId="
        f"{listener_id};ParentProcessId={listener_parent};"
        "ExecutablePath='C\u003a\u005cPython312\u005cpython.exe';"
        f"CommandLine='{listener_command}'"
        "}; return }; }; "
        "function Get-NetTCPConnection { [pscustomobject]@{LocalAddress='127.0.0.1';"
        f"LocalPort=7860;OwningProcess={listener_id}"
        "}; }; "
        "$message=''; try{Assert-OwnedGradioRuntime -Root 'E\u003a\u005crepo' "
        f"-ExpectedReleaseManifestSha256 '{digest}' -PreserveMode -RunId 'run-a'}}"
        "catch{$message=$_.Exception.Message}; $message"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == message, case


def test_tunnel_preexisting_owner_record_blocks_start_without_deletion() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Invoke-TunnelPreflight { [pscustomobject]@{Root='E\u003a\u005crepo';ReleaseManifestSha256=('a' * 64)} }; "
        "function Resolve-CloudflaredApplication { 'E\u003a\u005ccloudflared.exe' }; "
        "function Get-TunnelOwnerRecordPath { $script:events.Add('path'); "
        "'E\u003a\u005crepo\u005cdemo_runtime\u005clauncher\u005ctunnel.json' }; "
        "function Test-Path { $true }; "
        "function Start-Process { $script:events.Add('start'); throw 'must not start' }; "
        "function Remove-TunnelOwnerRecord { $script:events.Add('remove') }; "
        "$code=Invoke-TunnelLaunch; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "code": 5,
        "events": ["path"],
    }


def test_tunnel_owner_record_has_exact_incarnation_identity(tmp_path: Path) -> None:
    record_path = tmp_path / "tunnel.json"
    digest = _release_digest()
    body = (
        f"Write-TunnelOwnerRecord -RecordPath '{_powershell_literal(record_path)}' "
        "-ProcessId 42 "
        "-ProcessStartTimeUtc '2026-07-22T01:02:03.1234567Z' "
        "-OwnerNonce '0123456789abcdef0123456789abcdef' "
        "-ExecutablePath 'E\u003a\u005ccloudflared.exe' "
        f"-ReleaseManifestSha256 '{digest}'; "
        f"Get-Content -LiteralPath '{_powershell_literal(record_path)}' -Raw"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "schema_version": "imp.demo.tunnel-owner.v1",
        "process_id": 42,
        "process_start_time_utc": "2026-07-22T01:02:03.1234567Z",
        "owner_nonce": "0123456789abcdef0123456789abcdef",
        "executable_path": "E\u003a\u005ccloudflared.exe",
        "local_url": "http://127.0.0.1:7860",
        "release_manifest_sha256": digest,
    }


def test_tunnel_wait_failure_stops_exact_owned_process_before_record_removal() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Invoke-TunnelPreflight { [pscustomobject]@{Root='E\u003a\u005crepo';ReleaseManifestSha256=('a' * 64)} }; "
        "function Resolve-CloudflaredApplication { 'E\u003a\u005ccloudflared.exe' }; "
        "function Get-TunnelOwnerRecordPath { 'E\u003a\u005crepo\u005cdemo_runtime\u005clauncher\u005ctunnel.json' }; "
        "function Test-Path { $false }; "
        "function Start-Process { $script:events.Add('start'); "
        "$value=[pscustomobject]@{Id=42;ExitCode=7}; "
        "$value | Add-Member -MemberType ScriptMethod -Name WaitForExit "
        "-Value { throw 'wait failed' }; $value }; "
        "function Get-ExactProcess { [pscustomobject]@{ProcessId=42;"
        "CreationDate=[datetime]'2026-07-22T01:02:03.1234567Z';"
        "ExecutablePath='E\u003a\u005ccloudflared.exe';CommandLine='cloudflared tunnel "
        "--url http://127.0.0.1:7860'} }; "
        "function Write-TunnelOwnerRecord { $script:events.Add('record') }; "
        "function Stop-SpawnedProcessHandle { $script:events.Add('stop') }; "
        "function Remove-TunnelOwnerRecord { $script:events.Add('remove') }; "
        "$code=Invoke-TunnelLaunch; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "code": 5,
        "events": ["start", "record", "stop", "remove"],
    }


def test_tunnel_post_spawn_identity_failure_stops_returned_process_handle() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Invoke-TunnelPreflight { [pscustomobject]@{Root='E\u003a\u005crepo';ReleaseManifestSha256=('a' * 64)} }; "
        "function Resolve-CloudflaredApplication { 'E\u003a\u005ccloudflared.exe' }; "
        "function Get-TunnelOwnerRecordPath { 'E\u003a\u005crepo\u005cdemo_runtime\u005clauncher\u005ctunnel.json' }; "
        "function Test-Path { $false }; "
        "function Start-Process { $script:events.Add('start'); "
        "[pscustomobject]@{Id=42;HandleTag='exact-spawn'} }; "
        "function Get-ExactProcess { throw 'CIM identity unavailable' }; "
        "function Stop-SpawnedProcessHandle { param($Process); "
        "$script:events.Add(('handle:' + $Process.HandleTag)) }; "
        "function Remove-TunnelOwnerRecord { $script:events.Add('remove') }; "
        "$code=Invoke-TunnelLaunch; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "code": 5,
        "events": ["start", "handle:exact-spawn"],
    }


def test_tunnel_record_write_failure_uses_handle_not_reusable_pid() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Invoke-TunnelPreflight { [pscustomobject]@{Root='E\u003a\u005crepo';ReleaseManifestSha256=('a' * 64)} }; "
        "function Resolve-CloudflaredApplication { 'E\u003a\u005ccloudflared.exe' }; "
        "function Get-TunnelOwnerRecordPath { 'E\u003a\u005crepo\u005cdemo_runtime\u005clauncher\u005ctunnel.json' }; "
        "function Test-Path { $false }; "
        "function Start-Process { $script:events.Add('start'); "
        "[pscustomobject]@{Id=42;HandleTag='exact-spawn'} }; "
        "function Get-ExactProcess { [pscustomobject]@{ProcessId=42;"
        "CreationDate=[datetime]'2026-07-22T01:02:03.1234567Z';"
        "ExecutablePath='E\u003a\u005ccloudflared.exe';CommandLine='cloudflared tunnel "
        "--url http://127.0.0.1:7860'} }; "
        "function Write-TunnelOwnerRecord { $script:events.Add('write'); throw 'write failed' }; "
        "function Stop-Process { $script:events.Add('pid-stop') }; "
        "function Stop-SpawnedProcessHandle { param($Process); "
        "$script:events.Add(('handle:' + $Process.HandleTag)) }; "
        "$code=Invoke-TunnelLaunch; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "code": 5,
        "events": ["start", "write", "handle:exact-spawn"],
    }


def test_tunnel_get_exact_process_returns_null_when_pid_is_absent() -> None:
    body = (
        "function Get-CimInstance { @() }; "
        "$value=Get-ExactProcess -ProcessId 42; "
        "if($null -eq $value){'absent'}else{'present'}"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "absent"


def test_tunnel_stop_timeout_preserves_owner_record() -> None:
    digest = _release_digest()
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Resolve-DemoStopRoot { 'E\u003a\u005crepo' }; "
        f"function Assert-ReleaseManifest {{ '{digest}' }}; "
        "function Read-OwnedRecord { [pscustomobject]@{"
        "schema_version='imp.demo.tunnel-owner.v1';process_id=42;"
        "process_start_time_utc='2026-07-22T01:02:03.1234567Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef';"
        "executable_path='E\u003a\u005ccloudflared.exe';local_url='http://127.0.0.1:7860';"
        f"release_manifest_sha256='{digest}'}} }}; "
        "function Get-ExactProcess { [pscustomobject]@{ProcessId=42;"
        "CreationDate=[datetime]'2026-07-22T01:02:03.1234567Z';"
        "ExecutablePath='E\u003a\u005ccloudflared.exe';CommandLine='cloudflared tunnel "
        "--url http://127.0.0.1:7860'} }; "
        "function Stop-ProcessAndWait { $script:events.Add('stop'); throw 'timeout' }; "
        "function Remove-OwnedRecord { $script:events.Add('remove') }; "
        "$rejected=$false; try{Stop-OwnedCloudflared}catch{$rejected=$true}; "
        "[pscustomobject]@{rejected=$rejected;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "rejected": True,
        "events": ["stop"],
    }


def test_stop_gradio_accepts_json_deserialized_owner_identity() -> None:
    digest = _release_digest()
    owner_json = json.dumps(
        {
            "schema_version": "imp.demo.gradio-owner.v1",
            "release_manifest_sha256": digest,
            "public_tunnel_mode": True,
            "preserve_mode": True,
            "launcher_pid": 1752,
            "launcher_start_time_utc": "2026-07-22T01:02:03.1234567Z",
            "owner_nonce": "0123456789abcdef0123456789abcdef",
            "python_path": "E\u003a/repo/.venv-win/Scripts/python.exe",
            "session_path": "E\u003a/repo/demo_runtime/sessions/demo-0123456789abcdef0123456789abcdef",
            "host": "127.0.0.1",
            "port": 7860,
        }
    )
    body = (
        f"$record='{owner_json}' | ConvertFrom-Json; "
        "$record.launcher_pid=[long]$record.launcher_pid; $record.launcher_start_time_utc=[datetime]$record.launcher_start_time_utc; $record.port=[long]$record.port; "
        "Assert-GradioOwnerRecord -Record $record; "
        "$process=[pscustomobject]@{ProcessId=1752;CreationDate=[datetime]'2026-07-22T01:02:03.1234567Z';"
        "ExecutablePath='C\u003a\u005cWindows\u005cSystem32\u005cWindowsPowerShell\u005cv1.0\u005cpowershell.exe';"
        "CommandLine='powershell -File E\u003a\u005crepo\u005cscripts\u005cdemo\u005crun_demo.ps1'}; "
        "Assert-LauncherProcessIdentity -Record $record -Process $process -ExpectedScript 'E\u003a\u005crepo\u005cscripts\u005cdemo\u005crun_demo.ps1'; "
        "'accepted'"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "accepted"


def test_stop_tunnel_accepts_json_deserialized_owner_identity() -> None:
    digest = _release_digest()
    owner_json = json.dumps(
        {
            "schema_version": "imp.demo.tunnel-owner.v1",
            "release_manifest_sha256": digest,
            "process_id": 42,
            "process_start_time_utc": "2026-07-22T01:02:03.1234567Z",
            "owner_nonce": "0123456789abcdef0123456789abcdef",
            "executable_path": "E\u003a/cloudflared.exe",
            "local_url": "http://127.0.0.1:7860",
        }
    )
    body = (
        f"$record='{owner_json}' | ConvertFrom-Json; "
        "$record.process_id=[long]$record.process_id; $record.process_start_time_utc=[datetime]$record.process_start_time_utc; "
        "function Resolve-DemoStopRoot { 'E\u003a\u005crepo' }; "
        f"function Assert-ReleaseManifest {{ '{digest}' }}; "
        "function Read-OwnedRecord { $record }; "
        "function Get-ExactProcess { [pscustomobject]@{ProcessId=42;CreationDate=[datetime]'2026-07-22T01:02:03.1234567Z';"
        "ExecutablePath='E\u003a\u005ccloudflared.exe';CommandLine='cloudflared tunnel --url http://127.0.0.1:7860'} }; "
        "function Stop-ProcessAndWait { $script:events.Add('stop') }; "
        "function Remove-OwnedRecord { $script:events.Add('remove') }; "
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; Stop-OwnedCloudflared; $script:events | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == ["stop", "remove"]


@pytest.mark.parametrize("release_digest", (None, "0" * 64))
def test_stop_tunnel_rejects_missing_or_stale_release_digest(
    release_digest: str | None,
) -> None:
    digest = _release_digest()
    release_field = (
        "" if release_digest is None else f"release_manifest_sha256='{release_digest}';"
    )
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Resolve-DemoStopRoot { 'E\u003a\u005crepo' }; "
        f"function Assert-ReleaseManifest {{ '{digest}' }}; "
        "function Read-OwnedRecord { [pscustomobject]@{"
        "schema_version='imp.demo.tunnel-owner.v1';process_id=42;"
        "process_start_time_utc='2026-07-22T01:02:03.1234567Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef';"
        "executable_path='E\u003a\u005ccloudflared.exe';local_url='http://127.0.0.1:7860';"
        f"{release_field}}} }}; "
        "function Get-ExactProcess { $script:events.Add('process'); $null }; "
        "function Remove-OwnedRecord { $script:events.Add('remove') }; "
        "$rejected=$false; try{Stop-OwnedCloudflared}catch{$rejected=$true}; "
        "[pscustomobject]@{rejected=$rejected;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "rejected": True,
        "events": [],
    }


def test_tunnel_preserve_stop_event_contains_release_digest(tmp_path: Path) -> None:
    directory = tmp_path / "demo_runtime/preserved/run-a/tunnel"
    directory.mkdir(parents=True)
    digest = _release_digest()
    owner = directory / "owner-current.json"
    owner.write_text("{}", encoding="ascii")
    body = (
        f"$path=Write-PreserveComponentStopRecord -Root '{_powershell_literal(tmp_path)}' "
        f"-RunId 'run-a' -Component 'tunnel' -OwnerRecordPath '{_powershell_literal(owner)}' "
        f"-ReleaseManifestSha256 '{digest}'; "
        "Get-Content -LiteralPath $path -Raw"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    event = json.loads(result.stdout.strip().splitlines()[-1])
    assert event["release_manifest_sha256"] == digest


@pytest.mark.parametrize(
    "script_path", ["scripts/demo/run_tunnel.ps1", "scripts/demo/stop_demo.ps1"]
)
def test_tunnel_pid_reuse_with_start_time_mismatch_is_rejected(
    script_path: str,
) -> None:
    assert "function Assert-TunnelProcessIdentity" in _read(script_path)
    body = (
        "$record=[pscustomobject]@{process_id=42;"
        "process_start_time_utc='2026-07-22T00:00:00.0000000Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef';"
        "executable_path='E\u003a\u005ccloudflared.exe';local_url='http://127.0.0.1:7860'}; "
        "$process=[pscustomobject]@{ProcessId=42;"
        "CreationDate=[datetime]'2026-07-22T00:00:01Z';"
        "ExecutablePath='E\u003a\u005ccloudflared.exe';"
        "CommandLine='cloudflared tunnel --url http://127.0.0.1:7860'}; "
        "$rejected=$false; try{Assert-TunnelProcessIdentity -Record $record "
        "-Process $process}catch{$rejected=$true}; $rejected"
    )
    result = _run_launcher_function_harness(script_path, body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


@pytest.mark.parametrize("script_path", ["scripts/demo/run_tunnel.ps1"])
def test_gradio_listener_requires_owned_launcher_parent_and_nonce(
    script_path: str,
) -> None:
    body = (
        "$record=[pscustomobject]@{launcher_pid=42;"
        "owner_nonce='0123456789abcdef0123456789abcdef';"
        "python_path='E\u003a\u005cpython.exe'}; "
        "$connection=[pscustomobject]@{LocalAddress='127.0.0.1';LocalPort=7860;"
        "OwningProcess=84}; "
        "$process=[pscustomobject]@{ProcessId=84;ParentProcessId=99;"
        "ExecutablePath='E\u003a\u005cpython.exe';CommandLine='python.exe -m "
        "lesion_robustness.demo.app --host 127.0.0.1 --port 7860'}; "
        "$rejected=$false; try{Assert-GradioListenerProcessIdentity "
        "-Record $record -Connection $connection -Process $process}"
        "catch{$rejected=$true}; $rejected"
    )
    result = _run_launcher_function_harness(script_path, body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_stop_gradio_listener_rejects_parent_outside_valid_wrapper() -> None:
    body = (
        "$record=[pscustomobject]@{launcher_pid=42;"
        "owner_nonce='0123456789abcdef0123456789abcdef';"
        "python_path='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe'}; "
        "$connection=[pscustomobject]@{LocalAddress='127.0.0.1';LocalPort=7860;"
        "OwningProcess=84}; "
        "$wrapper=[pscustomobject]@{ProcessId=83;ParentProcessId=42;"
        "ExecutablePath='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe';"
        "CommandLine='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe -m "
        "lesion_robustness.demo.app --host 127.0.0.1 --port 7860'}; "
        "$process=[pscustomobject]@{ProcessId=84;ParentProcessId=99;"
        "ExecutablePath='C\u003a\u005cPython312\u005cpython.exe';CommandLine='C\u003a\u005cPython312\u005cpython.exe -m "
        "lesion_robustness.demo.app --host 127.0.0.1 --port 7860'}; "
        "$message=''; try{Assert-GradioListenerProcessIdentity "
        "-Record $record -Connection $connection -Process $process "
        "-WrapperProcess $wrapper}catch{$message=$_.Exception.Message}; $message"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Gradio listener process ownership proof failed."


def test_launcher_pid_reuse_with_start_time_mismatch_is_rejected() -> None:
    assert "function Assert-LauncherProcessIdentity" in _read(
        "scripts/demo/stop_demo.ps1"
    )
    body = (
        "$record=[pscustomobject]@{launcher_pid=42;"
        "launcher_start_time_utc='2026-07-21T00:00:00.0000000Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef'}; "
        "$process=[pscustomobject]@{ProcessId=42;CreationDate=[datetime]'2026-07-21T00:00:01Z';"
        "ExecutablePath='C\u003a\u005cWindows\u005cSystem32\u005cWindowsPowerShell\u005cv1.0\u005cpowershell.exe';"
        "CommandLine='powershell -File E\u003a\u005crepo\u005cscripts\u005cdemo\u005crun_demo.ps1'}; "
        "$rejected=$false; try{Assert-LauncherProcessIdentity -Record $record "
        "-Process $process -ExpectedScript 'E\u003a\u005crepo\u005cscripts\u005cdemo\u005crun_demo.ps1'}"
        "catch{$rejected=$true}; $rejected"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


@pytest.mark.parametrize(
    "script_path", ["scripts/demo/run_tunnel.ps1", "scripts/demo/stop_demo.ps1"]
)
def test_launcher_identity_accepts_submillisecond_cim_rounding(
    script_path: str,
) -> None:
    body = (
        "$record=[pscustomobject]@{launcher_pid=42;"
        "launcher_start_time_utc=[datetime]'2026-07-22T13:11:56.0272242Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef'}; "
        "$process=[pscustomobject]@{ProcessId=42;"
        "CreationDate=[datetime]'2026-07-22T13:11:56.0272240Z';"
        "ExecutablePath='C\u003a\u005cWindows\u005cSystem32\u005cWindowsPowerShell\u005cv1.0\u005cpowershell.exe';"
        "CommandLine='powershell -File E\u003a\u005crepo\u005cscripts\u005cdemo\u005crun_demo.ps1'}; "
        "Assert-LauncherProcessIdentity -Record $record -Process $process "
        "-ExpectedScript 'E\u003a\u005crepo\u005cscripts\u005cdemo\u005crun_demo.ps1'; 'accepted'"
    )
    result = _run_launcher_function_harness(script_path, body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "accepted"


@pytest.mark.parametrize(
    "script_path", ["scripts/demo/run_tunnel.ps1", "scripts/demo/stop_demo.ps1"]
)
def test_tunnel_identity_accepts_submillisecond_cim_rounding(
    script_path: str,
) -> None:
    body = (
        "$record=[pscustomobject]@{process_id=42;"
        "process_start_time_utc=[datetime]'2026-07-22T13:11:56.0272242Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef';"
        "executable_path='E\u003a\u005ccloudflared.exe';local_url='http://127.0.0.1:7860'}; "
        "$process=[pscustomobject]@{ProcessId=42;"
        "CreationDate=[datetime]'2026-07-22T13:11:56.0272240Z';"
        "ExecutablePath='E\u003a\u005ccloudflared.exe';"
        "CommandLine='cloudflared tunnel --url http://127.0.0.1:7860'}; "
        "Assert-TunnelProcessIdentity -Record $record -Process $process; 'accepted'"
    )
    result = _run_launcher_function_harness(script_path, body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "accepted"


@pytest.mark.parametrize(
    "script_path", ["scripts/demo/run_tunnel.ps1", "scripts/demo/stop_demo.ps1"]
)
@pytest.mark.parametrize(
    ("actual_start_time", "expected"),
    [
        ("2026-07-22T13:11:56.0282242Z", "True"),
        ("2026-07-22T13:11:56.0282243Z", "False"),
    ],
)
def test_process_start_time_tolerance_has_a_strict_one_millisecond_boundary(
    script_path: str, actual_start_time: str, expected: str
) -> None:
    body = (
        "$expected=[datetime]'2026-07-22T13:11:56.0272242Z'; "
        f"$actual=[datetime]'{actual_start_time}'; "
        "Test-ProcessStartTimeMatch -Expected $expected -Actual $actual"
    )
    result = _run_launcher_function_harness(script_path, body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_zero_listener_still_stops_valid_owned_launcher_before_cleanup() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Resolve-DemoStopRoot { 'E\u003a\u005crepo' }; "
        "function Read-OwnedRecord { [pscustomobject]@{launcher_pid=42;"
        "session_path='E\u003a\u005crepo\u005cdemo_runtime\u005csessions\u005cdemo-0123456789abcdef0123456789abcdef'} }; "
        "function Assert-GradioOwnerRecord {}; "
        "function Get-ExactProcess { [pscustomobject]@{ProcessId=42} }; "
        "function Assert-LauncherProcessIdentity {}; "
        "function Get-NetTCPConnection { @() }; "
        "function Stop-ProcessAndWait { param($ProcessId,$Label);"
        "$script:events.Add(($Label + ':' + $ProcessId)) }; "
        "function Remove-OwnedGradioState { $script:events.Add('cleanup') }; "
        "Stop-OwnedGradio; $script:events | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == [
        "Gradio launcher:42",
        "cleanup",
    ]


def test_preserve_stop_accepts_run_demo_gradio_owner_schema() -> None:
    digest = _release_digest()
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Resolve-DemoStopRoot { 'E\u003a\u005crepo' }; "
        "function Get-PreserveActiveOwnerRecordPath { 'E\u003a\u005crepo\u005cdemo_runtime\u005cpreserved\u005crun-a\u005cgradio\u005cowner-current.json' }; "
        "function Read-OwnedRecord { [pscustomobject]@{"
        "schema_version='imp.demo.gradio-owner.v1';"
        f"release_manifest_sha256='{digest}';"
        "public_tunnel_mode=$true;preserve_mode=$true;"
        "launcher_pid=42;launcher_start_time_utc='2026-07-22T01:02:03.1234567Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef';"
        "python_path='E\u003a\u005cpython.exe';session_path='E\u003a\u005crepo\u005cdemo_runtime\u005csessions\u005cdemo-0123456789abcdef0123456789abcdef';"
        "host='127.0.0.1';port=7860} }; "
        "function Get-ExactProcess { [pscustomobject]@{ProcessId=42} }; "
        "function Assert-LauncherProcessIdentity {}; "
        "function Get-NetTCPConnection { @() }; "
        "function Stop-ProcessAndWait { param($ProcessId,$Label); $script:events.Add(($Label + ':' + $ProcessId)) }; "
        "function Write-PreserveComponentStopRecord { $script:events.Add('journal') }; "
        "Stop-OwnedGradio -PreserveMode -RunId 'run-a'; "
        "$script:events | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == [
        "Gradio launcher:42",
        "journal",
    ]


def test_stop_gradio_proves_venv_wrapper_chain_and_stops_only_descendants() -> None:
    digest = _release_digest()
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Resolve-DemoStopRoot { 'E\u003a\u005crepo' }; "
        "function Get-PreserveActiveOwnerRecordPath { 'E\u003a\u005crepo\u005cdemo_runtime\u005cpreserved\u005crun-a\u005cgradio\u005cowner-current.json' }; "
        "function Read-OwnedRecord { [pscustomobject]@{"
        "schema_version='imp.demo.gradio-owner.v1';"
        f"release_manifest_sha256='{digest}';"
        "public_tunnel_mode=$true;preserve_mode=$true;"
        "launcher_pid=1752;launcher_start_time_utc='2026-07-22T01:02:03.1234567Z';"
        "owner_nonce='0123456789abcdef0123456789abcdef';"
        "python_path='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe';"
        "session_path='E\u003a\u005crepo\u005cdemo_runtime\u005csessions\u005cdemo-0123456789abcdef0123456789abcdef';"
        "host='127.0.0.1';port=7860} }; "
        "function Get-ExactProcess { param($ProcessId); switch($ProcessId){"
        "1752 { [pscustomobject]@{ProcessId=1752;ParentProcessId=1} }"
        "21476 { [pscustomobject]@{ProcessId=21476;ParentProcessId=1752;ExecutablePath='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe';CommandLine='E\u003a\u005crepo\u005c.venv-win\u005cScripts\u005cpython.exe -m lesion_robustness.demo.app --host 127.0.0.1 --port 7860'} }"
        "33760 { [pscustomobject]@{ProcessId=33760;ParentProcessId=21476;ExecutablePath='C\u003a\u005cPython312\u005cpython.exe';CommandLine='C\u003a\u005cPython312\u005cpython.exe -m lesion_robustness.demo.app --host 127.0.0.1 --port 7860'} }"
        "32048 { [pscustomobject]@{ProcessId=32048;ParentProcessId=33760;ExecutablePath='C\u003a\u005cPython312\u005cpython.exe';CommandLine='C\u003a\u005cPython312\u005cpython.exe worker'} }"
        "999 { [pscustomobject]@{ProcessId=999;ParentProcessId=1} } } }; "
        "function Assert-LauncherProcessIdentity {}; "
        "function Get-NetTCPConnection { [pscustomobject]@{LocalAddress='127.0.0.1';LocalPort=7860;OwningProcess=33760} }; "
        "function Get-CimInstance { @([pscustomobject]@{ProcessId=21476;ParentProcessId=1752},[pscustomobject]@{ProcessId=33760;ParentProcessId=21476},[pscustomobject]@{ProcessId=32048;ParentProcessId=33760},[pscustomobject]@{ProcessId=999;ParentProcessId=1}) }; "
        "function Stop-ProcessAndWait { param($ProcessId,$Label); $script:events.Add(($Label + ':' + $ProcessId)) }; "
        "function Write-PreserveComponentStopRecord { $script:events.Add('journal') }; "
        "Stop-OwnedGradio -PreserveMode -RunId 'run-a'; "
        "$script:events | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == [
        "Gradio descendant:32048",
        "Gradio descendant:33760",
        "Gradio descendant:21476",
        "Gradio launcher:1752",
        "journal",
    ]


def test_stop_gradio_accepts_launcher_that_exits_after_descendant_stop() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; $script:launcherGone=$false; "
        "function Resolve-DemoStopRoot { 'E\u003a\u005crepo' }; "
        "function Read-OwnedRecord { [pscustomobject]@{launcher_pid=42;session_path='E\u003a\u005crepo\u005cdemo_runtime\u005csessions\u005cdemo-0123456789abcdef0123456789abcdef'} }; "
        "function Assert-GradioOwnerRecord {}; function Assert-LauncherProcessIdentity {}; function Get-NetTCPConnection { @() }; "
        "function Get-CimInstance { @([pscustomobject]@{ProcessId=84;ParentProcessId=42}) }; "
        "function Get-ExactProcess { param($ProcessId); if($ProcessId -eq 42){if($script:launcherGone){return $null}; return [pscustomobject]@{ProcessId=42}}; if($ProcessId -eq 84){return [pscustomobject]@{ProcessId=84;ParentProcessId=42}} }; "
        "function Stop-ProcessAndWait { param($ProcessId,$Label); if($ProcessId -eq 84){$script:launcherGone=$true; $script:events.Add('descendant')}else{throw 'Cannot find a process'} }; "
        "function Remove-OwnedGradioState { $script:events.Add('cleanup') }; "
        "Stop-OwnedGradio; $script:events | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == [
        "descendant",
        "cleanup",
    ]


def test_stop_process_and_wait_accepts_descendant_that_vanishes_before_stop() -> None:
    body = (
        "function Get-ExactProcess { $null }; "
        "Stop-ProcessAndWait -ProcessId 2147483647 -Label 'Gradio descendant'; "
        "'accepted'"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "accepted"


def test_stop_process_and_wait_allows_gpu_descendant_teardown_window() -> None:
    body = (
        "$script:timeout=0; "
        "function Stop-Process { param($Id,[switch]$Force) }; "
        "function Wait-Process { param($Id,$Timeout,$ErrorAction); $script:timeout=$Timeout }; "
        "function Get-Process { $null }; "
        "Stop-ProcessAndWait -ProcessId 42 -Label 'Gradio descendant'; "
        "$script:timeout"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "30"


def test_stop_process_and_wait_rejects_still_present_or_reused_pid() -> None:
    body = (
        "function Get-ExactProcess { [pscustomobject]@{ProcessId=2147483647} }; "
        "$message=''; try{Stop-ProcessAndWait -ProcessId 2147483647 "
        "-Label 'Gradio descendant'}catch{$message=$_.Exception.Message}; $message"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert "Cannot find a process" in result.stdout


def test_stop_gradio_rejects_reused_launcher_pid_before_stop() -> None:
    body = (
        "$script:launcherReads=0; "
        "function Resolve-DemoStopRoot { 'E\u003a\u005crepo' }; "
        "function Read-OwnedRecord { [pscustomobject]@{launcher_pid=42;session_path='E\u003a\u005crepo\u005cdemo_runtime\u005csessions\u005cdemo-0123456789abcdef0123456789abcdef'} }; "
        "function Assert-GradioOwnerRecord {}; function Get-NetTCPConnection { @() }; function Get-CimInstance { @() }; "
        "function Get-ExactProcess { param($ProcessId); if($ProcessId -ne 42){return $null}; $script:launcherReads++; [pscustomobject]@{ProcessId=42;CreationDate=([datetime]::UtcNow.AddSeconds($script:launcherReads))} }; "
        "function Assert-LauncherProcessIdentity { param($Record,$Process,$ExpectedScript); if($script:launcherReads -gt 1){throw 'PID reuse detected'} }; "
        "function Stop-ProcessAndWait { throw 'must not stop reused launcher' }; "
        "$message=''; try{Stop-OwnedGradio}catch{$message=$_.Exception.Message}; $message"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PID reuse detected"


def test_shutdown_aggregates_first_error_and_still_proves_ports() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Stop-OwnedCloudflared { $script:events.Add('cloudflare'); throw 'first' }; "
        "function Stop-OwnedGradio { $script:events.Add('gradio') }; "
        "function Stop-OwnedSidecar { $script:events.Add('sidecar') }; "
        "function Remove-OwnedRuntimeFiles { $script:events.Add('cleanup') }; "
        "function Wait-DemoPortsClosed { $script:events.Add('ports'); $true }; "
        "$code=Invoke-DemoStop -Root 'ignored'; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "code": 5,
        "events": ["cloudflare", "gradio", "sidecar", "cleanup", "ports"],
    }


def test_sidecar_cleanup_preserves_record_when_stop_and_absence_proof_fail() -> None:
    root = _powershell_literal(ROOT)
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        f"function Resolve-SidecarContext {{ [pscustomobject]@{{Root='{root}';RuntimeRoot='RR';"
        "BundlePath='B';DockerPath='D';OwnerToken='0123456789abcdef0123456789abcdef'} }; "
        "function Assert-VerifiedBundle {}; function Assert-PinnedDockerImage {}; "
        "function Start-OwnedSidecar { 'container-id' }; "
        "function Wait-PinnedSidecarHealth { throw 'timeout' }; "
        "function Stop-OwnedSidecar { $script:events.Add('stop'); throw 'inspect failed' }; "
        "function Test-OwnedSidecarStoppedOrAbsent { $false }; "
        "function Remove-SidecarOwnerRecord { $script:events.Add('record') }; "
        "$code=Invoke-SidecarLaunch; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "code": 5,
        "events": ["stop"],
    }


def test_sidecar_start_record_failure_preserves_unresolved_owner_record() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Test-ContainerNamePresent { $false }; "
        "function New-SidecarRunArguments { @('run') }; "
        "function Invoke-DockerCommand { 'a' * 64 }; "
        "function Write-SidecarOwnerRecord { throw 'record write failed' }; "
        "function Stop-OwnedSidecar { $script:events.Add('stop'); throw 'inspect failed' }; "
        "function Test-SidecarContainerAbsent { $false }; "
        "function Remove-SidecarOwnerRecord { $script:events.Add('record') }; "
        "$context=[pscustomobject]@{DockerPath='D';BundlePath='B';"
        "OwnerToken='0123456789abcdef0123456789abcdef';RuntimeRoot='RR'}; "
        "$rejected=$false; try{Start-OwnedSidecar -Context $context}catch{$rejected=$true}; "
        "[pscustomobject]@{rejected=$rejected;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "rejected": True,
        "events": ["stop"],
    }


def test_sidecar_preexisting_owner_record_blocks_docker_run_without_deletion() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Test-ContainerNamePresent { $false }; "
        "function Get-SidecarOwnerRecordPath { $script:events.Add('preflight'); 'RR\u005csidecar.json' }; "
        "function Test-Path { $true }; "
        "function New-SidecarRunArguments { @('run') }; "
        "function Invoke-DockerCommand { $script:events.Add('run'); 'a' * 64 }; "
        "function Stop-OwnedSidecar { $script:events.Add('stop') }; "
        "function Remove-SidecarOwnerRecord { $script:events.Add('remove') }; "
        "$context=[pscustomobject]@{DockerPath='D';BundlePath='B';"
        "OwnerToken='0123456789abcdef0123456789abcdef';RuntimeRoot='RR'}; "
        "$rejected=$false; try{Start-OwnedSidecar -Context $context}catch{$rejected=$true}; "
        "[pscustomobject]@{rejected=$rejected;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "rejected": True,
        "events": ["preflight"],
    }


def test_sidecar_record_write_failure_never_removes_uncommitted_record() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Test-ContainerNamePresent { $false }; "
        "function Get-SidecarOwnerRecordPath { $script:events.Add('preflight'); 'RR\u005csidecar.json' }; "
        "function Test-Path { $false }; "
        "function New-SidecarRunArguments { @('run') }; "
        "function Invoke-DockerCommand { $script:events.Add('run'); 'a' * 64 }; "
        "function Write-SidecarOwnerRecord { $script:events.Add('write'); throw 'partial write' }; "
        "function Stop-OwnedSidecar { $script:events.Add('stop') }; "
        "function Remove-SidecarOwnerRecord { $script:events.Add('remove') }; "
        "$context=[pscustomobject]@{DockerPath='D';BundlePath='B';"
        "OwnerToken='0123456789abcdef0123456789abcdef';RuntimeRoot='RR'}; "
        "$rejected=$false; try{Start-OwnedSidecar -Context $context}catch{$rejected=$true}; "
        "[pscustomobject]@{rejected=$rejected;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "rejected": True,
        "events": ["preflight", "run", "write", "stop"],
    }


def test_sidecar_run_accepts_diagnostics_plus_one_exact_container_id() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Assert-SidecarOwnerRecordAbsent {}; "
        "function Test-ContainerNamePresent { $false }; "
        "function New-SidecarRunArguments { @('run') }; "
        "function Invoke-DockerCommand { @('diagnostic warning', ('a' * 64)) }; "
        "function Write-SidecarOwnerRecord { $script:events.Add('record') }; "
        "function Stop-OwnedSidecarByFixedName { $script:events.Add('cleanup') }; "
        "$context=[pscustomobject]@{DockerPath='D';BundlePath='B';"
        "OwnerToken='0123456789abcdef0123456789abcdef';RuntimeRoot='RR'}; "
        "$rejected=$false;$containerId=$null;"
        "try{$containerId=Start-OwnedSidecar -Context $context}catch{$rejected=$true}; "
        "[pscustomobject]@{rejected=$rejected;container_id=$containerId;"
        "events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "rejected": False,
        "container_id": "a" * 64,
        "events": ["record"],
    }


@pytest.mark.parametrize(
    "run_output",
    ["@('diagnostic only')", "@(('a' * 64), ('b' * 64))"],
)
def test_sidecar_malformed_run_output_cleans_fixed_owned_name(
    run_output: str,
) -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Assert-SidecarOwnerRecordAbsent {}; "
        "function Test-ContainerNamePresent { $false }; "
        "function New-SidecarRunArguments { @('run') }; "
        f"function Invoke-DockerCommand {{ {run_output} }}; "
        "function Stop-OwnedSidecarByFixedName { $script:events.Add('cleanup') }; "
        "$context=[pscustomobject]@{DockerPath='D';BundlePath='B';"
        "OwnerToken='0123456789abcdef0123456789abcdef';RuntimeRoot='RR'}; "
        "$rejected=$false;try{Start-OwnedSidecar -Context $context}catch{$rejected=$true}; "
        "[pscustomobject]@{rejected=$rejected;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "rejected": True,
        "events": ["cleanup"],
    }


def test_sidecar_cleanup_removes_record_after_absence_is_proven() -> None:
    root = _powershell_literal(ROOT)
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        f"function Resolve-SidecarContext {{ [pscustomobject]@{{Root='{root}';RuntimeRoot='RR';"
        "BundlePath='B';DockerPath='D';OwnerToken='0123456789abcdef0123456789abcdef'} }; "
        "function Assert-VerifiedBundle {}; function Assert-PinnedDockerImage {}; "
        "function Start-OwnedSidecar { 'container-id' }; "
        "function Wait-PinnedSidecarHealth { throw 'timeout' }; "
        "function Stop-OwnedSidecar { $script:events.Add('stop'); throw 'inspect failed' }; "
        "function Test-OwnedSidecarStoppedOrAbsent { $script:events.Add('absence'); $true }; "
        "function Remove-SidecarOwnerRecord { $script:events.Add('record') }; "
        "$code=Invoke-SidecarLaunch; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "code": 5,
        "events": ["stop", "absence", "record"],
    }


@pytest.mark.parametrize(
    ("absence_proven", "rejected", "expected_events"),
    [
        (False, True, ["inspect", "absence"]),
        (True, False, ["inspect", "absence", "record"]),
    ],
)
def test_stop_sidecar_preserves_ambiguous_inspect_but_removes_proven_absence(
    absence_proven: bool, rejected: bool, expected_events: list[str]
) -> None:
    powershell_absent = "$true" if absence_proven else "$false"
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Resolve-DemoStopRoot { 'E\u003a\u005crepo' }; "
        "function Read-OwnedRecord { [pscustomobject]@{"
        "schema_version='imp.demo.sidecar-owner.v1';"
        "container_id=('a' * 64);container_name='imp-nnunet-loop192';"
        "owner_token='0123456789abcdef0123456789abcdef';docker_path='D'} }; "
        "function Get-Item { [pscustomobject]@{PSIsContainer=$false;"
        "Attributes=[IO.FileAttributes]::Normal;Name='docker.exe';FullName='D'} }; "
        "function Get-SidecarContainerIdentity { $script:events.Add('inspect'); "
        "throw 'daemon or permission failure' }; "
        "function Test-SidecarContainerAbsent { $script:events.Add('absence'); "
        f"{powershell_absent} }}; "
        "function Remove-OwnedRecord { $script:events.Add('record') }; "
        "$wasRejected=$false; try{Stop-OwnedSidecar}catch{$wasRejected=$true}; "
        "[pscustomobject]@{rejected=$wasRejected;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "rejected": rejected,
        "events": expected_events,
    }


def test_stop_sidecar_escapes_exact_inspect_template_for_windows_powershell() -> None:
    body = (
        "$global:captured=$null; "
        "function Invoke-DockerCommand { param($DockerPath,$Arguments,$Label) "
        "$global:captured=@($Arguments) }; "
        "Get-SidecarContainerIdentity -DockerPath 'D' -ContainerId ('a' * 64) | Out-Null; "
        "[pscustomobject]@{major=$PSVersionTable.PSVersion.Major;"
        "arguments=@($global:captured)} | ConvertTo-Json -Depth 3 -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if payload["major"] != 5:
        pytest.skip("Windows PowerShell 5.1 regression")
    assert payload["arguments"] == [
        "container",
        "inspect",
        "--format",
        r'{{.Id}}|{{index .Config.Labels \"imp.demo.owner\"}}|{{.Name}}',
        "a" * 64,
    ]


def test_run_sidecar_escapes_all_inspect_templates_for_windows_powershell() -> None:
    body = (
        "$global:calls=New-Object 'System.Collections.Generic.List[object]'; "
        "function Invoke-DockerCommand { param($DockerPath,$Arguments,$Label) "
        "$global:calls.Add([pscustomobject]@{label=$Label;arguments=@($Arguments)}); "
        "if($Label -eq 'fixed-name sidecar ownership inspection'){"
        "return @((('a' * 64) + '|b|/imp-nnunet-run-a'))}; "
        "if($Label -eq 'owned sidecar inspection'){"
        "return @((('a' * 64) + '|b|/imp-nnunet-run-a'))}; "
        "if($Label -eq 'sidecar fixed-name absence proof'){return @()}; "
        "return @((('a' * 64) + '|b|/imp-nnunet-run-a|false')) }; "
        "Stop-OwnedSidecar -DockerPath 'D' -ContainerId ('a' * 64) "
        "-OwnerToken ('b' * 1) -ContainerName 'imp-nnunet-run-a'; "
        "Test-OwnedSidecarStoppedOrAbsent -DockerPath 'D' -ContainerId ('a' * 64) "
        "-OwnerToken ('b' * 1) -ContainerName 'imp-nnunet-run-a' -PreserveMode | Out-Null; "
        "Stop-OwnedSidecarByFixedName -DockerPath 'D' -OwnerToken 'b' "
        "-ContainerName 'imp-nnunet-run-a'; "
        "Test-SidecarContainerNameAbsent -DockerPath 'D' -OwnerToken 'b' "
        "-ContainerName 'imp-nnunet-run-a' | Out-Null; "
        "[pscustomobject]@{major=$PSVersionTable.PSVersion.Major;"
        "calls=@($global:calls | Where-Object {$_.label -match 'inspection|proof'} | "
        "ForEach-Object {[pscustomobject]@{label=$_.label;arguments=@($_.arguments)}})} "
        "| ConvertTo-Json -Depth 5 -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if payload["major"] != 5:
        pytest.skip("Windows PowerShell 5.1 regression")
    by_label = {entry["label"]: entry["arguments"] for entry in payload["calls"]}
    escaped = r'\"imp.demo.owner\"'
    assert by_label == {
        "owned sidecar inspection": [
            "container",
            "inspect",
            "--format",
            "{{.Id}}|{{index .Config.Labels " + escaped + "}}|{{.Name}}",
            "a" * 64,
        ],
        "stopped sidecar identity proof": [
            "container",
            "inspect",
            "--format",
            "{{.Id}}|{{index .Config.Labels " + escaped + "}}|{{.Name}}|{{.State.Running}}",
            "a" * 64,
        ],
        "fixed-name sidecar ownership inspection": [
            "container",
            "inspect",
            "--format",
            "{{.Id}}|{{index .Config.Labels " + escaped + "}}|{{.Name}}",
            "imp-nnunet-run-a",
        ],
        "sidecar fixed-name absence proof": [
            "container",
            "list",
            "--all",
            "--filter",
            "name=^/imp-nnunet-run-a$",
            "--format",
            "{{.Names}}|{{.Label " + escaped + "}}",
        ],
    }


def test_run_sidecar_ps5_name_probe_accepts_absence_but_rejects_daemon_failure(
    tmp_path: Path,
) -> None:
    fake_docker = tmp_path / "docker-probe.cmd"
    fake_docker.write_text(
        "@echo off\n"
        "if \"%2\"==\"list\" if \"%IMP_TEST_DOCKER_MODE%\"==\"absent\" exit /b 0\n"
        "if \"%IMP_TEST_DOCKER_MODE%\"==\"absent\" (\n"
        "  1>&2 echo Error response from daemon: No such container: %3\n"
        "  exit /b 1\n"
        ")\n"
        "1>&2 echo Error response from daemon: daemon unavailable\n"
        "exit /b 1\n",
        encoding="ascii",
    )
    body = (
        "$env:IMP_TEST_DOCKER_MODE='absent'; "
        f"$absent=Test-ContainerNamePresent -DockerPath '{_powershell_literal(fake_docker)}' "
        "-ContainerName 'imp-nnunet-run-a'; "
        "$env:IMP_TEST_DOCKER_MODE='daemon'; $rejected=$false; "
        f"try{{Test-ContainerNamePresent -DockerPath '{_powershell_literal(fake_docker)}' "
        "-ContainerName 'imp-nnunet-run-a' | Out-Null}catch{$rejected=$true}; "
        "[pscustomobject]@{major=$PSVersionTable.PSVersion.Major;"
        "absent=$absent;rejected=$rejected} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if payload["major"] != 5:
        pytest.skip("Windows PowerShell 5.1 regression")
    assert payload == {"major": 5, "absent": False, "rejected": True}


def test_stop_demo_wait_treats_test_listener_port_as_open() -> None:
    body = (
        "function Get-NetTCPConnection { [pscustomobject]@{LocalPort=7861} }; "
        "function Start-Sleep {}; "
        "$closed=Wait-DemoPortsClosed -TimeoutSeconds 0; "
        "[pscustomobject]@{closed=$closed} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"closed": False}


def test_demo_embedded_smoke_requires_task4_health_identity() -> None:
    script = _read("scripts/demo/run_demo.ps1")

    assert "health.device != \"cuda:0\"" in script
    assert "health.ready is not True" in script
    assert "health.status" not in script


def test_tunnel_exposes_only_gradio() -> None:
    script = _read("scripts/demo/run_tunnel.ps1")

    assert script.count("http://127.0.0.1:7860") >= 1
    assert "7862" not in script


def test_sidecar_arguments_lock_loopback_gpu_mount_and_lifecycle() -> None:
    result = _run_launcher_function_harness(
        "scripts/demo/run_sidecar.ps1",
        "$value=@(New-SidecarRunArguments -BundlePath 'E\u003a\u005cowned bundle' "
        "-OwnerToken '0123456789abcdef0123456789abcdef'); "
        "$value | ConvertTo-Json -Compress",
    )

    assert result.returncode == 0, result.stderr
    arguments = json.loads(result.stdout.strip().splitlines()[-1])
    command = " ".join(arguments)
    assert arguments[:4] == ["run", "--detach", "--rm", "--name"]
    assert "--gpus device=0" in command
    assert "--publish 127.0.0.1:7862:7862" in command
    assert "source=E\u003a\u005cowned bundle,target=/models/loop192,readonly" in command
    assert "--read-only" in arguments
    assert "/tmp:rw,noexec,nosuid,size=256m" in arguments
    assert "--restart no" in command


def test_sidecar_timeout_stops_only_container_started_by_this_launch() -> None:
    root = _powershell_literal(ROOT)
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        f"function Resolve-SidecarContext {{ [pscustomobject]@{{Root='{root}';RuntimeRoot='RR';"
        "BundlePath='B';DockerPath='D';OwnerToken='0123456789abcdef0123456789abcdef'} }; "
        "function Assert-VerifiedBundle { $script:events.Add('bundle') }; "
        "function Assert-PinnedDockerImage { $script:events.Add('image') }; "
        "function Start-OwnedSidecar { $script:events.Add('start'); 'container-id' }; "
        "function Wait-PinnedSidecarHealth { $script:events.Add('health'); throw 'timeout' }; "
        "function Stop-OwnedSidecar { param($DockerPath,$ContainerId,$OwnerToken); "
        "$script:events.Add(('stop:' + $ContainerId + ':' + $OwnerToken)) }; "
        "function Test-OwnedSidecarStoppedOrAbsent { $true }; "
        "$code=Invoke-SidecarLaunch -Root 'ignored' -BundlePath 'ignored' -DockerPath 'ignored'; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/run_sidecar.ps1", body)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "code": 5,
        "events": [
            "bundle",
            "image",
            "start",
            "health",
            "stop:container-id:0123456789abcdef0123456789abcdef",
        ],
    }


def test_stop_demo_orders_owned_resources_and_requires_closed_ports() -> None:
    body = (
        "$script:events=New-Object 'System.Collections.Generic.List[string]'; "
        "function Stop-OwnedCloudflared { $script:events.Add('cloudflare') }; "
        "function Stop-OwnedGradio { $script:events.Add('gradio') }; "
        "function Stop-OwnedSidecar { $script:events.Add('sidecar') }; "
        "function Remove-OwnedRuntimeFiles { $script:events.Add('cleanup') }; "
        "function Wait-DemoPortsClosed { $script:events.Add('ports'); $true }; "
        "$code=Invoke-DemoStop -Root 'ignored'; "
        "[pscustomobject]@{code=$code;events=@($script:events)} | ConvertTo-Json -Compress"
    )
    result = _run_launcher_function_harness("scripts/demo/stop_demo.ps1", body)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "code": 0,
        "events": ["cloudflare", "gradio", "sidecar", "cleanup", "ports"],
    }
    script = _read("scripts/demo/stop_demo.ps1")
    assert script.index("Stop-OwnedCloudflared") < script.index("Stop-OwnedGradio")
    assert script.index("Stop-OwnedGradio") < script.index("Stop-OwnedSidecar")
    for token in ("Get-NetTCPConnection", "7860", "7862", "Assert-OwnedSessionPath"):
        assert token in script


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
        "$LegacyLinuxHome/imp_cache/external_repos/loop170/nnUNet/pyproject.toml",
        "$LegacyLinuxHome/imp_cache/external_repos/loop170/nnUNet/nnunetv2.egg-info/PKG-INFO",
        "$LegacyLinuxHome/imp_cache/external_repos/loop170/nnUNet/.git/HEAD",
        "$LegacyLinuxHome/imp_cache/external_repos/loop170/nnUNet/.git/refs/heads/master",
    ):
        assert token in script
    assert "--privileged" not in script
    assert "--device" not in script
    assert "Mount-VHD -Path $resolvedVhd -ReadOnly" in script
    assert "mount -t ext4 -o ro,noload" in script


def test_recovery_requires_explicit_legacy_user_and_runbook_gates_environment() -> None:
    script = _read("scripts/demo/recover_nnunet_artifacts.ps1")
    parameter_block = script[: script.index("Set-StrictMode")]
    runbook = _read("docs/runbooks/demo-operations.md")

    assert re.search(
        r"\[Parameter\(Mandatory = \$true\)\]\s+"
        r"\[ValidateNotNullOrEmpty\(\)\]\s+"
        r"\[ValidatePattern\('[^']+'\)\]\s+"
        r"\[string\]\$LegacyLinuxUser",
        parameter_block,
    )
    assert "$LegacyLinuxUser =" not in parameter_block
    assert "IsNullOrWhiteSpace($env:IMP_LEGACY_LINUX_USER)" in runbook
    assert "-LegacyLinuxUser $env:IMP_LEGACY_LINUX_USER" in runbook


@pytest.mark.parametrize("legacy_user", [".", "..", "", "bad/user", r"bad\user"])
def test_recovery_rejects_unsafe_legacy_linux_user(legacy_user: str) -> None:
    result = _run_script(
        ROOT / "scripts/demo/recover_nnunet_artifacts.ps1",
        "-VhdPath",
        "missing.vhdx",
        "-ReportPath",
        "missing.json",
        "-OutputRoot",
        "missing-output",
        "-LegacyLinuxUser",
        legacy_user,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "LegacyLinuxUser" in output
    assert "Administrator token required" not in output


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
    assert script.count(".venv-win\u005cScripts\u005cpython.exe") == 1
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


def test_recovery_runbook_passes_explicit_trusted_python() -> None:
    runbook = _read("docs/runbooks/demo-operations.md")
    assert "recover_nnunet_artifacts.ps1" in runbook
    assert "-PythonExe '.venv-win\\Scripts\\python.exe'" in runbook


def test_recovery_container_arguments_lock_mounts_and_image() -> None:
    result = _run_recovery_function_harness(
        ("New-ContainerRecoveryArguments",),
        "$value=@(New-ContainerRecoveryArguments -VhdPath 'E\u003a\u005csource.vhdx' "
        "-OutputRoot 'E\u003a\u005cfresh output'); $value | ConvertTo-Json -Compress",
    )

    assert result.returncode == 0, result.stderr
    arguments = json.loads(result.stdout.strip())
    command = " ".join(arguments)
    assert arguments[:3] == ["run", "--rm", "-i"]
    assert "source=E\u003a\u005csource.vhdx,target=/input/source.vhdx,readonly" in command
    assert "source=E\u003a\u005cfresh output,target=/output" in command
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
        "CommandType='Application';Source='C\u003a\u005cDocker\u005cdocker.exe'} }; "
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
    assert payload["path"] == "C" + ":" + r"\Docker\docker.exe"
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
        "function Invoke-WslContext { '\u002fmnt/e/path with spaces' }; "
        "$value=ConvertTo-WslPath -ContextPrefix @('--system','--') "
        "-WindowsPath 'E\u003a\u005cpath with spaces'; "
        "[pscustomobject]@{type=$value.GetType().FullName;value=$value} | "
        "ConvertTo-Json -Compress",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == {
        "type": "System.String",
        "value": "\u002fmnt/e/path with spaces",
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
        "-LinuxMount '\u002fmnt/wsl/recovery' -PhysicalDrive '\u005c\u005c.\u005cPHYSICALDRIVE9' "
        "-ResolvedVhd 'E\u003a\u005csource.vhdx'); "
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
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\u005c/]", readme)


def test_runbook_documents_dual_live_limits_and_order() -> None:
    text = _read("docs/runbooks/demo-operations.md").lower()

    for token in (
        "run_sidecar.ps1",
        "run_demo.ps1",
        "run_tunnel.ps1",
        "stop_demo.ps1",
        "val_gate_failed_no_test",
        "no ground truth",
        "unauthenticated",
        "reconstructed runtime",
        "demo_runtime/nnunet/recovered-container-final2",
        "recover_nnunet_artifacts.ps1",
        "-vhdpath",
        "-reportpath",
        "-outputroot",
        "recovery=passed",
        "vhd detached",
    ):
        assert token in text
    assert text.index("sidecar") < text.index("gradio") < text.index("cloudflare")


def test_two_machine_runbook_documents_private_artifact_handoff() -> None:
    text = _read("docs/runbooks/two-machine-delivery.md").lower()

    for token in (
        "codex/dual-live-demo",
        "rtx 4060",
        "docker desktop",
        "cloudflared",
        "out-of-band",
        "recovery_receipt.json",
        "run_sidecar.ps1 -checkonly",
        "run_demo.ps1 -checkonly",
        "never push",
    ):
        assert token in text
    assert "quanntm1206/imp-lesion-evidence-demo" in text


def test_powershell_scripts_parse() -> None:
    shell = _powershell()
    for relative in (
        "scripts/demo/run_demo.ps1",
        "scripts/demo/run_tunnel.ps1",
        "scripts/demo/run_sidecar.ps1",
        "scripts/demo/stop_demo.ps1",
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
    _copy_release_manifest(root)
    runtime = root / "demo_runtime"
    runtime.mkdir()
    sentinel = runtime / "sibling-sentinel.txt"
    sentinel.write_text("preserve", encoding="ascii")
    fake_python = _fake_python(tmp_path, capture_environment=True)
    control, candidate = _fake_checkpoint_environment(tmp_path)
    observation = tmp_path / "app-env.txt"
    command = (
        f". '{_powershell_literal(script)}'; "
        "$env:GRADIO_TEMP_DIR='before-gradio'; $env:TMP='before-tmp'; "
        "$env:TEMP='before-temp'; "
        f"$env:IMP_TEST_OBSERVATION='{_powershell_literal(observation)}'; "
        f"$env:IMP_LOOP206_CONTROL_CHECKPOINT='{_powershell_literal(control)}'; "
        f"$env:IMP_LOOP206_CANDIDATE_CHECKPOINT='{_powershell_literal(candidate)}'; "
        "function Assert-DemoSidecarReady {}; "
        f"$code=Invoke-DemoLaunch -Device cuda -Root '{_powershell_literal(root)}' "
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
        "@((Join-Path $valid '..\u005c..\u005c..\u005coutside'),$runtime),"
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
    _copy_release_manifest(root)
    fake_python = _fake_python(tmp_path)
    control, candidate = _fake_checkpoint_environment(tmp_path)
    script = ROOT / "scripts/demo/run_demo.ps1"
    command = (
        f". '{_powershell_literal(script)}'; "
        f"$env:IMP_TEST_APP_EXIT='{app_exit}'; $script:guardCalls=0; "
        f"$env:IMP_LOOP206_CONTROL_CHECKPOINT='{_powershell_literal(control)}'; "
        f"$env:IMP_LOOP206_CANDIDATE_CHECKPOINT='{_powershell_literal(candidate)}'; "
        "function Assert-DemoSidecarReady {}; "
        "function Assert-OwnedSessionPath { param($SessionPath,$RuntimeRoot) "
        "$script:guardCalls++; if($script:guardCalls -gt 1){throw 'simulated cleanup failure'} "
        "[pscustomobject]@{RuntimeRoot=$RuntimeRoot;SessionPath=$SessionPath} }; "
        f"Invoke-DemoLaunch -Device cuda -Root '{_powershell_literal(root)}' "
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
    before = _cloudflared_pids()
    body = (
        "function Invoke-TunnelPreflight { [pscustomobject]@{Root='E\u003a\u005cowned-root';ReleaseManifestSha256=('a' * 64)} }; "
        f"Invoke-TunnelLaunch -CheckOnly -CloudflaredPath "
        f"'{_powershell_literal(tmp_path / 'must-not-be-resolved/cloudflared.exe')}'"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)
    after = _cloudflared_pids()

    assert result.returncode == 0, result.stderr
    assert "tunnel was not started" in result.stdout
    assert result.stdout.strip().splitlines()[-1] == "0"
    assert after == before


def test_tunnel_preserves_resolver_failure_exit_code(tmp_path: Path) -> None:
    body = (
        "function Invoke-TunnelPreflight { [pscustomobject]@{Root='E\u003a\u005cowned-root';ReleaseManifestSha256=('a' * 64)} }; "
        f"Invoke-TunnelLaunch -CloudflaredPath "
        f"'{_powershell_literal(tmp_path / 'missing/cloudflared.exe')}'"
    )
    result = _run_launcher_function_harness("scripts/demo/run_tunnel.ps1", body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "4"
