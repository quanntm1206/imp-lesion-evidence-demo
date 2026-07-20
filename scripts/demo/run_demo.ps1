[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cuda',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'

function Assert-OwnedSessionPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$SessionPath,
        [Parameter(Mandatory = $true)][string]$RuntimeRoot
    )

    $runtimeItem = Get-Item -LiteralPath $RuntimeRoot -Force -ErrorAction Stop
    $sessionItem = Get-Item -LiteralPath $SessionPath -Force -ErrorAction Stop
    if (-not $runtimeItem.PSIsContainer -or -not $sessionItem.PSIsContainer) {
        throw 'Demo runtime and session paths must be directories.'
    }
    $resolvedRuntime = [IO.Path]::GetFullPath($runtimeItem.FullName).TrimEnd(
        [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
    )
    $resolvedSession = [IO.Path]::GetFullPath($sessionItem.FullName).TrimEnd(
        [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
    )
    $prefix = $resolvedRuntime + [IO.Path]::DirectorySeparatorChar
    if (
        [string]::Equals($resolvedSession, $resolvedRuntime, [StringComparison]::OrdinalIgnoreCase) -or
        -not $resolvedSession.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
    ) {
        throw 'Demo session path is outside the owned runtime root.'
    }
    $expectedParent = [IO.Path]::GetFullPath(
        (Join-Path $resolvedRuntime 'sessions')
    ).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $actualParent = [IO.Path]::GetFullPath($sessionItem.Parent.FullName).TrimEnd(
        [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
    )
    if (-not [string]::Equals(
        $actualParent, $expectedParent, [StringComparison]::OrdinalIgnoreCase
    )) {
        throw 'Demo session path does not use the owned sessions parent.'
    }
    if ($sessionItem.Name -cnotmatch '^demo-[0-9a-f]{32}$') {
        throw 'Demo session path does not use an owned session name.'
    }
    foreach ($item in @(
        $runtimeItem,
        (Get-Item -LiteralPath $expectedParent -Force -ErrorAction Stop),
        $sessionItem
    )) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Demo session path contains a reparse point.'
        }
    }
    [pscustomobject]@{
        RuntimeRoot = $resolvedRuntime
        SessionPath = $resolvedSession
    }
}

function Restore-DemoEnvironment {
    param([hashtable]$Snapshot)
    foreach ($name in $Snapshot.Keys) {
        if ($Snapshot[$name].Present) {
            Set-Item -LiteralPath "Env:$name" -Value $Snapshot[$name].Value
        }
        else {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
    }
}

$Preflight = @'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "src"))
model_registry, evidence_registry, dataset_index, candidate_manifest, zero_manifest, live_config, control_checkpoint, candidate_checkpoint = map(Path, sys.argv[2:10])

from lesion_robustness.demo.fixed_cache import (
    DATASET_INDEX_SHA256,
    LIVE_CONFIG_SHA256,
    FixedCacheExpectations,
)
from lesion_robustness.demo.model_service import PINNED_REGISTRY
from lesion_robustness.evidence_registry import validate_registry

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

try:
    evidence = json.loads(evidence_registry.read_text(encoding="ascii"))
    validate_registry(evidence)
    expected_evidence_sha256 = "f6ed2eace90c49ee1b9f0c122e736920791b6301035bf8905c6a0ce27b755f32"
    if evidence.get("registry_sha256") != expected_evidence_sha256:
        raise ValueError("pinned evidence semantic hash")
    for source in evidence["sources"]:
        source_path = (root / source["path"]).resolve()
        source_path.relative_to(root)
        if not source_path.is_file() or sha256(source_path) != source["sha256"]:
            raise ValueError("evidence source hash")
    models = json.loads(model_registry.read_text(encoding="ascii"))
    if models != PINNED_REGISTRY:
        raise ValueError("model registry binding")

    expected = FixedCacheExpectations.loop206()
    artifacts = {
        control_checkpoint: models["control"]["checkpoint_sha256"],
        candidate_checkpoint: models["candidate"]["checkpoint_sha256"],
        candidate_manifest: expected.candidate_manifest_sha256,
        zero_manifest: expected.zero_manifest_sha256,
        dataset_index: DATASET_INDEX_SHA256,
        live_config: LIVE_CONFIG_SHA256,
    }
    for path, expected_sha256 in artifacts.items():
        if not path.is_file() or sha256(path) != expected_sha256:
            raise ValueError("release artifact hash")

    for manifest_path, expected_data_sha256 in (
        (candidate_manifest, expected.candidate_data_sha256),
        (zero_manifest, expected.zero_data_sha256),
    ):
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        data_path = manifest_path.parent / manifest["data"]["file"]
        if manifest["data"]["sha256"] != expected_data_sha256:
            raise ValueError("fixed cache data binding")
        if not data_path.is_file() or sha256(data_path) != expected_data_sha256:
            raise ValueError("fixed cache data hash")

    if models["prior_env"] != "IMP_LOOP206_PRIOR" or models["prior_receipt_env"] != "IMP_LOOP206_PRIOR_RECEIPT":
        raise ValueError("prior binding")
except Exception as exc:
    print(f"preflight_failed={type(exc).__name__}", file=sys.stderr)
    raise SystemExit(1)

print("preflight=passed")
print("evidence_class=train_screen / exact_fixed_cache / historical_cache_provenance_drift")
print("candidate_upload_authorized=false parity=0/76")
'@

function Invoke-DemoLaunch {
    [CmdletBinding()]
    param(
        [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda',
        [switch]$CheckOnly,
        [string]$Root = '',
        [string]$PythonExe = ''
    )

    try {
        if (-not $Root) {
            $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
        }
        else {
            $Root = (Resolve-Path -LiteralPath $Root).Path
        }
    }
    catch {
        [Console]::Error.WriteLine('Demo repository root is unavailable.')
        return 2
    }
    if (-not $PythonExe) {
        $PythonExe = Join-Path $Root '.venv-win\Scripts\python.exe'
    }
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        [Console]::Error.WriteLine(
            'Demo Python environment is unavailable. Run the Windows bootstrap first.'
        )
        return 2
    }

    $ModelRegistry = Join-Path $Root 'demo\model_registry.example.json'
    $EvidenceRegistry = Join-Path $Root 'demo\data\evidence_registry.json'
    $DatasetIndex = Join-Path $Root 'demo_runtime\loop206_dataset_index.json'
    $CandidateManifest = Join-Path $Root '.artifacts\preprocessing_search\loop206_leac_drlse\pilot_cache_v2_candidate\manifest.json'
    $ZeroManifest = Join-Path $Root '.artifacts\preprocessing_search\loop206_leac_drlse\pilot_cache_v2_zero_control\manifest.json'
    $LiveConfig = Join-Path $Root 'configs\demo\loop206_live.yaml'
    $ControlCheckpoint = Join-Path $Root 'runs\loop206-control-train-screen-pilot20-checkpoints\best.pt'
    $CandidateCheckpoint = Join-Path $Root 'runs\loop206-contour-channel-train-screen-pilot20-checkpoints\best.pt'

    $pythonPathPresent = Test-Path Env:PYTHONPATH
    $oldPythonPath = $env:PYTHONPATH
    $preflightExit = 1
    try {
        $env:PYTHONPATH = Join-Path $Root 'src'
        $Preflight |
            & $PythonExe - $Root $ModelRegistry $EvidenceRegistry $DatasetIndex $CandidateManifest $ZeroManifest $LiveConfig $ControlCheckpoint $CandidateCheckpoint |
            ForEach-Object { [Console]::Out.WriteLine([string]$_) }
        $preflightExit = $LASTEXITCODE
    }
    finally {
        if ($pythonPathPresent) {
            $env:PYTHONPATH = $oldPythonPath
        }
        else {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        }
    }
    if ($preflightExit -ne 0) {
        [Console]::Error.WriteLine(
            'Demo preflight failed. Check private runtime assets and release hashes.'
        )
        return $preflightExit
    }
    if ($CheckOnly) {
        return 0
    }

    $runtimeRoot = Join-Path $Root 'demo_runtime'
    $sessionParent = Join-Path $runtimeRoot 'sessions'
    $sessionPath = $null
    try {
        [void](New-Item -ItemType Directory -Path $sessionParent -Force)
        $sessionPath = Join-Path $sessionParent (
            'demo-' + [guid]::NewGuid().ToString('N')
        )
        [void](New-Item -ItemType Directory -Path $sessionPath)
        $owned = Assert-OwnedSessionPath -SessionPath $sessionPath -RuntimeRoot $runtimeRoot
        $sessionPath = $owned.SessionPath
    }
    catch {
        [Console]::Error.WriteLine('Unable to create an isolated demo upload session.')
        return 5
    }

    $snapshot = @{}
    foreach ($name in @(
        'PYTHONPATH', 'GRADIO_TEMP_DIR', 'TMP', 'TEMP',
        'IMP_LOOP206_DEMO_SESSION', 'IMP_LOOP206_PRIOR', 'IMP_LOOP206_PRIOR_RECEIPT'
    )) {
        $item = Get-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        $snapshot[$name] = @{
            Present = $null -ne $item
            Value = if ($null -ne $item) { $item.Value } else { $null }
        }
    }

    $appExit = 1
    $cleanupFailed = $false
    try {
        $env:PYTHONPATH = Join-Path $Root 'src'
        $env:GRADIO_TEMP_DIR = $sessionPath
        $env:TMP = $sessionPath
        $env:TEMP = $sessionPath
        $env:IMP_LOOP206_DEMO_SESSION = $sessionPath
        # Arbitrary candidate use stays locked even if the operator shell has stale prior variables.
        Remove-Item Env:IMP_LOOP206_PRIOR -ErrorAction SilentlyContinue
        Remove-Item Env:IMP_LOOP206_PRIOR_RECEIPT -ErrorAction SilentlyContinue

        [Console]::Out.WriteLine('Local demo: http://127.0.0.1:7860')
        & $PythonExe -m lesion_robustness.demo.app --host 127.0.0.1 --port 7860 --device $Device
        $appExit = $LASTEXITCODE
    }
    catch {
        [Console]::Error.WriteLine('Demo application failed.')
        $appExit = 1
    }
    finally {
        try {
            Restore-DemoEnvironment -Snapshot $snapshot
        }
        catch {
            $cleanupFailed = $true
        }
        try {
            $owned = Assert-OwnedSessionPath -SessionPath $sessionPath -RuntimeRoot $runtimeRoot
            $sessionPath = $owned.SessionPath
            Remove-Item -LiteralPath $sessionPath -Recurse -Force
        }
        catch {
            $cleanupFailed = $true
        }
    }

    if ($cleanupFailed) {
        [Console]::Error.WriteLine('Demo temporary-upload cleanup failed closed.')
        if ($appExit -eq 0) {
            return 5
        }
    }
    return $appExit
}

if ($MyInvocation.InvocationName -ne '.') {
    $exitCode = Invoke-DemoLaunch -Device $Device -CheckOnly:$CheckOnly
    exit $exitCode
}
