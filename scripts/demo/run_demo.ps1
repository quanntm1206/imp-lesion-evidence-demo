[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cuda',
    [switch]$CheckOnly,
    [switch]$PreserveMode,
    [switch]$PublicTunnelMode,
    [string]$RunId = '',
    [string]$PythonExe = ''
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
            [Environment]::SetEnvironmentVariable(
                $name, [string]$Snapshot[$name].Value, 'Process'
            )
        }
        else {
            [Environment]::SetEnvironmentVariable($name, $null, 'Process')
        }
    }
}

function Resolve-DemoRuntimeArtifactPath {
    param(
        [Parameter(Mandatory = $true)][string]$EnvironmentName,
        [Parameter(Mandatory = $true)][string]$DefaultPath
    )
    $configured = [Environment]::GetEnvironmentVariable($EnvironmentName, 'Process')
    $selected = if ($null -ne $configured -and $configured.Trim()) {
        $configured.Trim()
    }
    else {
        $DefaultPath
    }
    $item = Get-Item -LiteralPath $selected -Force -ErrorAction Stop
    if (
        $item.PSIsContainer -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    ) {
        throw "$EnvironmentName must identify a regular file."
    }
    return $item.FullName
}

function Resolve-DemoPythonApplication {
    param(
        [string]$ExplicitPath,
        [Parameter(Mandatory = $true)][string]$DefaultPath
    )
    $selected = if ($ExplicitPath) { $ExplicitPath } else { $DefaultPath }
    $item = Get-Item -LiteralPath $selected -Force -ErrorAction Stop
    if (
        $item.PSIsContainer -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        $item.Name -cne 'python.exe'
    ) {
        throw 'Demo Python must be a regular non-reparse file named exactly python.exe.'
    }
    return $item.FullName
}

$script:SidecarHealthUrl = 'http://127.0.0.1:7862/health'
$script:SidecarProtocol = ''
$script:SidecarModel = ''
$script:SidecarCheckpoint = ''
$script:ReleaseManifestSha256 = ''

function Initialize-ReleaseProjection {
    param([Parameter(Mandatory = $true)][string]$Root)
    $bytes = [IO.File]::ReadAllBytes((Join-Path $Root 'release\imp_release_manifest.json'))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $script:ReleaseManifestSha256 = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
    $release = [Text.Encoding]::ASCII.GetString($bytes) | ConvertFrom-Json
    $live = @($release.comparisons | Where-Object { $_.id -ceq 'live_demo' })
    if ($release.schema_version -cne 'imp.release.manifest.v1' -or $live.Count -ne 1) { throw 'release manifest projection mismatch' }
    $nnunet = $release.models.($live[0].right_model_id)
    if ($null -eq $nnunet) { throw 'release manifest projection missing' }
    $script:SidecarProtocol = [string]$nnunet.runtime.protocol
    $script:SidecarModel = [string]$live[0].right_model_id
    $script:SidecarCheckpoint = [string]$nnunet.checkpoint_sha256
}

function Test-ExactDemoSidecarHealth {
    param([Parameter(Mandatory = $true)]$Payload)
    $observed = @($Payload.PSObject.Properties.Name | Sort-Object)
    $expected = @('checkpoint_sha256', 'device', 'model_id', 'protocol', 'ready')
    if (($observed -join ',') -cne ($expected -join ',')) {
        return $false
    }
    return (
        $Payload.protocol -ceq $script:SidecarProtocol -and
        $Payload.model_id -ceq $script:SidecarModel -and
        $Payload.checkpoint_sha256 -ceq $script:SidecarCheckpoint -and
        $Payload.device -ceq 'cuda:0' -and
        $Payload.ready -is [bool] -and
        $Payload.ready
    )
}

function New-CryptographicOwnerNonce {
    $bytes = New-Object byte[] 16
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    return ([BitConverter]::ToString($bytes)).Replace('-', '').ToLowerInvariant()
}

function Assert-PreserveRunId {
    param([string]$RunId)
    if ($RunId -cnotmatch '^[a-z0-9][a-z0-9_-]{0,127}$') {
        throw 'Preserve run ID is unsafe.'
    }
    return $RunId
}

function Get-PreserveComponentDirectory {
    param([string]$Root, [string]$RunId, [string]$Component)
    $safeRunId = Assert-PreserveRunId -RunId $RunId
    if ($Component -cnotmatch '^[a-z0-9][a-z0-9_-]{0,63}$') {
        throw 'Preserve component is unsafe.'
    }
    $preservedRoot = [IO.Path]::GetFullPath((Join-Path $Root 'demo_runtime\preserved'))
    $directory = [IO.Path]::GetFullPath((Join-Path $preservedRoot "$safeRunId\$Component"))
    $prefix = $preservedRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $directory.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Preserve component escapes the runtime root.'
    }
    [void](New-Item -ItemType Directory -Path $directory -Force)
    return $directory
}

function Write-ExclusiveAsciiFile {
    param([string]$Path, [string]$Content)
    $bytes = [Text.Encoding]::ASCII.GetBytes($Content)
    $stream = [IO.File]::Open(
        $Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None
    )
    try {
        $stream.Write($bytes, 0, $bytes.Length)
    }
    finally {
        $stream.Dispose()
    }
}

function Get-CurrentProcessStartUtc {
    $process = Get-Process -Id $PID -ErrorAction Stop
    return $process.StartTime.ToUniversalTime().ToString(
        'yyyy-MM-ddTHH:mm:ss.fffffffZ',
        [Globalization.CultureInfo]::InvariantCulture
    )
}

function Assert-DemoSidecarReady {
    try {
        $health = Invoke-RestMethod `
            -Uri $script:SidecarHealthUrl -Method Get -TimeoutSec 5
    }
    catch {
        throw 'the pinned local nnU-Net sidecar is unavailable'
    }
    if (-not (Test-ExactDemoSidecarHealth -Payload $health)) {
        throw 'the local nnU-Net sidecar health identity is not exact'
    }
}

function Write-GradioOwnerRecord {
    param(
        [string]$Root,
        [string]$PythonExe,
        [string]$SessionPath,
        [switch]$PreserveMode,
        [switch]$PublicTunnelMode,
        [string]$RunId = ''
    )
    if ($script:ReleaseManifestSha256 -cnotmatch '^[0-9a-f]{64}$') {
        throw 'Release manifest identity is unavailable.'
    }
    $runtimeRoot = Join-Path $Root 'demo_runtime'
    $ownerRoot = if ($PreserveMode) {
        Get-PreserveComponentDirectory -Root $Root -RunId $RunId -Component 'gradio'
    }
    else {
        Join-Path $runtimeRoot 'launcher'
    }
    if (-not (Test-Path -LiteralPath $ownerRoot)) {
        [void](New-Item -ItemType Directory -Path $ownerRoot)
    }
    $ownerItem = Get-Item -LiteralPath $ownerRoot -Force
    if (-not $ownerItem.PSIsContainer -or ($ownerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'Gradio owner directory is unsafe.'
    }
    $recordPath = if ($PreserveMode) {
        Join-Path $ownerItem.FullName ("owner-" + (New-CryptographicOwnerNonce) + '.json')
    }
    else {
        Join-Path $ownerItem.FullName 'gradio.json'
    }
    if (-not $PreserveMode -and (Test-Path -LiteralPath $recordPath)) {
        throw 'A Gradio owner record already exists.'
    }
    $record = [ordered]@{
        schema_version = 'imp.demo.gradio-owner.v1'
        release_manifest_sha256 = $script:ReleaseManifestSha256
        public_tunnel_mode = [bool]$PublicTunnelMode
        preserve_mode = [bool]$PreserveMode
        launcher_pid = $PID
        launcher_start_time_utc = Get-CurrentProcessStartUtc
        owner_nonce = New-CryptographicOwnerNonce
        python_path = [IO.Path]::GetFullPath($PythonExe)
        session_path = [IO.Path]::GetFullPath($SessionPath)
        host = '127.0.0.1'
        port = 7860
    }
    $json = $record | ConvertTo-Json -Compress
    if ($PreserveMode) {
        Write-ExclusiveAsciiFile -Path $recordPath -Content $json
    }
    else {
        [IO.File]::WriteAllText($recordPath, $json, [Text.Encoding]::ASCII)
    }
    return $recordPath
}

function Remove-GradioOwnerRecord {
    param([string]$RecordPath, [string]$Root)
    if (-not $RecordPath -or -not (Test-Path -LiteralPath $RecordPath -PathType Leaf)) {
        return
    }
    $expectedParent = [IO.Path]::GetFullPath(
        (Join-Path $Root 'demo_runtime\launcher')
    ).TrimEnd('\', '/')
    $item = Get-Item -LiteralPath $RecordPath -Force
    if (
        $item.Name -cne 'gradio.json' -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        -not [string]::Equals(
            [IO.Path]::GetFullPath($item.Directory.FullName).TrimEnd('\', '/'),
            $expectedParent,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw 'Gradio owner record path is unsafe.'
    }
    Remove-Item -LiteralPath $item.FullName -Force
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
from lesion_robustness.release_manifest import live_demo_receipt_projection

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

try:
    evidence = json.loads(evidence_registry.read_text(encoding="ascii"))
    validate_registry(evidence)
    release_projection = live_demo_receipt_projection()
    if evidence.get("release_manifest_sha256") != release_projection["release_manifest_sha256"]:
        raise ValueError("evidence registry release manifest digest mismatch")
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

$DualSmoke = @'
import json
import re
import sys
from pathlib import Path

import numpy as np

root, model_registry, evidence_registry, control_checkpoint, candidate_checkpoint = map(Path, sys.argv[1:6])
sys.path.insert(0, str(root / "src"))

from lesion_robustness.demo.dual_live_protocol import (
    CHECKPOINT_SHA256,
    MODEL_ID,
    PROTOCOL_ID,
    rgb_sha256,
)
from lesion_robustness.demo.dual_live_service import DualLiveService
from lesion_robustness.demo.live_inputs import LiveInputEvidence
from lesion_robustness.demo.model_service import load_model_registry
from lesion_robustness.demo.nnunet_client import NnUNetClient
from lesion_robustness.demo.presentation import build_dual_live_receipt
from lesion_robustness.release_manifest import load_release_manifest

try:
    evidence_registry.read_text(encoding="ascii")
    release_manifest = load_release_manifest()
    models = load_model_registry(
        model_registry,
        environ={
            "IMP_LOOP206_CONTROL_CHECKPOINT": str(control_checkpoint),
            "IMP_LOOP206_CANDIDATE_CHECKPOINT": str(candidate_checkpoint),
        },
        device="cuda",
    )
    imp = models.build_service()
    client = NnUNetClient()
    health = client.health()
    if (
        health.protocol != PROTOCOL_ID
        or health.model_id != MODEL_ID
        or health.checkpoint_sha256 != CHECKPOINT_SHA256
        or health.device != "cuda:0"
        or health.ready is not True
    ):
        raise ValueError("sidecar health binding")

    height, width = 64, 64
    yy, xx = np.indices((height, width), dtype=np.uint16)
    image = np.stack(
        ((xx * 4) % 256, (yy * 4) % 256, ((xx + yy) * 2) % 256), axis=2
    ).astype(np.uint8)
    input_evidence = LiveInputEvidence(
        kind="synthetic",
        evidence_class="illustrative_synthetic_no_ground_truth",
        rgb_sha256=rgb_sha256(image),
        sample_id=None,
        source_dataset=None,
        source_page=None,
        image_license=None,
        training_exposure={},
        ground_truth_used=False,
        ground_truth_not_loaded=True,
    )
    if input_evidence.rgb_sha256 != rgb_sha256(image):
        raise ValueError("input evidence binding")
    result = DualLiveService(imp, client).run(image)
    if result.receipt_eligible is not True:
        raise ValueError("incomplete dual result")
    if result.input_sha256 != rgb_sha256(image) or not np.array_equal(result.original_rgb, image):
        raise ValueError("input binding")
    if result.imp is None or result.nnunet is None:
        raise ValueError("missing dual arm")
    for arm in (result.imp, result.nnunet):
        mask = np.asarray(arm.mask)
        if mask.shape != image.shape[:2] or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError("finite binary mask")
    if (
        not str(result.imp.device).startswith("cuda")
        or result.nnunet.device != "cuda:0"
        or result.nnunet.model_id != MODEL_ID
        or result.nnunet.checkpoint_sha256 != CHECKPOINT_SHA256
        or result.nnunet.protocol != PROTOCOL_ID
        or result.imp.model_id != models.control.model_id
        or result.imp.checkpoint_sha256 != models.control.checkpoint_sha256
    ):
        raise ValueError("model identity")

    receipt = build_dual_live_receipt(result, release_manifest, input_evidence)
    serialized = json.dumps(receipt, sort_keys=True, ensure_ascii=True, allow_nan=False)
    if receipt.get("schema_version") != "imp.dual_live.receipt.v2":
        raise ValueError("receipt completeness")
    if re.search(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|mnt)/)", serialized):
        raise ValueError("receipt local path")
except Exception as exc:
    print(f"dual_smoke_failed={type(exc).__name__}", file=sys.stderr)
    raise SystemExit(1)

print("dual_smoke=passed")
'@

function Invoke-DemoLaunch {
    [CmdletBinding()]
    param(
        [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda',
        [switch]$CheckOnly,
        [switch]$PreserveMode,
        [switch]$PublicTunnelMode,
        [string]$RunId = '',
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
    try { Initialize-ReleaseProjection -Root $Root }
    catch { [Console]::Error.WriteLine('Release manifest identity is unavailable.'); return 2 }
    if ($Device -cne 'cuda') {
        [Console]::Error.WriteLine('Dual-live launch requires CUDA; CPU fallback is disabled.')
        return 6
    }
    if ($PreserveMode) {
        $RunId = Assert-PreserveRunId -RunId $RunId
        [Console]::Out.WriteLine("preserve_run_id=$RunId")
    }
    elseif ($PublicTunnelMode) {
        [Console]::Error.WriteLine('Public tunnel mode requires preserve mode.')
        return 6
    }

    $ModelRegistry = Join-Path $Root 'demo\model_registry.example.json'
    $EvidenceRegistry = Join-Path $Root 'demo\data\evidence_registry.json'
    $DatasetIndex = Join-Path $Root 'demo_runtime\loop206_dataset_index.json'
    $CandidateManifest = Join-Path $Root '.artifacts\preprocessing_search\loop206_leac_drlse\pilot_cache_v2_candidate\manifest.json'
    $ZeroManifest = Join-Path $Root '.artifacts\preprocessing_search\loop206_leac_drlse\pilot_cache_v2_zero_control\manifest.json'
    $LiveConfig = Join-Path $Root 'configs\demo\loop206_live.yaml'
    $ControlCheckpoint = Resolve-DemoRuntimeArtifactPath -EnvironmentName 'IMP_LOOP206_CONTROL_CHECKPOINT' `
        -DefaultPath (Join-Path $Root 'runs\loop206-control-train-screen-pilot20-checkpoints\best.pt')
    $CandidateCheckpoint = Resolve-DemoRuntimeArtifactPath -EnvironmentName 'IMP_LOOP206_CANDIDATE_CHECKPOINT' `
        -DefaultPath (Join-Path $Root 'runs\loop206-contour-channel-train-screen-pilot20-checkpoints\best.pt')

    $oldPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
    $pythonPathPresent = $null -ne $oldPythonPath
    $preflightExit = 1
    try {
        $env:PYTHONPATH = Join-Path $Root 'src'
        $preflightArguments = @(
            '-', $Root, $ModelRegistry, $EvidenceRegistry, $DatasetIndex,
            $CandidateManifest, $ZeroManifest, $LiveConfig,
            $ControlCheckpoint, $CandidateCheckpoint
        )
        $Preflight |
            & $PythonExe @preflightArguments |
            ForEach-Object { [Console]::Out.WriteLine([string]$_) }
        $preflightExit = $LASTEXITCODE
    }
    finally {
        if ($pythonPathPresent) {
            [Environment]::SetEnvironmentVariable(
                'PYTHONPATH', $oldPythonPath, 'Process'
            )
        }
        else {
            [Environment]::SetEnvironmentVariable('PYTHONPATH', $null, 'Process')
        }
    }
    if ($preflightExit -ne 0) {
        [Console]::Error.WriteLine(
            'Demo preflight failed. Check private runtime assets and release hashes.'
        )
        return $preflightExit
    }

    try {
        Assert-DemoSidecarReady
    }
    catch {
        [Console]::Error.WriteLine("Demo sidecar gate failed: $($_.Exception.Message)")
        return 3
    }

    $smokeExit = 1
    $smokeLines = @()
    try {
        [Environment]::SetEnvironmentVariable(
            'PYTHONPATH', (Join-Path $Root 'src'), 'Process'
        )
        $smokeLines = @(
            $smokeArguments = @(
                '-', $Root, $ModelRegistry, $EvidenceRegistry,
                $ControlCheckpoint, $CandidateCheckpoint
            )
            $DualSmoke |
                & $PythonExe @smokeArguments |
                ForEach-Object { [string]$_ }
        )
        $smokeExit = $LASTEXITCODE
    }
    finally {
        if ($pythonPathPresent) {
            [Environment]::SetEnvironmentVariable(
                'PYTHONPATH', $oldPythonPath, 'Process'
            )
        }
        else {
            [Environment]::SetEnvironmentVariable('PYTHONPATH', $null, 'Process')
        }
    }
    foreach ($line in $smokeLines) {
        [Console]::Out.WriteLine($line)
    }
    if (
        $smokeExit -ne 0 -or
        @($smokeLines | Where-Object { $_.Trim() -ceq 'dual_smoke=passed' }).Count -ne 1
    ) {
        [Console]::Error.WriteLine('Demo dual smoke failed closed before Gradio bind.')
        return 4
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
        'IMP_LOOP206_PRESERVE_RUN_ID',
        'IMP_LOOP206_DEMO_SESSION', 'IMP_LOOP206_PRIOR', 'IMP_LOOP206_PRIOR_RECEIPT'
    )) {
        $value = [Environment]::GetEnvironmentVariable($name, 'Process')
        $snapshot[$name] = @{
            Present = $null -ne $value
            Value = $value
        }
    }

    $appExit = 1
    $cleanupFailed = $false
    $ownerRecordPath = $null
    try {
        [Environment]::SetEnvironmentVariable(
            'PYTHONPATH', (Join-Path $Root 'src'), 'Process'
        )
        $env:GRADIO_TEMP_DIR = $sessionPath
        $env:TMP = $sessionPath
        $env:TEMP = $sessionPath
        $env:IMP_LOOP206_DEMO_SESSION = $sessionPath
        if ($PreserveMode) {
            $env:IMP_LOOP206_PRESERVE_RUN_ID = $RunId
        }
        # Arbitrary candidate use stays locked even if the operator shell has stale prior variables.
        if (-not $PreserveMode) {
            Remove-Item Env:IMP_LOOP206_PRIOR -ErrorAction SilentlyContinue
            Remove-Item Env:IMP_LOOP206_PRIOR_RECEIPT -ErrorAction SilentlyContinue
        }

        $ownerRecordPath = Write-GradioOwnerRecord `
            -Root $Root -PythonExe $PythonExe -SessionPath $sessionPath `
            -PreserveMode:$PreserveMode -PublicTunnelMode:$PublicTunnelMode -RunId $RunId
        [Console]::Out.WriteLine('Local demo: http://127.0.0.1:7860')
        $appArguments = @(
            '-m', 'lesion_robustness.demo.app', '--host', '127.0.0.1',
            '--port', '7860', '--device', $Device
        )
        if ($PublicTunnelMode) {
            $appArguments += @(
                '--public-tunnel-mode', '--preserve-mode', '--run-id', $RunId
            )
        }
        elseif ($PreserveMode) {
            $appArguments += @('--preserve-mode', '--run-id', $RunId)
        }
        & $PythonExe @appArguments
        $appExit = $LASTEXITCODE
    }
    catch {
        [Console]::Error.WriteLine('Demo application failed.')
        $appExit = 1
    }
    finally {
        try {
            if (-not $PreserveMode) {
                Remove-GradioOwnerRecord -RecordPath $ownerRecordPath -Root $Root
            }
        }
        catch {
            $cleanupFailed = $true
        }
        try {
            Restore-DemoEnvironment -Snapshot $snapshot
        }
        catch {
            $cleanupFailed = $true
        }
        try {
            if (-not $PreserveMode) {
                $owned = Assert-OwnedSessionPath -SessionPath $sessionPath -RuntimeRoot $runtimeRoot
                $sessionPath = $owned.SessionPath
                Remove-Item -LiteralPath $sessionPath -Recurse -Force
            }
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
    try {
        $defaultRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
        $resolvedPython = Resolve-DemoPythonApplication `
            -ExplicitPath $PythonExe `
            -DefaultPath (Join-Path $defaultRoot '.venv-win\Scripts\python.exe')
    }
    catch {
        [Console]::Error.WriteLine('Demo Python environment is unavailable or unsafe.')
        exit 2
    }
    $exitCode = Invoke-DemoLaunch -Device $Device -CheckOnly:$CheckOnly `
        -PreserveMode:$PreserveMode -PublicTunnelMode:$PublicTunnelMode `
        -RunId $RunId -PythonExe $resolvedPython
    exit $exitCode
}
