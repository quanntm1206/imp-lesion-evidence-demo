[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cuda',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PythonExe = Join-Path $Root '.venv-win\Scripts\python.exe'

function Stop-DemoLaunch {
    param([string]$Message, [int]$Code)
    [Console]::Error.WriteLine($Message)
    exit $Code
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Stop-DemoLaunch 'Demo Python environment is unavailable. Run the Windows bootstrap first.' 2
}

$ModelRegistry = Join-Path $Root 'demo\model_registry.example.json'
$EvidenceRegistry = Join-Path $Root 'demo\data\evidence_registry.json'
$DatasetIndex = Join-Path $Root 'demo_runtime\loop206_dataset_index.json'
$CandidateManifest = Join-Path $Root '.artifacts\preprocessing_search\loop206_leac_drlse\pilot_cache_v2_candidate\manifest.json'
$ZeroManifest = Join-Path $Root '.artifacts\preprocessing_search\loop206_leac_drlse\pilot_cache_v2_zero_control\manifest.json'
$LiveConfig = Join-Path $Root 'configs\demo\loop206_live.yaml'
$ControlCheckpoint = Join-Path $Root 'runs\loop206-control-train-screen-pilot20-checkpoints\best.pt'
$CandidateCheckpoint = Join-Path $Root 'runs\loop206-contour-channel-train-screen-pilot20-checkpoints\best.pt'

# The public arbitrary-upload route stays control-only even if a shell contains old prior variables.
Remove-Item Env:IMP_LOOP206_PRIOR -ErrorAction SilentlyContinue
Remove-Item Env:IMP_LOOP206_PRIOR_RECEIPT -ErrorAction SilentlyContinue

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

$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $Root 'src'
try {
    $Preflight | & $PythonExe - $Root $ModelRegistry $EvidenceRegistry $DatasetIndex $CandidateManifest $ZeroManifest $LiveConfig $ControlCheckpoint $CandidateCheckpoint
    if ($LASTEXITCODE -ne 0) {
        $preflightExit = $LASTEXITCODE
        Stop-DemoLaunch 'Demo preflight failed. Check private runtime assets and release hashes.' $preflightExit
    }
    if ($CheckOnly) {
        exit 0
    }

    Write-Output 'Local demo: http://127.0.0.1:7860'
    & $PythonExe -m lesion_robustness.demo.app --host 127.0.0.1 --port 7860 --device $Device
    $appExit = $LASTEXITCODE
    exit $appExit
}
finally {
    $env:PYTHONPATH = $oldPythonPath
}
