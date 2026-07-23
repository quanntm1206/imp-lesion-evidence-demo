[CmdletBinding()]
param(
    [switch]$PreflightOnly,
    [switch]$DryRun,
    [string]$PythonExe = "",
    [string]$Protocol = "experiments/rq1_v2/protocol.json",
    [string]$ConfigDirectory = "experiments/rq1_v2/configs"
)

$ErrorActionPreference = "Stop"
if ($PreflightOnly -and $DryRun) {
    Write-Error "Choose either -PreflightOnly or -DryRun."
    exit 2
}
if (-not $PreflightOnly -and -not $DryRun) {
    Write-Error "Contract-only scaffold requires -PreflightOnly or -DryRun."
    exit 2
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Join-Path $Root ".venv-win\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    Write-Error "Python executable is missing."
    exit 2
}

$ProtocolPath = if ([IO.Path]::IsPathRooted($Protocol)) {
    (Resolve-Path -LiteralPath $Protocol).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $Root $Protocol)).Path
}
$ConfigRoot = if ([IO.Path]::IsPathRooted($ConfigDirectory)) {
    (Resolve-Path -LiteralPath $ConfigDirectory).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $Root $ConfigDirectory)).Path
}
$Configs = @(Get-ChildItem -LiteralPath $ConfigRoot -Filter "*.yaml" -File | Sort-Object Name)
$ExpectedConfigNames = @(
    "imp_seed1206.yaml", "imp_seed206.yaml", "imp_seed2206.yaml",
    "nnunet_seed1206.yaml", "nnunet_seed206.yaml", "nnunet_seed2206.yaml"
)
$ConfigArm = @{
    "imp_seed206.yaml" = "imp"
    "imp_seed1206.yaml" = "imp"
    "imp_seed2206.yaml" = "imp"
    "nnunet_seed206.yaml" = "nnunet"
    "nnunet_seed1206.yaml" = "nnunet"
    "nnunet_seed2206.yaml" = "nnunet"
}
if ($Configs.Count -ne 6 -or (@($Configs.Name) -join "|") -ne ($ExpectedConfigNames -join "|")) {
    [Console]::Error.WriteLine("The exact six canonical arm-by-seed config filenames are required.")
    exit 2
}

if ($PreflightOnly) {
    $RequiredEnvironment = @(
        "IMP_CLEAN_V3_INDEX",
        "IMP_RQ1_V2_EXPERIMENT_INPUT",
        "IMP_RQ1_V2_IMP_INITIALIZATION",
        "IMP_RQ1_V2_NNUNET_INITIALIZATION",
        "IMP_RQ1_V2_IMP_INPUT_SHA256",
        "IMP_RQ1_V2_NNUNET_INPUT_SHA256",
        "IMP_RQ1_V2_NNUNET_CHECKPOINT",
        "IMP_RQ1_V2_PARENT_RELEASE",
        "IMP_RQ1_V2_OUTPUT_ROOT"
    )
    $MissingEnvironment = @($RequiredEnvironment | Where-Object {
        [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($_))
    })
    if ($MissingEnvironment.Count -gt 0) {
        Write-Output (([ordered]@{
            status = "blocked_missing_prerequisite"
            missing_prerequisites = $MissingEnvironment
            jobs = 6
            data_open_count = 0
        } | ConvertTo-Json -Compress))
        exit 2
    }
}

foreach ($Config in $Configs) {
    $Arm = $ConfigArm[$Config.Name]
    if ($null -eq $Arm) {
        [Console]::Error.WriteLine("Config filename is not canonical: $($Config.Name)")
        exit 2
    }
    $Args = @(
        (Join-Path $Root "scripts\research\train_rq1_v2.py"),
        "--protocol", $ProtocolPath,
        "--config", $Config.FullName
    )
    if ($DryRun) {
        $Args += "--dry-run"
    } else {
        $DescribeArgs = @(
            (Join-Path $Root "scripts\research\train_rq1_v2.py"),
            "--protocol", $ProtocolPath,
            "--config", $Config.FullName,
            "--dry-run"
        )
        $DescriptionJson = & $PythonExe @DescribeArgs
        if ($LASTEXITCODE -ne 0) {
            exit 2
        }
        try {
            $Description = $DescriptionJson | ConvertFrom-Json -ErrorAction Stop
        } catch {
            [Console]::Error.WriteLine("Canonical checkpoint description is invalid.")
            exit 2
        }
        if ([string]::IsNullOrWhiteSpace([string]$Description.checkpoint_relative_path)) {
            [Console]::Error.WriteLine("Canonical checkpoint path is missing.")
            exit 2
        }
        $ArmInput = if ($Arm -eq "imp") {
            $env:IMP_RQ1_V2_IMP_INITIALIZATION
        } else {
            $env:IMP_RQ1_V2_NNUNET_INITIALIZATION
        }
        $ArmSha = if ($Arm -eq "imp") {
            $env:IMP_RQ1_V2_IMP_INPUT_SHA256
        } else {
            $env:IMP_RQ1_V2_NNUNET_INPUT_SHA256
        }
        $OutputCheckpoint = Join-Path $env:IMP_RQ1_V2_OUTPUT_ROOT ([string]$Description.checkpoint_relative_path)
        $Args += @(
            "--data-manifest", $env:IMP_CLEAN_V3_INDEX,
            "--experiment-manifest", $env:IMP_RQ1_V2_EXPERIMENT_INPUT,
            "--parent-release", $env:IMP_RQ1_V2_PARENT_RELEASE,
            "--imp-input-artifact", $env:IMP_RQ1_V2_IMP_INITIALIZATION,
            "--nnunet-input-artifact", $env:IMP_RQ1_V2_NNUNET_INITIALIZATION,
            "--nnunet-checkpoint", $env:IMP_RQ1_V2_NNUNET_CHECKPOINT,
            "--input-artifact", $ArmInput,
            "--input-artifact-sha256", $ArmSha,
            "--output-checkpoint", $OutputCheckpoint,
            "--preflight-only"
        )
    }
    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        exit 2
    }
}

Write-Output '{"status":"contract_checks_passed","jobs":6,"engine_available":false}'
exit 0
