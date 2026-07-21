[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$VhdPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ReportPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    $role = [Security.Principal.WindowsBuiltInRole]::Administrator
    if (-not $principal.IsInRole($role)) {
        throw 'Administrator token required'
    }
}

function Resolve-ExplicitFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $explicitPath = [IO.Path]::GetFullPath($Path)
    $resolvedPath = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).ProviderPath
    if (-not [string]::Equals($explicitPath, $resolvedPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label resolved path differs from explicit input"
    }
    $item = Get-Item -LiteralPath $resolvedPath -Force -ErrorAction Stop
    if ($item.PSIsContainer) {
        throw "$Label must be a file"
    }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "$Label must not be a reparse point"
    }
    return $resolvedPath
}

function Get-VhdSnapshot {
    param([Parameter(Mandatory = $true)][string]$Path)

    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    return [ordered]@{
        length = [int64]$item.Length
        creation_time_utc = $item.CreationTimeUtc.ToString('o', [Globalization.CultureInfo]::InvariantCulture)
        last_write_time_utc = $item.LastWriteTimeUtc.ToString('o', [Globalization.CultureInfo]::InvariantCulture)
    }
}

function Assert-SnapshotUnchanged {
    param(
        [Parameter(Mandatory = $true)]$Before,
        [Parameter(Mandatory = $true)]$After
    )

    foreach ($field in @('length', 'creation_time_utc', 'last_write_time_utc')) {
        if ($Before[$field] -cne $After[$field]) {
            throw "source VHD changed after recovery: $field"
        }
    }
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $output = & $FilePath @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code ${exitCode}: $($output -join [Environment]::NewLine)"
    }
    return @($output | ForEach-Object { "$_" })
}

function Invoke-WslContext {
    param(
        [Parameter(Mandatory = $true)][string[]]$ContextPrefix,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $wslArguments = @($ContextPrefix) + $Arguments
    return Invoke-NativeChecked -FilePath 'wsl.exe' -Arguments $wslArguments -Label $Label
}

function Invoke-WslContextScript {
    param(
        [Parameter(Mandatory = $true)][string[]]$ContextPrefix,
        [Parameter(Mandatory = $true)][string]$Script,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][string]$Label
    )

    $wslArguments = @($ContextPrefix) + @('sh', '-s', '--') + $Arguments
    $output = ($Script + "`n#") | & wsl.exe @wslArguments 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code ${exitCode}: $($output -join [Environment]::NewLine)"
    }
    return @($output | ForEach-Object { "$_" })
}

function Resolve-WslInspectionContext {
    $requiredCommands = @(
        'sh',
        'lsblk',
        'mount',
        'umount',
        'cp',
        'awk',
        'wslpath',
        'mountpoint',
        'mkdir',
        'sed',
        'head',
        'tr',
        'dirname'
    )
    $probeScript = @'
printf 'uid=%s\n' "$(id -u)"
for name in "$@"; do
    if command -v "$name" >/dev/null 2>&1; then
        printf 'tool=%s\n' "$name"
    else
        printf 'missing=%s\n' "$name"
    fi
done
'@
    $candidates = @(
        [pscustomobject]@{ Name = 'system'; Prefix = [string[]]@('--system', '--') },
        [pscustomobject]@{ Name = 'docker-desktop'; Prefix = [string[]]@('-d', 'docker-desktop', '-u', 'root', '--') }
    )
    $failures = New-Object 'System.Collections.Generic.List[string]'
    foreach ($candidate in $candidates) {
        try {
            $lines = @(Invoke-WslContextScript `
                -ContextPrefix $candidate.Prefix `
                -Script $probeScript `
                -Arguments $requiredCommands `
                -Label "$($candidate.Name) inspection preflight")
            $uidLines = @($lines | Where-Object { $_ -like 'uid=*' })
            if ($uidLines.Count -ne 1 -or $uidLines[0] -cne 'uid=0') {
                throw "$($candidate.Name) inspection context requires uid 0"
            }
            $available = @{}
            foreach ($line in $lines) {
                if ($line -like 'tool=*') {
                    $available[$line.Substring(5)] = $true
                }
            }
            $missing = @($requiredCommands | Where-Object { -not $available.ContainsKey($_) })
            if ($missing.Count -ne 0) {
                throw "$($candidate.Name) inspection context missing required commands: $($missing -join ', ')"
            }
            return [pscustomobject]@{
                Name = $candidate.Name
                Prefix = [string[]]$candidate.Prefix
            }
        }
        catch {
            [void]$failures.Add("$($candidate.Name): $($_.Exception.Message)")
        }
    }
    throw "no usable WSL inspection context: $($failures -join '; ')"
}

function ConvertTo-WslPath {
    param(
        [Parameter(Mandatory = $true)][string[]]$ContextPrefix,
        [Parameter(Mandatory = $true)][string]$WindowsPath
    )

    $values = @(Invoke-WslContext -ContextPrefix $ContextPrefix -Arguments @('wslpath', '-a', '-u', '--', $WindowsPath) -Label 'output path conversion')
    if ($values.Count -ne 1) {
        throw "output path conversion returned $($values.Count) lines"
    }
    $value = ([string]$values[0]).Trim()
    if (-not $value) {
        throw 'output path conversion returned an empty path'
    }
    return $value
}

function Invoke-RecoveryCleanup {
    param(
        [bool]$FilesystemMountAttempted,
        [bool]$WslAttachAttempted,
        [bool]$VhdMountAttempted,
        [string[]]$WslContextPrefix,
        [string]$LinuxMount,
        [string]$PhysicalDrive,
        [Parameter(Mandatory = $true)][string]$ResolvedVhd
    )

    $errors = New-Object 'System.Collections.Generic.List[string]'
    if ($FilesystemMountAttempted) {
        $unmountScript = @'
if mountpoint -q -- "$1"; then
    umount -- "$1"
fi
'@
        try {
            [void](Invoke-WslContextScript -ContextPrefix $WslContextPrefix -Script $unmountScript -Arguments @($LinuxMount) -Label 'ext4 unmount')
        }
        catch {
            [void]$errors.Add("ext4 unmount failed: $($_.Exception.Message)")
        }
    }
    if ($WslAttachAttempted -and $PhysicalDrive) {
        try {
            $output = & wsl.exe --unmount $PhysicalDrive 2>&1
            $exitCode = $LASTEXITCODE
            if ($exitCode -ne 0) {
                [void]$errors.Add("WSL disk detach failed with exit code ${exitCode}: $($output -join [Environment]::NewLine)")
            }
        }
        catch {
            [void]$errors.Add("WSL disk detach failed: $($_.Exception.Message)")
        }
    }
    if ($VhdMountAttempted) {
        try {
            $diskImage = Get-DiskImage -ImagePath $ResolvedVhd -ErrorAction Stop
            if ($diskImage.Attached) {
                [void](Dismount-VHD -Path $ResolvedVhd -ErrorAction Stop)
            }
        }
        catch {
            [void]$errors.Add("VHD detach failed: $($_.Exception.Message)")
        }
    }
    return $errors.ToArray()
}

function Assert-RecoveryCompleted {
    param(
        [Exception]$OperationError,
        [string[]]$CleanupErrors = @()
    )

    if ($null -eq $OperationError -and $CleanupErrors.Count -eq 0) {
        return
    }
    $messages = New-Object 'System.Collections.Generic.List[string]'
    if ($null -ne $OperationError) {
        [void]$messages.Add("recovery operation failed: $($OperationError.Message)")
    }
    foreach ($cleanupError in $CleanupErrors) {
        [void]$messages.Add($cleanupError)
    }
    throw ($messages -join [Environment]::NewLine)
}

function Get-FileIdentity {
    param([Parameter(Mandatory = $true)][string]$Path)

    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    return [ordered]@{
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        size = [int64]$item.Length
    }
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Value
    )

    $encoding = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($Path, $Value, $encoding)
}

function Write-RuntimeIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)]$PackageSpecs
    )

    $packages = @()
    $lockLines = @()
    foreach ($package in $PackageSpecs) {
        $identityRoot = Join-Path (Join-Path $Root 'package_metadata') $package.slug
        $metadataPath = Join-Path $identityRoot 'METADATA'
        $recordPath = Join-Path $identityRoot 'RECORD'
        $metadataText = [IO.File]::ReadAllText($metadataPath)
        $nameMatch = [regex]::Match($metadataText, '(?m)^Name:\s*(.+?)\r?$')
        $versionMatch = [regex]::Match($metadataText, '(?m)^Version:\s*(.+?)\r?$')
        if (-not $nameMatch.Success -or -not $versionMatch.Success) {
            throw "package identity is incomplete: $($package.slug)"
        }
        $name = $nameMatch.Groups[1].Value.Trim()
        $version = $versionMatch.Groups[1].Value.Trim()
        $normalized = ($name.ToLowerInvariant() -replace '[-_.]+', '-')
        if ($normalized -cne $package.normalized) {
            throw "package identity mismatch: $($package.slug)"
        }

        $files = [ordered]@{
            METADATA = Get-FileIdentity -Path $metadataPath
            RECORD = Get-FileIdentity -Path $recordPath
        }
        $directUrlPath = Join-Path $identityRoot 'direct_url.json'
        if (Test-Path -LiteralPath $directUrlPath -PathType Leaf) {
            $files['direct_url.json'] = Get-FileIdentity -Path $directUrlPath
        }
        $packages += [ordered]@{
            name = $name
            normalized_name = $normalized
            version = $version
            files = $files
        }
        $lockLines += "$name==$version"
    }

    $runtimeIdentity = [ordered]@{
        schema_version = 'loop192.runtime.identity.v1'
        python = '3.12'
        packages = $packages
    }
    $runtimeJson = $runtimeIdentity | ConvertTo-Json -Depth 8
    Write-Utf8NoBom -Path (Join-Path $Root 'runtime_identity.json') -Value ($runtimeJson + "`n")
    Write-Utf8NoBom -Path (Join-Path $Root 'requirements.lock') -Value (($lockLines -join "`n") + "`n")
}

function Invoke-DockerCommand {
    param(
        [Parameter(Mandatory = $true)][string]$DockerPath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $output = & $DockerPath @Arguments 2>&1
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Lines = @($output | ForEach-Object { "$_" })
        Label = $Label
    }
}

function Get-ContainerRecoveryContext {
    $image = 'alpine:3.22@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce'
    $requiredDigest = 'sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce'
    $commands = @(Get-Command 'docker.exe' -CommandType Application -ErrorAction SilentlyContinue)
    if ($commands.Count -ne 1) {
        return $null
    }
    $dockerPath = [IO.Path]::GetFullPath([string]$commands[0].Source)
    if ([IO.Path]::GetFileName($dockerPath) -cne 'docker.exe' -or -not (Test-Path -LiteralPath $dockerPath -PathType Leaf)) {
        return $null
    }
    $daemon = Invoke-DockerCommand -DockerPath $dockerPath -Arguments @('version', '--format', '{{.Server.Version}}') -Label 'Docker daemon preflight'
    if ($daemon.ExitCode -ne 0 -or @($daemon.Lines).Count -ne 1 -or -not ([string]$daemon.Lines[0]).Trim()) {
        return $null
    }
    $inspect = Invoke-DockerCommand -DockerPath $dockerPath -Arguments @('image', 'inspect', '--format', '{{range .RepoDigests}}{{println .}}{{end}}', $image) -Label 'Docker image digest preflight'
    if ($inspect.ExitCode -ne 0) {
        return $null
    }
    $digests = @($inspect.Lines | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ })
    if (@($digests | Where-Object { $_.EndsWith("@$requiredDigest", [StringComparison]::Ordinal) }).Count -eq 0) {
        throw 'Docker recovery image digest mismatch'
    }
    return [pscustomobject]@{
        DockerPath = $dockerPath
        Image = $image
    }
}

function New-ContainerRecoveryArguments {
    param(
        [Parameter(Mandatory = $true)][string]$VhdPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot
    )

    foreach ($path in @($VhdPath, $OutputRoot)) {
        if (-not [IO.Path]::IsPathRooted($path) -or $path.IndexOfAny([char[]]",`r`n") -ge 0) {
            throw 'Docker bind paths must be resolved absolute paths without commas or newlines'
        }
    }
    $image = 'alpine:3.22@sha256:14358309a308569c32bdc37e2e0e9694be33a9d99e68afb0f5ff33cc1f695dce'
    return @(
        'run', '--rm', '-i',
        '--pull', 'never',
        '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges:true',
        '--mount', "type=bind,source=$VhdPath,target=/input/source.vhdx,readonly",
        '--mount', "type=bind,source=$OutputRoot,target=/output",
        $image, 'sh', '-s', '--'
    )
}

function Get-ContainerRecoveryEntries {
    return @(
        [pscustomobject]@{ Source = 'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_results/Dataset192_IMPlesionCleanV3RGB256/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_all/checkpoint_final.pth'; Target = 'checkpoint_final.pth' },
        [pscustomobject]@{ Source = 'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_preprocessed/Dataset192_IMPlesionCleanV3RGB256/nnUNetPlans.json'; Target = 'nnUNetPlans.json' },
        [pscustomobject]@{ Source = 'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_preprocessed/Dataset192_IMPlesionCleanV3RGB256/dataset_fingerprint.json'; Target = 'dataset_fingerprint.json' },
        [pscustomobject]@{ Source = 'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_raw/Dataset192_IMPlesionCleanV3RGB256/dataset.json'; Target = 'dataset.json' },
        [pscustomobject]@{ Source = 'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_results/Dataset192_IMPlesionCleanV3RGB256/nnUNetTrainer_100epochs__nnUNetPlans__2d/plans.json'; Target = 'plans.json' },
        [pscustomobject]@{ Source = 'home/admin_mugen/imp_cache/external_repos/loop170/nnUNet/pyproject.toml'; Target = 'pyproject.toml' },
        [pscustomobject]@{ Source = 'home/admin_mugen/imp_cache/external_repos/loop170/nnUNet/nnunetv2.egg-info/PKG-INFO'; Target = 'PKG-INFO' },
        [pscustomobject]@{ Source = 'home/admin_mugen/imp_cache/external_repos/loop170/nnUNet/.git/HEAD'; Target = 'HEAD' },
        [pscustomobject]@{ Source = 'home/admin_mugen/imp_cache/external_repos/loop170/nnUNet/.git/refs/heads/master'; Target = 'master' }
    )
}

function Invoke-DockerParserProcess {
    param(
        [Parameter(Mandatory = $true)][string]$DockerPath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Script
    )

    $output = ($Script + "`n#") | & $DockerPath @Arguments 2>&1
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Lines = @($output | ForEach-Object { "$_" })
    }
}

function Assert-ContainerParserResult {
    param([Parameter(Mandatory = $true)]$ProcessResult)

    if ($ProcessResult.ExitCode -ne 0) {
        throw "Docker parser process failed with exit code $($ProcessResult.ExitCode)"
    }
    $lines = @($ProcessResult.Lines | ForEach-Object { ([string]$_).Trim() })
    if (@($lines | Where-Object { $_ -ceq 'recovery_7zip=7zip-24.09-r0' }).Count -ne 1) {
        throw 'Docker parser did not prove 7zip=24.09-r0'
    }
    $exitLines = @($lines | Where-Object { $_ -like 'recovery_7zip_exit=*' })
    if ($exitLines.Count -ne 1 -or $exitLines[0] -notmatch '^recovery_7zip_exit=[012]$') {
        throw 'Docker parser returned an invalid 7zip exit marker'
    }
    $allowedDiagnostics = @('ERRORS:', 'Headers Error', 'Archives with Errors: 1', 'Open Errors: 1')
    $diagnostics = @($lines | Where-Object { $_ -match '(?i)(warning|error)' })
    foreach ($diagnostic in $diagnostics) {
        if ($allowedDiagnostics -notcontains $diagnostic) {
            throw "unexpected parser diagnostic: $diagnostic"
        }
    }
    if (@($lines | Where-Object { $_ -ceq 'Headers Error' }).Count -ne 1) {
        throw 'Docker parser did not report the observed Headers Error warning'
    }
    return 'Headers Error'
}

function Assert-ExactRecoveryFiles {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string[]]$Expected
    )

    $items = @(Get-ChildItem -LiteralPath $Root -Force)
    if (@($items | Where-Object { $_.PSIsContainer }).Count -ne 0) {
        throw 'recovery output contains an unexpected directory'
    }
    $actual = @($items | ForEach-Object { $_.Name } | Sort-Object)
    $wanted = @($Expected | Sort-Object)
    if (($actual -join "`n") -cne ($wanted -join "`n")) {
        throw "recovery output allowlist mismatch: $($actual -join ', ')"
    }
}

function Write-ReconstructedRuntimeIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Backend,
        [Parameter(Mandatory = $true)][string]$ParserWarning
    )

    $version = '2.8.1'
    $commit = '3e9fdc5fec7c8164f8fc2c6263af8be73278130e'
    $pyproject = [IO.File]::ReadAllText((Join-Path $Root 'pyproject.toml'))
    $pkgInfo = [IO.File]::ReadAllText((Join-Path $Root 'PKG-INFO'))
    $head = [IO.File]::ReadAllText((Join-Path $Root 'HEAD')).Trim()
    $master = [IO.File]::ReadAllText((Join-Path $Root 'master')).Trim()
    if ($pyproject -notmatch 'version\s*=\s*"2\.8\.1"' -or $pkgInfo -notmatch '(?m)^Version:\s*2\.8\.1\s*$') {
        throw 'recovered nnunet source version mismatch'
    }
    if ($head -cne 'ref: refs/heads/master' -or $master -cne $commit) {
        throw 'recovered nnunet source commit mismatch'
    }
    $sourceFiles = [ordered]@{}
    foreach ($name in @('pyproject.toml', 'PKG-INFO', 'HEAD', 'master')) {
        $sourceFiles[$name] = Get-FileIdentity -Path (Join-Path $Root $name)
    }
    $identity = [ordered]@{
        schema_version = 'loop192.runtime.identity.v1'
        environment_status = 'reconstructed_required'
        recovery_backend = $Backend
        parser_warning = $ParserWarning
        source_identity = [ordered]@{
            distribution = 'nnunetv2'
            version = $version
            git_commit = $commit
            files = $sourceFiles
        }
        original_transitive_package_lock = 'unavailable'
        reconstruction_gate = @('Task 4 full transitive lock', 'checkpoint load', 'output replay')
    }
    Write-Utf8NoBom -Path (Join-Path $Root 'runtime_identity.json') -Value (($identity | ConvertTo-Json -Depth 8) + "`n")
    $lock = @(
        '# Original Loop192 transitive package lock unavailable; this is not the original environment.',
        '# Task 4 must resolve the full transitive lock and pass checkpoint load plus output replay.',
        "nnunetv2 @ git+https://github.com/MIC-DKFZ/nnUNet.git@$commit"
    ) -join "`n"
    Write-Utf8NoBom -Path (Join-Path $Root 'requirements.lock') -Value ($lock + "`n")
}

function Invoke-ContainerRecovery {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [Parameter(Mandatory = $true)][string]$VhdPath,
        [Parameter(Mandatory = $true)][string]$ReportPath,
        [Parameter(Mandatory = $true)][string]$OutputRoot
    )

    $resolvedVhd = Resolve-ExplicitFile -Path $VhdPath -Label 'VHD'
    $resolvedReport = Resolve-ExplicitFile -Path $ReportPath -Label 'report'
    $resolvedOutput = [IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutput) {
        if (-not (Get-Item -LiteralPath $resolvedOutput -Force).PSIsContainer -or @(Get-ChildItem -LiteralPath $resolvedOutput -Force).Count -ne 0) {
            throw 'OutputRoot must be an empty directory'
        }
    }
    else {
        [void](New-Item -ItemType Directory -Path $resolvedOutput)
    }
    $before = Get-VhdSnapshot -Path $resolvedVhd
    $entries = @(Get-ContainerRecoveryEntries)
    $arguments = @(New-ContainerRecoveryArguments -VhdPath $resolvedVhd -OutputRoot $resolvedOutput)
    $arguments += @($entries | ForEach-Object { $_.Source })
    $parserScript = @'
set -u
apk add --no-cache '7zip=24.09-r0' >/tmp/apk.log 2>&1 || { cat /tmp/apk.log; exit 71; }
apk info -e '7zip=24.09-r0' || exit 72
apk info -v 7zip | grep -Fx '7zip-24.09-r0' >/dev/null || exit 73
printf 'recovery_7zip=7zip-24.09-r0\n'
set +e
7zz e -y -o/output /input/source.vhdx "$@" >/tmp/7zip.log 2>&1
status=$?
set -e
cat /tmp/7zip.log
printf 'recovery_7zip_exit=%s\n' "$status"
exit 0
'@
    $result = Invoke-DockerParserProcess -DockerPath $Context.DockerPath -Arguments $arguments -Script $parserScript
    $warning = Assert-ContainerParserResult -ProcessResult $result
    $rawNames = @($entries | ForEach-Object { $_.Target })
    Assert-ExactRecoveryFiles -Root $resolvedOutput -Expected $rawNames
    if ((Get-Item -LiteralPath (Join-Path $resolvedOutput 'checkpoint_final.pth')).Length -ne 267947879) {
        throw 'checkpoint size does not match Loop192 recovery evidence'
    }
    $after = Get-VhdSnapshot -Path $resolvedVhd
    Assert-SnapshotUnchanged -Before $before -After $after
    Write-ReconstructedRuntimeIdentity -Root $resolvedOutput -Backend 'container-readonly-7zip' -ParserWarning $warning

    $sourceReport = Get-Content -LiteralPath $resolvedReport -Raw | ConvertFrom-Json
    $verificationReport = [ordered]@{
        candidate_id = $sourceReport.candidate_id
        provenance = $sourceReport.provenance
        source_vhd_proof = [ordered]@{ before = $before; after = $after }
    }
    $repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $pythonExe = Join-Path $repoRoot '.venv-win\Scripts\python.exe'
    $verifierPath = Join-Path $PSScriptRoot 'verify_nnunet_bundle.py'
    $receiptPath = Join-Path $resolvedOutput 'recovery_receipt.json'
    $verifierCode = @'
import importlib.util
import json
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location("loop192_verifier", sys.argv[1])
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load trusted verifier")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
receipt = module.verify_bundle(Path(sys.argv[2]), json.loads(sys.stdin.read()))
receipt.update(recovery_backend="container-readonly-7zip", parser_warning="Headers Error", runtime_status="reconstructed_required")
Path(sys.argv[3]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
'@
    ($verificationReport | ConvertTo-Json -Depth 8 -Compress) | & $pythonExe -c $verifierCode $verifierPath $resolvedOutput $receiptPath
    if ($LASTEXITCODE -ne 0) {
        throw "Loop192 bundle verification failed with exit code $LASTEXITCODE"
    }
    Assert-ExactRecoveryFiles -Root $resolvedOutput -Expected @($rawNames + 'runtime_identity.json' + 'requirements.lock' + 'recovery_receipt.json')
    Write-Output 'recovery=passed backend=container-readonly-7zip warning=Headers Error runtime_status=reconstructed_required'
}

function Invoke-WindowsAttachRecovery {
    Assert-Administrator

    foreach ($command in @('Mount-VHD', 'Dismount-VHD', 'Get-Disk', 'Get-DiskImage')) {
        if ($null -eq (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "required command unavailable: $command"
        }
    }
    if ($null -eq (Get-Command 'wsl.exe' -CommandType Application -ErrorAction SilentlyContinue)) {
        throw 'required command unavailable: wsl.exe'
    }

    $resolvedVhd = Resolve-ExplicitFile -Path $VhdPath -Label 'VHD'
    $resolvedReport = Resolve-ExplicitFile -Path $ReportPath -Label 'report'
    $resolvedOutput = [IO.Path]::GetFullPath($OutputRoot)
    if (Test-Path -LiteralPath $resolvedOutput) {
        $outputItem = Get-Item -LiteralPath $resolvedOutput -Force
        if (-not $outputItem.PSIsContainer) {
            throw 'OutputRoot must be a directory'
        }
        if (@(Get-ChildItem -LiteralPath $resolvedOutput -Force).Count -ne 0) {
            throw 'OutputRoot must be empty'
        }
    }
    else {
        [void](New-Item -ItemType Directory -Path $resolvedOutput)
    }

    $wslContext = Resolve-WslInspectionContext
    $before = Get-VhdSnapshot -Path $resolvedVhd
    [void](Invoke-NativeChecked -FilePath 'wsl.exe' -Arguments @('--terminate', 'Ubuntu-E') -Label 'Ubuntu-E termination')
    $beforeDevices = Invoke-WslContextScript -ContextPrefix $wslContext.Prefix -Script 'lsblk -pnro NAME' -Label 'pre-attachment block inventory'

    $diskImage = Get-DiskImage -ImagePath $resolvedVhd -ErrorAction Stop
    if ($diskImage.Attached) {
        throw 'source VHD is already attached'
    }

    $requiredArtifacts = @(
        'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_results/Dataset192_IMPlesionCleanV3RGB256/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_all/checkpoint_final.pth',
        'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_preprocessed/Dataset192_IMPlesionCleanV3RGB256/nnUNetPlans.json',
        'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_preprocessed/Dataset192_IMPlesionCleanV3RGB256/dataset_fingerprint.json',
        'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_raw/Dataset192_IMPlesionCleanV3RGB256/dataset.json',
        'home/admin_mugen/imp_cache/loop192_nnunet_clean_v3_results/Dataset192_IMPlesionCleanV3RGB256/nnUNetTrainer_100epochs__nnUNetPlans__2d/plans.json'
    )
    $packageSpecs = @(
        [ordered]@{ slug = 'nnunetv2'; normalized = 'nnunetv2' },
        [ordered]@{ slug = 'torch'; normalized = 'torch' },
        [ordered]@{ slug = 'dynamic-network-architectures'; normalized = 'dynamic-network-architectures' },
        [ordered]@{ slug = 'batchgenerators'; normalized = 'batchgenerators' },
        [ordered]@{ slug = 'batchgeneratorsv2'; normalized = 'batchgeneratorsv2' },
        [ordered]@{ slug = 'numpy'; normalized = 'numpy' },
        [ordered]@{ slug = 'scipy'; normalized = 'scipy' },
        [ordered]@{ slug = 'simpleitk'; normalized = 'simpleitk' },
        [ordered]@{ slug = 'acvl-utils'; normalized = 'acvl-utils' }
    )

    $vhdMountAttempted = $false
    $wslAttachAttempted = $false
    $filesystemMountAttempted = $false
    $physicalDrive = $null
    $linuxMount = "/mnt/wsl/loop192-recovery-$PID"
    $cleanupErrors = @()
    $operationError = $null

    try {
        $vhdMountAttempted = $true
        $mountedVhd = Mount-VHD -Path $resolvedVhd -ReadOnly -Passthru -ErrorAction Stop
        $diskNumber = [int]$mountedVhd.DiskNumber
        $disk = Get-Disk -Number $diskNumber -ErrorAction Stop
        if (-not $disk.IsReadOnly) {
            throw 'Windows disk is not read-only'
        }

        $physicalDrive = "\\.\PHYSICALDRIVE$diskNumber"
        $wslAttachAttempted = $true
        [void](Invoke-NativeChecked -FilePath 'wsl.exe' -Arguments @('--mount', $physicalDrive, '--bare') -Label 'bare WSL attachment')

        $beforeSet = @{}
        foreach ($line in $beforeDevices) {
            $name = $line.Trim()
            if ($name) {
                $beforeSet[$name] = $true
            }
        }
        $listCommand = @'
lsblk -pnro NAME,FSTYPE | while read -r name fstype rest; do
    printf '%s|%s\n' "$name" "$fstype"
done
'@
        $deviceRows = Invoke-WslContextScript -ContextPrefix $wslContext.Prefix -Script $listCommand -Label 'post-attachment block inventory'
        $candidates = @()
        foreach ($row in $deviceRows) {
            $separator = $row.IndexOf('|')
            if ($separator -lt 1) {
                continue
            }
            $name = $row.Substring(0, $separator).Trim()
            $fileSystem = $row.Substring($separator + 1).Trim()
            if (-not $beforeSet.ContainsKey($name) -and $fileSystem -ceq 'ext4') {
                $candidates += $name
            }
        }
        if ($candidates.Count -ne 1) {
            throw "expected exactly one new ext4 device, found $($candidates.Count)"
        }
        $linuxDevice = $candidates[0]

        $mountCommand = @'
set -eu
mkdir -p -- "$1"
mount -t ext4 -o ro,noload -- "$2" "$1"
'@
        $filesystemMountAttempted = $true
        [void](Invoke-WslContextScript -ContextPrefix $wslContext.Prefix -Script $mountCommand -Arguments @($linuxMount, $linuxDevice) -Label 'read-only ext4 mount')

        $proofCommand = @'
set -eu
options=$(awk -v target="$1" '$2 == target { print $4 }' /proc/mounts)
case ",$options," in
    *,ro,*) ;;
    *) exit 1 ;;
esac
case ",$options," in
    *,noload,*) ;;
    *) exit 1 ;;
esac
'@
        [void](Invoke-WslContextScript -ContextPrefix $wslContext.Prefix -Script $proofCommand -Arguments @($linuxMount) -Label 'ext4 read-only mount proof')

        $wslOutput = ConvertTo-WslPath -ContextPrefix $wslContext.Prefix -WindowsPath $resolvedOutput
        $copyCommand = @'
set -eu
test -f "$1/$2"
cp -- "$1/$2" "$3/$4"
'@
        foreach ($relative in $requiredArtifacts) {
            $filename = $relative.Split('/')[-1]
            [void](Invoke-WslContextScript -ContextPrefix $wslContext.Prefix -Script $copyCommand -Arguments @($linuxMount, $relative, $wslOutput, $filename) -Label "artifact copy: $filename")
        }

        $packageRoot = "$linuxMount/.venv/lib/python3.12/site-packages"
        $packageCopyCommand = @'
set -eu
root=$1
wanted=$2
destination=$3
found=
for metadata in "$root"/*.dist-info/METADATA; do
    test -f "$metadata" || continue
    name=$(sed -n 's/^Name:[[:space:]]*//p' "$metadata" | head -n 1)
    normalized=$(printf '%s' "$name" | tr '[:upper:]_' '[:lower:]-')
    if test "$normalized" = "$wanted"; then
        test -z "$found" || exit 4
        found=$(dirname "$metadata")
    fi
done
test -n "$found"
test -f "$found/METADATA"
test -f "$found/RECORD"
mkdir -p -- "$destination"
cp -- "$found/METADATA" "$destination/METADATA"
cp -- "$found/RECORD" "$destination/RECORD"
if test -f "$found/direct_url.json"; then
    cp -- "$found/direct_url.json" "$destination/direct_url.json"
fi
'@
        foreach ($package in $packageSpecs) {
            $destination = "$wslOutput/package_metadata/$($package.slug)"
            [void](Invoke-WslContextScript -ContextPrefix $wslContext.Prefix -Script $packageCopyCommand -Arguments @($packageRoot, $package.normalized, $destination) -Label "package identity copy: $($package.slug)")
        }
    }
    catch {
        $operationError = $_.Exception
    }
    finally {
        $cleanupErrors = @(Invoke-RecoveryCleanup `
            -FilesystemMountAttempted $filesystemMountAttempted `
            -WslAttachAttempted $wslAttachAttempted `
            -VhdMountAttempted $vhdMountAttempted `
            -WslContextPrefix $wslContext.Prefix `
            -LinuxMount $linuxMount `
            -PhysicalDrive $physicalDrive `
            -ResolvedVhd $resolvedVhd)
    }
    Assert-RecoveryCompleted -OperationError $operationError -CleanupErrors $cleanupErrors

    $postImage = Get-DiskImage -ImagePath $resolvedVhd -ErrorAction Stop
    if ($postImage.Attached) {
        throw 'source VHD remains attached after recovery'
    }
    $after = Get-VhdSnapshot -Path $resolvedVhd
    Assert-SnapshotUnchanged -Before $before -After $after

    Write-RuntimeIdentity -Root $resolvedOutput -PackageSpecs $packageSpecs
    $sourceReport = Get-Content -LiteralPath $resolvedReport -Raw | ConvertFrom-Json
    $verificationReport = [ordered]@{
        candidate_id = $sourceReport.candidate_id
        provenance = $sourceReport.provenance
        source_vhd_proof = [ordered]@{
            before = $before
            after = $after
        }
    }
    $verificationJson = $verificationReport | ConvertTo-Json -Depth 8 -Compress

    $repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $pythonExe = Join-Path $repoRoot '.venv-win\Scripts\python.exe'
    $verifierPath = Join-Path $PSScriptRoot 'verify_nnunet_bundle.py'
    $receiptPath = Join-Path $resolvedOutput 'recovery_receipt.json'
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
        throw 'trusted Python executable unavailable'
    }
    $verifierCode = @'
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("loop192_verifier", sys.argv[1])
if spec is None or spec.loader is None:
    raise RuntimeError("unable to load trusted verifier")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report = json.loads(sys.stdin.read())
receipt = module.verify_bundle(Path(sys.argv[2]), report)
Path(sys.argv[3]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
'@
    $verificationJson | & $pythonExe -c $verifierCode $verifierPath $resolvedOutput $receiptPath
    if ($LASTEXITCODE -ne 0) {
        throw "Loop192 bundle verification failed with exit code $LASTEXITCODE"
    }

    Write-Output 'recovery=passed'
}

function Invoke-AutomaticRecovery {
    $containerContext = Get-ContainerRecoveryContext
    if ($null -ne $containerContext) {
        Invoke-ContainerRecovery `
            -Context $containerContext `
            -VhdPath $VhdPath `
            -ReportPath $ReportPath `
            -OutputRoot $OutputRoot
        return
    }

    Assert-Administrator
    Invoke-WindowsAttachRecovery
}

try {
    Invoke-AutomaticRecovery
    exit 0
}
catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
