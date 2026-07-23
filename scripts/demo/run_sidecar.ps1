[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$PreserveMode,
    [string]$RunId = '',
    [string]$Root = '',
    [string]$BundlePath = '',
    [string]$DockerPath = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$script:ContainerName = 'imp-nnunet-loop192'
$script:ImageReference = 'imp-nnunet-sidecar:loop192'
$script:ImageId = 'sha256:86bd77c03c3918e3638565e29417cdf4360b499a0813fbc425dc36645f026f2d'
$script:HealthUrl = 'http://127.0.0.1:7862/health'
$script:ProtocolId = ''
$script:ModelId = ''
$script:CheckpointSha256 = ''
$script:CheckpointSize = 0
$script:DatasetSha256 = ''
$script:DatasetSize = 0
$script:FingerprintSha256 = ''
$script:FingerprintSize = 0
$script:PlansSha256 = ''
$script:PlansSize = 0
$script:RuntimeGitCommit = ''
$script:RuntimeStatus = ''
$script:RuntimeVersion = ''
$script:RecoveryReceiptSha256 = ''
$script:ReleaseManifestSha256 = ''

function Initialize-ReleaseProjection {
    param([Parameter(Mandatory = $true)][string]$Root)
    $path = Join-Path $Root 'release\imp_release_manifest.json'
    $bytes = [IO.File]::ReadAllBytes($path)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $script:ReleaseManifestSha256 = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
    $release = [Text.Encoding]::ASCII.GetString($bytes) | ConvertFrom-Json
    if ($release.schema_version -cne 'imp.release.manifest.v1') { throw 'release manifest schema mismatch' }
    $live = @($release.comparisons | Where-Object { $_.id -ceq 'live_demo' })
    if ($live.Count -ne 1) { throw 'release live projection mismatch' }
    $nnunet = $release.models.($live[0].right_model_id)
    $sidecar = $release.provenance.sidecar
    if ($null -eq $nnunet -or $null -eq $sidecar) { throw 'release manifest projection missing' }
    $script:ProtocolId = [string]$nnunet.runtime.protocol
    $script:ModelId = [string]$live[0].right_model_id
    $script:CheckpointSha256 = [string]$nnunet.checkpoint_sha256
    $script:CheckpointSize = [long]$sidecar.checkpoint_size
    $script:DatasetSha256 = [string]$sidecar.dataset_sha256
    $script:DatasetSize = [long]$sidecar.dataset_size
    $script:FingerprintSha256 = [string]$sidecar.fingerprint_sha256
    $script:FingerprintSize = [long]$sidecar.fingerprint_size
    $script:PlansSha256 = [string]$sidecar.plans_sha256
    $script:PlansSize = [long]$sidecar.plans_size
    $script:RuntimeGitCommit = [string]$sidecar.runtime_git_commit
    $script:RuntimeStatus = [string]$sidecar.runtime_status
    $script:RuntimeVersion = [string]$sidecar.runtime_version
    $script:RecoveryReceiptSha256 = [string]$sidecar.recovery_receipt_sha256
}

function Assert-PreserveRunId {
    param([string]$RunId)
    if ($RunId -cnotmatch '^[a-z0-9][a-z0-9_-]{0,127}$') {
        throw 'Preserve run ID is unsafe.'
    }
    return $RunId
}

function Get-SidecarDockerOwnerLabelKey {
    # Windows PowerShell strips embedded quotes from native argv unless escaped.
    if ($PSVersionTable.PSVersion.Major -lt 6) {
        return '\"imp.demo.owner\"'
    }
    return '"imp.demo.owner"'
}

function Get-PreserveComponentDirectory {
    param([string]$Root, [string]$RunId, [string]$Component)
    $safeRunId = Assert-PreserveRunId -RunId $RunId
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
    try { $stream.Write($bytes, 0, $bytes.Length) }
    finally { $stream.Dispose() }
}

function New-SidecarContainerName {
    param([string]$RunId, [string]$OwnerToken)
    $safeRunId = Assert-PreserveRunId -RunId $RunId
    if ($OwnerToken -cnotmatch '^[0-9a-f]{32}$') {
        throw 'Sidecar owner token is unsafe.'
    }
    $runPart = $safeRunId.Substring(0, [Math]::Min(24, $safeRunId.Length))
    return "imp-nnunet-$runPart-$($OwnerToken.Substring(0, 8))"
}

function Assert-RegularFile {
    param([Parameter(Mandatory = $true)][string]$Path, [string]$Label = 'file')
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "$Label must be a regular file"
    }
    return $item.FullName
}

function Assert-ContainedDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Parent,
        [string]$Label = 'directory'
    )
    $parentItem = Get-Item -LiteralPath $Parent -Force -ErrorAction Stop
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not $parentItem.PSIsContainer -or -not $item.PSIsContainer) {
        throw "$Label must be a directory"
    }
    $resolvedParent = [IO.Path]::GetFullPath($parentItem.FullName).TrimEnd(
        [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
    )
    $resolved = [IO.Path]::GetFullPath($item.FullName).TrimEnd(
        [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
    )
    $prefix = $resolvedParent + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label is outside the owned runtime root"
    }
    $cursor = $item
    while ($null -ne $cursor -and $cursor.FullName.StartsWith(
        $resolvedParent, [StringComparison]::OrdinalIgnoreCase
    )) {
        if (($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Label path contains a reparse point"
        }
        if ([string]::Equals(
            [IO.Path]::GetFullPath($cursor.FullName).TrimEnd('\', '/'),
            $resolvedParent,
            [StringComparison]::OrdinalIgnoreCase
        )) {
            break
        }
        $cursor = $cursor.Parent
    }
    return $resolved
}

function Resolve-DockerApplication {
    param([string]$ExplicitPath = '')
    if ($ExplicitPath) {
        $resolved = Assert-RegularFile -Path $ExplicitPath -Label 'Docker executable'
    }
    else {
        $command = Get-Command docker.exe -CommandType Application -ErrorAction Stop
        $resolved = Assert-RegularFile -Path $command.Source -Label 'Docker executable'
    }
    if ([IO.Path]::GetFileName($resolved) -cne 'docker.exe') {
        throw 'Docker executable filename must be exactly docker.exe'
    }
    return $resolved
}

function Resolve-SidecarContext {
    param(
        [string]$Root = '',
        [string]$BundlePath = '',
        [string]$DockerPath = '',
        [switch]$PreserveMode,
        [string]$RunId = ''
    )
    if (-not $Root) {
        $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    }
    else {
        $Root = (Resolve-Path -LiteralPath $Root -ErrorAction Stop).Path
    }
    $runtimeRoot = Join-Path $Root 'demo_runtime\nnunet'
    if (-not $BundlePath) {
        $BundlePath = Join-Path $runtimeRoot 'recovered-container-final2'
    }
    $resolvedBundle = Assert-ContainedDirectory `
        -Path $BundlePath -Parent $runtimeRoot -Label 'Loop192 bundle'
    $ownerToken = [guid]::NewGuid().ToString('N')
    if ($PreserveMode) {
        $RunId = Assert-PreserveRunId -RunId $RunId
    }
    [pscustomobject]@{
        Root = $Root
        RuntimeRoot = (Resolve-Path -LiteralPath $runtimeRoot).Path
        BundlePath = $resolvedBundle
        DockerPath = Resolve-DockerApplication -ExplicitPath $DockerPath
        OwnerToken = $ownerToken
        PreserveMode = [bool]$PreserveMode
        PreserveRunId = $RunId
        ContainerName = if ($PreserveMode) {
            New-SidecarContainerName -RunId $RunId -OwnerToken $ownerToken
        } else { $script:ContainerName }
        OwnerRecordPath = ''
    }
}

function Assert-Sha256 {
    param([string]$Path, [string]$Expected, [string]$Label)
    $resolved = Assert-RegularFile -Path $Path -Label $Label
    $observed = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observed -cne $Expected) {
        throw "$Label hash mismatch"
    }
}

function Test-ExactRecoveryReceipt {
    param(
        [Parameter(Mandatory = $true)]$Receipt,
        [Parameter(Mandatory = $true)]$Manifest
    )
    $metadataNames = @($Receipt.metadata.PSObject.Properties.Name | Sort-Object)
    $expectedMetadata = @(
        'dataset.json',
        'plans.json',
        'requirements.lock',
        'runtime_identity.json'
    ) | Sort-Object
    return (
        ($metadataNames -join ',') -ceq ($expectedMetadata -join ',') -and
        $Receipt.schema_version -ceq 'loop192.recovery.receipt.v1' -and
        $Receipt.model_id -ceq $script:ModelId -and
        $Receipt.checkpoint_sha256 -ceq $script:CheckpointSha256 -and
        $Receipt.plans_sha256 -ceq [string]$Manifest.artifacts.'plans.json'.sha256 -and
        $Receipt.fingerprint_sha256 -ceq [string]$Manifest.artifacts.'dataset_fingerprint.json'.sha256 -and
        $Receipt.source_vhd_unchanged -is [bool] -and
        $Receipt.source_vhd_unchanged -and
        $Receipt.runtime_status -ceq 'reconstructed_required'
    )
}

function Test-ExactModelInputSpacing {
    param([Parameter(Mandatory = $true)]$Spacing)
    $values = @($Spacing)
    if ($values.Count -ne 3) { return $false }
    $expected = @([double]999, [double]1, [double]1)
    $numericTypes = @(
        [TypeCode]::SByte, [TypeCode]::Byte,
        [TypeCode]::Int16, [TypeCode]::UInt16,
        [TypeCode]::Int32, [TypeCode]::UInt32,
        [TypeCode]::Int64, [TypeCode]::UInt64,
        [TypeCode]::Single, [TypeCode]::Double, [TypeCode]::Decimal
    )
    for ($index = 0; $index -lt $values.Count; $index++) {
        $value = $values[$index]
        if ($null -eq $value) { return $false }
        $typeCode = [Type]::GetTypeCode($value.GetType())
        if ($numericTypes -notcontains $typeCode) { return $false }
        $number = [double]$value
        if (
            [double]::IsNaN($number) -or
            [double]::IsInfinity($number) -or
            $number -ne $expected[$index]
        ) {
            return $false
        }
    }
    return $true
}

function Assert-VerifiedBundle {
    param([Parameter(Mandatory = $true)]$Context)
    $manifestPath = Join-Path $Context.Root 'sidecar\nnunet\model_manifest.example.json'
    [void](Assert-RegularFile -Path $manifestPath -Label 'model manifest')
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $manifestFields = @($manifest.PSObject.Properties.Name | Sort-Object)
    $expectedManifestFields = @(
        'artifacts', 'input', 'model_id', 'release_manifest_sha256',
        'runtime', 'schema_version'
    ) | Sort-Object
    $runtimeFields = @($manifest.runtime.PSObject.Properties.Name | Sort-Object)
    $expectedRuntimeFields = @(
        'distribution', 'environment_status', 'recovered_git_commit', 'version'
    ) | Sort-Object
    $inputFields = @($manifest.input.PSObject.Properties.Name | Sort-Object)
    $expectedInputFields = @('channels', 'layout', 'spacing') | Sort-Object
    if (
        ($manifestFields -join ',') -cne ($expectedManifestFields -join ',') -or
        ($runtimeFields -join ',') -cne ($expectedRuntimeFields -join ',') -or
        ($inputFields -join ',') -cne ($expectedInputFields -join ',') -or
        $manifest.schema_version -cne 'imp.nnunet.model-manifest.v1' -or
        $manifest.release_manifest_sha256 -cne $script:ReleaseManifestSha256 -or
        $manifest.model_id -cne $script:ModelId -or
        $manifest.runtime.distribution -cne 'nnunetv2' -or
        $manifest.runtime.version -cne $script:RuntimeVersion -or
        $manifest.runtime.recovered_git_commit -cne $script:RuntimeGitCommit -or
        $manifest.runtime.environment_status -cne $script:RuntimeStatus -or
        $manifest.input.layout -cne 'CZYX' -or
        $manifest.input.channels -ne 3 -or
        -not (Test-ExactModelInputSpacing -Spacing $manifest.input.spacing)
    ) {
        throw 'model manifest identity mismatch'
    }
    $expectedArtifacts = [ordered]@{
        'checkpoint_final.pth' = [pscustomobject]@{sha256=$script:CheckpointSha256;size=$script:CheckpointSize}
        'dataset.json' = [pscustomobject]@{sha256=$script:DatasetSha256;size=$script:DatasetSize}
        'dataset_fingerprint.json' = [pscustomobject]@{sha256=$script:FingerprintSha256;size=$script:FingerprintSize}
        'plans.json' = [pscustomobject]@{sha256=$script:PlansSha256;size=$script:PlansSize}
    }
    $artifactNames = @($manifest.artifacts.PSObject.Properties.Name | Sort-Object)
    $expectedArtifactNames = @($expectedArtifacts.Keys | Sort-Object)
    if (($artifactNames -join ',') -cne ($expectedArtifactNames -join ',')) {
        throw 'model manifest artifact set mismatch'
    }
    foreach ($property in $manifest.artifacts.PSObject.Properties) {
        $name = [string]$property.Name
        if ($name -cnotmatch '^[A-Za-z0-9_.-]+$') {
            throw 'model manifest artifact name is unsafe'
        }
        $expected = $expectedArtifacts[$name]
        $artifactFields = @($property.Value.PSObject.Properties.Name | Sort-Object)
        if (
            ($artifactFields -join ',') -cne 'sha256,size' -or
            [string]$property.Value.sha256 -cne [string]$expected.sha256 -or
            [long]$property.Value.size -ne [long]$expected.size
        ) {
            throw "bundle artifact $name semantic pin mismatch"
        }
        $artifact = Join-Path $Context.BundlePath $name
        Assert-Sha256 -Path $artifact -Expected ([string]$property.Value.sha256) -Label "bundle artifact $name"
        if ((Get-Item -LiteralPath $artifact).Length -ne [long]$property.Value.size) {
            throw "bundle artifact $name size mismatch"
        }
    }

    $receiptPath = Join-Path $Context.BundlePath 'recovery_receipt.json'
    [void](Assert-RegularFile -Path $receiptPath -Label 'recovery receipt')
    Assert-Sha256 `
        -Path $receiptPath `
        -Expected $script:RecoveryReceiptSha256 `
        -Label 'recovery receipt'
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if (-not (Test-ExactRecoveryReceipt -Receipt $receipt -Manifest $manifest)) {
        throw 'recovery receipt identity mismatch'
    }
    foreach ($property in $receipt.metadata.PSObject.Properties) {
        $name = [string]$property.Name
        if ($name -cnotmatch '^[A-Za-z0-9_.-]+$') {
            throw 'recovery receipt artifact name is unsafe'
        }
        $artifact = Join-Path $Context.BundlePath $name
        Assert-Sha256 -Path $artifact -Expected ([string]$property.Value.sha256) -Label "receipt artifact $name"
        if ((Get-Item -LiteralPath $artifact).Length -ne [long]$property.Value.size) {
            throw "receipt artifact $name size mismatch"
        }
    }
}

function Invoke-DockerCommand {
    param([string]$DockerPath, [string[]]$Arguments, [string]$Label)
    $lines = @(& $DockerPath @Arguments 2>&1 | ForEach-Object { [string]$_ })
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "$Label failed with exit code $code"
    }
    return $lines
}

function Assert-PinnedDockerImage {
    param([Parameter(Mandatory = $true)]$Context)
    [void](Invoke-DockerCommand -DockerPath $Context.DockerPath -Arguments @('version', '--format', '{{.Server.Version}}') -Label 'Docker service probe')
    $identity = @(Invoke-DockerCommand -DockerPath $Context.DockerPath -Arguments @('image', 'inspect', '--format', '{{.Id}}', $script:ImageReference) -Label 'pinned image probe')
    if ($identity.Count -ne 1 -or $identity[0].Trim() -cne $script:ImageId) {
        throw 'local sidecar image identity mismatch; rebuilding is required'
    }
}

function New-SidecarRunArguments {
    param(
        [string]$BundlePath,
        [string]$OwnerToken,
        [string]$ContainerName = $script:ContainerName,
        [switch]$PreserveMode
    )
    $arguments = @(
        'run', '--detach', '--name', $ContainerName,
        '--label', 'imp.demo.component=nnunet-sidecar',
        '--label', "imp.demo.owner=$OwnerToken",
        '--gpus', 'device=0',
        '--publish', '127.0.0.1:7862:7862',
        '--mount', "type=bind,source=$BundlePath,target=/models/loop192,readonly",
        '--read-only',
        '--tmpfs', '/tmp:rw,noexec,nosuid,size=256m',
        '--memory', '12g',
        '--restart', 'no',
        '--security-opt', 'no-new-privileges',
        '--cap-drop', 'ALL',
        $script:ImageReference
    )
    if (-not $PreserveMode) {
        $arguments = @('run', '--detach', '--rm') + $arguments[2..($arguments.Count - 1)]
    }
    return $arguments
}

function Get-SidecarOwnerRecordPath {
    param([Parameter(Mandatory = $true)]$Context, [switch]$PreserveMode)
    $isPreserved = $PreserveMode -or (
        $Context.PSObject.Properties.Name -contains 'PreserveMode' -and $Context.PreserveMode
    )
    $ownerRoot = if ($isPreserved) {
        Get-PreserveComponentDirectory `
            -Root $Context.Root -RunId $Context.PreserveRunId -Component 'sidecar'
    }
    else {
        Join-Path $Context.RuntimeRoot 'launcher'
    }
    if (-not (Test-Path -LiteralPath $ownerRoot)) {
        [void](New-Item -ItemType Directory -Path $ownerRoot)
    }
    $resolved = if ($isPreserved) {
        (Get-Item -LiteralPath $ownerRoot -Force -ErrorAction Stop).FullName
    }
    else {
        Assert-ContainedDirectory `
            -Path $ownerRoot -Parent $Context.RuntimeRoot -Label 'launcher owner directory'
    }
    if ($isPreserved) {
        return (Join-Path $resolved ("owner-" + [guid]::NewGuid().ToString('N') + '.json'))
    }
    return (Join-Path $resolved 'sidecar.json')
}

function Write-SidecarOwnerRecord {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [string]$ContainerId,
        [switch]$PreserveMode
    )
    $recordPath = Get-SidecarOwnerRecordPath `
        -Context $Context -PreserveMode:$PreserveMode
    if (-not $PreserveMode -and (Test-Path -LiteralPath $recordPath)) {
        throw 'a sidecar owner record already exists'
    }
    $record = [ordered]@{
        schema_version = 'imp.demo.sidecar-owner.v1'
        container_name = if ($Context.PSObject.Properties.Name -contains 'ContainerName') {
            $Context.ContainerName
        } else { $script:ContainerName }
        container_id = $ContainerId
        owner_token = $Context.OwnerToken
        docker_path = $Context.DockerPath
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

function Write-PreserveSidecarLifecycleRecord {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [Parameter(Mandatory = $true)][ValidateSet('started', 'stopped')][string]$Event,
        [Parameter(Mandatory = $true)][string]$ContainerId
    )
    if (-not $Context.PreserveMode) { return $null }
    $ownerPath = [string]$Context.OwnerRecordPath
    $ownerName = [IO.Path]::GetFileName($ownerPath)
    if ($ownerName -cnotmatch '^owner-[0-9a-f]{32}\.json$') {
        throw 'Preserved sidecar owner record identity is unsafe.'
    }
    $directory = Get-PreserveComponentDirectory `
        -Root $Context.Root -RunId $Context.PreserveRunId -Component 'sidecar'
    $expectedParent = [IO.Path]::GetFullPath($directory).TrimEnd('\', '/')
    $actualParent = [IO.Path]::GetFullPath(
        (Split-Path -Parent $ownerPath)
    ).TrimEnd('\', '/')
    if (-not [string]::Equals(
        $actualParent, $expectedParent, [StringComparison]::OrdinalIgnoreCase
    )) {
        throw 'Preserved sidecar owner record escapes its component directory.'
    }
    if ($ContainerId -cnotmatch '^[0-9a-f]{64}$') {
        throw 'Preserved sidecar container identity is invalid.'
    }
    $record = [ordered]@{
        event = $Event
        owner_record = $ownerName
        container_id = $ContainerId
        container_name = [string]$Context.ContainerName
        release_manifest_sha256 = $script:ReleaseManifestSha256
        recorded_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    $suffix = if ($Event -ceq 'started') { 'started' } else { 'stopped' }
    $path = Join-Path $directory (
        "$Event-" + [guid]::NewGuid().ToString('N') + ".$suffix.json"
    )
    Write-ExclusiveAsciiFile `
        -Path $path -Content ($record | ConvertTo-Json -Compress)
    return $path
}

function Remove-SidecarOwnerRecord {
    param([Parameter(Mandatory = $true)]$Context)
    $recordPath = Join-Path (Join-Path $Context.RuntimeRoot 'launcher') 'sidecar.json'
    if (Test-Path -LiteralPath $recordPath -PathType Leaf) {
        Remove-Item -LiteralPath $recordPath -Force
    }
}

function Assert-SidecarOwnerRecordAbsent {
    param([Parameter(Mandatory = $true)]$Context)
    if ($Context.PSObject.Properties.Name -contains 'PreserveMode' -and $Context.PreserveMode) {
        return
    }
    $recordPath = Get-SidecarOwnerRecordPath -Context $Context
    if (Test-Path -LiteralPath $recordPath) {
        throw 'a sidecar owner record already exists'
    }
}

function Test-ContainerNamePresent {
    param([string]$DockerPath, [string]$ContainerName = $script:ContainerName)
    $matches = @(Invoke-DockerCommand -DockerPath $DockerPath -Arguments @(
        'container', 'list', '--all', '--filter', "name=^/$ContainerName$", '--format',
        '{{.Names}}'
    ) -Label 'sidecar fixed-name presence proof')
    if ($matches.Count -eq 0) {
        return $false
    }
    if ($matches.Count -eq 1 -and $matches[0].Trim() -ceq $ContainerName) {
        return $true
    }
    throw 'sidecar fixed-name presence proof was ambiguous'
}

function Test-SidecarContainerAbsent {
    param([string]$DockerPath, [string]$ContainerId)
    $matches = @(Invoke-DockerCommand -DockerPath $DockerPath -Arguments @(
        'container', 'list', '--all', '--quiet', '--no-trunc', '--filter', "id=$ContainerId"
    ) -Label 'sidecar absence proof')
    if ($matches.Count -eq 0) {
        return $true
    }
    if ($matches.Count -eq 1 -and $matches[0].Trim() -ceq $ContainerId) {
        return $false
    }
    throw 'sidecar absence proof was ambiguous'
}

function Get-ExactContainerIdFromRunOutput {
    param([Parameter(Mandatory = $true)]$OutputLines)
    $containerIds = @(
        $OutputLines |
            ForEach-Object { ([string]$_).Trim() } |
            Where-Object { $_ -cmatch '^[0-9a-f]{64}$' }
    )
    if ($containerIds.Count -ne 1) {
        throw 'sidecar start did not return exactly one container ID'
    }
    return $containerIds[0]
}

function Test-SidecarContainerNameAbsent {
    param([string]$DockerPath, [string]$OwnerToken, [string]$ContainerName = $script:ContainerName)
    $labelKey = Get-SidecarDockerOwnerLabelKey
    $matches = @(Invoke-DockerCommand -DockerPath $DockerPath -Arguments @(
        'container', 'list', '--all', '--filter', "name=^/$ContainerName$", '--format',
        ('{{.Names}}|{{.Label ' + $labelKey + '}}')
    ) -Label 'sidecar fixed-name absence proof')
    if ($matches.Count -eq 0) {
        return $true
    }
    $expected = "$ContainerName|$OwnerToken"
    if ($matches.Count -eq 1 -and $matches[0].Trim() -ceq $expected) {
        return $false
    }
    throw 'sidecar fixed-name absence proof was ambiguous'
}

function Stop-OwnedSidecarByFixedName {
    param([string]$DockerPath, [string]$OwnerToken, [string]$ContainerName = $script:ContainerName)
    $labelKey = Get-SidecarDockerOwnerLabelKey
    try {
        $identity = @(Invoke-DockerCommand -DockerPath $DockerPath -Arguments @(
            'container', 'inspect', '--format',
            ('{{.Id}}|{{index .Config.Labels ' + $labelKey + '}}|{{.Name}}'),
            $ContainerName
        ) -Label 'fixed-name sidecar ownership inspection')
    }
    catch {
        $inspectionError = $_.Exception.Message
        if (Test-SidecarContainerNameAbsent `
            -DockerPath $DockerPath -OwnerToken $OwnerToken -ContainerName $ContainerName) {
            return
        }
        throw "fixed-name sidecar inspection failed and absence was not proven: $inspectionError"
    }
    if ($identity.Count -ne 1) {
        throw 'fixed-name sidecar ownership proof was ambiguous'
    }
    $parts = $identity[0].Trim().Split('|')
    if (
        $parts.Count -ne 3 -or
        $parts[0] -cnotmatch '^[0-9a-f]{64}$' -or
        $parts[1] -cne $OwnerToken -or
        $parts[2] -cne "/$ContainerName"
    ) {
        throw 'fixed-name sidecar ownership proof failed'
    }
    [void](Invoke-DockerCommand -DockerPath $DockerPath -Arguments @(
        'container', 'stop', '--time', '10', $parts[0]
    ) -Label 'fixed-name owned sidecar stop')
}

function Start-OwnedSidecar {
    param([Parameter(Mandatory = $true)]$Context, [switch]$PreserveMode)
    $containerName = if ($Context.PSObject.Properties.Name -contains 'ContainerName') {
        [string]$Context.ContainerName
    } else { $script:ContainerName }
    Assert-SidecarOwnerRecordAbsent -Context $Context
    if (Test-ContainerNamePresent `
        -DockerPath $Context.DockerPath -ContainerName $containerName) {
        throw 'the fixed sidecar container name is already in use'
    }
    $arguments = @(New-SidecarRunArguments `
        -BundlePath $Context.BundlePath -OwnerToken $Context.OwnerToken `
        -ContainerName $containerName -PreserveMode:$PreserveMode)
    $output = @(Invoke-DockerCommand -DockerPath $Context.DockerPath -Arguments $arguments -Label 'sidecar start')
    try {
        $containerId = Get-ExactContainerIdFromRunOutput -OutputLines $output
    }
    catch {
        $operationError = $_.Exception
        try {
            Stop-OwnedSidecarByFixedName `
                -DockerPath $Context.DockerPath -OwnerToken $Context.OwnerToken `
                -ContainerName $containerName
        }
        catch {
            throw "sidecar output cleanup failed after '$($operationError.Message)': $($_.Exception.Message)"
        }
        throw $operationError
    }
    $recordWritten = $false
    try {
        $ownerRecordPath = Write-SidecarOwnerRecord `
            -Context $Context -ContainerId $containerId -PreserveMode:$PreserveMode
        $recordWritten = $true
        if ($Context.PSObject.Properties.Name -contains 'OwnerRecordPath') {
            $Context.OwnerRecordPath = $ownerRecordPath
        }
        else {
            $Context | Add-Member -NotePropertyName OwnerRecordPath -NotePropertyValue $ownerRecordPath
        }
        if ($PreserveMode) {
            [void](Write-PreserveSidecarLifecycleRecord `
                -Context $Context -Event 'started' -ContainerId $containerId)
        }
    }
    catch {
        $operationError = $_.Exception
        $cleanupErrors = New-Object 'System.Collections.Generic.List[string]'
        $canRemoveRecord = $false
        try {
            Stop-OwnedSidecar `
                -DockerPath $Context.DockerPath -ContainerId $containerId `
                -OwnerToken $Context.OwnerToken -ContainerName $containerName
            $canRemoveRecord = $true
        }
        catch {
            $cleanupErrors.Add($_.Exception.Message)
            try {
                $canRemoveRecord = Test-SidecarContainerAbsent `
                    -DockerPath $Context.DockerPath -ContainerId $containerId
            }
            catch {
                $cleanupErrors.Add($_.Exception.Message)
            }
        }
        finally {
            if ($recordWritten -and $canRemoveRecord -and -not $PreserveMode) {
                try {
                    Remove-SidecarOwnerRecord -Context $Context
                }
                catch {
                    $cleanupErrors.Add($_.Exception.Message)
                }
            }
        }
        if ($cleanupErrors.Count -gt 0) {
            throw "sidecar start cleanup errors after '$($operationError.Message)': $($cleanupErrors -join '; ')"
        }
        throw $operationError
    }
    return $containerId
}

function Test-PinnedHealthPayload {
    param([Parameter(Mandatory = $true)]$Payload)
    $names = @($Payload.PSObject.Properties.Name | Sort-Object)
    $expected = @('checkpoint_sha256', 'device', 'model_id', 'protocol', 'ready')
    if (($names -join ',') -cne ($expected -join ',')) {
        return $false
    }
    return (
        $Payload.protocol -ceq $script:ProtocolId -and
        $Payload.model_id -ceq $script:ModelId -and
        $Payload.checkpoint_sha256 -ceq $script:CheckpointSha256 -and
        $Payload.device -ceq 'cuda:0' -and
        $Payload.ready -is [bool] -and
        $Payload.ready
    )
}

function Wait-PinnedSidecarHealth {
    param([int]$TimeoutSeconds = 120)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        try {
            $health = Invoke-RestMethod -Uri $script:HealthUrl -Method Get -TimeoutSec 3
            if (Test-PinnedHealthPayload -Payload $health) {
                return
            }
        }
        catch {
            # Startup connection failures are expected until the bounded deadline.
        }
        Start-Sleep -Milliseconds 1000
    } while ([DateTime]::UtcNow -lt $deadline)
    throw 'pinned sidecar health timed out'
}

function Stop-OwnedSidecar {
    param(
        [string]$DockerPath,
        [string]$ContainerId,
        [string]$OwnerToken,
        [string]$ContainerName = $script:ContainerName
    )
    $labelKey = Get-SidecarDockerOwnerLabelKey
    $identity = @(Invoke-DockerCommand -DockerPath $DockerPath -Arguments @(
        'container', 'inspect', '--format',
        ('{{.Id}}|{{index .Config.Labels ' + $labelKey + '}}|{{.Name}}'), $ContainerId
    ) -Label 'owned sidecar inspection')
    $expected = "$ContainerId|$OwnerToken|/$ContainerName"
    if ($identity.Count -ne 1 -or $identity[0].Trim() -cne $expected) {
        throw 'sidecar ownership proof failed; container was not stopped'
    }
    [void](Invoke-DockerCommand -DockerPath $DockerPath -Arguments @('container', 'stop', '--time', '10', $ContainerId) -Label 'owned sidecar stop')
}

function Test-OwnedSidecarStoppedOrAbsent {
    param(
        [string]$DockerPath,
        [string]$ContainerId,
        [string]$OwnerToken,
        [string]$ContainerName,
        [switch]$PreserveMode
    )
    $labelKey = Get-SidecarDockerOwnerLabelKey
    try {
        $identity = @(Invoke-DockerCommand -DockerPath $DockerPath -Arguments @(
            'container', 'inspect', '--format',
            ('{{.Id}}|{{index .Config.Labels ' + $labelKey + '}}|{{.Name}}|{{.State.Running}}'),
            $ContainerId
        ) -Label 'stopped sidecar identity proof')
    }
    catch {
        if ($PreserveMode) {
            throw 'preserved sidecar identity is absent; stopped container retention was not proven'
        }
        if (Test-SidecarContainerAbsent `
            -DockerPath $DockerPath -ContainerId $ContainerId) {
            return $true
        }
        throw 'sidecar identity was neither stopped nor absent'
    }
    $expected = "$ContainerId|$OwnerToken|/$ContainerName|false"
    if ($identity.Count -eq 1 -and $identity[0].Trim() -ceq $expected) {
        return $true
    }
    $running = "$ContainerId|$OwnerToken|/$ContainerName|true"
    if ($identity.Count -eq 1 -and $identity[0].Trim() -ceq $running) {
        return $false
    }
    throw 'stopped sidecar identity proof failed'
}

function Wait-SidecarPortClosed {
    param([int]$TimeoutSeconds = 15)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $listeners = @(Get-NetTCPConnection `
            -State Listen -LocalPort 7862 -ErrorAction SilentlyContinue)
        if ($listeners.Count -eq 0) { return }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    throw 'Sidecar listener 7862 remained open after owned container stop.'
}

function Complete-OwnedSidecarStop {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [Parameter(Mandatory = $true)][string]$ContainerId,
        [switch]$PreserveMode
    )
    $containerName = if ($Context.PSObject.Properties.Name -contains 'ContainerName') {
        [string]$Context.ContainerName
    }
    else {
        $script:ContainerName
    }
    $identityClosed = $false
    try {
        Stop-OwnedSidecar `
            -DockerPath $Context.DockerPath -ContainerId $ContainerId `
            -OwnerToken $Context.OwnerToken -ContainerName $containerName
    }
    catch {
        $stopError = $_.Exception.Message
        $identityClosed = Test-OwnedSidecarStoppedOrAbsent `
            -DockerPath $Context.DockerPath -ContainerId $ContainerId `
            -OwnerToken $Context.OwnerToken -ContainerName $containerName `
            -PreserveMode:$PreserveMode
        if (-not $identityClosed) {
            throw "owned sidecar stop failed and closed identity was not proven: $stopError"
        }
    }
    if (-not $identityClosed) {
        $identityClosed = Test-OwnedSidecarStoppedOrAbsent `
            -DockerPath $Context.DockerPath -ContainerId $ContainerId `
            -OwnerToken $Context.OwnerToken -ContainerName $containerName `
            -PreserveMode:$PreserveMode
        if (-not $identityClosed) {
            throw 'owned sidecar remained running after stop'
        }
    }
    Wait-SidecarPortClosed
    if ($PreserveMode) {
        [void](Write-PreserveSidecarLifecycleRecord `
            -Context $Context -Event 'stopped' -ContainerId $ContainerId)
    }
    else {
        Remove-SidecarOwnerRecord -Context $Context
    }
}

function Invoke-SidecarLaunch {
    [CmdletBinding()]
    param(
        [switch]$CheckOnly,
        [switch]$PreserveMode,
        [string]$RunId = '',
        [string]$Root = '',
        [string]$BundlePath = '',
        [string]$DockerPath = ''
    )
    try {
        $context = Resolve-SidecarContext `
            -Root $Root -BundlePath $BundlePath -DockerPath $DockerPath `
            -PreserveMode:$PreserveMode -RunId $RunId
        Initialize-ReleaseProjection -Root $context.Root
        Assert-VerifiedBundle -Context $context
        Assert-PinnedDockerImage -Context $context
        $contextContainerName = if ($context.PSObject.Properties.Name -contains 'ContainerName') {
            [string]$context.ContainerName
        } else { $script:ContainerName }
        $containerId = Start-OwnedSidecar -Context $context -PreserveMode:$PreserveMode
        try {
            Wait-PinnedSidecarHealth -TimeoutSeconds 120
        }
        catch {
            $operationError = $_.Exception
            try {
                Complete-OwnedSidecarStop `
                    -Context $context -ContainerId $containerId `
                    -PreserveMode:$PreserveMode
            }
            catch {
                throw "sidecar failure cleanup errors after '$($operationError.Message)': $($_.Exception.Message)"
            }
            throw $operationError
        }
        if ($CheckOnly) {
            Complete-OwnedSidecarStop `
                -Context $context -ContainerId $containerId `
                -PreserveMode:$PreserveMode
        }
        [Console]::Out.WriteLine(
            "sidecar_health=passed container=$containerId check_only=$([bool]$CheckOnly)"
        )
        return 0
    }
    catch {
        [Console]::Error.WriteLine("Sidecar launch failed closed: $($_.Exception.Message)")
        return 5
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    exit (Invoke-SidecarLaunch -CheckOnly:$CheckOnly -PreserveMode:$PreserveMode `
        -RunId $RunId -Root $Root -BundlePath $BundlePath -DockerPath $DockerPath)
}
