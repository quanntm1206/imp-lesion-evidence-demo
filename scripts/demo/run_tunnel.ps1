[CmdletBinding()]
param(
    [string]$CloudflaredPath = '',
    [string]$Root = '',
    [switch]$CheckOnly,
    [switch]$PreserveMode,
    [string]$RunId = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$script:LocalUrl = 'http://127.0.0.1:7860'

function Assert-ReleaseManifest {
    param([Parameter(Mandatory = $true)][string]$Root)
    $path = Join-Path $Root 'release\imp_release_manifest.json'
    $bytes = [IO.File]::ReadAllBytes($path)
    $payload = [Text.Encoding]::ASCII.GetString($bytes) | ConvertFrom-Json
    if ($payload.schema_version -cne 'imp.release.manifest.v1') { throw 'release manifest schema mismatch' }
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Resolve-TunnelRoot {
    param([string]$Root = '')
    if (-not $Root) {
        return (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    }
    return (Resolve-Path -LiteralPath $Root -ErrorAction Stop).Path
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

function Assert-SafeRegularFile {
    param([string]$Path, [string]$Label)
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "$Label must be a regular non-reparse file."
    }
    return $item.FullName
}

function Assert-GradioOwnerRecord {
    param(
        [Parameter(Mandatory = $true)]$Record,
        [Parameter(Mandatory = $true)][string]$ExpectedReleaseManifestSha256
    )
    try {
        $launcherPid = Convert-OwnedProcessId -Value $Record.launcher_pid
        $launcherStartTime = Convert-ProcessStartTimeUtc -Value $Record.launcher_start_time_utc
        $port = Convert-OwnedProcessId -Value $Record.port
    }
    catch {
        throw 'Gradio owner record identity mismatch.'
    }
    $fields = @($Record.PSObject.Properties.Name | Sort-Object)
    $expected = @(
        'host',
        'launcher_pid',
        'launcher_start_time_utc',
        'owner_nonce',
        'port',
        'preserve_mode',
        'public_tunnel_mode',
        'python_path',
        'release_manifest_sha256',
        'schema_version',
        'session_path'
    ) | Sort-Object
    if (
        ($fields -join ',') -cne ($expected -join ',') -or
        $Record.schema_version -cne 'imp.demo.gradio-owner.v1' -or
        $launcherPid -lt 1 -or
        $launcherStartTime -cnotmatch '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{7}Z$' -or
        [string]$Record.owner_nonce -cnotmatch '^[0-9a-f]{32}$' -or
        $Record.release_manifest_sha256 -cne $ExpectedReleaseManifestSha256 -or
        $Record.public_tunnel_mode -isnot [bool] -or
        -not $Record.public_tunnel_mode -or
        $Record.preserve_mode -isnot [bool] -or
        -not $Record.preserve_mode -or
        -not [IO.Path]::IsPathRooted([string]$Record.python_path) -or
        -not [IO.Path]::IsPathRooted([string]$Record.session_path) -or
        $Record.host -cne '127.0.0.1' -or
        $port -ne 7860
    ) {
        throw 'Gradio owner record identity mismatch.'
    }
}

function Read-SafeGradioOwnerRecord {
    param(
        [string]$Root,
        [Parameter(Mandatory = $true)][string]$ExpectedReleaseManifestSha256,
        [switch]$PreserveMode,
        [string]$RunId = ''
    )
    $runtimeRoot = Join-Path $Root 'demo_runtime'
    $ownerRoot = if ($PreserveMode) {
        Get-PreserveComponentDirectory -Root $Root -RunId $RunId -Component 'gradio'
    }
    else {
        Join-Path $runtimeRoot 'launcher'
    }
    $recordPath = if ($PreserveMode) {
        $records = @(Get-ChildItem -LiteralPath $ownerRoot -File -Filter 'owner-*.json' |
            Sort-Object LastWriteTimeUtc, Name)
        if ($records.Count -eq 0) { throw 'No preserved Gradio owner record exists.' }
        $records[-1].FullName
    }
    else {
        Join-Path $ownerRoot 'gradio.json'
    }
    foreach ($path in @($runtimeRoot, $ownerRoot)) {
        $item = Get-Item -LiteralPath $path -Force -ErrorAction Stop
        if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Gradio owner path is unsafe.'
        }
    }
    [void](Assert-SafeRegularFile -Path $recordPath -Label 'Gradio owner record')
    $recordItem = Get-Item -LiteralPath $recordPath -Force
    if (-not [string]::Equals(
        [IO.Path]::GetFullPath($recordItem.Directory.FullName).TrimEnd('\', '/'),
        [IO.Path]::GetFullPath($ownerRoot).TrimEnd('\', '/'),
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw 'Gradio owner record is outside the launcher directory.'
    }
    $record = Get-Content -LiteralPath $recordItem.FullName -Raw | ConvertFrom-Json
    Assert-GradioOwnerRecord `
        -Record $record `
        -ExpectedReleaseManifestSha256 $ExpectedReleaseManifestSha256

    $sessionsRoot = Join-Path $runtimeRoot 'sessions'
    $session = Get-Item -LiteralPath ([string]$record.session_path) -Force -ErrorAction Stop
    $sessions = Get-Item -LiteralPath $sessionsRoot -Force -ErrorAction Stop
    if (
        -not $session.PSIsContainer -or
        ($session.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        ($sessions.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        $session.Name -cnotmatch '^demo-[0-9a-f]{32}$' -or
        -not [string]::Equals(
            [IO.Path]::GetFullPath($session.Parent.FullName).TrimEnd('\', '/'),
            [IO.Path]::GetFullPath($sessions.FullName).TrimEnd('\', '/'),
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw 'Gradio session ownership path is unsafe.'
    }
    $record.python_path = Assert-SafeRegularFile `
        -Path ([string]$record.python_path) -Label 'Gradio Python executable'
    return $record
}

function Get-ExactProcess {
    param([int]$ProcessId)
    $values = @(Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId")
    if ($values.Count -gt 1) {
        throw 'Expected exactly one owned process incarnation.'
    }
    if ($values.Count -eq 0) {
        return $null
    }
    return $values[0]
}

function Convert-ProcessStartTimeUtc {
    param([Parameter(Mandatory = $true)]$Value)
    if ($Value -is [datetime]) {
        $instant = $Value.ToUniversalTime()
    }
    else {
        try {
            $instant = [datetime]::ParseExact(
                [string]$Value,
                'yyyy-MM-ddTHH:mm:ss.fffffffZ',
                [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::AssumeUniversal
            ).ToUniversalTime()
        }
        catch {
            $instant = [Management.ManagementDateTimeConverter]::ToDateTime(
                [string]$Value
            ).ToUniversalTime()
        }
    }
    return $instant.ToString(
        'yyyy-MM-ddTHH:mm:ss.fffffffZ',
        [Globalization.CultureInfo]::InvariantCulture
    )
}

function Test-ProcessStartTimeMatch {
    param(
        [Parameter(Mandatory = $true)]$Expected,
        [Parameter(Mandatory = $true)]$Actual
    )
    $format = 'yyyy-MM-ddTHH:mm:ss.fffffffZ'
    $style = [Globalization.DateTimeStyles]::AssumeUniversal
    $expectedInstant = [datetime]::ParseExact(
        (Convert-ProcessStartTimeUtc -Value $Expected),
        $format,
        [Globalization.CultureInfo]::InvariantCulture,
        $style
    )
    $actualInstant = [datetime]::ParseExact(
        (Convert-ProcessStartTimeUtc -Value $Actual),
        $format,
        [Globalization.CultureInfo]::InvariantCulture,
        $style
    )
    return [Math]::Abs(($expectedInstant - $actualInstant).Ticks) -le
        [TimeSpan]::FromMilliseconds(1).Ticks
}

function Convert-OwnedProcessId {
    param([Parameter(Mandatory = $true)]$Value)
    if (
        $Value -isnot [sbyte] -and
        $Value -isnot [byte] -and
        $Value -isnot [int16] -and
        $Value -isnot [uint16] -and
        $Value -isnot [int] -and
        $Value -isnot [uint32] -and
        $Value -isnot [long] -and
        $Value -isnot [uint64]
    ) {
        throw 'Owner process ID is not an integer.'
    }
    $processId = [int64]$Value
    if ($processId -lt 1 -or $processId -gt [int]::MaxValue) {
        throw 'Owner process ID is out of range.'
    }
    return [int]$processId
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

function Assert-TunnelProcessIdentity {
    param($Record, $Process)
    $recordProcessId = Convert-OwnedProcessId -Value $Record.process_id
    $recordStartTime = Convert-ProcessStartTimeUtc -Value $Record.process_start_time_utc
    $expectedPath = [IO.Path]::GetFullPath([string]$Record.executable_path)
    $actualPath = [IO.Path]::GetFullPath([string]$Process.ExecutablePath)
    if (
        [int]$Process.ProcessId -ne $recordProcessId -or
        -not (Test-ProcessStartTimeMatch -Expected $recordStartTime `
            -Actual $Process.CreationDate) -or
        [string]$Record.owner_nonce -cnotmatch '^[0-9a-f]{32}$' -or
        -not [string]::Equals(
            $actualPath, $expectedPath, [StringComparison]::OrdinalIgnoreCase
        ) -or
        [IO.Path]::GetFileName($actualPath) -notmatch '^cloudflared(?:\.exe)?$' -or
        $Record.local_url -cne $script:LocalUrl -or
        [string]$Process.CommandLine -notmatch
            'tunnel\s+--url\s+http://127\.0\.0\.1:7860(?:\s|$)'
    ) {
        throw 'Tunnel process incarnation mismatch.'
    }
}

function Assert-LauncherProcessIdentity {
    param($Record, $Process, [string]$ExpectedScript)
    if (
        [int]$Process.ProcessId -ne [int]$Record.launcher_pid -or
        -not (Test-ProcessStartTimeMatch `
            -Expected $Record.launcher_start_time_utc -Actual $Process.CreationDate)
    ) {
        throw 'Gradio launcher process incarnation mismatch.'
    }
    if ([IO.Path]::GetFileName([string]$Process.ExecutablePath) -notmatch '^(?:powershell|pwsh)\.exe$') {
        throw 'Gradio launcher executable identity mismatch.'
    }
    $command = ([string]$Process.CommandLine).Replace('/', '\')
    $fullScript = [IO.Path]::GetFullPath($ExpectedScript).Replace('/', '\')
    if (
        $command.IndexOf($fullScript, [StringComparison]::OrdinalIgnoreCase) -lt 0 -and
        $command -notmatch '(?i)(?:^|[\s"''])scripts\\demo\\run_demo\.ps1(?:[\s"'']|$)'
    ) {
        throw 'Gradio launcher script identity mismatch.'
    }
}

function Assert-GradioPythonWrapperIdentity {
    param($Record, $Process)
    $expectedPath = [IO.Path]::GetFullPath([string]$Record.python_path)
    if (
        [int]$Process.ParentProcessId -ne [int]$Record.launcher_pid -or
        -not [string]::Equals(
            [IO.Path]::GetFullPath([string]$Process.ExecutablePath),
            $expectedPath,
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        [string]$Process.CommandLine -notmatch '(?:^|\s)-m\s+lesion_robustness\.demo\.app(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--public-tunnel-mode(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--preserve-mode(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--host\s+127\.0\.0\.1(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--port\s+7860(?:\s|$)'
    ) {
        throw 'Gradio Python wrapper identity mismatch.'
    }
}

function Assert-GradioListenerProcessIdentity {
    param($Record, $Connection, $Process, $WrapperProcess)
    if (
        [string]$Connection.LocalAddress -cne '127.0.0.1' -or
        [int]$Connection.LocalPort -ne 7860 -or
        [int]$Connection.OwningProcess -ne [int]$Process.ProcessId -or
        [int]$Process.ParentProcessId -ne [int]$WrapperProcess.ProcessId -or
        [string]$Record.owner_nonce -cnotmatch '^[0-9a-f]{32}$' -or
        [IO.Path]::GetFileName([string]$Process.ExecutablePath) -cnotmatch '^python(?:\.exe)?$' -or
        [string]$Process.CommandLine -notmatch '(?:^|\s)-m\s+lesion_robustness\.demo\.app(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--public-tunnel-mode(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--preserve-mode(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--host\s+127\.0\.0\.1(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--port\s+7860(?:\s|$)'
    ) {
        throw 'Gradio listener process identity mismatch.'
    }
}

function Assert-OwnedGradioRuntime {
    param(
        [string]$Root,
        [Parameter(Mandatory = $true)][string]$ExpectedReleaseManifestSha256,
        [switch]$PreserveMode,
        [string]$RunId = ''
    )
    $record = Read-SafeGradioOwnerRecord `
        -Root $Root `
        -ExpectedReleaseManifestSha256 $ExpectedReleaseManifestSha256 `
        -PreserveMode:$PreserveMode -RunId $RunId
    $launcher = Get-ExactProcess -ProcessId ([int]$record.launcher_pid)
    Assert-LauncherProcessIdentity `
        -Record $record `
        -Process $launcher `
        -ExpectedScript (Join-Path $Root 'scripts\demo\run_demo.ps1')
    $connections = @(
        Get-NetTCPConnection -State Listen -LocalPort 7860 -ErrorAction Stop
    )
    if ($connections.Count -ne 1) {
        throw 'Expected exactly one Gradio listener on port 7860.'
    }
    $listener = Get-ExactProcess -ProcessId ([int]$connections[0].OwningProcess)
    if ($null -eq $listener) {
        throw 'Gradio listener process identity mismatch.'
    }
    $wrapper = Get-ExactProcess -ProcessId ([int]$listener.ParentProcessId)
    if ($null -eq $wrapper) {
        throw 'Gradio Python wrapper identity mismatch.'
    }
    Assert-GradioPythonWrapperIdentity -Record $record -Process $wrapper
    Assert-GradioListenerProcessIdentity `
        -Record $record -Connection $connections[0] -Process $listener `
        -WrapperProcess $wrapper
}

function ConvertFrom-GradioConfigJson {
    param([Parameter(Mandatory = $true)][string]$Json)
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        return ($Json | ConvertFrom-Json -AsHashtable)
    }
    Add-Type -AssemblyName System.Web.Extensions
    $serializer = New-Object System.Web.Script.Serialization.JavaScriptSerializer
    return $serializer.DeserializeObject($Json)
}

function Get-GradioPropertyValue {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    if ($Object -is [Collections.IDictionary]) { return $Object[$Name] }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Test-GradioPropertyExists {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $false }
    if ($Object -is [Collections.IDictionary]) {
        return (@($Object.Keys) -ccontains $Name)
    }
    return ($null -ne $Object.PSObject.Properties[$Name])
}

function Test-GradioUploadSource {
    param($Sources)
    foreach ($source in @($Sources)) {
        if (([string]$source).Trim() -ceq 'upload') { return $true }
    }
    return $false
}

function Get-ExactGradioComponentId {
    param(
        [Parameter(Mandatory = $true)][hashtable]$ComponentMap,
        [Parameter(Mandatory = $true)][string]$Type,
        [Parameter(Mandatory = $true)][string]$Property,
        [Parameter(Mandatory = $true)][string]$Value
    )
    $matches = @()
    foreach ($component in $ComponentMap.Values) {
        if ([string](Get-GradioPropertyValue -Object $component -Name 'type') -cne $Type) {
            continue
        }
        $props = Get-GradioPropertyValue -Object $component -Name 'props'
        $actual = Get-GradioPropertyValue -Object $props -Name $Property
        if (
            ($Property -ceq 'elem_classes' -and @($actual) -contains $Value) -or
            ($Property -cne 'elem_classes' -and [string]$actual -ceq $Value)
        ) {
            $matches += [int](Get-GradioPropertyValue -Object $component -Name 'id')
        }
    }
    if ($matches.Count -ne 1) { throw 'Gradio component identity mismatch.' }
    return $matches[0]
}

function Assert-GradioConfigEndpoint {
    param([switch]$PublicTunnelMode)
    $response = Invoke-WebRequest `
        -Uri "$($script:LocalUrl)/config" `
        -Method Get `
        -TimeoutSec 10 `
        -UseBasicParsing
    if ($response.StatusCode -ne 200) {
        throw 'Gradio config status mismatch.'
    }
    try {
        $config = ConvertFrom-GradioConfigJson -Json ([string]$response.Content)
    }
    catch {
        throw 'Gradio config is not valid JSON.'
    }
    $versionText = [string](Get-GradioPropertyValue -Object $config -Name 'version')
    $version = $null
    $versionValid = [Version]::TryParse($versionText, [ref]$version)
    $apiOpenExists = Test-GradioPropertyExists -Object $config -Name 'api_open'
    $apiOpen = Get-GradioPropertyValue -Object $config -Name 'api_open'
    if (
        [string](Get-GradioPropertyValue -Object $config -Name 'title') -cne 'Audited Dermoscopy Workbench' -or
        -not $versionValid -or
        $version.Major -notin @(5, 6) -or
        ($version.Major -eq 5 -and -not $apiOpenExists) -or
        ($apiOpenExists -and ($apiOpen -isnot [bool] -or $apiOpen))
    ) {
        throw 'Gradio config identity mismatch.'
    }
    $componentMap = @{}
    $publicNotice = $false
    foreach ($component in @($config.components)) {
        $id = Get-GradioPropertyValue -Object $component -Name 'id'
        $type = [string](Get-GradioPropertyValue -Object $component -Name 'type')
        $props = Get-GradioPropertyValue -Object $component -Name 'props'
        if (($id -isnot [int] -and $id -isnot [long]) -or -not $type -or $componentMap.ContainsKey([int]$id)) {
            throw 'Gradio component schema mismatch.'
        }
        $componentMap[[int]$id] = $component
        if ([string](Get-GradioPropertyValue -Object $props -Name 'info') -ceq 'Public tunnel: bundled public/synthetic inputs only') {
            $publicNotice = $true
        }
        if ($type -ceq 'file' -or (
            $type -ceq 'image' -and
            (Test-GradioUploadSource -Sources (Get-GradioPropertyValue -Object $props -Name 'sources'))
        )) {
            throw 'Gradio config exposes an upload-capable component.'
        }
    }
    if (-not $publicNotice) { throw 'Gradio config public-mode notice is missing.' }

    $dependencies = @($config.dependencies)
    $dependencyMap = @{}
    $dual = @()
    $thenDependencies = @()
    $runDualButtonId = Get-ExactGradioComponentId `
        -ComponentMap $componentMap -Type 'button' -Property 'elem_id' -Value 'run-dual'
    $runDualDependencies = @()
    foreach ($dependency in $dependencies) {
        $dependencyId = Get-GradioPropertyValue -Object $dependency -Name 'id'
        if (
            ($dependencyId -isnot [int] -and $dependencyId -isnot [long]) -or
            $dependencyMap.ContainsKey([int]$dependencyId)
        ) {
            throw 'Gradio dependency ID mismatch.'
        }
        $dependencyMap[[int]$dependencyId] = $dependency
        $apiName = [string](Get-GradioPropertyValue -Object $dependency -Name 'api_name')
        if ($apiName -match '(?i)upload') { throw 'Gradio config exposes an upload API.' }
        if ($apiName -ceq 'dual_live_compare') { $dual += $dependency }
        $targets = @(Get-GradioPropertyValue -Object $dependency -Name 'targets')
        foreach ($target in $targets) {
            if (@($target).Count -ne 2) { throw 'Gradio dependency target mismatch.' }
            if ([string]$target[1] -match '(?i)upload') { throw 'Gradio config exposes an upload event.' }
            if ([string]$target[1] -ceq 'then') { $thenDependencies += $dependency }
            if ($null -ne $target[0] -and -not $componentMap.ContainsKey([int]$target[0])) {
                throw 'Gradio dependency target references an unknown component.'
            }
        }
        if (
            $targets.Count -eq 1 -and
            @($targets[0]).Count -eq 2 -and
            ($targets[0][0] -is [int] -or $targets[0][0] -is [long]) -and
            [int]$targets[0][0] -eq $runDualButtonId -and
            [string]$targets[0][1] -ceq 'click'
        ) {
            $runDualDependencies += $dependency
        }
        foreach ($inputId in @(Get-GradioPropertyValue -Object $dependency -Name 'inputs')) {
            if (($inputId -isnot [int] -and $inputId -isnot [long]) -or -not $componentMap.ContainsKey([int]$inputId)) {
                throw 'Gradio dependency input mismatch.'
            }
            $inputType = [string](Get-GradioPropertyValue -Object $componentMap[[int]$inputId] -Name 'type')
            if ($inputType -ceq 'file' -or $inputType -ceq 'image') {
                throw 'Gradio dependency accepts an upload-capable input.'
            }
        }
        foreach ($outputId in @(Get-GradioPropertyValue -Object $dependency -Name 'outputs')) {
            if (($outputId -isnot [int] -and $outputId -isnot [long]) -or -not $componentMap.ContainsKey([int]$outputId)) {
                throw 'Gradio dependency output mismatch.'
            }
        }
    }
    if ($dual.Count -ne 1) { throw 'Gradio dual-live API graph mismatch.' }
    $dualDependency = $dual[0]
    $dualInputs = @(Get-GradioPropertyValue -Object $dualDependency -Name 'inputs')
    $dualOutputs = @(Get-GradioPropertyValue -Object $dualDependency -Name 'outputs')
    $dualTargets = @(Get-GradioPropertyValue -Object $dualDependency -Name 'targets')
    $expectedDualInputs = @(
        $(Get-ExactGradioComponentId -ComponentMap $componentMap -Type 'number' -Property 'elem_id' -Value 'dual-generation'),
        $(Get-ExactGradioComponentId -ComponentMap $componentMap -Type 'dropdown' -Property 'label' -Value 'Bundled public / synthetic sample')
    )
    $expectedDualOutputs = @(
        $(Get-ExactGradioComponentId -ComponentMap $componentMap -Type 'html' -Property 'elem_id' -Value 'dual-live-state'),
        $(Get-ExactGradioComponentId -ComponentMap $componentMap -Type 'image' -Property 'label' -Value 'Original RGB'),
        $(Get-ExactGradioComponentId -ComponentMap $componentMap -Type 'image' -Property 'label' -Value 'IMP overlay'),
        $(Get-ExactGradioComponentId -ComponentMap $componentMap -Type 'image' -Property 'label' -Value 'IMP mask'),
        $(Get-ExactGradioComponentId -ComponentMap $componentMap -Type 'image' -Property 'label' -Value 'nnU-Net overlay'),
        $(Get-ExactGradioComponentId -ComponentMap $componentMap -Type 'image' -Property 'label' -Value 'nnU-Net mask'),
        $(Get-ExactGradioComponentId -ComponentMap $componentMap -Type 'markdown' -Property 'elem_classes' -Value 'dual-ledger'),
        $(Get-ExactGradioComponentId -ComponentMap $componentMap -Type 'downloadbutton' -Property 'elem_id' -Value 'dual-receipt')
    )
    if ($runDualDependencies.Count -ne 1) { throw 'Gradio dual-live API graph mismatch.' }
    $runDualDependency = $runDualDependencies[0]
    $runDualDependencyId = [int](Get-GradioPropertyValue -Object $runDualDependency -Name 'id')
    $runDualInputs = @(Get-GradioPropertyValue -Object $runDualDependency -Name 'inputs')
    $runDualOutputs = @(Get-GradioPropertyValue -Object $runDualDependency -Name 'outputs')
    $expectedRunDualOutputs = @($expectedDualInputs[0]) + $expectedDualOutputs
    if (
        (Get-GradioPropertyValue -Object $runDualDependency -Name 'backend_fn') -isnot [bool] -or
        -not (Get-GradioPropertyValue -Object $runDualDependency -Name 'backend_fn') -or
        (Get-GradioPropertyValue -Object $runDualDependency -Name 'queue') -isnot [bool] -or
        (Get-GradioPropertyValue -Object $runDualDependency -Name 'queue') -or
        [string](Get-GradioPropertyValue -Object $runDualDependency -Name 'api_name') -cne 'false' -or
        [string](Get-GradioPropertyValue -Object $runDualDependency -Name 'api_visibility') -cne 'private' -or
        $runDualInputs.Count -ne 0 -or
        ($runDualOutputs -join ',') -cne ($expectedRunDualOutputs -join ',') -or
        (Get-GradioPropertyValue -Object $dualDependency -Name 'api_visibility') -cne 'public' -or
        (Get-GradioPropertyValue -Object $dualDependency -Name 'backend_fn') -isnot [bool] -or
        -not (Get-GradioPropertyValue -Object $dualDependency -Name 'backend_fn') -or
        (Get-GradioPropertyValue -Object $dualDependency -Name 'queue') -isnot [bool] -or
        -not (Get-GradioPropertyValue -Object $dualDependency -Name 'queue') -or
        ($dualInputs -join ',') -cne ($expectedDualInputs -join ',') -or
        ($dualOutputs -join ',') -cne ($expectedDualOutputs -join ',') -or
        $thenDependencies.Count -ne 1 -or
        [int](Get-GradioPropertyValue -Object $thenDependencies[0] -Name 'id') -ne [int](Get-GradioPropertyValue -Object $dualDependency -Name 'id') -or
        $dualTargets.Count -ne 1 -or
        @($dualTargets[0]).Count -ne 2 -or
        $null -ne $dualTargets[0][0] -or
        [string]$dualTargets[0][1] -cne 'then' -or
        ((Get-GradioPropertyValue -Object $dualDependency -Name 'trigger_after') -isnot [int] -and
            (Get-GradioPropertyValue -Object $dualDependency -Name 'trigger_after') -isnot [long]) -or
        [int](Get-GradioPropertyValue -Object $dualDependency -Name 'trigger_after') -ne $runDualDependencyId
    ) {
        throw 'Gradio dual-live API graph mismatch.'
    }
    $generation = $componentMap[[int]$dualInputs[0]]
    $selector = $componentMap[[int]$dualInputs[1]]
    $selectorProps = Get-GradioPropertyValue -Object $selector -Name 'props'
    if (
        [string](Get-GradioPropertyValue -Object $generation -Name 'type') -cne 'number' -or
        [string](Get-GradioPropertyValue -Object $selector -Name 'type') -cne 'dropdown' -or
        [string](Get-GradioPropertyValue -Object $selectorProps -Name 'label') -cne 'Bundled public / synthetic sample' -or
        (Get-GradioPropertyValue -Object $selectorProps -Name 'allow_custom_value') -isnot [bool] -or
        (Get-GradioPropertyValue -Object $selectorProps -Name 'allow_custom_value')
    ) {
        throw 'Gradio dual-live input surface mismatch.'
    }
}

function Invoke-TunnelPreflight {
    param([string]$Root = '', [switch]$PreserveMode, [string]$RunId = '')
    if (-not $PreserveMode) { throw 'Public tunnel requires preserve mode.' }
    $resolvedRoot = Resolve-TunnelRoot -Root $Root
    $releaseManifestSha256 = Assert-ReleaseManifest -Root $resolvedRoot
    Assert-OwnedGradioRuntime `
        -Root $resolvedRoot `
        -ExpectedReleaseManifestSha256 $releaseManifestSha256 `
        -PreserveMode:$PreserveMode -RunId $RunId
    Assert-GradioConfigEndpoint -PublicTunnelMode
    return [pscustomobject]@{
        Root = $resolvedRoot
        ReleaseManifestSha256 = $releaseManifestSha256
    }
}

function Resolve-CloudflaredApplication {
    param([string]$ExplicitPath = '')
    if ($ExplicitPath) {
        $resolved = (Resolve-Path -LiteralPath $ExplicitPath -ErrorAction Stop).Path
        [void](Assert-SafeRegularFile -Path $resolved -Label 'cloudflared executable')
    }
    else {
        $command = Get-Command cloudflared -CommandType Application -ErrorAction Stop
        $resolved = Assert-SafeRegularFile -Path $command.Source -Label 'cloudflared executable'
    }
    if ([IO.Path]::GetFileName($resolved) -notmatch '^cloudflared(?:\.exe)?$') {
        throw 'resolved executable has an unexpected name'
    }
    & $resolved --version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw 'cloudflared version probe failed'
    }
    return $resolved
}

function Get-TunnelOwnerRecordPath {
    param([string]$Root, [switch]$PreserveMode, [string]$RunId = '')
    $runtimeRoot = Join-Path $Root 'demo_runtime'
    $ownerRoot = if ($PreserveMode) {
        Get-PreserveComponentDirectory -Root $Root -RunId $RunId -Component 'tunnel'
    }
    else {
        Join-Path $runtimeRoot 'launcher'
    }
    foreach ($path in @($runtimeRoot, $ownerRoot)) {
        $item = Get-Item -LiteralPath $path -Force -ErrorAction Stop
        if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Tunnel owner path is unsafe.'
        }
    }
    if ($PreserveMode) {
        return (Join-Path $ownerRoot ("owner-" + (New-CryptographicOwnerNonce) + '.json'))
    }
    return (Join-Path $ownerRoot 'tunnel.json')
}

function Write-TunnelOwnerRecord {
    param(
        [string]$RecordPath,
        [int]$ProcessId,
        [string]$ProcessStartTimeUtc,
        [string]$OwnerNonce,
        [string]$ExecutablePath,
        [Parameter(Mandatory = $true)][string]$ReleaseManifestSha256
    )
    if ($ReleaseManifestSha256 -cnotmatch '^[0-9a-f]{64}$') {
        throw 'Tunnel release manifest identity mismatch.'
    }
    if (Test-Path -LiteralPath $RecordPath) {
        throw 'A tunnel owner record already exists.'
    }
    $record = [ordered]@{
        schema_version = 'imp.demo.tunnel-owner.v1'
        release_manifest_sha256 = $ReleaseManifestSha256
        process_id = $ProcessId
        process_start_time_utc = $ProcessStartTimeUtc
        owner_nonce = $OwnerNonce
        executable_path = [IO.Path]::GetFullPath($ExecutablePath)
        local_url = $script:LocalUrl
    }
    $json = $record | ConvertTo-Json -Compress
    Write-ExclusiveAsciiFile -Path $RecordPath -Content $json
}

function Remove-TunnelOwnerRecord {
    param([string]$RecordPath)
    if (Test-Path -LiteralPath $RecordPath -PathType Leaf) {
        $item = Get-Item -LiteralPath $RecordPath -Force
        if (
            $item.Name -cne 'tunnel.json' -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            $item.Directory.Name -cne 'launcher'
        ) {
            throw 'Tunnel owner record path is unsafe.'
        }
        Remove-Item -LiteralPath $item.FullName -Force
    }
}

function Stop-ProcessAndWait {
    param([int]$ProcessId, [string]$Label)
    Stop-Process -Id $ProcessId -Force
    Wait-Process -Id $ProcessId -Timeout 10 -ErrorAction SilentlyContinue
    if ($null -ne (Get-ExactProcess -ProcessId $ProcessId)) {
        throw "$Label did not stop within the timeout."
    }
}

function Stop-SpawnedProcessHandle {
    param([Parameter(Mandatory = $true)]$Process)
    if (-not $Process.HasExited) {
        $Process.Kill()
    }
    $Process.WaitForExit()
    if (-not $Process.HasExited) {
        throw 'cloudflared process handle did not reach an exited state.'
    }
}

function Invoke-TunnelLaunch {
    [CmdletBinding()]
    param(
        [string]$CloudflaredPath = '',
        [string]$Root = '',
        [switch]$CheckOnly,
        [switch]$PreserveMode,
        [string]$RunId = ''
    )
    try {
        if ($PreserveMode) {
            $RunId = Assert-PreserveRunId -RunId $RunId
        }
        $preflight = Invoke-TunnelPreflight `
            -Root $Root -PreserveMode:$PreserveMode -RunId $RunId
        $resolvedRoot = [string]$preflight.Root
        $releaseManifestSha256 = [string]$preflight.ReleaseManifestSha256
        if (
            -not $resolvedRoot -or
            $releaseManifestSha256 -cnotmatch '^[0-9a-f]{64}$'
        ) {
            throw 'Tunnel preflight context identity mismatch.'
        }
    }
    catch {
        [Console]::Error.WriteLine(
            "Local demo identity gate failed: $($_.Exception.Message)"
        )
        return 3
    }
    if ($CheckOnly) {
        [Console]::Out.WriteLine(
            'Owned Gradio identity and config passed; tunnel was not started.'
        )
        return 0
    }
    try {
        $resolved = Resolve-CloudflaredApplication -ExplicitPath $CloudflaredPath
    }
    catch {
        [Console]::Error.WriteLine('A valid cloudflared application was not found.')
        return 4
    }

    $recordPath = $null
    $recordWritten = $false
    $ownedProcessAbsent = $false
    $tunnelRecord = $null
    $spawnedProcessHandle = $null
    try {
        $recordPath = Get-TunnelOwnerRecordPath `
            -Root $resolvedRoot -PreserveMode:$PreserveMode -RunId $RunId
        if (-not $PreserveMode -and (Test-Path -LiteralPath $recordPath)) {
            throw 'A tunnel owner record already exists.'
        }
        $tunnelArguments = "tunnel --url $($script:LocalUrl)"
        $process = Start-Process `
            -FilePath $resolved `
            -ArgumentList $tunnelArguments `
            -NoNewWindow `
            -PassThru
        $spawnedProcessHandle = $process
        $processInfo = Get-ExactProcess -ProcessId ([int]$process.Id)
        $tunnelRecord = [pscustomobject]@{
            schema_version = 'imp.demo.tunnel-owner.v1'
            release_manifest_sha256 = $releaseManifestSha256
            process_id = [int]$process.Id
            process_start_time_utc = Convert-ProcessStartTimeUtc -Value $processInfo.CreationDate
            owner_nonce = New-CryptographicOwnerNonce
            executable_path = [IO.Path]::GetFullPath($resolved)
            local_url = $script:LocalUrl
        }
        Assert-TunnelProcessIdentity -Record $tunnelRecord -Process $processInfo
        Write-TunnelOwnerRecord `
            -RecordPath $recordPath `
            -ProcessId $process.Id `
            -ProcessStartTimeUtc $tunnelRecord.process_start_time_utc `
            -OwnerNonce $tunnelRecord.owner_nonce `
            -ExecutablePath $resolved `
            -ReleaseManifestSha256 $releaseManifestSha256
        $recordWritten = $true
        [Console]::Out.WriteLine(
            'Temporary public tunnel active. Press Ctrl+C to stop it.'
        )
        $process.WaitForExit()
        $ownedProcessAbsent = $true
        return $process.ExitCode
    }
    catch {
        [Console]::Error.WriteLine("Tunnel launch failed: $($_.Exception.Message)")
        return 5
    }
    finally {
        if ($null -ne $spawnedProcessHandle -and -not $ownedProcessAbsent) {
            try {
                Stop-SpawnedProcessHandle -Process $spawnedProcessHandle
                $ownedProcessAbsent = $true
            }
            catch {
                [Console]::Error.WriteLine(
                    "Tunnel cleanup failed; owner record was preserved: $($_.Exception.Message)"
                )
            }
        }
        if (-not $PreserveMode) {
            if ($recordWritten -and $ownedProcessAbsent) {
                Remove-TunnelOwnerRecord -RecordPath $recordPath
            }
        }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    exit (Invoke-TunnelLaunch `
        -CloudflaredPath $CloudflaredPath -Root $Root -CheckOnly:$CheckOnly `
        -PreserveMode:$PreserveMode -RunId $RunId)
}
