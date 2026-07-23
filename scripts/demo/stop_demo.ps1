[CmdletBinding()]
param([string]$Root = '', [switch]$PreserveMode, [string]$RunId = '')

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$script:StopRoot = $Root

function Resolve-DemoStopRoot {
    if (-not $script:StopRoot) {
        return (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    }
    return (Resolve-Path -LiteralPath $script:StopRoot -ErrorAction Stop).Path
}

function Assert-ReleaseManifest {
    param([Parameter(Mandatory = $true)][string]$Root)
    $path = Join-Path $Root 'release\imp_release_manifest.json'
    $bytes = [IO.File]::ReadAllBytes($path)
    $payload = [Text.Encoding]::ASCII.GetString($bytes) | ConvertFrom-Json
    if ($payload.schema_version -cne 'imp.release.manifest.v1') {
        throw 'release manifest schema mismatch'
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

function Assert-PreserveRunId {
    param([string]$RunId)
    if ($RunId -cnotmatch '^[a-z0-9][a-z0-9_-]{0,127}$') {
        throw 'Preserve run ID is unsafe.'
    }
    return $RunId
}

function Get-PreserveActiveOwnerRecordPath {
    param([string]$Root, [string]$RunId, [string]$Component)
    $safeRunId = Assert-PreserveRunId -RunId $RunId
    $preservedRoot = [IO.Path]::GetFullPath((Join-Path $Root 'demo_runtime\preserved'))
    $runRoot = [IO.Path]::GetFullPath((Join-Path $preservedRoot $safeRunId))
    $componentRoot = [IO.Path]::GetFullPath((Join-Path $runRoot $Component))
    $prefix = $preservedRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $componentRoot.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Preserve owner path escapes the runtime root.'
    }
    if (-not (Test-Path -LiteralPath $componentRoot -PathType Container)) {
        return $null
    }
    $owners = @(Get-ChildItem -LiteralPath $componentRoot -File -Filter 'owner-*.json' |
        Sort-Object LastWriteTimeUtc, Name)
    if ($owners.Count -eq 0) { return $null }
    $stoppedOwners = New-Object 'System.Collections.Generic.HashSet[string]' (
        [StringComparer]::OrdinalIgnoreCase
    )
    foreach ($stoppedFile in @(Get-ChildItem -LiteralPath $componentRoot -File -Filter '*.stopped.json')) {
        $stopped = Get-Content -LiteralPath $stoppedFile.FullName -Raw | ConvertFrom-Json
        if ([string]$stopped.owner_record -notmatch '^owner-[a-z0-9_-]+\.json$') {
            throw 'Preserve stopped record identity is unsafe.'
        }
        [void]$stoppedOwners.Add([string]$stopped.owner_record)
    }
    $newestOwner = $owners[-1]
    if ($stoppedOwners.Contains($newestOwner.Name)) {
        return $null
    }
    return $newestOwner.FullName
}

function Write-PreserveComponentStopRecord {
    param(
        [string]$Root,
        [string]$RunId,
        [string]$Component,
        [string]$OwnerRecordPath,
        [string]$ReleaseManifestSha256 = ''
    )
    $ownerName = [IO.Path]::GetFileName($OwnerRecordPath)
    if ($ownerName -cnotmatch '^owner-[a-z0-9_-]+\.json$') {
        throw 'Preserve owner record identity is unsafe.'
    }
    $directory = Join-Path $Root "demo_runtime\preserved\$RunId\$Component"
    $record = [ordered]@{
        event = 'stopped'
        owner_record = $ownerName
        recorded_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    if ($ReleaseManifestSha256) {
        if ($ReleaseManifestSha256 -cnotmatch '^[0-9a-f]{64}$') {
            throw 'Preserve release manifest identity is invalid.'
        }
        $record['release_manifest_sha256'] = $ReleaseManifestSha256
    }
    $bytes = [Text.Encoding]::ASCII.GetBytes(($record | ConvertTo-Json -Compress))
    for ($attempt = 0; $attempt -lt 16; $attempt++) {
        $path = Join-Path $directory (
            "stop-" + [guid]::NewGuid().ToString('N') + '.stopped.json'
        )
        try {
            $stream = [IO.File]::Open(
                $path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None
            )
            try { $stream.Write($bytes, 0, $bytes.Length) }
            finally { $stream.Dispose() }
            return $path
        }
        catch [IO.IOException] {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw }
        }
    }
    throw 'Unable to allocate a unique component stop record.'
}

function Assert-OwnedSessionPath {
    param([string]$SessionPath, [string]$RuntimeRoot)
    $runtime = Get-Item -LiteralPath $RuntimeRoot -Force -ErrorAction Stop
    $session = Get-Item -LiteralPath $SessionPath -Force -ErrorAction Stop
    if (-not $runtime.PSIsContainer -or -not $session.PSIsContainer) {
        throw 'Owned runtime and session must be directories.'
    }
    $resolvedRuntime = [IO.Path]::GetFullPath($runtime.FullName).TrimEnd('\', '/')
    $resolvedSession = [IO.Path]::GetFullPath($session.FullName).TrimEnd('\', '/')
    $expectedParent = [IO.Path]::GetFullPath(
        (Join-Path $resolvedRuntime 'sessions')
    ).TrimEnd('\', '/')
    if (
        -not $resolvedSession.StartsWith(
            $resolvedRuntime + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        -not [string]::Equals(
            [IO.Path]::GetFullPath($session.Parent.FullName).TrimEnd('\', '/'),
            $expectedParent,
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        $session.Name -cnotmatch '^demo-[0-9a-f]{32}$'
    ) {
        throw 'Session is outside launcher ownership.'
    }
    foreach ($item in @($runtime, (Get-Item -LiteralPath $expectedParent -Force), $session)) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Owned session path contains a reparse point.'
        }
    }
    return $resolvedSession
}

function Read-OwnedRecord {
    param([string]$Path, [string]$ExpectedParent, [string]$ExpectedName)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    $item = Get-Item -LiteralPath $Path -Force
    $parent = Get-Item -LiteralPath $ExpectedParent -Force
    if (
        $item.Name -cne $ExpectedName -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        ($parent.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        -not [string]::Equals(
            [IO.Path]::GetFullPath($item.Directory.FullName).TrimEnd('\', '/'),
            [IO.Path]::GetFullPath($parent.FullName).TrimEnd('\', '/'),
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw 'Launcher owner record path is unsafe.'
    }
    return (Get-Content -LiteralPath $item.FullName -Raw | ConvertFrom-Json)
}

function Remove-OwnedRecord {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $item = Get-Item -LiteralPath $Path -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Launcher owner record is a reparse point.'
        }
        Remove-Item -LiteralPath $item.FullName -Force
    }
}

function Get-ExactProcess {
    param([int]$ProcessId)
    $values = @(Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId")
    if ($values.Count -gt 1) {
        throw 'Process identity is ambiguous.'
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
        $Record.local_url -cne 'http://127.0.0.1:7860' -or
        [string]$Process.CommandLine -notmatch
            'tunnel\s+--url\s+http://127\.0\.0\.1:7860(?:\s|$)'
    ) {
        throw 'Tunnel process incarnation mismatch.'
    }
}

function Assert-GradioOwnerRecord {
    param([Parameter(Mandatory = $true)]$Record)
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
        'python_path',
        'public_tunnel_mode',
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
        [string]$Record.release_manifest_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
        $Record.public_tunnel_mode -isnot [bool] -or
        $Record.preserve_mode -isnot [bool] -or
        -not [IO.Path]::IsPathRooted([string]$Record.python_path) -or
        -not [IO.Path]::IsPathRooted([string]$Record.session_path) -or
        $Record.host -cne '127.0.0.1' -or
        $port -ne 7860
    ) {
        throw 'Gradio owner record identity mismatch.'
    }
}

function Assert-LauncherProcessIdentity {
    param(
        [Parameter(Mandatory = $true)]$Record,
        [Parameter(Mandatory = $true)]$Process,
        [Parameter(Mandatory = $true)][string]$ExpectedScript
    )
    if (
        [int]$Process.ProcessId -ne [int]$Record.launcher_pid -or
        [string]$Record.owner_nonce -cnotmatch '^[0-9a-f]{32}$' -or
        -not (Test-ProcessStartTimeMatch `
            -Expected $Record.launcher_start_time_utc -Actual $Process.CreationDate)
    ) {
        throw 'Gradio launcher process incarnation mismatch.'
    }
    $executableName = [IO.Path]::GetFileName([string]$Process.ExecutablePath)
    if ($executableName -notmatch '^(?:powershell|pwsh)\.exe$') {
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
    param(
        [Parameter(Mandatory = $true)]$Record,
        [Parameter(Mandatory = $true)]$Process
    )
    $expectedPath = [IO.Path]::GetFullPath([string]$Record.python_path)
    if (
        [int]$Process.ParentProcessId -ne [int]$Record.launcher_pid -or
        -not [string]::Equals(
            [IO.Path]::GetFullPath([string]$Process.ExecutablePath),
            $expectedPath,
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        [string]$Process.CommandLine -notmatch '(?:^|\s)-m\s+lesion_robustness\.demo\.app(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--host\s+127\.0\.0\.1(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--port\s+7860(?:\s|$)'
    ) {
        throw 'Gradio Python wrapper ownership proof failed.'
    }
}

function Assert-GradioListenerProcessIdentity {
    param(
        [Parameter(Mandatory = $true)]$Record,
        [Parameter(Mandatory = $true)]$Connection,
        [Parameter(Mandatory = $true)]$Process,
        [Parameter(Mandatory = $true)]$WrapperProcess
    )
    if (
        [string]$Connection.LocalAddress -cne '127.0.0.1' -or
        [int]$Connection.LocalPort -ne 7860 -or
        [int]$Connection.OwningProcess -ne [int]$Process.ProcessId -or
        [int]$Process.ParentProcessId -ne [int]$WrapperProcess.ProcessId -or
        [string]$Record.owner_nonce -cnotmatch '^[0-9a-f]{32}$' -or
        [IO.Path]::GetFileName([string]$Process.ExecutablePath) -cnotmatch '^python(?:\.exe)?$' -or
        [string]$Process.CommandLine -notmatch '(?:^|\s)-m\s+lesion_robustness\.demo\.app(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--host\s+127\.0\.0\.1(?:\s|$)' -or
        [string]$Process.CommandLine -notmatch '--port\s+7860(?:\s|$)'
    ) {
        throw 'Gradio listener process ownership proof failed.'
    }
}

function Get-OwnedGradioDescendantProcesses {
    param([Parameter(Mandatory = $true)][int]$LauncherProcessId)
    $allProcesses = @(Get-CimInstance Win32_Process)
    $seen = New-Object 'System.Collections.Generic.HashSet[int]'
    [void]$seen.Add($LauncherProcessId)
    $frontier = New-Object 'System.Collections.Generic.List[int]'
    $frontier.Add($LauncherProcessId)
    $descendants = New-Object 'System.Collections.Generic.List[object]'

    while ($frontier.Count -gt 0) {
        $next = New-Object 'System.Collections.Generic.List[int]'
        foreach ($process in $allProcesses) {
            $processId = [int]$process.ProcessId
            if (
                $frontier.Contains([int]$process.ParentProcessId) -and
                $seen.Add($processId)
            ) {
                $descendants.Add($process)
                $next.Add($processId)
            }
        }
        $frontier = $next
    }
    return @($descendants.ToArray())
}

function Stop-ProcessAndWait {
    param([int]$ProcessId, [string]$Label)
    try {
        Stop-Process -Id $ProcessId -Force
    }
    catch {
        if (
            $_.CategoryInfo.Category -cne 'ObjectNotFound' -or
            $_.FullyQualifiedErrorId -cne
                'NoProcessFoundForGivenId,Microsoft.PowerShell.Commands.StopProcessCommand'
        ) {
            throw
        }
        if ($null -ne (Get-ExactProcess -ProcessId $ProcessId)) {
            throw
        }
        return
    }
    # CUDA worker teardown can outlast the normal process-stop window.
    Wait-Process -Id $ProcessId -Timeout 30 -ErrorAction SilentlyContinue
    if ($null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
        throw "$Label did not stop within the timeout."
    }
}

function Remove-OwnedGradioState {
    param($Record, [string]$RuntimeRoot, [string]$RecordPath)
    if (Test-Path -LiteralPath ([string]$Record.session_path) -PathType Container) {
        $session = Assert-OwnedSessionPath `
            -SessionPath ([string]$Record.session_path) `
            -RuntimeRoot $RuntimeRoot
        Remove-Item -LiteralPath $session -Recurse -Force
    }
    Remove-OwnedRecord -Path $RecordPath
}

function Stop-OwnedCloudflared {
    param([switch]$PreserveMode, [string]$RunId = '')
    $root = Resolve-DemoStopRoot
    $releaseManifestSha256 = Assert-ReleaseManifest -Root $root
    $ownerRoot = if ($PreserveMode) {
        Join-Path $root "demo_runtime\preserved\$RunId\tunnel"
    } else { Join-Path $root 'demo_runtime\launcher' }
    $recordPath = if ($PreserveMode) {
        Get-PreserveActiveOwnerRecordPath -Root $root -RunId $RunId -Component 'tunnel'
    } else { Join-Path $ownerRoot 'tunnel.json' }
    if ($null -eq $recordPath) { return }
    $record = Read-OwnedRecord `
        -Path $recordPath -ExpectedParent $ownerRoot `
        -ExpectedName ([IO.Path]::GetFileName($recordPath))
    if ($null -eq $record) {
        return
    }
    try {
        $processId = Convert-OwnedProcessId -Value $record.process_id
        $recordStartTime = Convert-ProcessStartTimeUtc -Value $record.process_start_time_utc
    }
    catch {
        throw 'Tunnel owner record identity mismatch.'
    }
    $fields = @($record.PSObject.Properties.Name | Sort-Object)
    if (
        ($fields -join ',') -cne 'executable_path,local_url,owner_nonce,process_id,process_start_time_utc,release_manifest_sha256,schema_version' -or
        $record.schema_version -cne 'imp.demo.tunnel-owner.v1' -or
        $record.release_manifest_sha256 -cne $releaseManifestSha256 -or
        $record.local_url -cne 'http://127.0.0.1:7860' -or
        $processId -lt 1 -or
        $recordStartTime -cnotmatch '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{7}Z$' -or
        [string]$record.owner_nonce -cnotmatch '^[0-9a-f]{32}$'
    ) {
        throw 'Tunnel owner record identity mismatch.'
    }
    $process = Get-ExactProcess -ProcessId $processId
    if ($null -ne $process) {
        Assert-TunnelProcessIdentity -Record $record -Process $process
        Stop-ProcessAndWait -ProcessId $processId -Label 'cloudflared tunnel'
    }
    if ($PreserveMode) {
        [void](Write-PreserveComponentStopRecord `
            -Root $root -RunId $RunId -Component 'tunnel' `
            -OwnerRecordPath $recordPath `
            -ReleaseManifestSha256 $releaseManifestSha256)
    }
    if (-not $PreserveMode) {
        Remove-OwnedRecord -Path $recordPath
    }
}

function Stop-OwnedGradio {
    param([switch]$PreserveMode, [string]$RunId = '')
    $root = Resolve-DemoStopRoot
    $runtimeRoot = Join-Path $root 'demo_runtime'
    $ownerRoot = if ($PreserveMode) {
        Join-Path $runtimeRoot "preserved\$RunId\gradio"
    } else { Join-Path $runtimeRoot 'launcher' }
    $recordPath = if ($PreserveMode) {
        Get-PreserveActiveOwnerRecordPath -Root $root -RunId $RunId -Component 'gradio'
    } else { Join-Path $ownerRoot 'gradio.json' }
    if ($null -eq $recordPath) { return }
    $record = Read-OwnedRecord `
        -Path $recordPath -ExpectedParent $ownerRoot `
        -ExpectedName ([IO.Path]::GetFileName($recordPath))
    if ($null -eq $record) {
        return
    }
    Assert-GradioOwnerRecord -Record $record
    $launcher = Get-ExactProcess -ProcessId ([int]$record.launcher_pid)
    if ($null -eq $launcher) {
        throw 'Gradio launcher process is unavailable; ownership state was preserved.'
    }
    Assert-LauncherProcessIdentity `
        -Record $record `
        -Process $launcher `
        -ExpectedScript (Join-Path $root 'scripts\demo\run_demo.ps1')

    $connections = @(Get-NetTCPConnection -State Listen -LocalPort 7860 -ErrorAction SilentlyContinue)
    if ($connections.Count -gt 1) {
        throw 'Gradio listener ownership is ambiguous.'
    }
    $listener = $null
    $wrapper = $null
    if ($connections.Count -eq 1) {
        $listener = Get-ExactProcess -ProcessId ([int]$connections[0].OwningProcess)
        if ($null -eq $listener) {
            throw 'Gradio listener process is unavailable.'
        }
        $wrapper = Get-ExactProcess -ProcessId ([int]$listener.ParentProcessId)
        if ($null -eq $wrapper) {
            throw 'Gradio Python wrapper process is unavailable.'
        }
        Assert-GradioPythonWrapperIdentity -Record $record -Process $wrapper
        Assert-GradioListenerProcessIdentity `
            -Record $record -Connection $connections[0] -Process $listener `
            -WrapperProcess $wrapper
    }

    $stopErrors = New-Object 'System.Collections.Generic.List[string]'
    $descendants = @(Get-OwnedGradioDescendantProcesses `
        -LauncherProcessId ([int]$launcher.ProcessId))
    if (
        $null -ne $listener -and (
            $descendants.ProcessId -notcontains [int]$listener.ProcessId -or
            $descendants.ProcessId -notcontains [int]$wrapper.ProcessId
        )
    ) {
        throw 'Gradio process lineage changed during ownership verification.'
    }
    for ($index = $descendants.Count - 1; $index -ge 0; $index--) {
        $expected = $descendants[$index]
        $process = Get-ExactProcess -ProcessId ([int]$expected.ProcessId)
        if ($null -eq $process) {
            continue
        }
        if ([int]$process.ParentProcessId -ne [int]$expected.ParentProcessId) {
            throw 'Gradio descendant process lineage changed during shutdown.'
        }
        try {
            Stop-ProcessAndWait -ProcessId ([int]$process.ProcessId) -Label 'Gradio descendant'
        }
        catch {
            $stopErrors.Add($_.Exception.Message)
        }
    }
    $launcherAtStop = Get-ExactProcess -ProcessId ([int]$launcher.ProcessId)
    if ($null -ne $launcherAtStop) {
        Assert-LauncherProcessIdentity `
            -Record $record `
            -Process $launcherAtStop `
            -ExpectedScript (Join-Path $root 'scripts\demo\run_demo.ps1')
        try {
            Stop-ProcessAndWait -ProcessId ([int]$launcherAtStop.ProcessId) -Label 'Gradio launcher'
        }
        catch {
            $stopErrors.Add($_.Exception.Message)
        }
    }
    if ($stopErrors.Count -gt 0) {
        throw "Gradio stop errors: $($stopErrors -join '; ')"
    }
    if ($PreserveMode) {
        [void](Write-PreserveComponentStopRecord `
            -Root $root -RunId $RunId -Component 'gradio' `
            -OwnerRecordPath $recordPath)
    }
    if (-not $PreserveMode) {
        Remove-OwnedGradioState `
            -Record $record -RuntimeRoot $runtimeRoot -RecordPath $recordPath
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

function Get-SidecarContainerIdentity {
    param([string]$DockerPath, [string]$ContainerId)
    # Windows PowerShell strips embedded quotes when marshalling native argv.
    $labelKey = '"imp.demo.owner"'
    if ($PSVersionTable.PSVersion.Major -lt 6) {
        $labelKey = '\"imp.demo.owner\"'
    }
    $format = '{{.Id}}|{{index .Config.Labels ' + $labelKey + '}}|{{.Name}}'
    return @(Invoke-DockerCommand -DockerPath $DockerPath -Arguments @(
        'container', 'inspect', '--format',
        $format,
        $ContainerId
    ) -Label 'sidecar ownership inspection')
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
    throw 'Sidecar absence proof was ambiguous.'
}

function Stop-OwnedSidecar {
    param([switch]$PreserveMode, [string]$RunId = '')
    $root = Resolve-DemoStopRoot
    $ownerRoot = if ($PreserveMode) {
        Join-Path $root "demo_runtime\preserved\$RunId\sidecar"
    } else { Join-Path $root 'demo_runtime\nnunet\launcher' }
    $recordPath = if ($PreserveMode) {
        Get-PreserveActiveOwnerRecordPath -Root $root -RunId $RunId -Component 'sidecar'
    } else { Join-Path $ownerRoot 'sidecar.json' }
    if ($null -eq $recordPath) { return }
    $record = Read-OwnedRecord `
        -Path $recordPath -ExpectedParent $ownerRoot `
        -ExpectedName ([IO.Path]::GetFileName($recordPath))
    if ($null -eq $record) {
        return
    }
    $fields = @($record.PSObject.Properties.Name | Sort-Object)
    if (
        ($fields -join ',') -cne 'container_id,container_name,docker_path,owner_token,schema_version' -or
        $record.schema_version -cne 'imp.demo.sidecar-owner.v1' -or
        [string]$record.container_name -cnotmatch '^imp-nnunet-[a-z0-9_-]+$' -or
        [string]$record.container_id -cnotmatch '^[0-9a-f]{64}$' -or
        [string]$record.owner_token -cnotmatch '^[0-9a-f]{32}$'
    ) {
        throw 'Sidecar owner record identity mismatch.'
    }
    $docker = Get-Item -LiteralPath ([string]$record.docker_path) -Force
    if (
        $docker.PSIsContainer -or
        ($docker.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        $docker.Name -cne 'docker.exe'
    ) {
        throw 'Sidecar Docker identity mismatch.'
    }
    try {
        $identity = @(Get-SidecarContainerIdentity `
            -DockerPath $docker.FullName -ContainerId ([string]$record.container_id))
    }
    catch {
        $inspectionError = $_.Exception.Message
        if (Test-SidecarContainerAbsent `
            -DockerPath $docker.FullName -ContainerId ([string]$record.container_id)) {
            if ($PreserveMode) {
                [void](Write-PreserveComponentStopRecord `
                    -Root $root -RunId $RunId -Component 'sidecar' `
                    -OwnerRecordPath $recordPath)
            }
            else {
                Remove-OwnedRecord -Path $recordPath
            }
            return
        }
        throw "Sidecar inspection failed and absence was not proven: $inspectionError"
    }
    $expected = "$($record.container_id)|$($record.owner_token)|/$($record.container_name)"
    if ($identity.Count -ne 1 -or ([string]$identity[0]).Trim() -cne $expected) {
        throw 'Sidecar container ownership proof failed.'
    }
    [void](Invoke-DockerCommand -DockerPath $docker.FullName -Arguments @(
        'container', 'stop', '--time', '10', ([string]$record.container_id)
    ) -Label 'owned sidecar stop')
    if ($PreserveMode) {
        [void](Write-PreserveComponentStopRecord `
            -Root $root -RunId $RunId -Component 'sidecar' `
            -OwnerRecordPath $recordPath)
    }
    else {
        Remove-OwnedRecord -Path $recordPath
    }
}

function Remove-OwnedRuntimeFiles {
    $root = Resolve-DemoStopRoot
    foreach ($path in @(
        (Join-Path $root 'demo_runtime\launcher'),
        (Join-Path $root 'demo_runtime\nnunet\launcher')
    )) {
        if (Test-Path -LiteralPath $path -PathType Container) {
            $item = Get-Item -LiteralPath $path -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw 'Launcher directory cleanup path is unsafe.'
            }
            if (@(Get-ChildItem -LiteralPath $item.FullName -Force).Count -eq 0) {
                Remove-Item -LiteralPath $item.FullName -Force
            }
        }
    }
}

function Wait-DemoPortsClosed {
    param([int]$TimeoutSeconds = 30)
    $ports = @(7860, 7861, 7862)
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $open = @(
            Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
                Where-Object { $_.LocalPort -in $ports }
        )
        if ($open.Count -eq 0) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Write-PreserveStopRecord {
    param(
        [string]$Root,
        [string]$RunId,
        [scriptblock]$NonceFactory = { [guid]::NewGuid().ToString('N') }
    )
    $RunId = Assert-PreserveRunId -RunId $RunId
    $directory = Join-Path $Root "demo_runtime\preserved\$RunId\stop"
    [void](New-Item -ItemType Directory -Path $directory -Force)
    $record = [ordered]@{
        event = 'stopped'
        ports = @(7860, 7861, 7862)
        recorded_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    $bytes = [Text.Encoding]::ASCII.GetBytes(($record | ConvertTo-Json -Compress))
    for ($attempt = 0; $attempt -lt 16; $attempt++) {
        $nonce = [string](& $NonceFactory)
        if ($nonce -cnotmatch '^[a-z0-9][a-z0-9_-]{0,127}$') {
            throw 'Stop record nonce is unsafe.'
        }
        $path = Join-Path $directory ("stop-$nonce.stopped.json")
        try {
            $stream = [IO.File]::Open(
                $path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None
            )
            try { $stream.Write($bytes, 0, $bytes.Length) }
            finally { $stream.Dispose() }
            return $path
        }
        catch [IO.IOException] {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw }
        }
    }
    throw 'Unable to allocate a unique stop record.'
}

function Invoke-DemoStop {
    [CmdletBinding()]
    param([string]$Root = '', [switch]$PreserveMode, [string]$RunId = '')
    $script:StopRoot = $Root
    if ($PreserveMode) {
        $RunId = Assert-PreserveRunId -RunId $RunId
    }
    $errors = New-Object 'System.Collections.Generic.List[string]'
    foreach ($operation in @(
        @{ Label = 'Cloudflare'; Action = { Stop-OwnedCloudflared -PreserveMode:$PreserveMode -RunId $RunId } },
        @{ Label = 'Gradio'; Action = { Stop-OwnedGradio -PreserveMode:$PreserveMode -RunId $RunId } },
        @{ Label = 'sidecar'; Action = { Stop-OwnedSidecar -PreserveMode:$PreserveMode -RunId $RunId } },
        @{ Label = 'runtime cleanup'; Action = { if (-not $PreserveMode) { Remove-OwnedRuntimeFiles } } }
    )) {
        try {
            & $operation.Action | Out-Null
        }
        catch {
            $errors.Add("$($operation.Label): $($_.Exception.Message)")
        }
    }
    $portsClosed = $false
    try {
        $portsClosed = Wait-DemoPortsClosed -TimeoutSeconds 30
    }
    catch {
        $errors.Add("port proof: $($_.Exception.Message)")
    }
    if (-not $portsClosed -and $errors.Count -eq 0) {
        [Console]::Error.WriteLine('Demo shutdown failed: ports 7860, 7861, or 7862 remain open.')
        return 7
    }
    if ($errors.Count -gt 0) {
        [Console]::Error.WriteLine("Demo shutdown errors: $($errors -join '; ')")
        return 5
    }
    if ($portsClosed) {
        if ($PreserveMode) {
            [void](Write-PreserveStopRecord `
                -Root (Resolve-DemoStopRoot) -RunId $RunId)
        }
        [Console]::Out.WriteLine('demo_stop=passed ports_closed=7860,7861,7862')
        return 0
    }
    return 7
}

if ($MyInvocation.InvocationName -ne '.') {
    exit (Invoke-DemoStop `
        -Root $Root -PreserveMode:$PreserveMode -RunId $RunId)
}
