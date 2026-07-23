Set-StrictMode -Version Latest

function ConvertTo-LfText {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Value)

    $Value.Replace("`r`n", "`n").Replace("`r", "`n")
}

function New-BlockedDeckManifest {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$PriorManifestPath,
        [Parameter(Mandatory)][string]$HtmlPath,
        [Parameter(Mandatory)][string]$PdfPath,
        [Parameter(Mandatory)][string]$PptxPath,
        [Parameter(Mandatory)][string]$CurrentReleaseManifestSha256,
        [Parameter(Mandatory)][string]$CurrentContentSha256,
        [Parameter(Mandatory)][int]$SlideCount
    )

    try {
        $getFileSha256Hex = {
            param([Parameter(Mandatory)][string]$Path)

            $stream = [IO.File]::OpenRead($Path)
            $sha256 = [Security.Cryptography.SHA256]::Create()
            try {
                ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
            }
            finally {
                $sha256.Dispose()
                $stream.Dispose()
            }
        }
        $sha256Pattern = '^[0-9a-f]{64}$'
        $integerTypeCodes = @(
            [TypeCode]::Byte,
            [TypeCode]::SByte,
            [TypeCode]::Int16,
            [TypeCode]::UInt16,
            [TypeCode]::Int32,
            [TypeCode]::UInt32,
            [TypeCode]::Int64,
            [TypeCode]::UInt64
        )
        if (
            $CurrentReleaseManifestSha256 -cnotmatch $sha256Pattern -or
            $CurrentContentSha256 -cnotmatch $sha256Pattern -or
            $SlideCount -le 0
        ) {
            throw 'invalid blocked manifest authority'
        }

        $priorManifestItem = Get-Item -LiteralPath $PriorManifestPath -Force -ErrorAction Stop
        if (
            $priorManifestItem.PSIsContainer -or
            ($priorManifestItem.Attributes -band [IO.FileAttributes]::ReparsePoint)
        ) {
            throw 'invalid prior manifest file'
        }
        $prior = [IO.File]::ReadAllText($priorManifestItem.FullName) |
            ConvertFrom-Json -ErrorAction Stop
        if ($null -eq $prior -or $prior -is [Array]) {
            throw 'invalid prior manifest root'
        }

        $requiredTopLevelFields = @(
            'schema',
            'package_state',
            'current_release_manifest_sha256',
            'slide_count',
            'content_sha256',
            'generated_utc',
            'files'
        )
        $priorFields = @($prior.PSObject.Properties.Name)
        foreach ($field in $requiredTopLevelFields) {
            if ($priorFields -cnotcontains $field) {
                throw 'missing prior manifest field'
            }
        }
        if (
            $prior.schema -cne 'imp.presentation.delivery/v2' -or
            @('complete', 'incomplete_blocked') -cnotcontains $prior.package_state -or
            $prior.current_release_manifest_sha256 -cnotmatch $sha256Pattern -or
            $prior.content_sha256 -cnotmatch $sha256Pattern -or
            ($prior.generated_utc -isnot [DateTime] -and (
                $prior.generated_utc -isnot [string] -or
                [string]::IsNullOrWhiteSpace($prior.generated_utc)
            )) -or
            $integerTypeCodes -notcontains [Type]::GetTypeCode($prior.slide_count.GetType()) -or
            [decimal]$prior.slide_count -le 0
        ) {
            throw 'invalid prior manifest authority'
        }

        $priorFiles = @($prior.files)
        if ($priorFiles.Count -ne 3) {
            throw 'invalid prior file count'
        }
        $expectedFiles = @(
            [pscustomobject]@{
                Path = 'outputs/' + [IO.Path]::GetFileName($HtmlPath)
                LivePath = $HtmlPath
                Extension = '.html'
            },
            [pscustomobject]@{
                Path = 'outputs/' + [IO.Path]::GetFileName($PdfPath)
                LivePath = $PdfPath
                Extension = '.pdf'
            },
            [pscustomobject]@{
                Path = 'outputs/' + [IO.Path]::GetFileName($PptxPath)
                LivePath = $PptxPath
                Extension = '.pptx'
            }
        )
        $expectedPaths = [Collections.Generic.HashSet[string]]::new(
            [StringComparer]::Ordinal
        )
        foreach ($expected in $expectedFiles) {
            if (
                [IO.Path]::GetExtension($expected.LivePath) -cne $expected.Extension -or
                -not $expectedPaths.Add($expected.Path)
            ) {
                throw 'invalid live artifact paths'
            }
        }

        $requiredFileFields = @(
            'path',
            'status',
            'built_release_manifest_sha256',
            'current_release_manifest_sha256',
            'content_sha256',
            'bytes',
            'sha256'
        )
        $allowedStatuses = @('current', 'stale_unregenerated', 'stale_rebuild_blocked')
        $blockedFiles = foreach ($expected in $expectedFiles) {
            $matches = @($priorFiles | Where-Object { $_.path -ceq $expected.Path })
            if ($matches.Count -ne 1) {
                throw 'invalid prior artifact paths'
            }
            $entry = $matches[0]
            $entryFields = @($entry.PSObject.Properties.Name)
            foreach ($field in $requiredFileFields) {
                if ($entryFields -cnotcontains $field) {
                    throw 'missing prior artifact field'
                }
            }
            if (
                $entry.status -isnot [string] -or
                $allowedStatuses -cnotcontains $entry.status -or
                $entry.built_release_manifest_sha256 -cnotmatch $sha256Pattern -or
                $entry.current_release_manifest_sha256 -cne $prior.current_release_manifest_sha256 -or
                $entry.content_sha256 -cnotmatch $sha256Pattern -or
                $entry.sha256 -cnotmatch $sha256Pattern -or
                $integerTypeCodes -notcontains [Type]::GetTypeCode($entry.bytes.GetType()) -or
                [decimal]$entry.bytes -lt 0 -or
                ($entry.status -ceq 'current' -and
                    $entry.content_sha256 -cne $prior.content_sha256)
            ) {
                throw 'invalid prior artifact authority'
            }

            $liveItem = Get-Item -LiteralPath $expected.LivePath -Force -ErrorAction Stop
            if (
                $liveItem.PSIsContainer -or
                ($liveItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
                [decimal]$entry.bytes -ne [decimal]$liveItem.Length
            ) {
                throw 'invalid live artifact'
            }
            $liveSha256 = & $getFileSha256Hex $liveItem.FullName
            if ($liveSha256 -cne $entry.sha256) {
                throw 'invalid live artifact digest'
            }

            [ordered]@{
                path = $expected.Path
                status = 'stale_rebuild_blocked'
                built_release_manifest_sha256 = $entry.built_release_manifest_sha256
                current_release_manifest_sha256 = $CurrentReleaseManifestSha256
                content_sha256 = $entry.content_sha256
                bytes = $liveItem.Length
                sha256 = $liveSha256
            }
        }

        $priorCurrentCount = @($priorFiles | Where-Object { $_.status -ceq 'current' }).Count
        if (
            ($prior.package_state -ceq 'complete' -and (
                $priorCurrentCount -ne 3 -or
                @($priorFiles | Where-Object {
                    $_.built_release_manifest_sha256 -cne $prior.current_release_manifest_sha256
                }).Count -ne 0
            )) -or
            ($prior.package_state -ceq 'incomplete_blocked' -and $priorCurrentCount -eq 3)
        ) {
            throw 'invalid prior package state'
        }

        [ordered]@{
            schema = 'imp.presentation.delivery/v2'
            package_state = 'incomplete_blocked'
            current_release_manifest_sha256 = $CurrentReleaseManifestSha256
            slide_count = $SlideCount
            content_sha256 = $CurrentContentSha256
            generated_utc = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
            files = @($blockedFiles)
        }
    }
    catch {
        throw 'prior delivery receipt invalid'
    }
}

function Invoke-DeckPublishTransaction {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$StagedHtmlPath,
        [Parameter(Mandatory)][string]$StagedPdfPath,
        [Parameter(Mandatory)][string]$StagedManifestPath,
        [Parameter(Mandatory)][string]$HtmlPath,
        [Parameter(Mandatory)][string]$PdfPath,
        [Parameter(Mandatory)][string]$ManifestPath,
        [Parameter(Mandatory)][string]$CommitMarkerPath,
        [Parameter(Mandatory)][string]$RollbackDir,
        [Parameter(Mandatory)][System.Collections.IDictionary]$BlockedManifest,
        [scriptblock]$MoveOperation,
        [scriptblock]$RemoveOperation
    )

    if ($null -eq $MoveOperation) {
        $MoveOperation = {
            param([string]$Source, [string]$Destination)
            Move-Item -LiteralPath $Source -Destination $Destination -Force
        }
    }
    if ($null -eq $RemoveOperation) {
        $RemoveOperation = {
            param([string]$Path, [bool]$Recurse)
            if ($Recurse) {
                Remove-Item -LiteralPath $Path -Recurse -Force
            }
            else {
                Remove-Item -LiteralPath $Path -Force
            }
        }
    }

    $rollbackHtmlPath = Join-Path $RollbackDir ([IO.Path]::GetFileName($HtmlPath))
    $rollbackPdfPath = Join-Path $RollbackDir ([IO.Path]::GetFileName($PdfPath))
    $commitMarker = [ordered]@{
        schema = 'imp.presentation.delivery.commit/v1'
        package_state = 'in_progress'
        stage_id = Split-Path $StagedHtmlPath -Parent | Split-Path -Leaf
        targets = @(
            [IO.Path]::GetFileName($HtmlPath),
            [IO.Path]::GetFileName($PdfPath),
            [IO.Path]::GetFileName($ManifestPath)
        )
    }
    $backedUp = [ordered]@{}
    $installed = New-Object 'System.Collections.Generic.List[string]'

    New-Item -ItemType Directory -Path $RollbackDir -Force | Out-Null
    [IO.File]::WriteAllText(
        $CommitMarkerPath,
        ($commitMarker | ConvertTo-Json -Depth 6),
        [Text.UTF8Encoding]::new($false)
    )
    # The manifest is the authority. Block it before changing either artifact.
    [IO.File]::WriteAllText(
        $ManifestPath,
        ($BlockedManifest | ConvertTo-Json -Depth 6),
        [Text.UTF8Encoding]::new($false)
    )

    try {
        foreach ($pair in @(
            @($HtmlPath, $rollbackHtmlPath),
            @($PdfPath, $rollbackPdfPath)
        )) {
            if (Test-Path -LiteralPath $pair[0]) {
                & $MoveOperation $pair[0] $pair[1]
                $backedUp[$pair[0]] = $pair[1]
            }
        }
        & $MoveOperation $StagedHtmlPath $HtmlPath
        $installed.Add($HtmlPath)
        & $MoveOperation $StagedPdfPath $PdfPath
        $installed.Add($PdfPath)
        # Complete authority is installed last.
        & $MoveOperation $StagedManifestPath $ManifestPath
    }
    catch {
        $publishError = $_.Exception.Message
        $rollbackErrors = New-Object 'System.Collections.Generic.List[string]'
        try {
            [IO.File]::WriteAllText(
                $ManifestPath,
                ($BlockedManifest | ConvertTo-Json -Depth 6),
                [Text.UTF8Encoding]::new($false)
            )
        }
        catch {
            $rollbackErrors.Add('blocked manifest write failed')
        }
        foreach ($path in $installed) {
            if (Test-Path -LiteralPath $path) {
                try { & $RemoveOperation $path $false }
                catch { $rollbackErrors.Add("installed artifact removal failed: $([IO.Path]::GetFileName($path))") }
            }
        }
        foreach ($source in $backedUp.Keys) {
            $backup = $backedUp[$source]
            if (Test-Path -LiteralPath $backup) {
                try { & $MoveOperation $backup $source }
                catch { $rollbackErrors.Add("artifact restore failed: $([IO.Path]::GetFileName($source))") }
            }
        }
        $commitMarker.package_state = 'incomplete_blocked'
        $commitMarker.rollback_status = if ($rollbackErrors.Count) { 'failed' } else { 'restored' }
        $commitMarker.rollback_error_count = $rollbackErrors.Count
        [IO.File]::WriteAllText(
            $CommitMarkerPath,
            ($commitMarker | ConvertTo-Json -Depth 6),
            [Text.UTF8Encoding]::new($false)
        )
        $suffix = if ($rollbackErrors.Count) {
            '; rollback: ' + ($rollbackErrors -join '; ')
        }
        else { '' }
        throw "deck publish failed: $publishError$suffix"
    }

    # Cleanup occurs only after the complete manifest is authoritative.
    & $RemoveOperation $CommitMarkerPath $false
    & $RemoveOperation $RollbackDir $true
}
