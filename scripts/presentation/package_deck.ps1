[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Add-Type -AssemblyName System.IO.Compression.FileSystem
. (Join-Path $PSScriptRoot 'publish_deck_transaction.ps1')

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$sourceDir = Join-Path $root 'presentation\interactive'
$outputDir = Join-Path $root 'outputs'
$pptxPath = Join-Path $outputDir 'imp-lesion-evidence-defense.pptx'
$pdfPath = Join-Path $outputDir 'imp-lesion-evidence-defense.pdf'
$htmlPath = Join-Path $outputDir 'imp-lesion-evidence-defense.html'
$manifestPath = Join-Path $outputDir 'imp-lesion-evidence-defense-manifest.json'
$stageDir = Join-Path $outputDir ('.imp-lesion-evidence-defense-stage-' + [Guid]::NewGuid().ToString('N'))
$stagedHtmlPath = Join-Path $stageDir 'imp-lesion-evidence-defense.html'
$stagedPdfPath = Join-Path $stageDir 'imp-lesion-evidence-defense.pdf'
$stagedManifestPath = Join-Path $stageDir 'imp-lesion-evidence-defense-manifest.json'
$commitMarkerPath = Join-Path $outputDir '.imp-lesion-evidence-defense-commit.json'
$rollbackDir = Join-Path $outputDir ('.imp-lesion-evidence-defense-rollback-' + [Guid]::NewGuid().ToString('N'))
$releaseManifestPath = Join-Path $root 'release\imp_release_manifest.json'

function ConvertTo-Base64Utf8 {
    param([Parameter(Mandatory)][string]$Value)
    [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Value))
}

function Get-RelativeDeliveryPath {
    param([Parameter(Mandatory)][string]$Path)
    'outputs/' + [IO.Path]::GetFileName($Path)
}

function Get-Sha256Hex {
    param([Parameter(Mandatory)][byte[]]$Bytes)
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        ([BitConverter]::ToString($sha256.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
}

function Assert-PresentationSourcesUnchanged {
    param(
        [Parameter(Mandatory)][string]$ContentPath,
        [Parameter(Mandatory)][string]$ContentSha256,
        [Parameter(Mandatory)][string]$BuilderPath,
        [Parameter(Mandatory)][string]$BuilderSha256,
        [Parameter(Mandatory)][string]$ReleasePath,
        [Parameter(Mandatory)][string]$ReleaseSha256
    )
    $bindings = @(
        [pscustomobject]@{ Path = $ContentPath; Sha256 = $ContentSha256 }
        [pscustomobject]@{ Path = $BuilderPath; Sha256 = $BuilderSha256 }
        [pscustomobject]@{ Path = $ReleasePath; Sha256 = $ReleaseSha256 }
    )
    foreach ($binding in $bindings) {
        $current = Get-Sha256Hex -Bytes ([IO.File]::ReadAllBytes($binding.Path))
        if ($current -cne $binding.Sha256) {
            throw 'presentation source rotated before packaging'
        }
    }
}

function Resolve-TrustedPdfInfo {
    $candidate = $env:IMP_PDFINFO_EXE
    if ([string]::IsNullOrWhiteSpace($candidate) -and -not [string]::IsNullOrWhiteSpace($env:IMP_PYTHON_EXE)) {
        $runtimeRoot = Split-Path (Split-Path $env:IMP_PYTHON_EXE -Parent) -Parent
        $candidate = Join-Path $runtimeRoot 'native\poppler\Library\bin\pdfinfo.exe'
    }
    if ([string]::IsNullOrWhiteSpace($candidate) -or -not [IO.Path]::IsPathRooted($candidate)) {
        throw 'Trusted absolute pdfinfo executable unavailable'
    }
    $item = Get-Item -LiteralPath $candidate -Force -ErrorAction Stop
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Trusted absolute pdfinfo executable unavailable'
    }
    $item.FullName
}

function Get-ZipEntryText {
    param(
        [Parameter(Mandatory)][IO.Compression.ZipArchive]$Archive,
        [Parameter(Mandatory)][string]$Name
    )
    $entry = $Archive.GetEntry($Name)
    if ($null -eq $entry) {
        throw "Injected PPTX missing OOXML part: $Name"
    }
    $stream = $entry.Open()
    $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8, $true)
    try {
        return $reader.ReadToEnd()
    }
    finally {
        $reader.Dispose()
        $stream.Dispose()
    }
}

function Get-SlideShapeNode {
    param(
        [Parameter(Mandatory)][IO.Compression.ZipArchive]$Archive,
        [Parameter(Mandatory)][int]$SlideNumber,
        [Parameter(Mandatory)][string]$ShapeName
    )
    $document = [Xml.XmlDocument]::new()
    $document.LoadXml((Get-ZipEntryText -Archive $Archive -Name "ppt/slides/slide$SlideNumber.xml"))
    $namespaces = [Xml.XmlNamespaceManager]::new($document.NameTable)
    $namespaces.AddNamespace('p', 'http://schemas.openxmlformats.org/presentationml/2006/main')
    $document.SelectSingleNode("//p:cNvPr[@name='$ShapeName']", $namespaces)
}

function Resolve-InternalSlideJumpTarget {
    param(
        [Parameter(Mandatory)][IO.Compression.ZipArchive]$Archive,
        [Parameter(Mandatory)][int]$SlideNumber,
        [Parameter(Mandatory)][string]$ShapeName
    )
    $shape = Get-SlideShapeNode -Archive $Archive -SlideNumber $SlideNumber -ShapeName $ShapeName
    if ($null -eq $shape) {
        throw "PPTX navigation shape missing: slide $SlideNumber / $ShapeName"
    }
    $slideNamespaces = [Xml.XmlNamespaceManager]::new($shape.OwnerDocument.NameTable)
    $slideNamespaces.AddNamespace('a', 'http://schemas.openxmlformats.org/drawingml/2006/main')
    $click = $shape.SelectSingleNode('a:hlinkClick', $slideNamespaces)
    if ($null -eq $click -or $click.GetAttribute('action') -cne 'ppaction://hlinksldjump') {
        throw "PPTX navigation action missing: slide $SlideNumber / $ShapeName"
    }
    $relationshipId = $click.GetAttribute(
        'id',
        'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    )
    if ([string]::IsNullOrWhiteSpace($relationshipId)) {
        throw "PPTX navigation relationship missing: slide $SlideNumber / $ShapeName"
    }

    $relationships = [Xml.XmlDocument]::new()
    $relationships.LoadXml((Get-ZipEntryText -Archive $Archive -Name "ppt/slides/_rels/slide$SlideNumber.xml.rels"))
    $relationshipNamespaces = [Xml.XmlNamespaceManager]::new($relationships.NameTable)
    $relationshipNamespaces.AddNamespace('pr', 'http://schemas.openxmlformats.org/package/2006/relationships')
    $relationship = $relationships.SelectSingleNode(
        "//pr:Relationship[@Id='$relationshipId']",
        $relationshipNamespaces
    )
    if (
        $null -eq $relationship -or
        $relationship.GetAttribute('Type') -cne 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide' -or
        -not [string]::IsNullOrWhiteSpace($relationship.GetAttribute('TargetMode'))
    ) {
        throw "PPTX navigation relationship invalid: slide $SlideNumber / $ShapeName"
    }
    $target = $relationship.GetAttribute('Target')
    if ($target -notmatch '(?:^|/)slide(\d+)\.xml$') {
        throw "PPTX navigation target invalid: slide $SlideNumber / $ShapeName"
    }
    [int]$Matches[1]
}

function Assert-InjectedPptxNavigation {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][int]$ExpectedSlideCount
    )
    $archive = [IO.Compression.ZipFile]::OpenRead($Path)
    try {
        for ($number = 1; $number -le $ExpectedSlideCount; $number++) {
            $slideXml = Get-ZipEntryText -Archive $archive -Name "ppt/slides/slide$number.xml"
            if ($slideXml -notmatch '<p:transition[^>]*spd="med"[^>]*>\s*<p:fade') {
                throw "PPTX navigation injection missing medium fade on slide $number"
            }
        }
        $expectedPipelineTargets = @(5, 6, 6, 7, 8, 10)
        for ($index = 0; $index -lt $expectedPipelineTargets.Count; $index++) {
            $actualTarget = Resolve-InternalSlideJumpTarget `
                -Archive $archive `
                -SlideNumber 4 `
                -ShapeName "pipeline-node-$index"
            if ($actualTarget -ne $expectedPipelineTargets[$index]) {
                throw "Unexpected pipeline target for node $index"
            }
        }
        for ($number = 5; $number -le 10; $number++) {
            $actualTarget = Resolve-InternalSlideJumpTarget `
                -Archive $archive `
                -SlideNumber $number `
                -ShapeName 'back-to-pipeline'
            if ($actualTarget -ne 4) {
                throw "Unexpected Back to Pipeline target on slide $number"
            }
        }
        for ($number = 1; $number -le 4; $number++) {
            if ($null -ne (Get-SlideShapeNode -Archive $archive -SlideNumber $number -ShapeName 'back-to-pipeline')) {
                throw "Unexpected Back to Pipeline on slide $number"
            }
        }
        for ($number = 11; $number -le $ExpectedSlideCount; $number++) {
            if ($null -ne (Get-SlideShapeNode -Archive $archive -SlideNumber $number -ShapeName 'back-to-pipeline')) {
                throw "Unexpected Back to Pipeline on slide $number"
            }
        }
    }
    finally {
        $archive.Dispose()
    }
}

function Assert-PresentationReleaseContract {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$BuilderSource,
        [Parameter(Mandatory)][object]$ContentProjection,
        [Parameter(Mandatory)][object]$Release,
        [Parameter(Mandatory)][string]$ReleaseDigest
    )
    $leftModelId = 'L206-control-s206'
    $rightModelId = 'L192-nnUNet-v2-raw-100ep'
    $liveComparisons = @($Release.comparisons | Where-Object { $_.id -ceq 'live_demo' })
    $demoSlides = @($ContentProjection.slides | Where-Object { $_.id -ceq 's10-demo' })
    $expectedPresenterRoute = @(
        's01-title', 's02-leakage', 's03-questions', 's04-pipeline',
        's05-data', 's06-models', 's07-validation', 's08-ablation-design',
        's09-negative-result', 's10-demo', 's16-reproducibility', 's17-conclusion'
    )
    $expectedContentLine = "Live public/synthetic RGB: $leftModelId, then reconstructed $rightModelId"
    $expectedBuilderText = "SAME RGB, SEQUENTIAL\n$leftModelId\n$rightModelId"
    $expectedBoundaryText = 'illustrative fixed-cache examples; not protected-test evidence'
    if (
        $liveComparisons.Count -ne 1 -or
        $liveComparisons[0].left_model_id -cne $leftModelId -or
        $liveComparisons[0].right_model_id -cne $rightModelId -or
        $demoSlides.Count -ne 1 -or
        -not (@($demoSlides[0].body) -ccontains $expectedContentLine) -or
        (@($ContentProjection.meta.presenter_route) -join ',') -cne ($expectedPresenterRoute -join ',') -or
        -not (@($demoSlides[0].body) -ccontains $expectedBoundaryText) -or
        (@($demoSlides[0].live_ground_truth_state) -cne 'ground_truth_not_loaded') -or
        -not $BuilderSource.Contains($expectedBuilderText)
    ) {
        throw 'presentation source provenance mismatch'
    }
    $archive = [IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $slideXml = Get-ZipEntryText -Archive $archive -Name 'ppt/slides/slide10.xml'
        if (-not $slideXml.Contains($leftModelId) -or -not $slideXml.Contains($rightModelId)) {
            throw 'PPTX Slide 10 identity mismatch'
        }
        $notesXml = Get-ZipEntryText -Archive $archive -Name 'ppt/notesSlides/notesSlide10.xml'
        if (-not $notesXml.Contains($ReleaseDigest)) {
            throw 'PPTX release provenance mismatch'
        }
    }
    finally {
        $archive.Dispose()
    }
}

$contentPath = Join-Path $sourceDir 'content.json'
$builderPath = Join-Path $root 'scripts\presentation\build_pptx.mjs'
$html = [IO.File]::ReadAllText((Join-Path $sourceDir 'index.html'))
$css = [IO.File]::ReadAllText((Join-Path $sourceDir 'deck.css'))
$script = [IO.File]::ReadAllText((Join-Path $sourceDir 'deck.js'))
$contentBytes = [IO.File]::ReadAllBytes($contentPath)
$builderBytes = [IO.File]::ReadAllBytes($builderPath)
$expectedSlideCount = 17
$content = [Text.UTF8Encoding]::new($false, $true).GetString($contentBytes)
$builderSource = [Text.UTF8Encoding]::new($false, $true).GetString($builderBytes)
$releaseBytes = [IO.File]::ReadAllBytes($releaseManifestPath)
$contentDigest = Get-Sha256Hex -Bytes $contentBytes
$builderDigest = Get-Sha256Hex -Bytes $builderBytes
$releaseDigest = Get-Sha256Hex -Bytes $releaseBytes
$release = [Text.Encoding]::ASCII.GetString($releaseBytes) | ConvertFrom-Json
$contentProjection = $content | ConvertFrom-Json
$expectedSlideIds = @(
    's01-title', 's02-leakage', 's03-questions', 's04-pipeline',
    's05-data', 's06-models', 's07-validation', 's08-ablation-design',
    's09-negative-result', 's10-demo', 's11-challenge-leakage',
    's12-challenge-fairness', 's13-challenge-uncertainty',
    's14-challenge-demo', 's15-challenge-repro', 's16-reproducibility',
    's17-conclusion'
)
if (@($contentProjection.slides).Count -ne $expectedSlideCount) {
    throw "Expected $expectedSlideCount source slides"
}
for ($index = 0; $index -lt $expectedSlideCount; $index++) {
    if ($contentProjection.slides[$index].id -cne $expectedSlideIds[$index]) {
        throw "Unexpected source slide ID at position $($index + 1)"
    }
}
$releaseComparisons = @($release.comparisons | Where-Object {
    $_.id -ceq 'paper_rq1' -or $_.id -ceq 'paper_rq2'
})
if (
    $release.schema_version -cne 'imp.release.manifest.v1' -or
    $contentProjection.release_manifest_sha256 -cne $releaseDigest -or
    @($contentProjection.release_comparisons).Count -ne $releaseComparisons.Count
) {
    throw 'release manifest projection mismatch'
}
for ($index = 0; $index -lt $releaseComparisons.Count; $index++) {
    foreach ($field in @('id', 'left_model_id', 'right_model_id', 'claim_policy', 'scope')) {
        if ($contentProjection.release_comparisons[$index].$field -cne $releaseComparisons[$index].$field) {
            throw 'release manifest projection mismatch'
        }
    }
}
$deltaUri = 'data:image/png;base64,' + [Convert]::ToBase64String(
    [IO.File]::ReadAllBytes((Join-Path $sourceDir 'assets\loop206-delta.png'))
)
$demoMiddleUri = 'data:image/png;base64,' + [Convert]::ToBase64String(
    [IO.File]::ReadAllBytes((Join-Path $sourceDir 'assets\qualitative-demo-middle.png'))
)

# Keep the portable deck independent of fetch/file:// restrictions. JSON lives in
# a non-executable script tag; deck.js parses it directly at startup.
function ConvertTo-SafeInlineJson {
    param([Parameter(Mandatory)][string]$Value)
    return $Value.Replace('<', '\u003c').Replace('>', '\u003e').Replace('&', '\u0026')
}
$embeddedContent = ConvertTo-SafeInlineJson $content
$html = $html.Replace(
    '    <div id="slides" class="slides"></div>',
    "    <script id=`"deck-content`" type=`"application/json`">`n$embeddedContent`n    </script>`n    <div id=`"slides`" class=`"slides`"></div>"
)
$script = $script.Replace('"assets/loop206-delta.png"', '"' + $deltaUri + '"')
$script = $script.Replace('"assets/qualitative-demo-middle.png"', '"' + $demoMiddleUri + '"')
$html = $html.Replace('<link rel="stylesheet" href="deck.css">', "<style>`n$css`n</style>")
$html = $html.Replace('<script src="deck.js" defer></script>', "<script>`n$script`n</script>")
$html = ConvertTo-LfText -Value $html

if (-not (Test-Path -LiteralPath $pptxPath)) {
    throw "PPTX not found: $pptxPath"
}
Assert-InjectedPptxNavigation -Path $pptxPath -ExpectedSlideCount $expectedSlideCount
Assert-PresentationReleaseContract `
    -Path $pptxPath `
    -BuilderSource $builderSource `
    -ContentProjection $contentProjection `
    -Release $release `
    -ReleaseDigest $releaseDigest
$pdfInfoExe = Resolve-TrustedPdfInfo
Assert-PresentationSourcesUnchanged `
    -ContentPath $contentPath `
    -ContentSha256 $contentDigest `
    -BuilderPath $builderPath `
    -BuilderSha256 $builderDigest `
    -ReleasePath $releaseManifestPath `
    -ReleaseSha256 $releaseDigest

# Validate the prior receipt before staging can mutate the filesystem.
$blockedManifest = New-BlockedDeckManifest `
    -PriorManifestPath $manifestPath `
    -HtmlPath $htmlPath `
    -PdfPath $pdfPath `
    -PptxPath $pptxPath `
    -CurrentReleaseManifestSha256 $releaseDigest `
    -CurrentContentSha256 $contentDigest `
    -SlideCount $expectedSlideCount

$powerPoint = $null
$deck = $null
$powerPoint = New-Object -ComObject PowerPoint.Application
try {
    New-Item -ItemType Directory -Path $stageDir -Force | Out-Null
    [IO.File]::WriteAllText($stagedHtmlPath, $html, [Text.UTF8Encoding]::new($false))
    $deck = $powerPoint.Presentations.Open($pptxPath, $true, $false, $false)
    $deck.SaveAs($stagedPdfPath, 32)
}
finally {
    if ($null -ne $deck) {
        $deck.Close()
        [Runtime.InteropServices.Marshal]::FinalReleaseComObject($deck) | Out-Null
    }
    if ($null -ne $powerPoint) {
        $powerPoint.Quit()
        [Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

$pdfMetadata = & $pdfInfoExe $stagedPdfPath 2>&1
$pageLine = $pdfMetadata | Where-Object { $_ -match '^Pages:\s+\d+' } | Select-Object -First 1
$pageCount = if ($null -ne $pageLine -and $pageLine -match '^Pages:\s+(\d+)') { [int]$Matches[1] } else { 0 }
if ($pageCount -ne $expectedSlideCount) {
    throw "Expected a $expectedSlideCount-page PDF. pdfinfo output: $($pdfMetadata -join '; ')"
}

$files = @($stagedHtmlPath, $stagedPdfPath, $pptxPath) | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    [ordered]@{
        path = Get-RelativeDeliveryPath $item.FullName
        status = 'current'
        built_release_manifest_sha256 = $releaseDigest
        current_release_manifest_sha256 = $releaseDigest
        content_sha256 = $contentDigest
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$manifest = [ordered]@{
    schema = 'imp.presentation.delivery/v2'
    package_state = 'complete'
    current_release_manifest_sha256 = $releaseDigest
    slide_count = $expectedSlideCount
    content_sha256 = $contentDigest
    generated_utc = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    files = $files
}
[IO.File]::WriteAllText(
    $stagedManifestPath,
    ($manifest | ConvertTo-Json -Depth 5),
    [Text.UTF8Encoding]::new($false)
)

# Current artifacts remain untouched until every staged output and receipt check succeeds.
Assert-PresentationSourcesUnchanged `
    -ContentPath $contentPath `
    -ContentSha256 $contentDigest `
    -BuilderPath $builderPath `
    -BuilderSha256 $builderDigest `
    -ReleasePath $releaseManifestPath `
    -ReleaseSha256 $releaseDigest

# Publish through a manifest-first transaction with a commit marker. During commit, the authoritative
# receipt is explicitly blocked, so a failed move cannot look like a mixed
# current package. Existing HTML/PDF are restored on any move failure.
Invoke-DeckPublishTransaction `
    -StagedHtmlPath $stagedHtmlPath `
    -StagedPdfPath $stagedPdfPath `
    -StagedManifestPath $stagedManifestPath `
    -HtmlPath $htmlPath `
    -PdfPath $pdfPath `
    -ManifestPath $manifestPath `
    -CommitMarkerPath $commitMarkerPath `
    -RollbackDir $rollbackDir `
    -BlockedManifest $blockedManifest

[ordered]@{
    html = $htmlPath
    pdf = $pdfPath
    pptx = $pptxPath
    manifest = $manifestPath
} | ConvertTo-Json -Compress
