[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$sourceDir = Join-Path $root 'presentation\interactive'
$outputDir = Join-Path $root 'outputs'
$pptxPath = Join-Path $outputDir 'imp-lesion-evidence-defense.pptx'
$pdfPath = Join-Path $outputDir 'imp-lesion-evidence-defense.pdf'
$htmlPath = Join-Path $outputDir 'imp-lesion-evidence-defense.html'
$manifestPath = Join-Path $outputDir 'imp-lesion-evidence-defense-manifest.json'

function ConvertTo-Base64Utf8 {
    param([Parameter(Mandatory)][string]$Value)
    [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Value))
}

function Get-RelativeDeliveryPath {
    param([Parameter(Mandatory)][string]$Path)
    'outputs/' + [IO.Path]::GetFileName($Path)
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$html = [IO.File]::ReadAllText((Join-Path $sourceDir 'index.html'))
$css = [IO.File]::ReadAllText((Join-Path $sourceDir 'deck.css'))
$script = [IO.File]::ReadAllText((Join-Path $sourceDir 'deck.js'))
$content = [IO.File]::ReadAllText((Join-Path $sourceDir 'content.json'))

$contentUri = 'data:application/json;base64,' + (ConvertTo-Base64Utf8 $content)
$deltaUri = 'data:image/png;base64,' + [Convert]::ToBase64String(
    [IO.File]::ReadAllBytes((Join-Path $sourceDir 'assets\loop206-delta.png'))
)
$demoUri = 'data:image/png;base64,' + [Convert]::ToBase64String(
    [IO.File]::ReadAllBytes((Join-Path $sourceDir 'assets\qualitative-demo.png'))
)

$script = $script.Replace('"content.json"', '"' + $contentUri + '"')
$script = $script.Replace('"assets/loop206-delta.png"', '"' + $deltaUri + '"')
$script = $script.Replace('"assets/qualitative-demo.png"', '"' + $demoUri + '"')
$html = $html.Replace('<link rel="stylesheet" href="deck.css">', "<style>`n$css`n</style>")
$html = $html.Replace('<script src="deck.js" defer></script>', "<script>`n$script`n</script>")
[IO.File]::WriteAllText($htmlPath, $html, [Text.UTF8Encoding]::new($false))

if (-not (Test-Path -LiteralPath $pptxPath)) {
    throw "PPTX not found: $pptxPath"
}

$powerPoint = $null
$deck = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $deck = $powerPoint.Presentations.Open($pptxPath, $true, $false, $false)
    $deck.SaveAs($pdfPath, 32)
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

$pdfInfo = Get-Command pdfinfo -ErrorAction Stop
$dependencyRoot = Split-Path (Split-Path (Split-Path $pdfInfo.Source -Parent) -Parent) -Parent
$pdfInfoExe = Join-Path $dependencyRoot 'native\poppler\Library\bin\pdfinfo.exe'
if (-not (Test-Path -LiteralPath $pdfInfoExe)) {
    $pdfInfoExe = $pdfInfo.Source
}
$pdfMetadata = & $pdfInfoExe $pdfPath 2>&1
$pageLine = $pdfMetadata | Where-Object { $_ -match '^Pages:\s+\d+' } | Select-Object -First 1
$pageCount = if ($null -ne $pageLine -and $pageLine -match '^Pages:\s+(\d+)') { [int]$Matches[1] } else { 0 }
if ($pageCount -ne 12) {
    throw "Expected a 12-page PDF. pdfinfo output: $($pdfMetadata -join '; ')"
}

$files = @($htmlPath, $pdfPath, $pptxPath) | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    [ordered]@{
        path = Get-RelativeDeliveryPath $item.FullName
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$manifest = [ordered]@{
    schema = 'imp.presentation.delivery/v1'
    slide_count = 12
    generated_utc = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    files = $files
}
[IO.File]::WriteAllText(
    $manifestPath,
    ($manifest | ConvertTo-Json -Depth 5),
    [Text.UTF8Encoding]::new($false)
)

[ordered]@{
    html = $htmlPath
    pdf = $pdfPath
    pptx = $pptxPath
    manifest = $manifestPath
} | ConvertTo-Json -Compress
