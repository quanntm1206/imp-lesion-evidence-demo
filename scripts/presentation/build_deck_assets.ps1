[CmdletBinding()]
param(
    [string]$RepoRoot
)

$ErrorActionPreference = 'Stop'
if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}

$outputDir = Join-Path $RepoRoot 'presentation\interactive\assets'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$pdfToPpm = Get-Command pdftoppm -All -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandType -eq 'Application' -and $_.Source -notlike '*.cmd' } |
    Select-Object -First 1
if (-not $pdfToPpm) {
    throw 'pdftoppm executable not found. Install Poppler or MiKTeX Poppler tools.'
}

$specs = @(
    [ordered]@{
        name = 'loop206-delta'
        source = 'paper/clean_v3_loop206/figures/loop206_delta.pdf'
        output = 'presentation/interactive/assets/loop206-delta.png'
        dpi = 180
    },
    [ordered]@{
        name = 'qualitative-demo'
        source = 'paper/clean_v3_loop206/figures/qualitative_demo.pdf'
        output = 'presentation/interactive/assets/qualitative-demo.png'
        dpi = 120
    }
)

$assets = foreach ($spec in $specs) {
    $sourcePath = Join-Path $RepoRoot ($spec.source -replace '/', '\')
    $outputPath = Join-Path $RepoRoot ($spec.output -replace '/', '\')
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Missing source figure: $($spec.source)"
    }

    $prefix = [IO.Path]::Combine(
        [IO.Path]::GetDirectoryName($outputPath),
        [IO.Path]::GetFileNameWithoutExtension($outputPath)
    )
    & $pdfToPpm.Source -f 1 -singlefile -png -r $spec.dpi $sourcePath $prefix
    if ($LASTEXITCODE -ne 0) {
        throw "pdftoppm failed for $($spec.source) with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path -LiteralPath $outputPath)) {
        throw "Expected raster was not created: $($spec.output)"
    }

    [ordered]@{
        name = $spec.name
        source = $spec.source
        source_sha256 = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
        output = $spec.output
        output_sha256 = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
        bytes = (Get-Item -LiteralPath $outputPath).Length
        dpi = $spec.dpi
    }
}

$manifest = [ordered]@{
    schema = 'imp.presentation.assets/v1'
    source = 'tracked-paper-figures'
    assets = @($assets)
}

$manifestPath = Join-Path $outputDir 'asset-manifest.json'
$json = $manifest | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText($manifestPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
Write-Output "assets_status=valid count=$($assets.Count) manifest=presentation/interactive/assets/asset-manifest.json"
