[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cu130')]
    [string]$Compute = 'cpu',
    [string]$Venv = '.venv-win'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-LastExitCode {
    param([Parameter(Mandatory = $true)][string]$Operation)

    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/ and retry.'
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ProjectEnvironment = if ([IO.Path]::IsPathRooted($Venv)) {
    [IO.Path]::GetFullPath($Venv)
} else {
    [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Venv))
}
$Python = Join-Path $ProjectEnvironment 'Scripts/python.exe'
$ReceiptDir = Join-Path $ProjectRoot '.artifacts/task12'
$GeneratedPaperDir = Join-Path $ReceiptDir 'generated-paper'
$PaperDir = Join-Path $ProjectRoot 'paper/clean_v3_loop206'
$Registry = Join-Path $ProjectRoot 'demo/data/evidence_registry.json'
$PreviousProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT

try {
    Set-Location $ProjectRoot
    $env:UV_PROJECT_ENVIRONMENT = $ProjectEnvironment

    if ($Compute -eq 'cu130') {
        uv sync --python 3.12 --extra dev --extra analysis --extra demo --extra train
        Assert-LastExitCode 'uv dependency sync'

        # uv.lock resolves Windows torch to CPU. Overlay only after the final sync.
        uv pip install --python $Python --index-url https://download.pytorch.org/whl/cu130/ torch==2.12.0+cu130 torchvision==0.27.0+cu130
        Assert-LastExitCode 'CUDA 13.0 PyTorch overlay'

        & $Python -c "import torch, torchvision; assert torch.__version__ == '2.12.0+cu130', torch.__version__; assert torchvision.__version__ == '0.27.0+cu130', torchvision.__version__; assert torch.cuda.is_available(), 'CUDA is unavailable after the cu130 overlay'; print(f'torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.cuda.is_available()}')"
        Assert-LastExitCode 'CUDA overlay verification'
    } else {
        uv sync --python 3.12 --extra dev --extra analysis --extra demo
        Assert-LastExitCode 'uv dependency sync'
    }

    New-Item -ItemType Directory -Force -Path $ReceiptDir | Out-Null

    # --no-sync prevents later commands from replacing an explicit CUDA overlay.
    uv run --no-sync --python 3.12 python -m pytest tests/demo -q
    Assert-LastExitCode 'demo test suite'

    uv run --no-sync --python 3.12 python scripts/paper/build_clean_v3_tables.py --registry $Registry --paper-dir $GeneratedPaperDir
    Assert-LastExitCode 'deterministic paper table build'

    foreach ($Table in @('evidence_scope.tex', 'clean_v3_validation.tex', 'loop206_ablation.tex', 'legacy_loop170.tex')) {
        $GeneratedHash = (Get-FileHash -Algorithm SHA256 (Join-Path $GeneratedPaperDir "tables/$Table")).Hash
        $TrackedHash = (Get-FileHash -Algorithm SHA256 (Join-Path $PaperDir "tables/$Table")).Hash
        if ($GeneratedHash -ne $TrackedHash) {
            throw "Deterministic paper table drift: $Table"
        }
    }

    uv run --no-sync --python 3.12 python scripts/paper/audit_clean_v3_paper.py --paper $PaperDir --registry $Registry --receipt (Join-Path $ReceiptDir 'paper-audit.json')
    Assert-LastExitCode 'paper evidence audit'

    Push-Location $PaperDir
    try {
        $PaperBuilt = $false
        $LatexmkExitCode = $null
        if (Get-Command latexmk -ErrorAction SilentlyContinue) {
            $PreviousErrorPreference = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
            $LatexmkExitCode = $LASTEXITCODE
            $ErrorActionPreference = $PreviousErrorPreference
            if ($LatexmkExitCode -eq 0) {
                $PaperBuilt = $true
            } else {
                Write-Warning "latexmk failed; trying pdflatex/bibtex fallback (exit $LatexmkExitCode)"
            }
        }

        if (-not $PaperBuilt -and (Get-Command pdflatex -ErrorAction SilentlyContinue) -and (Get-Command bibtex -ErrorAction SilentlyContinue)) {
            pdflatex -interaction=nonstopmode -halt-on-error main.tex
            Assert-LastExitCode 'first pdflatex pass'
            bibtex main
            Assert-LastExitCode 'bibtex pass'
            pdflatex -interaction=nonstopmode -halt-on-error main.tex
            Assert-LastExitCode 'second pdflatex pass'
            pdflatex -interaction=nonstopmode -halt-on-error main.tex
            Assert-LastExitCode 'final pdflatex pass'
            $PaperBuilt = $true
        }

        if (-not $PaperBuilt -and $null -ne $LatexmkExitCode) {
            throw "latexmk failed with exit code $LatexmkExitCode and pdflatex/bibtex fallback is unavailable. Install Perl for latexmk or install both fallback commands."
        } elseif (-not $PaperBuilt) {
            throw 'No TeX toolchain found. Install latexmk, or install both pdflatex and bibtex, then rerun this script.'
        }
    } finally {
        Pop-Location
    }
} finally {
    $env:UV_PROJECT_ENVIRONMENT = $PreviousProjectEnvironment
    Set-Location $ProjectRoot
}
