# Two-Machine Private Delivery

## Fixed Responsibilities

- Main workstation, RTX 5060 Ti: artifact generation, model execution, and demo serving. Use `main` for integration and `demo-runtime` for source changes.
- Laptop, RTX 4060 8 GB GPU and 16 GB RAM: clean-clone bootstrap, paper build, citation/read-only review, CPU tests, and an optional single-GPU smoke check. Use `paper-review` for source changes.
- Do not load two models concurrently or claim laptop training capacity. The laptop assignment is review and validation, not full training.
- Browser rendering and desktop/mobile screenshots remain unverified. Any visual-review steps in this runbook are pending operator checks, not release evidence.
- Exchange code only by private GitHub push/pull. Exchange weights, priors, caches, and data only by private LAN/USB after SHA-256 verification; never through GitHub.

## Windows Bootstrap

Prerequisites: Git, `uv`, Python install access through `uv`, and either `latexmk` or both `pdflatex` and `bibtex`. The script creates the configurable `.venv-win` environment with Python 3.12 and the `dev`, `analysis`, and `demo` extras, then runs CPU tests, deterministic table generation, a portable paper audit, and the paper build.

CPU bootstrap for a clean laptop clone:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1
```

Different environment directory:

```powershell
$laptopVenv = Join-Path $env:LOCALAPPDATA 'IMP/venvs/paper-review'
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1 -Venv $laptopVenv
```

Optional RTX 4060 smoke environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1 -Compute cu130
```

`-Compute cu130` first installs the `train` extra, then overlays the tested `torch==2.12.0+cu130` and `torchvision==0.27.0+cu130` wheels from the official CUDA 13.0 index. It fails unless both versions match and `torch.cuda.is_available()` is true. A normal `uv sync` resolves the Windows lock to CPU wheels and can replace this overlay. After the overlay, run GPU-related project commands with `.venv-win\Scripts\python.exe` directly or `uv run --no-sync`; do not run an ordinary `uv sync` in that environment.

The smoke option verifies the environment only. Run a single model at a time. Keep weights outside GitHub.

### Audit Levels

Clean-clone bootstrap and CI are portable verification. They use `--source-verification registry-only`: registry semantic integrity, citations, claims, paper artifacts, manifest bindings, and every present source byte are validated. Compact source reports intentionally absent from Git are recorded as sorted `missing_source_ids` with the warning `source bytes unavailable; strict local release audit required`. This receipt is not full evidence reproduction.

Before a local release, run the strict local release audit on the main workstation where all registry sources are present:

```powershell
.venv-win\Scripts\python.exe scripts/paper/audit_clean_v3_paper.py --paper paper/clean_v3_loop206 --registry demo/data/evidence_registry.json --receipt .artifacts/task12/paper-audit-strict.json
```

Strict is the audit CLI default. It fails on any missing or mismatched source byte. Do not relabel a `source_verification=registry-only` receipt as strict.

## Private GitHub Provisioning

The owner is `quanntm1206`. Never substitute a public repository. Authenticate, create the private repository, then verify visibility before any push:

```powershell
gh auth status
gh repo create quanntm1206/imp-lesion-evidence-demo --private --source . --remote origin
gh repo view quanntm1206/imp-lesion-evidence-demo --json isPrivate,sshUrl,url
```

Require `isPrivate: true`. Stop on false, missing, or ambiguous output. If the repository exists, resolve its authenticated SSH URL and add the remote only when `origin` is absent:

```powershell
$repo = gh repo view quanntm1206/imp-lesion-evidence-demo --json isPrivate,sshUrl | ConvertFrom-Json
if ($repo.isPrivate -ne $true) { throw 'Repository privacy is not verified' }
git remote add origin $repo.sshUrl
```

Push without force. Task 12 preserves local `main`; integration/default-branch changes belong to final delivery:

```powershell
git push -u origin rescue/paper-demo
git push origin main
git push origin rescue/paper-demo:paper-review
git push origin rescue/paper-demo:demo-runtime
```

## Laptop Handoff

Physical laptop execution remains unverified until the operator runs it. One-command CPU handoff from PowerShell:

```powershell
$ErrorActionPreference = 'Stop'; $repoJson = gh repo view quanntm1206/imp-lesion-evidence-demo --json isPrivate,sshUrl; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; $repo = $repoJson | ConvertFrom-Json; if ($repo.isPrivate -ne $true) { throw 'Repository privacy is not verified' }; $target = Join-Path (Get-Location) 'imp-lesion-evidence-demo'; git clone --branch paper-review $repo.sshUrl $target; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; Set-Location $target; powershell -ExecutionPolicy Bypass -File scripts/bootstrap_windows.ps1; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; git status --short; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

Expected operator receipt: bootstrap exits zero; portable demo tests pass with only named external-runtime integration skips; paper audit reports `passed=true errors=0 source_verification=registry-only` plus missing-source warnings; the PDF builds; `git status --short` emits nothing. Record the command exit status and commit SHA. Do not report physical-laptop verification before that receipt exists.

## Artifact Transfer

Use PowerShell 7. On the main workstation, set `IMP_ARTIFACT_TRANSFER_ROOT` to the private transfer directory, then write a recursive, path-safe hash manifest before LAN/USB transfer:

```powershell
$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path -LiteralPath $env:IMP_ARTIFACT_TRANSFER_ROOT).Path
$manifestName = 'sha256-manifest.json'
$manifestPath = Join-Path $sourceRoot $manifestName
$files = @(
    Get-ChildItem -LiteralPath $sourceRoot -Recurse -File |
        Where-Object { $_.FullName -ne $manifestPath } |
        ForEach-Object {
            $relative = [IO.Path]::GetRelativePath($sourceRoot, $_.FullName).Replace('\', '/')
            if ([IO.Path]::IsPathRooted($relative) -or $relative -eq '..' -or $relative.StartsWith('../')) {
                throw "Unsafe transfer path: $relative"
            }
            [ordered]@{
                path = $relative
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
                bytes = [int64]$_.Length
            }
        } |
        Sort-Object path
)
$payload = [ordered]@{ schema_version = 'imp.private_transfer.v1'; files = $files }
$json = $payload | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText($manifestPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
```

Transfer the entire directory, including `sha256-manifest.json`. On the laptop, set `IMP_ARTIFACT_TRANSFER_ROOT` to the received directory, recompute the recursive manifest records, then require exact relative-path, size, and SHA-256 equality:

```powershell
$ErrorActionPreference = 'Stop'
$destinationRoot = (Resolve-Path -LiteralPath $env:IMP_ARTIFACT_TRANSFER_ROOT).Path
$manifestName = 'sha256-manifest.json'
$manifestPath = Join-Path $destinationRoot $manifestName
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if ($manifest.schema_version -ne 'imp.private_transfer.v1') { throw 'Transfer manifest schema mismatch' }
foreach ($entry in $manifest.files) {
    $relative = [string]$entry.path
    if ([IO.Path]::IsPathRooted($relative) -or $relative -eq '..' -or $relative.StartsWith('../') -or $relative.Contains('/../')) {
        throw "Unsafe manifest path: $relative"
    }
}
$actual = @(
    Get-ChildItem -LiteralPath $destinationRoot -Recurse -File |
        Where-Object { $_.FullName -ne $manifestPath } |
        ForEach-Object {
            $relative = [IO.Path]::GetRelativePath($destinationRoot, $_.FullName).Replace('\', '/')
            [pscustomobject]@{
                path = $relative
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
                bytes = [int64]$_.Length
            }
        } |
        Sort-Object path
)
$difference = Compare-Object -ReferenceObject @($manifest.files) -DifferenceObject $actual -Property path,sha256,bytes
if ($difference) { $difference | Out-String | Write-Error; throw 'Artifact transfer verification failed' }
```

The manifest contains canonical relative paths only; it never records absolute paths. Stop on any missing, extra, changed, rooted, or traversal path. Do not commit, upload, paste, or attach weights, priors, datasets, caches, environment values, tokens, or absolute private paths to GitHub issues or CI logs.

## CI Contract

CI starts from GitHub's clean checkout on an Ubuntu CPU runner. It installs pinned `uv` and Python 3.12, syncs only `dev`, `analysis`, and `demo`, runs portable `tests/demo`, rebuilds tracked tables, rejects deterministic drift, performs the registry-only paper audit, compiles LaTeX when a runner is installed, and uploads test/paper receipts. It requires no model weights, private cache, dataset, CUDA, or GPU. Missing external-runtime integration assets produce only the explicit `external runtime assets; local release gate required` skips. GPU integration and the strict source-byte audit remain local receipts.

## Official Sources

Fetched 2026-07-20:

- `UV_PROJECT_ENVIRONMENT`: https://docs.astral.sh/uv/reference/environment/#uv_project_environment
- `uv run --no-sync`: https://docs.astral.sh/uv/reference/cli/#uv-run
- GitHub Actions integration for `uv`: https://docs.astral.sh/uv/guides/integration/github/
- Full-commit action pinning guidance: https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions
- `uv` 0.11.29 release used by CI: https://github.com/astral-sh/uv/releases/tag/0.11.29
- `actions/checkout` v7.0.1 source: https://github.com/actions/checkout/releases/tag/v7.0.1
- `astral-sh/setup-uv` v8.3.2 source: https://github.com/astral-sh/setup-uv/releases/tag/v8.3.2
- `actions/upload-artifact` v7.0.1 source: https://github.com/actions/upload-artifact/releases/tag/v7.0.1
- Exact CUDA 13.0 PyTorch wheel indexes: https://download.pytorch.org/whl/cu130/torch/ and https://download.pytorch.org/whl/cu130/torchvision/
