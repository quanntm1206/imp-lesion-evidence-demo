[CmdletBinding()]
param(
    [string]$CloudflaredPath = ''
)

$ErrorActionPreference = 'Stop'
$LocalUrl = 'http://127.0.0.1:7860'

function Stop-DemoTunnel {
    param([string]$Message, [int]$Code)
    [Console]::Error.WriteLine($Message)
    exit $Code
}

try {
    $health = Invoke-WebRequest -Uri $LocalUrl -Method Get -TimeoutSec 10 -UseBasicParsing
    if ($health.StatusCode -ne 200) {
        throw 'health status mismatch'
    }
}
catch {
    Stop-DemoTunnel 'Local demo is unavailable. Start run_demo.ps1 before opening a tunnel.' 3
}

try {
    if ($CloudflaredPath) {
        $resolved = (Resolve-Path -LiteralPath $CloudflaredPath -ErrorAction Stop).Path
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw 'cloudflared path is not a file'
        }
    }
    else {
        $command = Get-Command cloudflared -CommandType Application -ErrorAction Stop
        $resolved = $command.Source
    }
    if ([System.IO.Path]::GetFileName($resolved) -notmatch '^cloudflared(?:\.exe)?$') {
        throw 'resolved executable has an unexpected name'
    }
    & $resolved --version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw 'cloudflared version probe failed'
    }
}
catch {
    Stop-DemoTunnel 'A valid cloudflared application was not found.' 4
}

Write-Output 'Temporary public tunnel active. Press Ctrl+C to stop it.'
& $resolved tunnel --url http://127.0.0.1:7860
$exitCode = $LASTEXITCODE
exit $exitCode
