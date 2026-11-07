<#
.SYNOPSIS
Run Heads A and B for one recommendation day with live OpenRouter model calls.
.DESCRIPTION
Uses the frozen historical archive, with a 9 a.m. America/New_York cutoff.
Date is the session being recommended, not the date of the model call.
Both heads run by default. BudgetUsd controls B; HeadABudgetUsd controls A.
Both default to $0.80. Head A uses its existing usage-based cost guard.
DryRun performs a no-provider-call preflight. Offline rehearses with scripted responses.
.EXAMPLE
.\scripts\run_live_demo.ps1 -DryRun
.EXAMPLE
.\scripts\run_live_demo.ps1 -Date 2026-08-27
.EXAMPLE
.\scripts\run_live_demo.ps1 -Date 2026-08-27 -RequireDifferences -EnvFile C:\path\to\.env
.EXAMPLE
.\scripts\run_live_demo.ps1 -Offline
.EXAMPLE
.\scripts\run_live_demo.ps1 -ReportRunId live_20260827_4c79e0
.EXAMPLE
.\scripts\run_live_demo.ps1 -Head A -Date 2026-08-27
#>
[CmdletBinding()]
param(
    [string]$Date = "2026-08-27",
    [ValidateRange(0.05, 10.0)]
    [double]$BudgetUsd = 0.8,
    [ValidateRange(0.05, 10.0)]
    [double]$HeadABudgetUsd = 0.8,
    [ValidateSet("A", "B", "Both")]
    [string]$Head = "Both",
    [ValidateRange(1, 8)]
    [int]$MaxBatches = 3,
    [string]$EnvFile = "",
    [ValidatePattern('^[A-Za-z0-9_-]{1,24}$')]
    [string]$RunId = "",
    [ValidatePattern('^[A-Za-z0-9_-]{1,24}$')]
    [string]$ReportRunId = "",
    [switch]$DryRun,
    [switch]$Offline,
    [switch]$RequireDifferences
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project environment missing. From '$repoRoot', run: uv sync --extra dev"
}

# Resolve an explicitly supplied credential file before changing the working folder.
if ($EnvFile) {
    $EnvFile = (Resolve-Path -LiteralPath $EnvFile -ErrorAction Stop).Path
}

$pythonArguments = @(
    "-X", "utf8", "-u", "-m", "scripts.run_live_demo",
    "--date", $Date,
    "--head", $(if ($Head -eq "Both") { "Both" } else { $Head.ToUpperInvariant() }),
    "--budget-usd", $BudgetUsd.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--head-a-budget-usd", $HeadABudgetUsd.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--max-batches", $MaxBatches.ToString()
)
if ($EnvFile) { $pythonArguments += @("--env-file", $EnvFile) }
if ($RunId) { $pythonArguments += @("--run-id", $RunId) }
if ($ReportRunId) { $pythonArguments += @("--report-run-id", $ReportRunId) }
if ($DryRun) { $pythonArguments += "--dry-run" }
if ($Offline) { $pythonArguments += "--offline" }
if ($RequireDifferences) { $pythonArguments += "--require-differences" }

$previousUtf8 = $env:PYTHONUTF8
$previousUnbuffered = $env:PYTHONUNBUFFERED
$previousPythonPath = $env:PYTHONPATH
$demoExitCode = 0
Push-Location -LiteralPath $repoRoot
try {
    $env:PYTHONUTF8 = "1"
    $env:PYTHONUNBUFFERED = "1"
    $runtimeSource = Join-Path $repoRoot "src"
    $env:PYTHONPATH = if ($previousPythonPath) {
        $runtimeSource + [IO.Path]::PathSeparator + $previousPythonPath
    } else { $runtimeSource }
    & $pythonPath @pythonArguments
    $demoExitCode = $LASTEXITCODE
}
finally {
    $env:PYTHONUTF8 = $previousUtf8
    $env:PYTHONUNBUFFERED = $previousUnbuffered
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
if ($demoExitCode -ne 0) {
    Write-Host "Recommendation demo exited with code $demoExitCode. See the message above."
    exit $demoExitCode
}
