# =============================================================================
#  Windows Defender exclusions for the vo_coach dev environment
# =============================================================================
#  RUN AS ADMINISTRATOR:
#    right-click PowerShell -> "Run as administrator", then:
#    powershell -ExecutionPolicy Bypass -File F:\wierd_bio\defender_exclusions.ps1
#
#  WHAT THIS TRADES AWAY - read before running.
#
#  Excluding a folder means Defender stops scanning files in it. The .venv
#  holds executable Python code and native DLLs, so if a malicious package
#  were ever installed there via pip, it would not be caught. That risk is
#  accepted here because the packages are version-pinned in requirements.txt
#  and installed from PyPI deliberately, not downloaded at random.
#
#  This deliberately does NOT exclude python.exe globally, and does NOT
#  exclude all of F:\ - either would let ANY Python script on the machine run
#  unscanned, which is a much larger hole than this problem justifies.
#
#  TO UNDO: same commands with Remove-MpPreference instead of Add-MpPreference.
# =============================================================================

$ErrorActionPreference = "Stop"

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$pr = New-Object Security.Principal.WindowsPrincipal($id)
if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Not elevated. Re-run this from an ADMIN PowerShell." -ForegroundColor Red
    exit 1
}

# 19,643 files. Written en masse by pip, re-read en masse by every pytest run.
# This is the one that actually matters.
$venv = "F:\wierd_bio\vo_coach\.venv"

# The channels venv is where test_pipeline.py's 97 checks actually run from, so
# it gets re-read as often as vo_coach's does. Same trade, same reasoning.
$venvChannels = "F:\wierd_bio\channels\.venv"

# ~1 GB of model weights, re-read whenever Whisper loads.
$hf = Join-Path $env:USERPROFILE ".cache\huggingface"

foreach ($p in @($venv, $venvChannels, $hf)) {
    if (Test-Path $p) {
        Add-MpPreference -ExclusionPath $p
        Write-Host "excluded: $p" -ForegroundColor Green
    } else {
        Write-Host "skipped (not found): $p" -ForegroundColor Yellow
    }
}

# Scheduled scans: cap CPU and only run when the machine is idle. This costs
# no protection at all - scans just take longer and stay out of your way.
Set-MpPreference -ScanAvgCPULoadFactor 20
Set-MpPreference -ScanOnlyIfIdleEnabled $true
Write-Host "scheduled-scan CPU cap set to 20%, idle-only enabled" -ForegroundColor Green

Write-Host ""
Write-Host "Current exclusions:" -ForegroundColor Cyan
(Get-MpPreference).ExclusionPath
