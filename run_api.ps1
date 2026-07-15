# Run from FYP-BioFitCoach folder: checks DB then starts FastAPI.
# Usage: .\run_api.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Error "venv not found. Create it: python -m venv venv && .\venv\Scripts\pip install -r requirements.txt"
    exit 1
}

& .\venv\Scripts\python.exe database.py check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& .\venv\Scripts\python.exe database.py serve
