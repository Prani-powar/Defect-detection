param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$LocalPackages = Join-Path $ProjectRoot ".train_packages"
if (Test-Path -LiteralPath $LocalPackages) {
    $env:PYTHONPATH = $LocalPackages
}

$PythonCandidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "C:\Users\prani\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    "python"
)

$Python = $null
foreach ($Candidate in $PythonCandidates) {
    if ($Candidate -eq "python") {
        $Command = Get-Command python -ErrorAction SilentlyContinue
        if ($Command) {
            $Python = $Command.Source
            break
        }
    }
    elseif (Test-Path -LiteralPath $Candidate) {
        $Python = $Candidate
        break
    }
}

if (-not $Python) {
    throw "Python was not found. Install Python 3.11, then run: py -3.11 -m venv .venv"
}

Write-Host "Starting Apple AI dashboard..."
Write-Host "Project: $ProjectRoot"
Write-Host "Python:  $Python"
Write-Host "URL:     http://localhost:$Port"

& $Python -B -m streamlit run "$ProjectRoot\src\dashboard.py" `
    --global.developmentMode=false `
    --server.port $Port `
    --server.headless false
