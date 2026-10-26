# Create and populate the project virtual environment (Windows PowerShell).
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment in .venv ..."
    python -m venv .venv
} else {
    Write-Host "Using existing virtual environment in .venv ."
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")

Write-Host ""
Write-Host "Verifying imports and versions ..."
& $VenvPython -c "import numpy, pandas, sklearn, tensorflow, matplotlib, seaborn, scipy; print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('scikit-learn', sklearn.__version__); print('tensorflow', tensorflow.__version__); print('matplotlib', matplotlib.__version__); print('seaborn', seaborn.__version__); print('scipy', scipy.__version__)"

Write-Host ""
Write-Host "Environment ready."
Write-Host "Run: .\.venv\Scripts\python.exe .\run_all.py"
