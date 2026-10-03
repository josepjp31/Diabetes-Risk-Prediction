# Runs the whole pipeline on Windows (PowerShell). Usage (from the repository root):
#   .\run_all.ps1          # full pipeline (long: hyper-parameter searches)
#   .\run_all.ps1 -Fast    # skips the searches and uses the best parameters reported in the paper
#
# The script creates .venv and installs requirements.txt if needed, and ALWAYS uses the
# interpreter inside .venv (never the global Python). It stops at the first failing step.
# If PowerShell blocks the script:  Set-ExecutionPolicy -Scope Process Bypass
param([switch]$Fast)

Set-Location $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# 1. Virtual environment
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        Write-Host "ERROR: could not create .venv. Is Python 3.10+ installed and in PATH?"
        exit 1
    }
}

# 2. Dependencies (installed only if an import fails)
& $venvPython -c "import numpy, pandas, matplotlib, sklearn, lightgbm, imblearn, shap, openpyxl, ucimlrepo" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies - this takes a few minutes, do NOT press Ctrl+C..."
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: pip install failed or was interrupted. Run .\run_all.ps1 again to resume."
        exit 1
    }
}

# 3. Pipeline. UsesFast = the script accepts --fast (steps 03, 04 and 05).
$steps = @(
    @{ Script = "src/01_data_preparation.py";     UsesFast = $false },
    @{ Script = "src/02_logistic_regression.py";  UsesFast = $false },
    @{ Script = "src/03_random_forest.py";        UsesFast = $true  },
    @{ Script = "src/04_gradient_boosting.py";    UsesFast = $true  },
    @{ Script = "src/05_mlp_neural_network.py";   UsesFast = $true  },
    @{ Script = "src/06_cross_model_comparison.py"; UsesFast = $false }
)
foreach ($step in $steps) {
    $pyArgs = @($step.Script)
    if ($Fast -and $step.UsesFast) { $pyArgs += "--fast" }
    Write-Host ""
    Write-Host ">>> python $($pyArgs -join ' ')"
    & $venvPython @pyArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: step failed: $($step.Script)"
        exit $LASTEXITCODE
    }
}
Write-Host ""
Write-Host "Done. See figures/ and results/."
