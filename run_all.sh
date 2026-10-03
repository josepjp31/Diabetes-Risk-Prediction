#!/usr/bin/env bash
# Same as run_all.ps1 for WSL / Linux / macOS.
#   bash run_all.sh          # full pipeline (long)
#   bash run_all.sh --fast   # skip hyper-parameter searches (uses the paper's best params)
# Tip for WSL: clone/run the repo inside the Linux filesystem (~/...) rather than /mnt/c for speed,
# and do not mix it with the Windows .venv (Scripts\) - this script creates its own .venv-linux.
set -euo pipefail
cd "$(dirname "$0")"

FAST=""
[[ "${1:-}" == "--fast" ]] && FAST="--fast"

[ -x .venv-linux/bin/python ] || python3 -m venv .venv-linux
PY=.venv-linux/bin/python

if ! "$PY" -c "import numpy, pandas, matplotlib, sklearn, lightgbm, imblearn, shap, openpyxl, ucimlrepo" 2>/dev/null; then
  echo "Installing dependencies (do not press Ctrl+C)..."
  "$PY" -m pip install -r requirements.txt
fi

"$PY" src/01_data_preparation.py
"$PY" src/02_logistic_regression.py
"$PY" src/03_random_forest.py $FAST
"$PY" src/04_gradient_boosting.py $FAST
"$PY" src/05_mlp_neural_network.py $FAST
"$PY" src/06_cross_model_comparison.py
echo "Done. See figures/ and results/."
