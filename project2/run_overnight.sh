#!/usr/bin/env bash
#
# One-shot overnight training launcher.
#
# Creates or updates the `AI` conda environment from environment.yml, then
# runs the Optuna harness (hparam_optuna.py) followed by the meta-shootout
# (meta_shootout.py). Produces runs/overnight_champion.pkl as the final
# artefact.
#
# Usage:
#   ./run_overnight.sh
#
# Prereqs:
#   - conda (Miniconda or Anaconda) on PATH
#   - Repository cloned; working directory = project root
#
# Safe to interrupt (Ctrl-C): Optuna writes per-trial rows to CSV as it goes,
# so a partial run still leaves usable artefacts in runs/optuna_<ts>/.

set -euo pipefail

# ── Check conda is available ──────────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found on PATH."
    echo "Install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Enable `conda activate` inside this script (fixes sourcing issues on some shells).
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── Create or update the AI environment ───────────────────────────────────────
if conda env list | grep -qE "^AI\s"; then
    echo "Updating conda env 'AI' from environment.yml..."
    conda env update -n AI -f environment.yml --prune
else
    echo "Creating conda env 'AI' from environment.yml..."
    conda env create -f environment.yml
fi

# ── Run the Optuna harness ────────────────────────────────────────────────────
echo ""
echo "===================================================================="
echo "  Starting Optuna overnight training"
echo "  Logs stream to stdout. Output goes to runs/optuna_<timestamp>/"
echo "===================================================================="
echo ""

conda run -n AI --live-stream python hparam_optuna.py

# ── Pick overnight winner via 1000-game shootout ──────────────────────────────
echo ""
echo "===================================================================="
echo "  Running meta-shootout on all trial champions (1000 games each)"
echo "===================================================================="
echo ""

conda run -n AI --live-stream python meta_shootout.py

echo ""
echo "===================================================================="
echo "  DONE. Overnight champion: runs/overnight_champion.pkl"
echo "  Watch it play:   conda run -n AI python watch_cartpole.py runs/overnight_champion.pkl"
echo "  Record best MP4: conda run -n AI python best_cartpole.py   runs/overnight_champion.pkl 20"
echo "===================================================================="
