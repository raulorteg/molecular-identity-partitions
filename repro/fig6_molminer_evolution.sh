#!/usr/bin/env bash
# Fig 6 — MolMiner 2D section tessellation across training epochs.
#   A compact ladder of fixed-coordinate slices packed into a 6x4 grid: every
#   other checkpoint (final epoch pinned), with the top-6 persistent-identity
#   molecules shown as a strip above the slices.
#
# Analyze (default): plot from the shipped per-epoch slices under
#   data/fig6_molminer_evolution/2d-slices/. No GPU. See SOURCE.md to re-create it.
#
# The plotter reads its inputs from a relative "2d-slices/" directory, so we run
# it from the figure's data dir and write the output back to figures/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"

O="$REPO_ROOT/figures/reproducibility/fig6_molminer_evolution"
mkdir -p "$O"

activate_env tess-molminer
cd "$REPO_ROOT/data/fig6_molminer_evolution"
python "$REPO_ROOT/scripts/molminer/plot_tessellation_evolution_mod.py" \
    --cols 6 --out "$O/evolution.png"

echo "Output: figures/fig6_molminer_evolution/evolution.png"
ls -la "$O"
