#!/usr/bin/env bash
# Fig S11 — HierVAE 2D section tessellation across training steps.
#   A compact ladder of (z0,z1) slices packed into a 6x4 grid:
#   every other checkpoint over steps 5k-240k (final step pinned), with the
#   top-6 persistent-identity molecules shown as a strip above the slices.
#
# Analyze (default): plot from the shipped per-step slices under
#   data/figS11_hiervae_evolution/2d-slices/. No GPU. See SOURCE.md to re-create it.
#
# The plotter reads a relative "2d-slices/" directory, so we run it from the
# figure's data dir and write the output back to figures/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"

O="$REPO_ROOT/figures/reproducibility/figS11_hiervae_evolution"
mkdir -p "$O"

activate_env tess-molminer
cd "$REPO_ROOT/data/figS11_hiervae_evolution"
python "$REPO_ROOT/scripts/hiervae/plot_tessellation_evolution_mod.py" \
    --cols 6 --top 6 --out "$O/evolution.png"

echo "Output: figures/figS11_hiervae_evolution/evolution.png"
ls -la "$O"
