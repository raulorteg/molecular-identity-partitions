#!/usr/bin/env bash
# Fig S6 — GDSS 2D section under coarsened equivalence conventions.
#   The Fig 1c grid re-quotiented under 5 progressively coarser relations
#   (InChIKey, Murcko, Murcko-generic, formula, composition).
#
# Analyze (default): plot from the shipped Fig 1 grid (reused, not duplicated).
# Re-create the data: that grid — see data/fig1_sections/SOURCE.md.
# Runs under tess-gdss (rdkit 2020.09.1) for GDSS valence-expanded SMILES.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

IN=data/fig1_sections/gdss_grid2d
O=figures/reproducibility/figS6_gdss_coarsen
mkdir -p "$O"
CONVENTIONS=(inchikey murcko murcko_generic formula composition)

activate_env tess-gdss
for CONV in "${CONVENTIONS[@]}"; do
    python scripts/gdss/gdss_plot_grid2d.py --in_dir "$IN" \
        --out_dir "$O" --label gdss_grid --convention "$CONV"
done

echo "Outputs in $O:"; ls "$O"
