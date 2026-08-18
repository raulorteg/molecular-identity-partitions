#!/usr/bin/env bash
# Fig 1 — two-dimensional fixed-eta sections.
#   Panels a-c: 2D coordinate sections (MolMiner, HierVAE, GDSS). Each grid
#   point is decoded with a fixed random seed and colored by canonical-SMILES
#   identity. For MolMiner the section varies the first two conditioning
#   coordinates (logP, QED).
#
# Fig S14 reads from the same data/fig1_sections/ — see repro/figS14_paths.sh.
#
# Analyze (default): plot from shipped data/fig1_sections/. No GPU, no model.
# Re-create: rebuild the inputs from checkpoints — see data/fig1_sections/SOURCE.md.
#
# The molecule grids call rdkit. GDSS panels run under tess-gdss
# (rdkit 2020.09.1); their SMILES include valence constructs rdkit >= 2023
# rejects.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

D=data/fig1_sections
O=figures/reproducibility/fig1_sections
mkdir -p "$O"

# --- panels a, b: MolMiner / HierVAE 2D sections ---
activate_env tess-molminer
python scripts/molminer/molminer_plot_2d_slice.py --csv "$D/molminer_2dslice_seed42.txt"     --out_dir "$O" --label fig1a_molminer
python scripts/hiervae/plot_tessellation_final.py --csv "$D/hiervae_2dslice_step240000.txt"  --out_dir "$O" --label fig1b_hiervae

# --- panel c: GDSS 2D grid ---
activate_env tess-gdss
python scripts/gdss/gdss_plot_grid2d.py --in_dir "$D/gdss_grid2d" --out_dir "$O" --label fig1c_gdss

echo "Outputs in $O:"
ls -la "$O"
