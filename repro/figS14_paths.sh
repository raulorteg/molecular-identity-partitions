#!/usr/bin/env bash
# Fig S14 — one-dimensional fixed-randomness paths.
#   Panels a-c: decoded molecules along straight-line paths between generative
#   coordinates (MolMiner, HierVAE, GDSS). Molecular identities remain constant
#   over intervals and switch abruptly, revealing 1D intersections of identity
#   cells. For GDSS, red segments mark invalid outputs (the null identity).
#
# Shares Fig 1's input directory rather than duplicating it.
#
# Analyze (default): plot from shipped data/fig1_sections/. No GPU, no model.
# Re-create: rebuild the inputs from checkpoints — see data/fig1_sections/SOURCE.md.
#
# Path bars color by the raw SMILES string, but the GDSS panel runs under
# tess-gdss (rdkit 2020.09.1), matching how its interpolation log was parsed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

D=data/fig1_sections
O=figures/reproducibility/figS14_paths
mkdir -p "$O"

# --- panels a, b: MolMiner / HierVAE 1D paths ---
activate_env tess-molminer
python scripts/common/plot_path_bar.py --in "$D/molminer_path_seed42.json" --out_dir "$O" --label figS14a_molminer
python scripts/common/plot_path_bar.py --in "$D/hiervae_path_seed7.json"   --out_dir "$O" --label figS14b_hiervae

# --- panel c: GDSS 1D interpolation path ---
activate_env tess-gdss
python scripts/common/plot_path_bar.py --in "$D/gdss_interp_log.json" --out_dir "$O" --label figS14c_gdss

echo "Outputs in $O:"
ls -la "$O"
