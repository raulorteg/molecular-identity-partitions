#!/usr/bin/env bash
# Fig 2 — stochastic redecoding flows along a probabilistic walk.
#   How the identity mass redistributes across α-steps when the same latent
#   path is decoded many times (MolMiner, HierVAE, GDSS), canonical convention.
#
# Analyze (default): plot from the shipped walk CSVs under data/fig2_flows/.
#   No GPU, no model. See data/fig2_flows/SOURCE.md to re-create it.
# The 6-convention sweep is split out into the S7-S9 entrypoints.
#
# GDSS runs under tess-gdss (rdkit 2020.09.1); its decoded SMILES include
# valence-expanded constructs newer rdkit rejects.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

D=data/fig2_flows
O=figures/reproducibility/fig2_flows
# Per-model subdirs: the flow plotter writes generic pie_{start,end}.png, so
# separate dirs keep the three models' pies from overwriting each other.
mkdir -p "$O/molminer" "$O/hiervae" "$O/gdss"

# --- MolMiner + HierVAE panels (newer rdkit parses their in-distribution SMILES) ---
activate_env tess-molminer
python scripts/common/compute_class_counts.py --data "$D/molminer_walk_seed42.csv.gz" \
    --out_csv "$O/molminer/class_counts.csv" --conventions canonical
python scripts/common/plot_flowplot.py --csv "$D/molminer_walk_seed42.csv.gz" \
    --out "$O/molminer/walk.png"
python scripts/common/compute_class_counts.py --data "$D/hiervae_walk_seed7.csv.gz" \
    --out_csv "$O/hiervae/class_counts.csv" --conventions canonical
python scripts/common/plot_flowplot.py --csv "$D/hiervae_walk_seed7.csv.gz" \
    --out "$O/hiervae/walk.png"

# --- GDSS panel (rdkit 2020.09.1) ---
activate_env tess-gdss
python scripts/common/compute_class_counts.py --data "$D/gdss_walk_seed42.csv.gz" \
    --out_csv "$O/gdss/class_counts.csv" --conventions canonical
python scripts/common/plot_flowplot.py --csv "$D/gdss_walk_seed42.csv.gz" \
    --out "$O/gdss/walk.png"

echo "Outputs in $O:"
ls -la "$O"
