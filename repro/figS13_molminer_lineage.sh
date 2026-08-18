#!/usr/bin/env bash
# Fig S13 — second MolMiner branching lineage trace, on the (TPSA, MolWt) grid
#   (the Fig 4 trace is on logP×qed; this is the same machinery, different axes).
#
# Analyze (default): plot from the shipped lineage JSONL under
#   data/figS13_molminer_lineage/. No GPU. See SOURCE.md to re-create it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

D=data/figS13_molminer_lineage
O=figures/reproducibility/figS13_molminer_lineage
mkdir -p "$O"
JSONL="$D/molminer_lineages_tpsa_molwt.jsonl"

activate_env tess-molminer
python scripts/molminer/molminer_2d_tree_plot.py --jsonl "$JSONL" --out "$O/2d_tree_plot.png"
python scripts/molminer/molminer_lineage_plot.py --jsonl "$JSONL" --out "$O/lineage.png"

echo "Outputs in $O:"; ls -la "$O"
