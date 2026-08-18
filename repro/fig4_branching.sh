#!/usr/bin/env bash
# Fig 4 — autoregressive branching of MolMiner decoding lineages.
#   2D tessellation tree (8 decode steps), the same with lineage overlays, and
#   the lineage decoding DAG.
#
# Analyze (default): plot from the shipped lineage JSONL under
#   data/fig4_branching/. No GPU, no model. See SOURCE.md to re-create it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

D=data/fig4_branching
O=figures/reproducibility/fig4_branching
mkdir -p "$O"
JSONL="$D/molminer_lineages.jsonl"

activate_env tess-molminer

# Combined 8-step tessellation tree (lineage fills are intrinsic to this view).
python scripts/molminer/molminer_2d_tree_plot.py --jsonl "$JSONL" --out "$O/2d_tree_plot.png"
# Lineage decoding DAG.
python scripts/molminer/molminer_lineage_plot.py --jsonl "$JSONL" --out "$O/lineage.png"

echo "Outputs in $O:"
ls -la "$O"
