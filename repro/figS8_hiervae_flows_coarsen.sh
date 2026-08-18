#!/usr/bin/env bash
# Fig S8 — HierVAE redecoding flows under coarsened equivalence conventions.
#   The Fig 2 HierVAE walk re-bucketed under 5 coarser relations.
#
# Analyze (default): plot from the shipped Fig 2 walk CSV (reused, not duplicated).
# Re-create the data: that walk — see data/fig2_flows/SOURCE.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

CSV=data/fig2_flows/hiervae_walk_seed7.csv.gz
O=figures/reproducibility/figS8_hiervae_flows_coarsen
mkdir -p "$O"
CONVENTIONS=(inchikey murcko murcko_generic formula composition)

activate_env tess-molminer
for CONV in "${CONVENTIONS[@]}"; do
    python scripts/common/plot_flowplot.py --csv "$CSV" \
        --out "$O/walk.png" --convention "$CONV"
done

echo "Outputs in $O:"; ls "$O"
