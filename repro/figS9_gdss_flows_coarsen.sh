#!/usr/bin/env bash
# Fig S9 — GDSS redecoding flows under coarsened equivalence conventions.
#   The Fig 2 GDSS walk re-bucketed under 5 coarser relations.
#
# Analyze (default): plot from the shipped Fig 2 walk CSV (reused, not duplicated).
# Re-create the data: that walk — see data/fig2_flows/SOURCE.md.
# Runs under tess-gdss (rdkit 2020.09.1). SLOW: ~1.5 h for the 5-convention
# sweep over the 1.0M-row GDSS walk.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

CSV=data/fig2_flows/gdss_walk_seed42.csv.gz
O=figures/reproducibility/figS9_gdss_flows_coarsen
mkdir -p "$O"
CONVENTIONS=(inchikey murcko murcko_generic formula composition)

activate_env tess-gdss
for CONV in "${CONVENTIONS[@]}"; do
    python scripts/common/plot_flowplot.py --csv "$CSV" \
        --out "$O/walk.png" --convention "$CONV"
done

echo "Outputs in $O:"; ls "$O"
