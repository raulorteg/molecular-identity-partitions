#!/usr/bin/env bash
# Fig S4 — MolMiner 2D section under coarsened equivalence conventions.
#   The Fig 1a slice re-quotiented under 5 progressively coarser relations
#   (InChIKey, Murcko, Murcko-generic, formula, composition).
#
# Analyze (default): plot from the shipped Fig 1 slice (reused, not duplicated).
# Re-create the data: that slice — see data/fig1_sections/SOURCE.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

CSV=data/fig1_sections/molminer_2dslice_seed42.txt
O=figures/reproducibility/figS4_molminer_coarsen
mkdir -p "$O"
CONVENTIONS=(inchikey murcko murcko_generic formula composition)

activate_env tess-molminer
for CONV in "${CONVENTIONS[@]}"; do
    python scripts/molminer/molminer_plot_2d_slice.py --csv "$CSV" \
        --out_dir "$O" --label molminer_slice --convention "$CONV"
done
python scripts/common/compute_class_counts.py --data "$CSV" --out_csv "$O/class_counts.csv"

echo "Outputs in $O:"; ls "$O"
