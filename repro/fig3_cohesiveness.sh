#!/usr/bin/env bash
# Fig 3 — chemical cohesiveness of identity balls (within/across/random Tanimoto).
# Also writes the per-architecture identity-overlap .npz that Table 1 reads.
#
# Analyze (default): from data/chemical_cohesiveness/<arch>/ and the cached
# identity sets under data/<arch>/identity_sets*/. No GPU.
# Re-create: see data/fig3_cohesiveness/SOURCE.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

O=figures/reproducibility/fig3_cohesiveness
mkdir -p "$O"

activate_env tess-molminer

# --- per-architecture identity-overlap stats (from cached identity sets) ---
# Writes .npz only; tab1_jaccard.sh reads these back for the identity-overlap
# CSVs behind Table 1.
python scripts/cohesion/identity_overlap_stats.py --arch "MolMiner" \
    --identity_dir data/molminer/identity_sets        --out "$O/column_molminer_cohesiveness.npz"
python scripts/cohesion/identity_overlap_stats.py --arch "HierVAE" \
    --identity_dir data/hiervae/identity_sets_pooled  --out "$O/column_hiervae_cohesiveness.npz"
python scripts/cohesion/identity_overlap_stats.py --arch "GDSS" \
    --identity_dir data/gdss/identity_sets            --out "$O/column_gdss_cohesiveness.npz"

# --- Fig 3 itself: the cross-architecture KDE composite ---
INPUT_DIRS=(data/chemical_cohesiveness/molminer data/chemical_cohesiveness/hiervae data/chemical_cohesiveness/gdss)
python scripts/cohesion/plot_chemical_kde_composite.py \
    --input_dirs "${INPUT_DIRS[@]}" --out "$O/chemical_cohesiveness_kde.png"

echo "Outputs in $O:"
ls -la "$O"
