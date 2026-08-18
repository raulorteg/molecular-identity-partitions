#!/usr/bin/env bash
# Fig 5 (+ Tables S1/S2) — local-tessellation cohesiveness across training.
#   Matched deterministic balls (r=0.5) re-decoded at a ladder of checkpoints;
#   tracks within/across-ball Tanimoto, AUC(W,A), Jaccard, and n_unique as the
#   model trains. Top row MolMiner, bottom row HierVAE.
#
# Analyze (default): plot from the shipped compact caches under
#   data/fig5_convergence/<model>/{cohesiveness_cache,jaccard_cache}. No GPU.
#   See data/fig5_convergence/SOURCE.md to re-create it (decode + cache build).
#
# MolMiner is restricted to the K=20 balls that decode completely at every
# checkpoint; HierVAE uses all K=30. The caches already encode that selection,
# but --balls is passed so a cache rebuild reproduces it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

O=figures/reproducibility/fig5_convergence
mkdir -p "$O/molminer" "$O/hiervae"

MOLMINER_K20="0,1,4,5,6,7,8,9,11,12,15,16,17,18,20,22,24,27,28,29"

activate_env tess-molminer

# --- MolMiner (K=20, epochs) ---
python scripts/molminer/molminer_ball_cohesiveness_evolution.py \
    --in_dir   data/fig5_convergence/molminer \
    --coh_cache data/fig5_convergence/molminer/cohesiveness_cache \
    --jac_cache data/fig5_convergence/molminer/jaccard_cache \
    --balls "$MOLMINER_K20" --model_name MolMiner \
    --fig   "$O/molminer/evolution.png" \
    --table "$O/molminer/ball_convergence_table.csv"

# --- HierVAE (K=30, training steps) ---
python scripts/molminer/molminer_ball_cohesiveness_evolution.py \
    --in_dir   data/fig5_convergence/hiervae \
    --coh_cache data/fig5_convergence/hiervae/cohesiveness_cache \
    --jac_cache data/fig5_convergence/hiervae/jaccard_cache \
    --model_name HierVAE --x_label step --x_sci \
    --fig   "$O/hiervae/evolution.png" \
    --table "$O/hiervae/ball_convergence_table.csv"

echo "Outputs in $O:"; ls -R "$O"
