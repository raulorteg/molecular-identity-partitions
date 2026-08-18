#!/usr/bin/env bash
# Fig S2 — HierVAE 2D slices across alternative latent-dim pairs.
#   Same anchor as Fig 1b (seed 7, checkpoint 240000, ~40k samples); probes
#   alternate (z_i, z_j) pairs with the remaining latent dims held fixed.
#
# RE-CREATE ONLY. The alt-axis slice data is not shipped, so there is no analyze
# replot — it regenerates from the HierVAE checkpoint each run. Requires the
# model sources + checkpoint; see models/SETUP.md. CPU, slow.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

CKPT="${HIERVAE_CKPT:-checkpoints/hiervae/model.ckpt.240000}"
VOCAB="${HIERVAE_VOCAB:-common/hiervae/vocab.txt}"
D=data/figS2_hiervae_slices
O=figures/reproducibility/figS2_hiervae_slices
mkdir -p "$D" "$O"

# alternate latent-dim pairs (axis_a axis_b)
PAIRS=("0 1" "0 2" "1 2" "0 3" "1 3" "2 3")

activate_env tess-hiervae
for pair in "${PAIRS[@]}"; do
  read -r A B <<< "$pair"
  TAG="z${A}_z${B}"
  echo "=== slice $TAG ==="
  python scripts/hiervae/tessellation_map.py \
      --vocab "$VOCAB" --model "$CKPT" --num_evals 200 --seed 7 \
      --axis_a "$A" --axis_b "$B" --out_dir "$D/$TAG"
  python scripts/hiervae/plot_tessellation_final.py \
      --csv "$D/$TAG"/*.txt --out_dir "$O" --label "$TAG"
done

echo "Outputs in $O:"; ls "$O"
