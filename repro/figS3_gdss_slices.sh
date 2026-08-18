#!/usr/bin/env bash
# Fig S3 — GDSS 2D slices across alternative axes of the prior-noise tensor.
#   seed 42, 24 active nodes; six 197x197 slices over different (entry_a, entry_b)
#   pairs of the node-feature (x) and adjacency (adj) noise tensors.
#
# RE-CREATE ONLY. The alt-axis slice data is not shipped, so there is no analyze
# replot — it regenerates from the GDSS checkpoint each run (via the upstream
# `sample_zinc250k` config). Requires the GDSS source + checkpoint; see
# models/SETUP.md. Runs under tess-gdss (rdkit 2020.09.1). Dual-GPU, slow.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

D=data/figS3_gdss_slices
O=figures/reproducibility/figS3_gdss_slices
mkdir -p "$D" "$O"
COMMON=(--n 197 --z0_min -2 --z0_max 2 --z1_min -2 --z1_max 2)

# tag  slice_tensor  axis_a  axis_b
SLICES=(
  "x_a0f0_a0f1  x   0,0 0,1"
  "x_a0f1_a0f2  x   0,1 0,2"
  "x_a0f0_a1f0  x   0,0 1,0"
  "x_a0f0_a1f1  x   0,0 1,1"
  "adj_0-1_0-2  adj 0,1 0,2"
  "adj_0-1_2-3  adj 0,1 2,3"
)

activate_env tess-gdss
for spec in "${SLICES[@]}"; do
  read -r TAG TENSOR A B <<< "$spec"
  echo "=== slice $TAG ($TENSOR: $A x $B) ==="
  python scripts/gdss/gdss_grid_2d.py "${COMMON[@]}" \
      --slice_tensor "$TENSOR" --axis_a "$A" --axis_b "$B" --out_dir "$D/$TAG"
  python scripts/gdss/gdss_plot_grid2d.py --in_dir "$D/$TAG" --out_dir "$O" --label "$TAG"
done

echo "Outputs in $O:"; ls "$O"
