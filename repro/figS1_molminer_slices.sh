#!/usr/bin/env bash
# Fig S1 — MolMiner 2D slices across alternative property-pair axes.
#   Same anchor as Fig 1a (seed 42, druglike defaults on the 10 unswept
#   properties); six 100x100 slices over different pairs of the 12-dim
#   conditioning vector.
#
# RE-CREATE ONLY. The alt-axis slice data is not shipped, so there is no analyze
# replot for this figure — it regenerates from the MolMiner checkpoint each run.
# Requires the model sources + checkpoints; see models/SETUP.md. GPU/CPU, slow.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

# Checkpoint/vocab roots — override via env if your layout differs (SETUP.md).
CKPT_DIR="${MOLMINER_CKPT_DIR:-checkpoints/molminer}"
COMMON="${MOLMINER_COMMON_DIR:-common/molminer}"
GRID="${GRID:-100}"
D=data/figS1_molminer_slices
O=figures/reproducibility/figS1_molminer_slices
mkdir -p "$D" "$O"

# tag  axis_a axis_b  a_min a_max  b_min b_max
SLICES=(
  "logP_qed            logP qed           1 3      0.6 0.8"
  "logP_TPSA           logP TPSA          1 3      40 100"
  "qed_SAS             qed  SAS           0.6 0.8  2 4"
  "TPSA_molWt          TPSA molWt         40 100   250 450"
  "FractionCSP3_molWt  FractionCSP3 molWt 0.2 0.6  250 450"
  "logP_SAS            logP SAS           1 3      2 4"
)

activate_env tess-molminer
for spec in "${SLICES[@]}"; do
  read -r TAG A B AMIN AMAX BMIN BMAX <<< "$spec"
  echo "=== slice $TAG ($A x $B) ==="
  python scripts/molminer/molminer_2d_slice.py \
      --ckpt_molminer "$CKPT_DIR/best_molminer-2026-tess.pth" \
      --ckpt_starter "$COMMON/best_starter.pth" --ckpt_gmm "$COMMON/gmm_model.pkl" \
      --stats_path "$COMMON/stats.json" \
      --vocab_fragments "$COMMON/vocab_fragments.csv" \
      --vocab_attachments "$COMMON/vocab_attachments.csv" \
      --vocab_anchors "$COMMON/vocab_anchors.csv" \
      --device cpu --seed 42 --axis_a "$A" --axis_b "$B" \
      --"${A}_min" "$AMIN" --"${A}_max" "$AMAX" --"${A}_steps" "$GRID" \
      --"${B}_min" "$BMIN" --"${B}_max" "$BMAX" --"${B}_steps" "$GRID" \
      --out_dir "$D/$TAG"
  python scripts/molminer/molminer_plot_2d_slice.py \
      --csv "$D/$TAG"/*.txt --out_dir "$O" --label "$TAG"
done

echo "Outputs in $O:"; ls "$O"
