# Fig 6: data

Shipped analysis inputs: 50 per-epoch 2D section slices (one decoded tessellation
slice per training epoch).

| Path | Content |
|------|---------|
| `2d-slices/last_molminer-2026-tess_{1..50}.pth.txt` | per-epoch fixed-coordinate 2D slice (grid points → SMILES) |

The plotter reads `2d-slices/` relative to the working dir, so `fig6_molminer_evolution.sh`
runs from this directory.

## Analyze (default): replot, no GPU
```
bash repro/fig6_molminer_evolution.sh
```

## Re-create: the slices from checkpoints (GPU/CPU)
Decode a fixed 2D conditioning slice at each epoch checkpoint (see
models/SETUP.md), with the same vocab/stats bundle as
`data/fig1_sections/SOURCE.md`. Output is named after the checkpoint, so the
shipped filenames need no renaming.

```
python scripts/molminer/molminer_2d_slice.py \
    --ckpt_molminer <ckpt>/last_molminer-2026-tess_{epoch}.pth \
    --ckpt_starter ... --vocab_anchors ... --seed 42 \
    --out_dir data/fig6_molminer_evolution/2d-slices
```
