# Fig S11: data

Shipped analysis inputs: 48 per-step 2D section slices (one decoded tessellation
slice per HierVAE training step, 5k-240k).

| Path | Content |
|------|---------|
| `2d-slices/tessellation_{step}.txt` | per-step (z0,z1) 2D slice (grid points → SMILES) |

The plotter reads `2d-slices/` relative to the working dir, so
`figS11_hiervae_evolution.sh` runs from this directory.

## Analyze (default): replot, no GPU
```
bash repro/figS11_hiervae_evolution.sh
```

## Re-create: the slices from checkpoints (CPU)
Decode a fixed (z0,z1) slice at each step checkpoint (see models/SETUP.md).
Output is named `tessellation_{step}.txt` from the checkpoint's step, so the
shipped filenames need no renaming.

```
python scripts/hiervae/tessellation_map.py --vocab common/hiervae/vocab.txt \
    --model <ckpt>/model.ckpt.{step} --num_evals 200 --seed 7 \
    --out_dir data/figS11_hiervae_evolution/2d-slices
```
