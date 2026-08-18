# Fig 1 (+ Fig S14): analysis inputs

Shipped, plot-ready. `repro/fig1_sections.sh` replots the Fig 1 sections and
`repro/figS14_paths.sh` the Fig S14 paths, both without a GPU. Figs S4-S6
re-quotient the same three slices under coarser conventions.

| File | Panel | Content |
|------|-------|---------|
| `molminer_2dslice_seed42.txt`      | 1a  | MolMiner 2D conditioning-vector slice, seed 42 |
| `hiervae_2dslice_step240000.txt`   | 1b  | HierVAE (z0,z1) slice at training step 240000 |
| `gdss_grid2d/grid_log.json`        | 1c  | GDSS 2D prior-noise grid |
| `molminer_path_seed42.json`        | S14a | MolMiner deterministic walk, 200 α-steps, seed 42 |
| `hiervae_path_seed7.json`          | S14b | HierVAE deterministic walk, 200 α-steps, seed 7 |
| `gdss_interp_log.json`             | S14c | GDSS latent interpolation between two noise samples |

## Re-create: regenerate from checkpoints

Requires the model sources + checkpoints (see `models/SETUP.md`).

MolMiner needs its full checkpoint + vocab bundle, abbreviated `$CK`/`$CO` below
(`checkpoints/molminer/`, `common/molminer/`):

```bash
CK=checkpoints/molminer; CO=common/molminer
MM="--ckpt_molminer $CK/best_molminer-2026-tess.pth --ckpt_starter $CO/best_starter.pth \
    --ckpt_gmm $CO/gmm_model.pkl --stats_path $CO/stats.json \
    --vocab_fragments $CO/vocab_fragments.csv --vocab_attachments $CO/vocab_attachments.csv \
    --vocab_anchors $CO/vocab_anchors.csv"

# S14a/S14b  paths
python scripts/molminer/walk_molminer.py $MM --seed 42 \
    --out data/fig1_sections/molminer_path_seed42.json
python scripts/hiervae/walk_hiervae.py --vocab common/hiervae/vocab.txt \
    --model checkpoints/hiervae/model.ckpt.240000 --seed 7 \
    --out data/fig1_sections/hiervae_path_seed7.json
# S14c  GDSS interpolation
python scripts/gdss/gdss_interpolate.py --out_dir data/fig1_sections   # writes gdss_interp_log.json

# 1a/1b  2D sections. Both auto-name their output inside --out_dir; the shipped
# files were renamed by hand (arrows below). Default axes are the paper's.
python scripts/molminer/molminer_2d_slice.py $MM --seed 42 \
    --out_dir data/fig1_sections     # best_molminer-2026-tess.pth.txt -> molminer_2dslice_seed42.txt
python scripts/hiervae/tessellation_map.py --vocab common/hiervae/vocab.txt \
    --model checkpoints/hiervae/model.ckpt.240000 --num_evals 200 \
    --out_dir data/fig1_sections     # tessellation_240000.txt -> hiervae_2dslice_step240000.txt
# 1c  GDSS 2D grid
python scripts/gdss/gdss_grid_2d.py --out_dir data/fig1_sections/gdss_grid2d   # writes grid_log.json
```
