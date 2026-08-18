# Fig 5 (+ Tables S1/S2): data

Shipped analysis inputs: the compact per-checkpoint cohesiveness/Jaccard caches.
The figure and tables render straight from these (no GPU, no decode).

| Path | Model | Content |
|------|-------|---------|
| `molminer/cohesiveness_cache/*.npz` | MolMiner | W/A/R Tanimoto per checkpoint, K=20 balls |
| `molminer/jaccard_cache/*.json`     | MolMiner | per-convention Jaccard per checkpoint |
| `hiervae/cohesiveness_cache/*.npz`  | HierVAE  | W/A/R Tanimoto per checkpoint, K=30 balls |
| `hiervae/jaccard_cache/*.json`      | HierVAE  | per-convention Jaccard per checkpoint |

MolMiner is the K=20 subset of balls that decode completely at every
checkpoint: `0,1,4,5,6,7,8,9,11,12,15,16,17,18,20,22,24,27,28,29`. HierVAE uses
all K=30 at r=0.5.

## Analyze (default): replot, no GPU
```
bash repro/fig5_convergence.sh
```
Regenerates `evolution.png` + `ball_convergence_table.csv` per model.

## Re-create: caches from checkpoint decodes (GPU/CPU, hours-days)
Decode the r=0.5 deterministic balls at each checkpoint (see models/SETUP.md),
producing `<in_dir>/<ckpt>/ball_*_*_*_deterministic.csv`, then let the evolution
script build the caches:
```
python scripts/molminer/molminer_ball_cohesiveness_evolution.py \
    --in_dir <decodes> --coh_cache <out>/cohesiveness_cache --jac_cache <out>/jaccard_cache \
    --balls 0,1,4,5,6,7,8,9,11,12,15,16,17,18,20,22,24,27,28,29 --model_name MolMiner \
    --fig <fig.png> --table <table.csv>
```
Precompute/decode: `scripts/molminer/molminer_precompute_balls.py`,
`molminer_decode_ball_deterministic.py`; HierVAE: `hiervae_precompute_balls.py`,
`hiervae_decode_ball_deterministic.py`.
