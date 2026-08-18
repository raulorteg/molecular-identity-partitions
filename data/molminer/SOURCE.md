# MolMiner identity balls: analysis inputs

Shipped, analysis-ready. Feeds Fig 3 and Tables 1-2 through the cohesion
suite; see `data/fig3_cohesiveness/SOURCE.md`.

| Path | Content |
|------|---------|
| `identity_sets/ball_NN.pkl` | one pickled `frozenset` of canonical SMILES per ball, NN = 00..29 |
| `identity_sets/centers.npy` | `(30, 12)` ball centers in the scaled conditioning space |

30 balls. The 12 dimensions are MolMiner's physicochemical conditioning
vector, the first two of which are logP and QED (the Fig 1a axes). `ball_00`
holds 37 distinct molecules, against 1,270 for HierVAE and 86,649 for GDSS.

## Analyze (default): no GPU

```bash
bash repro/fig3_cohesiveness.sh     # Fig 3
bash repro/tab1_jaccard.sh          # Table 1
bash repro/tab2_metric_scaling.sh   # Table 2
```

## Re-create: GPU, from checkpoints

Needs the MolMiner source + checkpoint (`models/SETUP.md`) and the raw ball
decodes, which are multi-GB and not shipped.

```bash
# 1. sample ball centers by log-density and points within radius r of each
#    (defaults: --n_balls 30 --n_samples 100000 --radius 1.0 --seed 42)
python scripts/molminer/molminer_precompute_balls.py \
    --ckpt_gmm common/molminer/gmm_model.pkl --out_dir mci

# 2. decode every sampled point (the expensive step), once per ball
python scripts/molminer/molminer_decode_ball_stochastic.py \
    --ball_id <0..29> --idx_start 0 --idx_end 100000 --mci_dir mci --device cuda \
    --ckpt_molminer checkpoints/molminer/best_molminer-2026-tess.pth \
    --ckpt_starter common/molminer/best_starter.pth \
    --ckpt_gmm common/molminer/gmm_model.pkl \
    --stats_path common/molminer/stats.json \
    --vocab_fragments common/molminer/vocab_fragments.csv \
    --vocab_attachments common/molminer/vocab_attachments.csv \
    --vocab_anchors common/molminer/vocab_anchors.csv

# 3. collapse the decodes into per-ball identity sets
python scripts/cohesion/build_identity_sets.py \
    --balls_dirs mci \
    --output_dir data/molminer/identity_sets --nproc 30
```

Step 2 must be the **stochastic** decoder; the sibling
`molminer_decode_ball_deterministic.py` belongs to the Fig 5 convergence
experiment (`data/fig5_convergence/SOURCE.md`).

Step 2 writes `mci/results/ball_NN_<start>_<end>_stochastic.csv`. Point
`--balls_dirs` at the directory holding `results/`; step 3 matches only
`ball_NN_<start>_<end>.csv` or `..._stochastic.csv`, so `_deterministic` chunks
are rejected with `no chunks for ball 0`.
