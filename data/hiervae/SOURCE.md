# HierVAE identity balls: analysis inputs

Shipped, analysis-ready. Feeds Fig 3 and Tables 1-2 through the cohesion
suite; see `data/fig3_cohesiveness/SOURCE.md`.

| Path | Content |
|------|---------|
| `identity_sets_pooled/ball_NN.pkl` | one pickled `frozenset` of canonical SMILES per ball, NN = 00..59 |
| `identity_sets_pooled/centers.npy` | `(60, 32)` ball centers in the latent space |

60 balls, not 30: HierVAE pools two seeds, `balls` and `balls_seed43`, 30
each, hence `_pooled` and K=60 against K=30 for the other two. Balls are
renumbered on build, so the first seed keeps 00..29 and the second becomes
30..59; index 30 is the seed boundary. `ball_00` holds 1,270 distinct
molecules.

## Analyze (default): no GPU

```bash
bash repro/fig3_cohesiveness.sh     # Fig 3
bash repro/tab1_jaccard.sh          # Table 1
bash repro/tab2_metric_scaling.sh   # Table 2
```

## Re-create: GPU, from checkpoints

Needs the hgraph2graph source + checkpoint (`models/SETUP.md`) and the raw ball
decodes, which are multi-GB and not shipped.

```bash
# 1. sample ball centers and points within radius r, once per seed
python scripts/hiervae/hiervae_precompute_balls.py --out_dir mci --seed 42
python scripts/hiervae/hiervae_precompute_balls.py --out_dir mci_seed43 --seed 43

# 2. decode every sampled point (the expensive step), once per ball per seed
python scripts/hiervae/hiervae_decode_ball.py \
    --ball_id <0..29> --idx_start 0 --idx_end 100000 --mci_dir mci \
    --vocab common/hiervae/vocab.txt \
    --model checkpoints/hiervae/model.ckpt.240000

# 3. pool both seeds into one 60-ball identity-set cache
python scripts/cohesion/build_identity_sets.py \
    --balls_dirs mci mci_seed43 \
    --output_dir data/hiervae/identity_sets_pooled --nproc 60
```

Step 2 must be `hiervae_decode_ball.py`, the **stochastic** (`greedy=False`)
decoder. The sibling `hiervae_decode_ball_deterministic.py` is the greedy Fig 5
convergence decoder (`data/fig5_convergence/SOURCE.md`); step 3 matches only
`ball_NN_<start>_<end>.csv` or `..._stochastic.csv`.

Passing both `--balls_dirs` in that order produces the 0..29 / 30..59 numbering
the downstream scripts assume.
