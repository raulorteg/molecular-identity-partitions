# GDSS identity balls: analysis inputs

Shipped, analysis-ready. Feeds Fig 3 and Tables 1-2 through the cohesion
suite; see `data/fig3_cohesiveness/SOURCE.md`.

| Path | Content |
|------|---------|
| `identity_sets/ball_NN.pkl` | one pickled `frozenset` of canonical SMILES per ball, NN = 00..29 |
| `identity_sets/centers.npy` | `(30, 492)` ball centers in the flattened prior-noise space |

30 balls. The 492 dimensions are GDSS's prior-noise tensor for the active nodes,
flattened; the Fig 1c section varies two scalar entries of it.

At 97 MB this is the largest of the three: `ball_00` holds 86,649 distinct
molecules, against 1,270 for HierVAE and 37 for MolMiner.

## Analyze (default): no GPU

```bash
bash repro/fig3_cohesiveness.sh     # Fig 3
bash repro/tab1_jaccard.sh          # Table 1
bash repro/tab2_metric_scaling.sh   # Table 2
```

**Parse GDSS SMILES under `tess-gdss` (rdkit 2020.09.1).** These identity sets
were built with that version; rdkit >= 2023 rejects about 13% of GDSS decodes
as invalid, which silently shrinks every set. See
`data/chemical_cohesiveness/gdss_gdssrdkit/SOURCE.md`.

## Re-create: GPU, from checkpoints

Needs the GDSS source + the official upstream checkpoints (`models/SETUP.md`)
and the raw ball decodes, which are multi-GB and not shipped.

```bash
# 1. sample ball centers and points within radius r of each
#    (defaults: --d 492 --n_balls 30 --n_samples 100000 --radius 0.1 --seed 42)
python scripts/gdss/gdss_precompute_balls.py --out_dir mci

# 2. decode every sampled point through the PC sampler (the expensive step)
python scripts/gdss/gdss_decode_ball.py \
    --ball_id <0..29> --idx_start 0 --idx_end 100000 --mci_dir mci

# 3. collapse the decodes into per-ball identity sets
python scripts/cohesion/build_identity_sets.py \
    --balls_dirs mci \
    --output_dir data/gdss/identity_sets --nproc 30
```

Run all three under `tess-gdss`. Step 2 writes
`mci/results/ball_NN_<start>_<end>.csv`, which step 3 matches directly; unlike
MolMiner and HierVAE, GDSS has only the one ball decoder.
