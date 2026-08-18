# Fig 2 (+ S7-S9): data

Shipped analysis inputs: the merged probabilistic-walk CSVs (one decoded SMILES
per row, ~1.0M rows = 200 α-steps × ~5120 redecodes).

| File | Model |
|------|-------|
| `molminer_walk_seed42.csv.gz` | MolMiner |
| `hiervae_walk_seed7.csv.gz`   | HierVAE  |
| `gdss_walk_seed42.csv.gz`     | GDSS     |

## Analyze (default): replot, no GPU
```
bash repro/fig2_flows.sh              # canonical-convention flows (Fig 2)
bash repro/figS7_molminer_flows_coarsen.sh   # + S8, S9: 6-convention sweep
```

## Re-create: the walks (GPU/CPU, hours)
Decode the probabilistic walk from each model checkpoint (see models/SETUP.md):
```
python scripts/molminer/probabilistic_walk.py     ...   # MolMiner
python scripts/hiervae/hiervae_probabilistic_walk.py ... # HierVAE
python scripts/gdss/gdss_probabilistic_walk.py    ...   # GDSS
```
Each emits per-shard CSVs; concatenate and gzip them to the filenames above.
