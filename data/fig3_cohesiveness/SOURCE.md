# Fig 3 (+ Tables 1, 2): data

Cohesion data keeps its model-structured layout, addressed directly by the
codebase, rather than being copied under this dir:

| Path | Content |
|------|---------|
| `data/<arch>/identity_sets*/ball_NN.pkl`        | cached canonical-SMILES sets per identity ball (analyze) |
| `data/chemical_cohesiveness/<arch>/`            | Tanimoto W/A/R arrays, per-ball AUCs, `summary.json`, MACCS twin under `maccs/` (analyze) |
| `data/chemical_cohesiveness/<arch>/jaccard_by_convention.json` | per-convention Jaccard (analyze, feeds Table 1) |

`<arch>` ∈ {molminer, hiervae, gdss}. HierVAE identity sets are pooled across
two seeds (`identity_sets_pooled`, K=60); `--split_at 30` marks the seed boundary.

## Analyze (default): replot, no GPU
```
bash repro/fig3_cohesiveness.sh     # Fig 3 columns + composites
bash repro/tab1_jaccard.sh          # Table 1
bash repro/tab2_metric_scaling.sh   # Table 2
```

## Re-create: from ball decodes
Needs the raw `data/<arch>/balls{,_seed43}/results/ball_*.csv` decodes (see models/SETUP.md), then:
```
# build identity-set, then W/A/R, per arch:
python scripts/cohesion/build_identity_sets.py --balls_dirs data/<arch>/balls --output_dir data/<arch>/identity_sets --nproc 30
python scripts/cohesion/chemical_cohesiveness.py run --arch <Label> --identity_dir data/<arch>/identity_sets --output_dir data/chemical_cohesiveness/<arch> --fingerprint morgan
```
HierVAE pools `data/hiervae/balls data/hiervae/balls_seed43` (30+30=60) with `--split_at 30`.
