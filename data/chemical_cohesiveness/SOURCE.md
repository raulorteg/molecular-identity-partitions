# Chemical cohesiveness: analysis outputs

Shipped, plot-ready. The computed layer over the identity balls in
`data/molminer/`, `data/hiervae/` and `data/gdss/`. Fig 3 and Tables 1-2 read
from here.

## Per architecture

`<arch>` is one of `molminer/`, `hiervae/`, `gdss/`.

| Path | Content |
|------|---------|
| `<arch>/W.npy`, `A.npy`, `R.npy` | pooled Tanimoto similarities: within-ball, across-ball, random baseline |
| `<arch>/W_per_ball.npy`, `A_per_ball.npy` | the same, kept per ball |
| `<arch>/per_ball_AUC.npy`, `per_ball_AUC_WR.npy` | per-ball AUC(W,A) and AUC(W,R) |
| `<arch>/summary.json` | headline medians, quartiles and the three pooled AUCs (feeds Fig 3) |
| `<arch>/per_ball.csv` | per-ball table exported for the SI |
| `<arch>/jaccard_by_convention.json` | median Jaccard under each of the six conventions (feeds Table 1) |
| `<arch>/pairwise_tanimoto_distance.{csv,json}` | per-pair Euclidean vs chemical distance, and the R^2 / Spearman rho headline (feeds Table 2) |
| `<arch>/fingerprints/ball_NN.pkl` | the subsampled canonical-SMILES lists that were fingerprinted |
| `<arch>/maccs/` | the same outputs recomputed on MACCS keys |

## Cross-architecture tables

| File | Paper artifact |
|------|----------------|
| `JACCARD_BY_CONVENTION.csv` | Table 1 |
| `CHEMICAL_COHESION_TWIN_TABLES.csv` | Table 2 (ECFP primary + MACCS twin) |
| `CROSS_ARCH_SUMMARY.csv` | SI summary across architectures |
| `GDSS_RDKIT_COMPARISON_{jaccard,chemistry}.csv` | the rdkit-version comparison described below |

## `gdss_gdssrdkit/`: the rdkit-version control

Same layout as `gdss/`, but the GDSS SMILES are re-fingerprinted under rdkit
2020.09.1 (`tess-gdss`) instead of the modern rdkit used for the cross-
architecture numbers. Newer rdkit rejects about 13% of GDSS decodes as invalid.
This is a control, not a fourth architecture.

## Analyze (default): no GPU

```bash
bash repro/fig3_cohesiveness.sh     # Fig 3
bash repro/tab1_jaccard.sh          # Table 1
bash repro/tab2_metric_scaling.sh   # Table 2
```

Both table entrypoints rewrite files here in place, byte-identically: a clean
`git status` afterwards is the check that your numbers match the shipped ones.

## Re-create: from the identity sets

No GPU, but hours. Needs `data/<arch>/identity_sets*/`; see each
architecture's SOURCE.md.

```bash
# per architecture, ECFP primary then the MACCS twin
python scripts/cohesion/chemical_cohesiveness.py run \
    --arch MolMiner \
    --identity_dir data/molminer/identity_sets \
    --output_dir data/chemical_cohesiveness/molminer \
    --fingerprint morgan
python scripts/cohesion/chemical_cohesiveness.py run \
    --arch MolMiner \
    --identity_dir data/molminer/identity_sets \
    --output_dir data/chemical_cohesiveness/molminer/maccs \
    --fingerprint maccs
```

Repeat with `--arch HierVAE --identity_dir data/hiervae/identity_sets_pooled`
and `--arch GDSS --identity_dir data/gdss/identity_sets`. Build
`gdss_gdssrdkit/` the same way from the GDSS identity sets, but under
`tess-gdss`. Then run the three entrypoints above to rebuild the tables.

The shipped pooled AUCs are 0.8426 (MolMiner), 0.8684 (HierVAE) and 0.5234
(GDSS), computed on 10,000 sampled pairs per ball. Re-running the probe does not
reproduce them exactly. Expect agreement to about the third decimal, not
byte-identical output.
