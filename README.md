# Molecular Identity Partitions

[![arXiv](https://img.shields.io/badge/arXiv-2608.06956-b31b1b.svg)](https://arxiv.org/abs/2608.06956)
[![MolMiner checkpoints](https://img.shields.io/badge/checkpoints-MolMiner-1682D4.svg)](https://zenodo.org/records/21631560)
[![HierVAE checkpoints](https://img.shields.io/badge/checkpoints-HierVAE-1682D4.svg)](https://zenodo.org/records/21632650)
[![GDSS checkpoints](https://img.shields.io/badge/checkpoints-GDSS%20(upstream)-lightgrey.svg)](https://github.com/harryjo97/GDSS)

Code, data, and figures for *"How Molecular Generative Models Organize Molecular
Identity"* (Ortega-Ochoa et al., 2026). Probes three molecular generators
(**MolMiner**, **HierVAE**, **GDSS**) for how each organizes the space of
molecular identities. Figure numbering follows arXiv:2608.06956v1.

## Two ways to run anything

**Analyze the shipped data**: no GPU, seconds to about an hour depending on the
figure (see the `analyze` column below). The plot-ready inputs are under
`data/<figure>/`. Every entrypoint does this by default and writes PNGs
into `figures/reproducibility/<figure>/`. Tables are written as CSV next to their data:

```bash
bash repro/fig3_cohesiveness.sh    # → figures/reproducibility/fig3_cohesiveness/
bash repro/tab2_metric_scaling.sh  # → tables + figures/reproducibility/tab2_metric_scaling/
```

**Re-create the data**: GPU, hours to days. Regenerates those inputs from the
model checkpoints. Each `data/<figure>/SOURCE.md` gives the exact command and a
rough wall-clock; checkpoints and the raw multi-GB decode dumps are not in the
repo (see [`models/SETUP.md`](models/SETUP.md) and the checkpoint badges above).

Build the environments first: `conda env create -f envs/<name>.yml`.

## Figure index

One entrypoint per artifact, named for its paper number. Each script's header
names the producer modules it calls and the conda env each step needs.

`analyze` is measured wall clock against the shipped data, no GPU, rounded to
the minute; it varies from system to system.

| Paper | Entrypoint (`repro/`) | Analyze | Data (`data/`) |
|-------|----------------------|---------|----------------|
| Fig 1 | `fig1_sections.sh` | <1 min | `fig1_sections/` |
| Fig 2 | `fig2_flows.sh` | 24 min | `fig2_flows/` |
| Tbl 1 | `tab1_jaccard.sh` | 45 min | `fig3_cohesiveness/`, `chemical_cohesiveness/` |
| Fig 3 | `fig3_cohesiveness.sh` | <1 min | `fig3_cohesiveness/`, `chemical_cohesiveness/`, `<arch>/` |
| Tbl 2 | `tab2_metric_scaling.sh` | <1 min | `chemical_cohesiveness/`, `<arch>/` |
| Fig 4 | `fig4_branching.sh` | <1 min | `fig4_branching/` |
| Fig 5 + Tbls S1, S2 | `fig5_convergence.sh` | <1 min | `fig5_convergence/` |
| Fig 6 | `fig6_molminer_evolution.sh` | <1 min | `fig6_molminer_evolution/` |
| Fig S1 | `figS1_molminer_slices.sh` | GPU | not shipped, re-create only |
| Fig S2 | `figS2_hiervae_slices.sh` | GPU | not shipped, re-create only |
| Fig S3 | `figS3_gdss_slices.sh` | GPU | not shipped, re-create only |
| Fig S4 | `figS4_molminer_coarsen.sh` | <1 min | reuses `fig1_sections/` |
| Fig S5 | `figS5_hiervae_coarsen.sh` | 2 min | reuses `fig1_sections/` |
| Fig S6 | `figS6_gdss_coarsen.sh` | <1 min | reuses `fig1_sections/` |
| Fig S7 | `figS7_molminer_flows_coarsen.sh` | 23 min | reuses `fig2_flows/` |
| Fig S8 | `figS8_hiervae_flows_coarsen.sh` | 26 min | reuses `fig2_flows/` |
| Fig S9 | `figS9_gdss_flows_coarsen.sh` | 61 min | reuses `fig2_flows/` |
| Fig S11 | `figS11_hiervae_evolution.sh` | 3 min | `figS11_hiervae_evolution/` |
| Fig S13 | `figS13_molminer_lineage.sh` | <1 min | `figS13_molminer_lineage/` |
| Fig S14 | `figS14_paths.sh` | <1 min | reuses `fig1_sections/` |


## Layout

```
repro/     one entrypoint per paper artifact
scripts/   producer libraries: common/ (equivalence relations), cohesion/
           (Fig 3 + Tables 1-2), and molminer/ hiervae/ gdss/
data/      shipped plot-ready inputs, one dir per figure (+ SOURCE.md each)
figures/   published figures, one dir per figure
  reproducibility/   where re-running the entrypoints writes, so the
                     published figures are never overwritten
envs/      conda environments, one per model + a plotting env
models/    patches and upstream pins needed only to re-create data
```


## Source models

| Model | Upstream code | Checkpoints |
|-------|---------------|-------------|
| MolMiner | [`raulorteg/molminer`](https://github.com/raulorteg/molminer) | [zenodo.org/records/21631560](https://zenodo.org/records/21631560) (retrained in-house) |
| HierVAE | [`wengong-jin/hgraph2graph`](https://github.com/wengong-jin/hgraph2graph) | [zenodo.org/records/21632650](https://zenodo.org/records/21632650) (retrained in-house) |
| GDSS | [`harryjo97/GDSS`](https://github.com/harryjo97/GDSS) | official upstream checkpoints |

Pinned commits and the required patches are in
[`models/SETUP.md`](models/SETUP.md). GDSS-decoded SMILES must be parsed under
`tess-gdss` (rdkit 2020.09.1), the version it was developed on; newer rdkit
rejects about 13% of them.

## License

The material authored here -- `scripts/`, `repro/`, `data/`, `figures/`,
`envs/`, and the docs -- is licensed [CC BY
4.0](https://creativecommons.org/licenses/by/4.0/)

This does **not** extend to the three generative models probed here. MolMiner,
HierVAE and GDSS are third-party software, each under its own license, as are
their checkpoints, the ZINC training data, and the patches under `models/`
(diffs against upstream, governed by the upstream license). See
[`LICENSE`](LICENSE) for the full scope.

## Citation

```bibtex
@misc{ortegaochoa2026molecularidentity,
  title         = {How Molecular Generative Models Organize Molecular Identity},
  author        = {Ortega-Ochoa, Raul and
                   Vegge, Tejs and
                   Bakander, Jens S. and
                   Mantilla Calderon, Luis and
                   Aspuru-Guzik, Alan and
                   Buonassisi, Tonio},
  year          = {2026},
  eprint        = {2608.06956},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.06956}
}
```
