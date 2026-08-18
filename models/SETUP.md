# Re-creation setup

Only needed to re-create the raw data from checkpoints. Analyzing the shipped
data needs none of this.

Clone the three upstream models as siblings of this repo, at these commits:

| Model    | Repo                                        | Commit    |
|----------|---------------------------------------------|-----------|
| MolMiner | github.com/raulorteg/molminer               | `e2ab283` |
| HierVAE  | github.com/wengong-jin/hgraph2graph         | `e396dba` |
| GDSS     | github.com/harryjo97/GDSS                   | `24cc490` |

Each is third-party software under its own license; see [`LICENSE`](../LICENSE).

```bash
git clone https://github.com/harryjo97/GDSS.git           ../GDSS         && git -C ../GDSS         checkout 24cc490
git clone https://github.com/wengong-jin/hgraph2graph.git ../hgraph2graph && git -C ../hgraph2graph checkout e396dba
git clone https://github.com/raulorteg/molminer.git       ../molminer     && git -C ../molminer     checkout e2ab283
```

Apply the patches:

```bash
git -C ../hgraph2graph apply models/hiervae.patch
git -C ../GDSS         apply models/gdss.patch
```

`hiervae.patch` runs `hgraph/` on CPU (the decoder holds CPU-only state). `gdss.patch` gives the PC
sampler optional `initial_x`/`initial_adj`, so the interpolation probe can fix
the initial noise.

Create the environments:

```bash
conda env create -f envs/tess-molminer.yml
conda env create -f envs/tess-hiervae.yml --solver=libmamba
conda env create -f envs/tess-gdss.yml
conda env create -f envs/tess-plot.yml
```

Checkpoints are not shipped in this repo. The two retrained in-house models are archived on Zenodo; GDSS uses the official upstream
checkpoints.

| Model    | Checkpoints                                                      |
|----------|------------------------------------------------------------------|
| MolMiner | https://zenodo.org/records/21631560                              |
| HierVAE  | https://zenodo.org/records/21632650                              |
| GDSS     | official upstream (see the GDSS repo above)                      |

Unpack them so the re-create commands in each `data/*/SOURCE.md` resolve:

```
checkpoints/molminer/best_molminer-2026-tess.pth   common/molminer/best_starter.pth
checkpoints/hiervae/model.ckpt.240000              common/molminer/gmm_model.pkl
                                                   common/molminer/stats.json
                                                   common/molminer/vocab_{fragments,attachments,anchors}.csv
                                                   common/hiervae/vocab.txt
```

Flag names differ per model, MolMiner takes a `--ckpt_*`/`--vocab_*`
bundle, HierVAE takes `--vocab` and `--model`. Override the defaults with the
`MOLMINER_CKPT_DIR`, `MOLMINER_COMMON_DIR`, `HIERVAE_CKPT` and `HIERVAE_VOCAB`
environment variables if your layout differs.

GDSS-decoded SMILES must be parsed under `tess-gdss` (rdkit 2020.09.1); newer
rdkit rejects ~13% of them.
