# Fig S13: data

Shipped analysis input: a second MolMiner decode-lineage JSONL, on the
(TPSA, MolWt) property grid. Same machinery as Fig 4, different axes.

| File | Content |
|------|---------|
| `molminer_lineages_tpsa_molwt.jsonl` | greedy-rollout decode tree over the TPSA×MolWt grid |

## Analyze (default): replot, no GPU
```
bash repro/figS13_molminer_lineage.sh
```

## Re-create: from checkpoints (GPU/CPU)
```
python scripts/molminer/molminer_tree_tessellation.py \
    --ckpt_molminer ... --ckpt_starter ... --ckpt_gmm ... --stats_path ... \
    --vocab_fragments ... --vocab_attachments ... --vocab_anchors ... \
    --x_prop TPSA --y_prop molWt --out_jsonl molminer_lineages_tpsa_molwt.jsonl
```
See models/SETUP.md for checkpoint/vocab locations.
