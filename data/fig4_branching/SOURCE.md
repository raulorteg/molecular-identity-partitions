# Fig 4 (+ S13): data

Shipped analysis input: the MolMiner decode-lineage JSONL (2525 records, one per
(logP, qed) grid point, with coordinates and the full intermediate decode
rollout).

| File | Content |
|------|---------|
| `molminer_lineages.jsonl` | greedy-rollout decode tree over the logP×qed grid |

## Analyze (default): replot, no GPU
```
bash repro/fig4_branching.sh              # 2D tree + lineages + DAG (Fig 4)
bash repro/figS13_molminer_lineage.sh     # extra lineage trace (S13)
```

## Re-create: the lineages from checkpoints (GPU/CPU)
```
python scripts/molminer/molminer_tree_tessellation.py \
    --ckpt_molminer ... --ckpt_starter ... --ckpt_gmm ... \
    --stats_path ... --vocab_fragments ... --vocab_attachments ... --vocab_anchors ... \
    --logP_min 1.0 --logP_max 3.0 --logP_steps 5 --qed_min 0.6 --qed_max 0.8 --qed_steps 5
```
See models/SETUP.md for checkpoint/vocab locations.
