# figS4_molminer_coarsen: data

This figure re-quotients an existing input under 5 coarser equivalence
conventions (InChIKey, Murcko, Murcko-generic, formula, composition); it ships
no data of its own and reuses the **Fig 1a MolMiner slice** under `data/fig1_sections/`.

## Analyze (default): replot, no GPU
```
bash repro/figS4_molminer_coarsen.sh
```

## Re-create the data
Regenerate the reused input: see `data/fig1_sections/SOURCE.md`.
