# figS1_molminer_slices: data (RE-CREATE ONLY)

Panels: property-pair axes (logP_qed, logP_TPSA, qed_SAS, TPSA_molWt, FractionCSP3_molWt, logP_SAS), seed 42.

## Re-create: from checkpoint, then plot
Requires the model source + checkpoint (see models/SETUP.md), then:
```
bash repro/figS1_molminer_slices.sh
```
The entrypoint decodes each slice into `data/figS1_molminer_slices/<tag>/` and plots into
`figures/figS1_molminer_slices/`. GPU/CPU, slow.
