# figS3_gdss_slices: data (RE-CREATE ONLY)

Panels: alternate prior-noise tensor axes (x and adj), seed 42, 24 active nodes.

## Re-create: from checkpoint, then plot
Requires the model source + checkpoint (see models/SETUP.md), then:
```
bash repro/figS3_gdss_slices.sh
```
The entrypoint decodes each slice into `data/figS3_gdss_slices/<tag>/` and plots into
`figures/figS3_gdss_slices/`. GPU/CPU, slow.
