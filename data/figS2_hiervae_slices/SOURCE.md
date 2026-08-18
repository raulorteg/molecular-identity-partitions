# figS2_hiervae_slices: data (RE-CREATE ONLY)

Panels: alternate latent-dim pairs, seed 7, ckpt 240000.

## Re-create: from checkpoint, then plot
Requires the model source + checkpoint (see models/SETUP.md), then:
```
bash repro/figS2_hiervae_slices.sh
```
The entrypoint decodes each slice into `data/figS2_hiervae_slices/<tag>/` and plots into
`figures/figS2_hiervae_slices/`. GPU/CPU, slow.
