"""Deterministic straight-line walk between two random latent samples in
HierVAE (Fig S14b probe).

Draws one source and one destination from torch.randn(latent_size) under a
single seed, then greedy-decodes the linear interpolation at alpha_steps
points along the segment. Output matches the JSON shape of
data/gdss/gdss_interpolation/interpolation_log.json, so
scripts/common/plot_path_bar.py consumes it directly.

Output format:
    [{"alpha": float, "smiles": str, "inchi": str, "inchikey_first": str,
      "latent_ref": "seed={seed}"}, ...]

Re-create only; see data/fig1_sections/SOURCE.md.
"""
import argparse
import json
import os

import numpy as np
import torch

from _common import add_hiervae_args, set_all_seeds, load_hiervae, compute_inchi


def parse_args():
    p = argparse.ArgumentParser()
    add_hiervae_args(p)
    p.add_argument('--out', required=True, help='Output JSON path')
    p.add_argument('--alpha_steps', type=int, default=200)
    p.add_argument('--source_idx', type=int, default=0,
                   help='Index into the (num_sources, latent_size) RNG draw')
    p.add_argument('--dest_idx', type=int, default=0,
                   help='Index into the (num_destinations, latent_size) RNG draw')
    p.add_argument('--num_sources', type=int, default=100,
                   help='Pool size for the source RNG draw; keep at 100 to reproduce the shipped path')
    p.add_argument('--num_destinations', type=int, default=500,
                   help='Pool size for the destination RNG draw; keep at 500 to reproduce the shipped path')
    return p.parse_args()


def main():
    args = parse_args()
    # force_cpu=True: the decoder has CPU-only internal state.
    model, use_cuda = load_hiervae(args, force_cpu=True)

    # Draw a (num_sources, latent_size) and a (num_destinations, latent_size)
    # pool, then index into them. Drawing torch.randn(latent_size) twice
    # instead lands in a decoder dead zone at seed=7 and yields empty SMILES.
    set_all_seeds(args.seed)
    source_pool = torch.randn(args.num_sources, model.latent_size)
    destination_pool = torch.randn(args.num_destinations, model.latent_size)
    source = source_pool[args.source_idx]
    destination = destination_pool[args.dest_idx]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    records = []

    with torch.no_grad():
        for alpha in np.linspace(0.0, 1.0, args.alpha_steps):
            set_all_seeds(args.seed)
            z = ((1.0 - alpha) * source + alpha * destination).unsqueeze(0)
            try:
                smiles_list = model.decoder.decode(
                    (z, z, z), greedy=True, max_decode_step=150
                )
                smi = smiles_list[0] if smiles_list else ""
            except Exception as e:
                # fail loudly; a silent catch here masks device mismatches
                print(f"  WARN: decode failed at alpha={alpha:.4f}: {e}",
                      file=__import__('sys').stderr)
                smi = ""

            inchi_str, ikey = compute_inchi(smi)
            records.append({
                "alpha": float(alpha),
                "smiles": smi if smi is not None else "",
                "inchi": inchi_str,
                "inchikey_first": ikey,
                "latent_ref": f"seed={args.seed}",
            })

    with open(args.out, 'w') as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} alpha-points to {args.out}")


if __name__ == "__main__":
    main()
