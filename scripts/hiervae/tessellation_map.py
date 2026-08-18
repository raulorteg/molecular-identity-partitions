"""HierVAE 2D-slice probe — decodes a 200x200 (z_a, z_b) grid greedily, with
the remaining latent dims held fixed at torch.randn(1, latent_size) under
--seed. Produces data/figS11_hiervae_evolution/2d-slices/tessellation_{step}.txt:
one slice for Fig 1b, 48 per-step slices for Fig S11.

The perturbed dims come from --axis_a/--axis_b; the defaults 0 and 1
reproduce the paper probe.

CPU-only: the 40k greedy decodes are bottlenecked by RDKit InChI work, not the
decoder.

Re-create only; see data/figS11_hiervae_evolution/SOURCE.md.
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from _common import add_hiervae_args, set_all_seeds, load_hiervae, compute_inchi


def parse_args():
    p = argparse.ArgumentParser()
    add_hiervae_args(p)
    p.add_argument('--nsample', type=int, default=10000,
                   help='Unused.')
    p.add_argument('--num_evals', type=int, default=200,
                   help='Grid resolution per axis (200 → 200x200 = 40000 cells; reduce for smoke tests)')
    p.add_argument('--axis_a', type=int, default=0,
                   help='Latent dim swept on z_a axis. Default 0 (paper baseline).')
    p.add_argument('--axis_b', type=int, default=1,
                   help='Latent dim swept on z_b axis. Default 1 (paper baseline).')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Output directory. Defaults to ./tessellation_data/.')
    return p.parse_args()


def main():
    args = parse_args()
    model, _ = load_hiervae(args, force_cpu=True)

    # Validate slice spec against loaded latent size.
    L = model.latent_size
    for label, v in (('axis_a', args.axis_a), ('axis_b', args.axis_b)):
        if not (0 <= v < L):
            raise SystemExit('{}: dim {} outside [0, {})'.format(label, v, L))
    if args.axis_a == args.axis_b:
        raise SystemExit('axis_a and axis_b must be distinct latent dims.')

    set_all_seeds(args.seed)
    z_source = torch.randn(1, L)

    z0_vec = np.linspace(-2, 2, args.num_evals)
    z1_vec = np.linspace(-2, 2, args.num_evals)

    step = args.model.split("/")[-1].split(".")[-1]
    out_dir = args.out_dir if args.out_dir else 'tessellation_data'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'tessellation_{step}.txt')

    # Write slice metadata sidecar (matches GDSS / MolMiner SI convention).
    slice_meta = {
        'axis_a': args.axis_a,
        'axis_b': args.axis_b,
        'num_evals': args.num_evals,
        'z0_range': [-2.0, 2.0],
        'z1_range': [-2.0, 2.0],
        'seed': args.seed,
        'latent_size': L,
        'checkpoint_step': step,
    }
    with open(os.path.join(out_dir, 'slice_meta.json'), 'w') as f:
        json.dump(slice_meta, f, indent=2)

    with open(out_path, 'w') as outfile, torch.no_grad():
        outfile.write('z0,z1,smiles,inchi,inchikey_first\n')
        for z0 in tqdm(z0_vec, total=len(z0_vec)):
            for z1 in z1_vec:
                set_all_seeds(args.seed)
                root_vecs = z_source.clone()
                root_vecs[:, [args.axis_a]] = z0
                root_vecs[:, [args.axis_b]] = z1
                try:
                    smiles_list = model.decoder.decode(
                        (root_vecs, root_vecs, root_vecs),
                        greedy=True, max_decode_step=150
                    )
                    for smiles in smiles_list:
                        smi = smiles if smiles is not None else ""
                        inchi_str, ikey = compute_inchi(smi)
                        outfile.write(f"{z0},{z1},{smi},{inchi_str},{ikey}\n")
                except Exception:
                    outfile.write(f"{z0},{z1},,,\n")


if __name__ == "__main__":
    main()
