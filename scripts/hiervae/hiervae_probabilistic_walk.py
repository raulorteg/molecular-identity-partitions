"""HierVAE probabilistic latent walk — the Fig 2 flow probe.

Samples N molecules per alpha step with no fixed seed, using greedy=False so
the topology head draws with torch.bernoulli and the fragment head with
torch.multinomial. Source/destination pool draws match walk_hiervae.py.

Writes one row per sample (alpha, sample_idx, smiles, canonical_smiles);
plot_flowplot.py reads only alpha and canonical_smiles.

Re-create only; see data/fig2_flows/SOURCE.md.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import add_hiervae_args, set_all_seeds, load_hiervae


def to_canonical(smi):
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def main():
    p = argparse.ArgumentParser(
        description="HierVAE probabilistic latent walk (greedy=False stochastic decode)."
    )
    add_hiervae_args(p)
    p.add_argument("--out", required=True, type=pathlib.Path,
                   help="Output CSV path.")
    p.add_argument("--alpha_steps", type=int, default=50,
                   help="Number of points along the interpolation.")
    p.add_argument("--n_samples", type=int, default=50,
                   help="Stochastic decodes per alpha step (greedy=False).")
    p.add_argument("--decode_seed", type=int, default=None,
                   help="Seed for the stochastic decode loop. If None "
                        "(default), uses --seed for both endpoint draws and "
                        "decode RNG. When sample-sharding "
                        "across many CPU processes, pass --seed 7 to every "
                        "shard (so they share endpoints z1, z2 = Fig S14b "
                        "baseline) but a distinct --decode_seed per shard "
                        "so the bernoulli/multinomial decode noise is "
                        "independent across shards.")
    p.add_argument("--source_idx", type=int, default=0)
    p.add_argument("--dest_idx", type=int, default=0)
    p.add_argument("--num_sources", type=int, default=100)
    p.add_argument("--num_destinations", type=int, default=500)
    args = p.parse_args()

    model, _ = load_hiervae(args, force_cpu=True)

    # one seed reset for the pool draw, matching walk_hiervae.py
    set_all_seeds(args.seed)
    source_pool = torch.randn(args.num_sources, model.latent_size)
    destination_pool = torch.randn(args.num_destinations, model.latent_size)
    source = source_pool[args.source_idx]
    destination = destination_pool[args.dest_idx]

    # decode_seed defaults to args.seed; distinct values give shards independent draws
    decode_seed = args.decode_seed if args.decode_seed is not None else args.seed
    set_all_seeds(decode_seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "sample_idx", "smiles", "canonical_smiles"])

        with torch.no_grad():
            for alpha in tqdm(np.linspace(0.0, 1.0, args.alpha_steps),
                              desc="alphas", total=args.alpha_steps):
                z = ((1.0 - alpha) * source + alpha * destination).unsqueeze(0)
                # no per-sample seed reset: let the RNG advance
                for s in range(args.n_samples):
                    try:
                        out = model.decoder.decode(
                            (z, z, z), greedy=False, max_decode_step=150)
                        smi = out[0] if out else ""
                    except Exception as e:
                        print(f"  decode failed at alpha={alpha:.4f} sample={s}: {e}",
                              file=sys.stderr)
                        smi = ""
                    writer.writerow([float(alpha), s, smi or "",
                                     to_canonical(smi)])

    print(f"Wrote alpha_steps={args.alpha_steps} × n_samples={args.n_samples} "
          f"= {args.alpha_steps * args.n_samples} rows → {args.out}")


if __name__ == "__main__":
    main()
