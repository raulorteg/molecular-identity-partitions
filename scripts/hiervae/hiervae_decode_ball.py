"""HierVAE stochastic ball decoder, for the Fig 3 balls.

Decodes a slice of precomputed ball samples through HierVAE's stochastic
decoder (greedy=False), which draws fresh bernoulli/categorical samples from
the topology and label heads, so the same z can decode to different molecules
across calls.

Each (ball_id, idx_start) chunk gets an independent RNG stream derived from
args.seed, so parallel workers do not share a post-seed state.

Writes <mci_dir>/results/ball_{ball_id:02d}_{start}_{end}.csv with columns
ball_id, idx, smiles.

Re-create only; see data/hiervae/SOURCE.md.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _common import add_hiervae_args, set_all_seeds, load_hiervae


def main():
    p = argparse.ArgumentParser(
        description="Decode HierVAE ball points with the stochastic (greedy=False) decoder."
    )
    p.add_argument("--ball_id", type=int, required=True)
    p.add_argument("--idx_start", type=int, required=True)
    p.add_argument("--idx_end", type=int, required=True)
    p.add_argument("--mci_dir", type=pathlib.Path, required=True)
    add_hiervae_args(p)
    args = p.parse_args()

    results_dir = args.mci_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"ball_{args.ball_id:02d}_{args.idx_start}_{args.idx_end}.csv"

    ball_path = args.mci_dir / f"ball_{args.ball_id:02d}.npy"
    ball = np.load(ball_path)
    vectors = ball[args.idx_start: args.idx_end]

    # HierVAE decoder is CPU-only; matches tessellation_map.py and walk_hiervae.py.
    model, _ = load_hiervae(args, force_cpu=True)

    # per-chunk seed: distinct (ball_id, idx_start) chunks get independent streams
    chunk_seed = (args.seed * 1_000_003 + args.ball_id * 1009 + args.idx_start) & 0x7FFF_FFFF
    set_all_seeds(chunk_seed)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ball_id", "idx", "smiles"])

        with torch.no_grad():
            for local_i, vec in enumerate(tqdm(
                    vectors,
                    desc=f"hiervae ball {args.ball_id:02d} [{args.idx_start}:{args.idx_end}]")):
                global_idx = args.idx_start + local_i
                z = torch.tensor(vec, dtype=torch.float).unsqueeze(0)
                try:
                    out = model.decoder.decode((z, z, z), greedy=False, max_decode_step=150)
                    smi = out[0] if out else ""
                except Exception as e:
                    print(f"  decode failed at idx={global_idx}: {e}",
                          file=sys.stderr)
                    smi = ""
                writer.writerow([args.ball_id, global_idx, smi or ""])

    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
