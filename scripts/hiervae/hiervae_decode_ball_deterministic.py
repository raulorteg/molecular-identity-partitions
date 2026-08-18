"""HierVAE deterministic ball decoder, for the convergence experiment.

Decodes a slice of precomputed ball points through HierVAE's greedy decoder
(greedy=True), resetting every RNG to args.seed before each point, so the
decode is a deterministic function of z.

The sibling hiervae_decode_ball.py is the stochastic (greedy=False) decoder
used for the Fig 3 cohesiveness balls.

Writes <mci_dir>/results/ball_{ball_id:02d}_{start}_{end}_deterministic.csv with
columns ball_id, idx, smiles.

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
        description="Decode HierVAE ball points DETERMINISTICALLY (greedy, seed reset per point)."
    )
    p.add_argument("--ball_id", type=int, required=True)
    p.add_argument("--idx_start", type=int, required=True)
    p.add_argument("--idx_end", type=int, required=True)
    p.add_argument("--mci_dir", type=pathlib.Path, required=True)
    p.add_argument("--out_dir", type=pathlib.Path, default=None,
                   help="Override output dir (default: <mci_dir>/results).")
    p.add_argument("--max_decode_step", type=int, default=150)
    add_hiervae_args(p)
    args = p.parse_args()

    results_dir = args.out_dir if args.out_dir is not None else args.mci_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        results_dir
        / f"ball_{args.ball_id:02d}_{args.idx_start}_{args.idx_end}_deterministic.csv"
    )

    ball_path = args.mci_dir / f"ball_{args.ball_id:02d}.npy"
    ball = np.load(ball_path)
    vectors = ball[args.idx_start: args.idx_end]

    # HierVAE decoder is CPU-only; matches tessellation_map.py / walk_hiervae.py.
    model, _ = load_hiervae(args, force_cpu=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ball_id", "idx", "smiles"])

        with torch.no_grad():
            for local_i, vec in enumerate(tqdm(
                    vectors,
                    desc=f"hiervae ball {args.ball_id:02d} [{args.idx_start}:{args.idx_end}] (det)")):
                global_idx = args.idx_start + local_i
                set_all_seeds(args.seed)            # ← deterministic per point (protocol parity)
                z = torch.tensor(vec, dtype=torch.float).unsqueeze(0)
                try:
                    out = model.decoder.decode(
                        (z, z, z), greedy=True, max_decode_step=args.max_decode_step)
                    smi = out[0] if out else ""
                except Exception as e:
                    print(f"  decode failed at idx={global_idx}: {e}", file=sys.stderr)
                    smi = ""
                writer.writerow([args.ball_id, global_idx, smi or ""])

    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
