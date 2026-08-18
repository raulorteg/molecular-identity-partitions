"""Decode a slice of precomputed ball samples deterministically: greedy
MolMiner, weighted starter, seed reset before every sample.

Same pipeline as molminer_decode_ball_stochastic.py, except set_seed(seed) runs
before each point and seed=seed is passed into _sample, so reruns match
bit-for-bit and every (ball-point, model) pair maps to one molecule. This is the
Fig 5 convergence probe.

Writes mci/results/ball_{ball_id:02d}_{start}_{end}_deterministic.csv.

Re-create only; see data/molminer/SOURCE.md.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")
from molminer.generator import MolecularGenerator
from molminer.utils import set_seed


def main():
    p = argparse.ArgumentParser(
        description="Decode ball points deterministically (seed reset per point)."
    )
    p.add_argument("--ball_id", type=int, required=True)
    p.add_argument("--idx_start", type=int, required=True)
    p.add_argument("--idx_end", type=int, required=True)
    p.add_argument("--seed", type=int, default=42,
                   help="Reset global RNG to this and pass into _sample before EVERY point.")
    p.add_argument("--ckpt_molminer", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_starter", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_gmm", required=True, type=pathlib.Path)
    p.add_argument("--stats_path", required=True, type=pathlib.Path)
    p.add_argument("--vocab_fragments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_attachments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_anchors", required=True, type=pathlib.Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mci_dir", type=pathlib.Path, default=pathlib.Path("mci"))
    p.add_argument("--out_dir", type=pathlib.Path, default=None,
                   help="Override output dir (default: <mci_dir>/results).")
    args = p.parse_args()

    results_dir = args.out_dir if args.out_dir is not None else args.mci_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        results_dir
        / f"ball_{args.ball_id:02d}_{args.idx_start}_{args.idx_end}_deterministic.csv"
    )

    ball_path = args.mci_dir / f"ball_{args.ball_id:02d}.npy"
    ball = np.load(ball_path)
    vectors = ball[args.idx_start : args.idx_end]

    gen = MolecularGenerator(
        ckpt_molminer=args.ckpt_molminer,
        ckpt_starter=args.ckpt_starter,
        ckpt_gmm=args.ckpt_gmm,
        stats_path=args.stats_path,
        vocab_fragments=args.vocab_fragments,
        vocab_attachments=args.vocab_attachments,
        vocab_anchors=args.vocab_anchors,
        device=args.device,
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ball_id", "idx", "smiles"])

        for local_i, vec in enumerate(
            tqdm(
                vectors,
                desc=f"ball {args.ball_id:02d} [{args.idx_start}:{args.idx_end}] (det)",
            )
        ):
            global_idx = args.idx_start + local_i

            # vec is already in scaled space; _sample() avoids double scaling
            c = torch.tensor(vec, dtype=torch.float).to(args.device)
            set_seed(args.seed)                       # ← deterministic per-point
            smiles, failed, _ = gen._sample(
                c=c,
                topk=5,
                weighted=True,
                greedy=True,
                max_tries=10,
                seed=args.seed,                       # ← passed through
            )

            smiles = smiles if not failed else ""
            writer.writerow([args.ball_id, global_idx, smiles])

    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
