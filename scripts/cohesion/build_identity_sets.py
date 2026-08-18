"""Build per-ball canonical-SMILES identity-set caches.

Reads every chunk CSV under <balls_dir>/results/, RDKit-canonicalizes the
smiles column, deduplicates, and writes one frozenset pickle per ball at
<output_dir>/ball_NN.pkl, plus centers.npy.

HierVAE pools two seeds: passing both ball dirs renumbers the balls so the
first keeps 0..K-1 and the second becomes K..2K-1.

Re-create only; see data/<arch>/SOURCE.md for the per-architecture invocation.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import re
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # silence per-SMILES parse warnings

CHUNK_RE = re.compile(r"^ball_(\d+)_\d+_\d+(?:_stochastic)?\.csv$", re.IGNORECASE)


def canonical_or_none(smi):
    if smi is None or pd.isna(smi):
        return None
    s = str(smi).strip()
    if not s or s.lower() == "nan":
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def process_ball(args):
    out_path, csv_paths, ball_id_in, ball_id_out = args
    smis = []
    for p in csv_paths:
        df = pd.read_csv(p, usecols=["smiles"])
        smis.extend(df["smiles"].tolist())
    canon = set()
    for s in smis:
        c = canonical_or_none(s)
        if c is not None:
            canon.add(c)
    canon = frozenset(canon)
    with open(out_path, "wb") as f:
        pickle.dump(canon, f, protocol=4)
    return (ball_id_in, ball_id_out, len(smis), len(canon))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balls_dirs", type=pathlib.Path, nargs="+", required=True,
                    help="One or more ball directories (each contains "
                         "centers.npy + results/ + ball_NN.npy). For pooled "
                         "HierVAE pass orig dir then new dir.")
    ap.add_argument("--output_dir", type=pathlib.Path, required=True)
    ap.add_argument("--nproc", type=int, default=24)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Gather (ball_id_in_source_dir, source_dir, ball_id_out) triples.
    tasks = []
    centers_parts = []
    ball_id_out = 0
    per_dir_summary = []
    for balls_dir in args.balls_dirs:
        results_dir = balls_dir / "results"
        if not results_dir.is_dir():
            raise SystemExit(f"missing {results_dir}")
        chunks_by_ball: dict[int, list[pathlib.Path]] = defaultdict(list)
        for p in results_dir.iterdir():
            m = CHUNK_RE.match(p.name)
            if m:
                chunks_by_ball[int(m.group(1))].append(p)
        centers_path = balls_dir / "centers.npy"
        if not centers_path.is_file():
            raise SystemExit(f"missing {centers_path}")
        c = np.load(centers_path)
        n_balls = c.shape[0]
        for b_in in range(n_balls):
            csvs = sorted(chunks_by_ball.get(b_in, []))
            if not csvs:
                raise SystemExit(f"no chunks for ball {b_in} in {balls_dir}")
            out_path = args.output_dir / f"ball_{ball_id_out:02d}.pkl"
            tasks.append((out_path, csvs, b_in, ball_id_out))
            ball_id_out += 1
        centers_parts.append(c)
        per_dir_summary.append((balls_dir, n_balls, sum(len(v) for v in chunks_by_ball.values())))

    centers = np.concatenate(centers_parts, axis=0)
    np.save(args.output_dir / "centers.npy", centers)

    K = len(tasks)
    print(f"output_dir: {args.output_dir}")
    for d, nb, nc in per_dir_summary:
        print(f"  {d}: {nb} balls, {nc} chunks")
    print(f"total: K={K} balls, pooled centers shape {centers.shape}")
    print(f"canonicalizing with {args.nproc} workers...")

    t0 = time.time()
    with Pool(args.nproc) as pool:
        for i, (b_in, b_out, n_raw, n_canon) in enumerate(pool.imap_unordered(process_ball, tasks)):
            print(f"  [{i+1:>3}/{K}] ball_{b_out:02d} (src ball_{b_in:02d})  "
                  f"raw={n_raw}  canon={n_canon}", flush=True)
    t1 = time.time()
    print(f"done in {t1-t0:.1f}s")


if __name__ == "__main__":
    main()
