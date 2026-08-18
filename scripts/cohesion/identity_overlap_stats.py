"""Per-architecture identity-overlap statistics between decoded balls.

Reads the per-ball identity-set pickles + centers.npy, computes pairwise
Jaccard, raw overlap and Euclidean distance, and writes them as a .npz.

export_cohesion_csvs.py reads the .npz to build the identity-overlap summary
and per-pair CSVs. Table 1 is built separately from Jaccard-by-convention data.

Invoked by repro/fig3_cohesiveness.sh.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr


def load_identity_sets(identity_dir: pathlib.Path):
    pkls = sorted(identity_dir.glob("ball_*.pkl"))
    if not pkls:
        raise SystemExit(f"no ball_*.pkl in {identity_dir}")
    sets = []
    for p in pkls:
        with open(p, "rb") as f:
            sets.append(pickle.load(f))
    centers = np.load(identity_dir / "centers.npy")
    assert centers.shape[0] == len(sets), \
        f"centers/sets mismatch: {centers.shape[0]} vs {len(sets)}"
    return sets, centers


def pairwise_stats(sets, centers):
    K = len(sets)
    universe = set()
    for s in sets:
        universe |= s
    N = len(universe)
    sizes = np.array([len(s) for s in sets], dtype=int)

    # per-identity ball coverage, for the singleton fraction
    from collections import defaultdict
    cover = defaultdict(int)
    for s in sets:
        for smi in s:
            cover[smi] += 1
    n_singleton = sum(1 for c in cover.values() if c == 1)
    f_singleton = n_singleton / N

    rows = []
    for i, j in combinations(range(K), 2):
        Si, Sj = sets[i], sets[j]
        O = len(Si & Sj)
        U = len(Si | Sj)
        J = O / U if U > 0 else 0.0
        Ki, Kj = sizes[i], sizes[j]
        d = float(np.linalg.norm(centers[i] - centers[j]))
        rows.append({
            "i": i, "j": j,
            "d": d, "J": J, "O": O, "U": U, "Ki": Ki, "Kj": Kj,
        })
    return rows, N, sizes, f_singleton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--identity_dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    sets, centers = load_identity_sets(args.identity_dir)
    K = len(sets)
    rows, N, sizes, f_singleton = pairwise_stats(sets, centers)
    n_pairs = len(rows)

    d_arr = np.array([r["d"] for r in rows])
    J_arr = np.array([r["J"] for r in rows])
    O_arr = np.array([r["O"] for r in rows])
    i_arr = np.array([r["i"] for r in rows])
    j_arr = np.array([r["j"] for r in rows])

    rho_dJ, p_dJ = spearmanr(d_arr, J_arr, alternative="less")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(args.out.with_suffix(".npz"),
             d=d_arr, J=J_arr, O=O_arr,
             Ki=np.array([sizes[i] for i in i_arr]),
             Kj=np.array([sizes[j] for j in j_arr]),
             i=i_arr, j=j_arr,
             N=N, K=K, f_singleton=f_singleton)
    print(f"wrote {args.out.with_suffix('.npz')}")
    print()
    print(f"SUMMARY {args.arch}: K={K}, {n_pairs} pairs, universe={N}")
    print(f"  Spearman rho(d,J) = {rho_dJ:+.3f}, p = {p_dJ:.3g}")
    print(f"  f_singleton       = {f_singleton:.3f}")


if __name__ == "__main__":
    main()
