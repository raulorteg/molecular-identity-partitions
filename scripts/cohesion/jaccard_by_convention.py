"""Jaccard identity-overlap per equivalence convention, behind Table 1.

Reports J(i,j) = |S_i n S_j| / |S_i u S_j| over all K(K-1)/2 ball pairs
(median / mean / max) under each of the 6 conventions. Reads the cached
identity sets (data/<arch>/identity_sets*/ball_NN.pkl), frozensets of
canonical SMILES, re-keyed through scripts/common/equivalence.py.

Invoked by repro/tab1_jaccard.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.equivalence import INVALID_KEY, NO_SCAFFOLD_KEY, equiv_key  # noqa: E402

CONVENTIONS = (
    "canonical",
    "inchikey",
    "murcko",
    "murcko_generic",
    "formula",
    "composition",
)
SENTINELS = {INVALID_KEY, NO_SCAFFOLD_KEY}


def load_canonical_sets(identity_dir: pathlib.Path):
    pkls = sorted(identity_dir.glob("ball_*.pkl"))
    if not pkls:
        raise SystemExit(f"no ball_*.pkl in {identity_dir}")
    sets = []
    for p in pkls:
        with open(p, "rb") as f:
            sets.append(set(pickle.load(f)))
    return sets


def rekey(canonical_set, convention):
    """Map a ball's canonical-SMILES set to its class set under `convention`.

    Returns (frozenset_of_classes, n_dropped) where n_dropped counts canonical
    SMILES whose class is a sentinel (invalid / no-scaffold) under this
    convention.
    """
    if convention == "canonical":
        kept = {s for s in canonical_set if s not in SENTINELS}
        return frozenset(kept), len(canonical_set) - len(kept)
    classes = set()
    n_dropped = 0
    for s in canonical_set:
        k = equiv_key(s, convention)
        if k in SENTINELS:
            n_dropped += 1
        else:
            classes.add(k)
    return frozenset(classes), n_dropped


def pairwise_jaccard(sets):
    out = []
    for i, j in combinations(range(len(sets)), 2):
        union = len(sets[i] | sets[j])
        out.append((len(sets[i] & sets[j]) / union) if union else 0.0)
    return np.asarray(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--identity_dir", type=pathlib.Path, required=True)
    ap.add_argument("--out_json", type=pathlib.Path, required=True)
    args = ap.parse_args()

    canon_sets = load_canonical_sets(args.identity_dir)
    K = len(canon_sets)
    n_pairs = K * (K - 1) // 2
    print(f"=== {args.arch}: K={K} balls, {n_pairs} pairs, "
          f"identity_dir={args.identity_dir} ===")

    result = {"arch": args.arch, "K": K, "n_pairs": n_pairs, "conventions": {}}
    for conv in CONVENTIONS:
        rekeyed = []
        total_in = total_dropped = 0
        for s in canon_sets:
            cls, dropped = rekey(s, conv)
            rekeyed.append(cls)
            total_in += len(s)
            total_dropped += dropped
        universe = set().union(*rekeyed) if rekeyed else set()
        sizes = [len(s) for s in rekeyed]
        J = pairwise_jaccard(rekeyed)
        result["conventions"][conv] = {
            "universe_N": len(universe),
            "median_ball_classes": float(np.median(sizes)),
            "median_J": float(np.median(J)),
            "mean_J": float(np.mean(J)),
            "max_J": float(np.max(J)),
            "dropped_frac": (total_dropped / total_in) if total_in else 0.0,
        }
        print(f"  {conv:16s} medJ={np.median(J):.4f}  meanJ={np.mean(J):.4f}  "
              f"maxJ={np.max(J):.4f}  |U|={len(universe):>8d}  "
              f"dropped={100*total_dropped/total_in:.1f}%")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
