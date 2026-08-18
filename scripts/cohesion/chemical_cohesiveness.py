"""Chemical cohesiveness probe: within-ball vs across-ball vs random Tanimoto
distributions, per architecture.

Loads the cached identity sets (data/<arch>/identity_sets*/ball_NN.pkl),
subsamples 1000 molecules per ball, fingerprints them, and builds three
Tanimoto distributions: W (within-ball pairs), A (ball vs pool-of-others) and
R (random pairs from the pooled subsample), plus per-ball AUC(W_i vs A_i).
Writes the raw arrays and summary.json for one architecture per `run`.

Re-create only; see data/fig3_cohesiveness/SOURCE.md.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

RDLogger.DisableLog("rdApp.*")

SEED = 20260520
N_SUB_PER_BALL = 1000
N_PAIRS_PER_BALL = 10_000
N_PAIRS_R = 10_000
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

FINGERPRINTS = ("morgan", "maccs")


def make_fp_fn(fingerprint):
    """Return a smiles->ExplicitBitVect function.

    morgan: Morgan/ECFP radius=2, 2048-bit.  maccs: 166-bit MACCS keys.
    """
    if fingerprint == "morgan":
        def fn(smi):
            mol = Chem.MolFromSmiles(smi)
            return None if mol is None else AllChem.GetMorganFingerprintAsBitVect(
                mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        return fn
    if fingerprint == "maccs":
        def fn(smi):
            mol = Chem.MolFromSmiles(smi)
            return None if mol is None else MACCSkeys.GenMACCSKeys(mol)
        return fn
    raise ValueError(f"unknown fingerprint {fingerprint!r}; expected {FINGERPRINTS}")


def load_identity_sets(identity_dir: pathlib.Path):
    pkls = sorted(identity_dir.glob("ball_*.pkl"))
    sets = []
    for p in pkls:
        with open(p, "rb") as f:
            sets.append(pickle.load(f))
    return sets


def subsample_and_fingerprint(sets, n_sub: int, rng, fp_fn):
    """Subsample each ball's identity set to n_sub molecules and fingerprint."""
    K = len(sets)
    fps_per_ball = []
    sub_smiles_per_ball = []
    for i, S in enumerate(sets):
        smiles = list(S)
        if len(smiles) > n_sub:
            idx = rng.choice(len(smiles), size=n_sub, replace=False)
            smiles = [smiles[k] for k in idx]
        fps = []
        kept_smiles = []
        for s in smiles:
            fp = fp_fn(s)
            if fp is not None:
                fps.append(fp)
                kept_smiles.append(s)
        fps_per_ball.append(fps)
        sub_smiles_per_ball.append(kept_smiles)
        print(f"  ball_{i:02d}  |S_i|={len(S):>6d}  subsample kept={len(fps)}", flush=True)
    return fps_per_ball, sub_smiles_per_ball


def tanimoto_pairs(fps_a, fps_b, idx_pairs):
    """Compute Tanimoto for each pair (a_idx, b_idx). fps_a and fps_b are
    lists of RDKit ExplicitBitVect."""
    return np.array([
        DataStructs.TanimotoSimilarity(fps_a[a], fps_b[b]) for a, b in idx_pairs
    ], dtype=np.float32)


def sample_within_pairs(n_mols: int, n_pairs: int, rng):
    if n_mols < 2:
        return np.empty((0, 2), dtype=int)
    # pairs with i != j
    out = []
    while len(out) < n_pairs:
        a = rng.integers(0, n_mols, size=n_pairs)
        b = rng.integers(0, n_mols, size=n_pairs)
        mask = a != b
        out.extend(zip(a[mask].tolist(), b[mask].tolist()))
    return np.array(out[:n_pairs], dtype=int)


def auc_dominance(x, y, n_max: int = 100_000):
    """P(X > Y) where X drawn from `x` and Y drawn from `y` independently.
    Uses subsample-by-rank to scale to large arrays."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if len(x) > n_max:
        x = np.random.default_rng(SEED).choice(x, size=n_max, replace=False)
    if len(y) > n_max:
        y = np.random.default_rng(SEED + 1).choice(y, size=n_max, replace=False)
    # exact O(n log n) via merged-sort rank
    combined = np.concatenate([x, y])
    labels = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Mann-Whitney U for "x > y" (greater)
    rank_x_sum = ranks[labels == 0].sum()
    n_x = len(x); n_y = len(y)
    U = rank_x_sum - n_x * (n_x + 1) / 2
    return float(U / (n_x * n_y))


def run_one_arch(arch: str, identity_dir: pathlib.Path, output_dir: pathlib.Path,
                 fingerprint: str = "morgan"):
    output_dir.mkdir(parents=True, exist_ok=True)
    fp_dir = output_dir / "fingerprints"
    fp_dir.mkdir(exist_ok=True)
    fp_fn = make_fp_fn(fingerprint)

    rng = np.random.default_rng(SEED)
    sets = load_identity_sets(identity_dir)
    K = len(sets)
    print(f"\n=== {arch}  (K={K} balls, fingerprint={fingerprint}) ===", flush=True)

    print(f"subsampling {N_SUB_PER_BALL} molecules per ball + fingerprinting...", flush=True)
    t0 = time.time()
    fps_per_ball, sub_smiles_per_ball = subsample_and_fingerprint(
        sets, N_SUB_PER_BALL, rng, fp_fn)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    for i, (smis, fps) in enumerate(zip(sub_smiles_per_ball, fps_per_ball)):
        with open(fp_dir / f"ball_{i:02d}.pkl", "wb") as f:
            pickle.dump({"smiles": smis, "n": len(fps)}, f)

    # per-ball W and A
    print(f"sampling {N_PAIRS_PER_BALL} W pairs + {N_PAIRS_PER_BALL} A pairs per ball...",
          flush=True)
    W_all = []
    A_all = []
    per_ball_AUC = []
    per_ball_med_W = []
    per_ball_med_A = []
    # Build a flat (ball, idx) index of all subsampled molecules across balls
    flat = []
    for j in range(K):
        for k in range(len(fps_per_ball[j])):
            flat.append((j, k))
    flat = np.array(flat, dtype=int)  # shape (M_total, 2)

    M_total = len(flat)
    print(f"  total subsampled molecules across balls: {M_total}", flush=True)

    for i in range(K):
        fps_i = fps_per_ball[i]
        n_i = len(fps_i)

        # Within: random pairs (a,b) with a != b within ball i
        w_pairs = sample_within_pairs(n_i, N_PAIRS_PER_BALL, rng)
        if len(w_pairs) > 0:
            W_i = tanimoto_pairs(fps_i, fps_i, w_pairs.tolist())
        else:
            W_i = np.empty(0, dtype=np.float32)

        # Across: random pairs (m from ball i, m' from any other ball)
        other_mask = flat[:, 0] != i
        other_flat = flat[other_mask]
        # Sample N_PAIRS pairs: (a in 0..n_i-1, b is a (ball_j, mol_k) from other)
        a_idx = rng.integers(0, n_i, size=N_PAIRS_PER_BALL)
        b_choice = rng.integers(0, len(other_flat), size=N_PAIRS_PER_BALL)
        b_ball = other_flat[b_choice, 0]
        b_mol = other_flat[b_choice, 1]
        A_i_vals = np.empty(N_PAIRS_PER_BALL, dtype=np.float32)
        for p in range(N_PAIRS_PER_BALL):
            A_i_vals[p] = DataStructs.TanimotoSimilarity(
                fps_i[int(a_idx[p])], fps_per_ball[int(b_ball[p])][int(b_mol[p])]
            )

        W_all.append(W_i)
        A_all.append(A_i_vals)
        per_ball_AUC.append(auc_dominance(W_i, A_i_vals))
        per_ball_med_W.append(float(np.median(W_i)) if len(W_i) else float("nan"))
        per_ball_med_A.append(float(np.median(A_i_vals)))
        if (i + 1) % 5 == 0 or i == K - 1:
            print(f"  ball_{i:02d} done  med(W)={per_ball_med_W[-1]:.3f}  "
                  f"med(A)={per_ball_med_A[-1]:.3f}  AUC={per_ball_AUC[-1]:.3f}",
                  flush=True)

    W = np.concatenate(W_all) if W_all else np.empty(0, dtype=np.float32)
    A = np.concatenate(A_all) if A_all else np.empty(0, dtype=np.float32)

    # R: random pairs from pooled subsample
    print(f"sampling {N_PAIRS_R} R pairs from pooled subsample...", flush=True)
    R_a = rng.integers(0, M_total, size=N_PAIRS_R)
    R_b = rng.integers(0, M_total, size=N_PAIRS_R)
    R_vals = np.empty(N_PAIRS_R, dtype=np.float32)
    for p in range(N_PAIRS_R):
        if R_a[p] == R_b[p]:
            R_vals[p] = 1.0  # self-pair (rare); will skip downstream
            continue
        ball_a, mol_a = flat[R_a[p]]
        ball_b, mol_b = flat[R_b[p]]
        R_vals[p] = DataStructs.TanimotoSimilarity(
            fps_per_ball[int(ball_a)][int(mol_a)],
            fps_per_ball[int(ball_b)][int(mol_b)],
        )
    # Drop self-pairs
    R_vals = R_vals[R_vals < 1.0 - 1e-9]

    np.save(output_dir / "W.npy", W)
    np.save(output_dir / "A.npy", A)
    np.save(output_dir / "R.npy", R_vals)
    np.save(output_dir / "per_ball_AUC.npy", np.array(per_ball_AUC))
    # Per-ball arrays for KDE composite (shape K x N_PAIRS_PER_BALL when all
    # balls produced full pairs).
    K_actual = len(W_all)
    target = N_PAIRS_PER_BALL
    W_pb = np.full((K_actual, target), np.nan, dtype=np.float32)
    A_pb = np.full((K_actual, target), np.nan, dtype=np.float32)
    for k, (wi, ai) in enumerate(zip(W_all, A_all)):
        if len(wi) >= target:
            W_pb[k] = wi[:target]
        elif len(wi) > 0:
            W_pb[k, : len(wi)] = wi
        if len(ai) >= target:
            A_pb[k] = ai[:target]
        elif len(ai) > 0:
            A_pb[k, : len(ai)] = ai
    np.save(output_dir / "W_per_ball.npy", W_pb)
    np.save(output_dir / "A_per_ball.npy", A_pb)

    auc_WA = auc_dominance(W, A)
    auc_WR = auc_dominance(W, R_vals)
    auc_AR = auc_dominance(A, R_vals)
    per_ball_AUC_arr = np.array(per_ball_AUC)
    # Per-ball AUC(W_i vs the pooled random baseline R)
    per_ball_AUC_WR_arr = np.array([auc_dominance(w, R_vals) for w in W_all])
    np.save(output_dir / "per_ball_AUC_WR.npy", per_ball_AUC_WR_arr)
    n_pass = int((per_ball_AUC_arr > 0.5).sum())

    def iqr(a):
        return float(np.percentile(a, 25)), float(np.percentile(a, 75))
    W_q25, W_q75 = iqr(W)
    A_q25, A_q75 = iqr(A)
    R_q25, R_q75 = iqr(R_vals)
    pbAUC_q25, pbAUC_q75 = iqr(per_ball_AUC_arr)
    pbAUC_WR_q25, pbAUC_WR_q75 = iqr(per_ball_AUC_WR_arr)

    summary = {
        "arch": arch,
        "fingerprint": fingerprint,
        "K": K,
        "median_W": float(np.median(W)),
        "median_A": float(np.median(A)),
        "median_R": float(np.median(R_vals)),
        "W_q25": W_q25, "W_q75": W_q75,
        "A_q25": A_q25, "A_q75": A_q75,
        "R_q25": R_q25, "R_q75": R_q75,
        "AUC_W_vs_A": auc_WA,
        "AUC_W_vs_R": auc_WR,
        "AUC_A_vs_R": auc_AR,
        "n_pairs_W_per_ball": N_PAIRS_PER_BALL,
        "n_pairs_A_per_ball": N_PAIRS_PER_BALL,
        "n_pairs_R": int(len(R_vals)),
        "per_ball_AUC_median": float(np.median(per_ball_AUC_arr)),
        "per_ball_AUC_q25": pbAUC_q25,
        "per_ball_AUC_q75": pbAUC_q75,
        "per_ball_AUC_min": float(per_ball_AUC_arr.min()),
        "per_ball_AUC_max": float(per_ball_AUC_arr.max()),
        "per_ball_AUC_WR_median": float(np.median(per_ball_AUC_WR_arr)),
        "per_ball_AUC_WR_q25": pbAUC_WR_q25,
        "per_ball_AUC_WR_q75": pbAUC_WR_q75,
        "n_balls_AUC_gt_0p5": n_pass,
        "frac_balls_AUC_gt_0p5": n_pass / K,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"=== {arch} SUMMARY ===")
    print(f"  median(W)={summary['median_W']:.3f}  "
          f"median(A)={summary['median_A']:.3f}  "
          f"median(R)={summary['median_R']:.3f}")
    print(f"  AUC(W vs A)={auc_WA:.3f}  "
          f"AUC(W vs R)={auc_WR:.3f}  "
          f"AUC(A vs R)={auc_AR:.3f}")
    print(f"  per-ball AUC median={summary['per_ball_AUC_median']:.3f}  "
          f"({n_pass}/{K} balls > 0.5)")

    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run", help="Run analysis for one architecture")
    ap_run.add_argument("--arch", required=True)
    ap_run.add_argument("--identity_dir", type=pathlib.Path, required=True)
    ap_run.add_argument("--output_dir", type=pathlib.Path, required=True)
    ap_run.add_argument("--fingerprint", choices=FINGERPRINTS, default="morgan",
                        help="morgan (ECFP r=2 2048b, primary) or maccs (166-bit robustness twin)")

    args = ap.parse_args()

    if args.cmd == "run":
        run_one_arch(args.arch, args.identity_dir, args.output_dir, args.fingerprint)


if __name__ == "__main__":
    main()
