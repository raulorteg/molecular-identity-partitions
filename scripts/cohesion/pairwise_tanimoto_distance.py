"""Euclidean-distance vs. chemical-distance probe.

Per architecture and per ball pair (i,j): d_ij, the Euclidean distance between
ball centers, against med D_ij = 1 - median pairwise Tanimoto similarity of the
two balls' subsampled SMILES sets. A cosine twin (cdist_ij = 1 - cosine
similarity) is computed alongside; its correlation carries the flipped sign.

Reads data/<arch>/identity_sets*/centers.npy and the already-fingerprinted
data/chemical_cohesiveness/<arch>/fingerprints/ball_NN.pkl; writes
pairwise_tanimoto_distance.{csv,json} beside them.

Invoked by repro/tab2_metric_scaling.sh.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import pickle

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")

SEED = 20260520
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
N_SUB_PER_BALL = 200          # cap per-ball FP subsample for the full cross matrix
MIN_FPS_PER_BALL = 5          # skip pairs where either ball has fewer FPs

FINGERPRINTS = ("morgan", "maccs")


def make_fp_fn(fingerprint):
    """mol-SMILES -> ExplicitBitVect for the chosen fingerprint (see
    chemical_cohesiveness.make_fp_fn; kept in sync)."""
    if fingerprint == "morgan":
        return lambda m: AllChem.GetMorganFingerprintAsBitVect(
            m, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    if fingerprint == "maccs":
        return lambda m: MACCSkeys.GenMACCSKeys(m)
    raise ValueError(f"unknown fingerprint {fingerprint!r}; expected {FINGERPRINTS}")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

ARCHS = [
    ("MolMiner", "data/molminer/identity_sets",        "data/chemical_cohesiveness/molminer"),
    ("HierVAE",  "data/hiervae/identity_sets_pooled",  "data/chemical_cohesiveness/hiervae"),
    ("GDSS",     "data/gdss/identity_sets",            "data/chemical_cohesiveness/gdss"),
]



def load_fp_list(fp_pkl_path: pathlib.Path, rng, fp_fn) -> list:
    """Return a list of RDKit ExplicitBitVect built from the cached SMILES
    subsample, capped at N_SUB_PER_BALL. `fp_fn` is the mol->FP builder."""
    with open(fp_pkl_path, "rb") as f:
        d = pickle.load(f)
    smiles = d["smiles"]
    if len(smiles) > N_SUB_PER_BALL:
        idx = rng.choice(len(smiles), size=N_SUB_PER_BALL, replace=False)
        smiles = [smiles[k] for k in idx]
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(fp_fn(m))
    return fps


def median_pairwise_tanimoto(fps_a, fps_b) -> float:
    """Median Tanimoto similarity over the full |A| x |B| cross-product."""
    sims = []
    for fa in fps_a:
        sims.extend(DataStructs.BulkTanimotoSimilarity(fa, fps_b))
    return float(np.median(sims))


def run_arch(label: str, identity_dir: pathlib.Path,
             chemcoh_dir: pathlib.Path, fingerprint: str = "morgan") -> dict:
    centers = np.load(identity_dir / "centers.npy")
    K = centers.shape[0]
    fp_fn = make_fp_fn(fingerprint)

    rng = np.random.default_rng(SEED)
    print(f"== {label}  K={K}, dim={centers.shape[1]}, fingerprint={fingerprint} ==", flush=True)
    print("  loading + fingerprinting subsamples...", flush=True)
    fps_per_ball = []
    for i in range(K):
        fp_pkl = chemcoh_dir / "fingerprints" / f"ball_{i:02d}.pkl"
        if not fp_pkl.is_file():
            raise SystemExit(f"missing {fp_pkl}")
        fps_per_ball.append(load_fp_list(fp_pkl, rng, fp_fn))
    sizes = [len(f) for f in fps_per_ball]
    print(f"  per-ball FP counts: min={min(sizes)} median={int(np.median(sizes))} max={max(sizes)}",
          flush=True)

    norms = np.linalg.norm(centers, axis=1)
    rows = []
    skipped = 0
    print(f"  computing median Tanimoto for K(K-1)/2={K*(K-1)//2} pairs...", flush=True)
    for i in range(K):
        for j in range(i + 1, K):
            n_i, n_j = sizes[i], sizes[j]
            if n_i < MIN_FPS_PER_BALL or n_j < MIN_FPS_PER_BALL:
                skipped += 1
                continue
            d_ij = float(np.linalg.norm(centers[i] - centers[j]))
            denom = norms[i] * norms[j]
            cos_ij = float(np.dot(centers[i], centers[j]) / denom) if denom > 0 else float("nan")
            med_T = median_pairwise_tanimoto(fps_per_ball[i], fps_per_ball[j])
            rows.append({
                "i": i, "j": j, "n_i": n_i, "n_j": n_j,
                "n_pairs_used": n_i * n_j,
                "d_ij": d_ij,
                "cos_ij": cos_ij,
                "cos_dist": 1.0 - cos_ij,
                "med_T_sim": med_T,
                "med_T_dist": 1.0 - med_T,
            })
    print(f"  {len(rows)} pairs kept ({skipped} skipped for low FP count)", flush=True)

    out_csv = chemcoh_dir / "pairwise_tanimoto_distance.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_csv}", flush=True)

    d = np.array([r["d_ij"] for r in rows])
    cosd = np.array([r["cos_dist"] for r in rows])
    cos = np.array([r["cos_ij"] for r in rows])
    D = np.array([r["med_T_dist"] for r in rows])
    T = np.array([r["med_T_sim"] for r in rows])

    def ols_r2(x, y):
        """OLS R^2 of y on x, dropping non-finite pairs."""
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        slope, intercept = np.polyfit(x, y, deg=1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(slope), float(intercept), float(r2)

    # Euclidean probe: chemical DISTANCE rises with center distance; chemical
    # SIMILARITY falls with it
    slope, intercept, r2 = ols_r2(d, D)
    rho, p_rho = spearmanr(d, T, alternative="less")

    # Cosine twin: chemical DISTANCE rises with cosine DISTANCE; chemical
    # SIMILARITY rises with cosine SIMILARITY
    cm = np.isfinite(cos) & np.isfinite(T)
    slope_c, intercept_c, r2_c = ols_r2(cosd, D)
    rho_c, p_rho_c = spearmanr(cos[cm], T[cm], alternative="greater")

    stats = {
        "arch": label,
        "fingerprint": fingerprint,
        "K": int(K),
        "n_pairs": len(rows),
        "n_pairs_skipped_low_fp_count": skipped,
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "R2_distance_vs_chem_distance": float(r2),
        "spearman_rho_d_vs_sim": float(rho),
        "spearman_p_alt_less": float(p_rho),
        "ols_slope_cos": float(slope_c),
        "ols_intercept_cos": float(intercept_c),
        "R2_cosdist_vs_chem_distance": float(r2_c),
        "spearman_rho_cos_vs_sim": float(rho_c),
        "spearman_p_cos_alt_greater": float(p_rho_c),
    }
    with open(chemcoh_dir / "pairwise_tanimoto_distance.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  EUCLID  R^2(d, D) = {r2:.3f}   "
          f"Spearman rho(d, T) = {rho:+.3f}  (p_less = {p_rho:.2e})",
          flush=True)
    print(f"  COSINE  R^2(cosd, D) = {r2_c:.3f}   "
          f"Spearman rho(cos, T) = {rho_c:+.3f}  (p_greater = {p_rho_c:.2e})",
          flush=True)
    return {"d": d, "D": D, "T": T, "stats": stats,
            "ols": (slope, intercept)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint", choices=FINGERPRINTS, default="morgan",
                    help="morgan (ECFP r=2 2048b, primary) or maccs (166-bit robustness twin)")
    ap.add_argument("--label", default=None, help="custom arch label (single mode)")
    ap.add_argument("--identity_dir", type=pathlib.Path, default=None)
    ap.add_argument("--chemcoh_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    # morgan writes to the base chemcoh dir; maccs to a maccs/ subdir.
    sub = "" if args.fingerprint == "morgan" else args.fingerprint

    if args.identity_dir and args.chemcoh_dir:
        label = args.label or "GDSS"
        arch_specs = [(label, args.identity_dir, args.chemcoh_dir)]
    else:
        arch_specs = [(label, REPO_ROOT / id_rel,
                       (REPO_ROOT / chem_rel / sub) if sub else (REPO_ROOT / chem_rel))
                      for label, id_rel, chem_rel in ARCHS]

    per_arch = {}
    for label, id_dir, chem_dir in arch_specs:
        if not (id_dir / "centers.npy").is_file():
            print(f"[skip] {label}: missing {id_dir/'centers.npy'}")
            continue
        if not (chem_dir / "fingerprints").is_dir():
            print(f"[skip] {label}: missing fingerprints under {chem_dir}")
            continue
        per_arch[label] = run_arch(label, id_dir, chem_dir, args.fingerprint)

    if not per_arch:
        raise SystemExit("no architectures processed")



if __name__ == "__main__":
    main()
