"""GDSS ball sampler for Fig 3.

Samples 30 ball centers from GDSS's analytic prior N(0, I_d), stratified by
log-likelihood quantiles, then samples N points uniformly from a d-ball of
radius r around each.

The coordinate dimension d covers only GDSS's active structured noise —
z_x 24 nodes x 9 feats = 216, plus z_adj 24*23/2 = 276 — for d = 492. Padding
nodes 24-37 and the adj diagonal are zero by mask convention and omitted here;
gdss_decode_ball.py inflates them back at decode time.

Writes centers.npy (n_balls, d) and ball_NN.npy (N, d) into --out_dir.

Re-create only; see data/gdss/SOURCE.md.
"""
import argparse
import pathlib

import numpy as np
from tqdm import tqdm


N_ACTIVE_NODES = 24
MAX_FEAT_NUM = 9
D_X = N_ACTIVE_NODES * MAX_FEAT_NUM                                     # 216
D_ADJ = N_ACTIVE_NODES * (N_ACTIVE_NODES - 1) // 2                      # 276
D_TOTAL = D_X + D_ADJ                                                   # 492


def sample_dball_uniform(center: np.ndarray, r: float, n: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly from a d-ball of radius r around center.

    Paper formula (Section 3): z = z_0 + r * u^(1/d) * v/||v||,
    where v ~ N(0, I_d) and u ~ U(0, 1).
    """
    d = len(center)
    directions = rng.standard_normal((n, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    u = rng.uniform(0, 1, size=(n, 1))
    radii = r * (u ** (1.0 / d))
    return center + radii * directions


def log_prior(z: np.ndarray) -> np.ndarray:
    """log N(0, I_d)(z) for a batch of z. Shape (B, d) -> (B,)."""
    d = z.shape[1]
    return -0.5 * (z ** 2).sum(axis=1) - 0.5 * d * np.log(2.0 * np.pi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=D_TOTAL,
                   help=f"Flat generative-coordinate dim (default {D_TOTAL} = "
                        f"{D_X} z_x + {D_ADJ} z_adj symmetric).")
    p.add_argument("--n_balls", type=int, default=30)
    p.add_argument("--n_samples", type=int, default=100000,
                   help="Points sampled per ball")
    p.add_argument("--radius", type=float, default=0.1,
                   help="Ball radius in standardized coords (paper uses 0.1)")
    p.add_argument("--n_candidates", type=int, default=3000,
                   help="Prior samples drawn before log-prior stratification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=pathlib.Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    candidates = rng.standard_normal((args.n_candidates, args.d))
    log_p = log_prior(candidates)

    quantiles = np.linspace(0, 100, args.n_balls + 1)
    boundaries = np.percentile(log_p, quantiles)
    centers = []
    for i in range(args.n_balls):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (log_p >= lo) & (log_p <= hi if i == args.n_balls - 1 else log_p < hi)
        idx = rng.choice(np.where(mask)[0])
        centers.append(candidates[idx])

    centers = np.array(centers)
    np.save(args.out_dir / "centers.npy", centers)
    print(f"Saved centers: {centers.shape} -> {args.out_dir / 'centers.npy'}")

    for i, center in enumerate(tqdm(centers, desc="Sampling balls")):
        rng_i = np.random.default_rng(args.seed + i)
        pts = sample_dball_uniform(center, r=args.radius, n=args.n_samples, rng=rng_i)
        fname = args.out_dir / f"ball_{i:02d}.npy"
        np.save(fname, pts)
        tqdm.write(f"Ball {i:02d}: {pts.shape} -> {fname}")


if __name__ == "__main__":
    main()
