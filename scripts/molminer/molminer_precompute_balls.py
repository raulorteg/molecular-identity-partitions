"""MolMiner ball sampler for the Fig 3 identity balls.

Samples 30 ball centers from a GMM fitted to the scaled training conditions,
stratified by log-likelihood, then draws N points uniformly from a d-ball of
radius r around each.

Writes centers.npy (30, 12) and ball_{00..29}.npy (N, 12), in scaled space,
into --out_dir.

Re-create only; see data/molminer/SOURCE.md.
"""

import argparse
import pathlib
import joblib
import numpy as np
from tqdm import tqdm


def sample_dball_uniform(center: np.ndarray, r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly from a d-ball of radius r around center."""
    d = len(center)
    directions = rng.standard_normal((n, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # scale radii with u^(1/d) for uniform volume sampling
    u = rng.uniform(0, 1, size=(n, 1))
    radii = r * (u ** (1.0 / d))
    return center + radii * directions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_gmm", required=True, type=pathlib.Path)
    p.add_argument("--n_balls", type=int, default=30)
    p.add_argument("--n_samples", type=int, default=100000)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--n_candidates", type=int, default=3000, help="GMM samples to draw before stratification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("mci"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    gmm = joblib.load(args.ckpt_gmm)

    # sample candidates and compute log-likelihoods
    candidates, _ = gmm.sample(n_samples=args.n_candidates)
    log_liks = gmm.score_samples(candidates)

    # stratify: bin by log-likelihood quantiles, pick one per bin
    quantiles = np.linspace(0, 100, args.n_balls + 1)
    boundaries = np.percentile(log_liks, quantiles)
    centers = []
    for i in range(args.n_balls):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (log_liks >= lo) & (log_liks <= hi if i == args.n_balls - 1 else log_liks < hi)
        idx = rng.choice(np.where(mask)[0])
        centers.append(candidates[idx])

    centers = np.array(centers)  # (30, 12)
    np.save(args.out_dir / "centers.npy", centers)
    print(f"Saved centers: {centers.shape} -> {args.out_dir / 'centers.npy'}")

    # sample d-ball points for each center, each ball has its own independent rng
    for i, center in enumerate(tqdm(centers, desc="Sampling balls")):
        rng_i = np.random.default_rng(args.seed + i)
        pts = sample_dball_uniform(center, r=args.radius, n=args.n_samples, rng=rng_i)
        fname = args.out_dir / f"ball_{i:02d}.npy"
        np.save(fname, pts)
        tqdm.write(f"Ball {i:02d}: {pts.shape} -> {fname}")


if __name__ == "__main__":
    main()