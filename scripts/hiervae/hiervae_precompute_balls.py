"""HierVAE ball sampler for Fig 3.

Samples 30 ball centers from HierVAE's analytic prior N(0, I_latent_size),
stratified by log-likelihood quantiles, then samples N points uniformly from a
d-ball of radius r around each.

Writes centers.npy (30, latent_size) and ball_{00..29}.npy (N, latent_size)
into --out_dir.

Re-create only; see data/hiervae/SOURCE.md.
"""
import argparse
import pathlib

import numpy as np
from tqdm import tqdm


def sample_dball_uniform(center: np.ndarray, r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points uniformly from a d-ball of radius r around center."""
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
    p.add_argument("--latent_size", type=int, default=32,
                   help="HierVAE latent dim (matches add_hiervae_args default)")
    p.add_argument("--n_balls", type=int, default=30)
    p.add_argument("--n_samples", type=int, default=100000,
                   help="Points sampled per ball")
    p.add_argument("--radius", type=float, default=0.1,
                   help="Ball radius in standardized latent coords (paper uses 0.1)")
    p.add_argument("--n_candidates", type=int, default=3000,
                   help="Prior samples drawn before log-prior stratification")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=pathlib.Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    candidates = rng.standard_normal((args.n_candidates, args.latent_size))
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
