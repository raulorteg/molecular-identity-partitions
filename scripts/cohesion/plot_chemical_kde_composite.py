"""Fig 3 per-ball Tanimoto KDE, one column per architecture.

Per column: thin per-ball within-pair KDEs, then pooled within (W, solid),
across (A, dashed) and the random baseline (R, dotted).

Reads data/chemical_cohesiveness/<arch>/: W_per_ball.npy (K x N_PAIRS,
NaN-padded), A.npy, R.npy, summary.json (AUCs for the stats box).

Invoked by repro/fig3_cohesiveness.sh.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def kde_eval(values: np.ndarray, x: np.ndarray, bw: str = "scott"):
    v = values[np.isfinite(values)]
    if len(v) < 2 or v.std() < 1e-9:
        return np.zeros_like(x)
    try:
        return gaussian_kde(v, bw_method=bw).evaluate(x)
    except Exception:
        return np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dirs", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--xmax", type=float, default=0.70,
                    help="Trim Tanimoto axis (default 0.70; right tail is sparse).")
    ap.add_argument("--per_ball_alpha", type=float, default=0.30)
    ap.add_argument("--width_in", type=float, default=6.75,
                    help="two-column figure width (default 6.75 in = 17.15 cm).")
    ap.add_argument("--height_in", type=float, default=2.55)
    args = ap.parse_args()

    n_archs = len(args.input_dirs)
    fig, axes = plt.subplots(
        1, n_archs,
        figsize=(args.width_in, args.height_in),
        sharey=False, sharex=True,
    )
    if n_archs == 1:
        axes = [axes]

    x_eval = np.linspace(0.0, args.xmax, 400)

    for ax, in_dir in zip(axes, args.input_dirs):
        summary = json.load(open(in_dir / "summary.json"))
        W_pb = np.load(in_dir / "W_per_ball.npy")  # (K, N_PAIRS)
        W = np.load(in_dir / "W.npy")
        A = np.load(in_dir / "A.npy")
        R = np.load(in_dir / "R.npy")
        K = W_pb.shape[0]
        cmap = plt.cm.viridis

        ymax = 0.0
        for k in range(K):
            y = kde_eval(W_pb[k], x_eval)
            ax.plot(x_eval, y,
                    color=cmap(k / max(K - 1, 1)),
                    alpha=args.per_ball_alpha,
                    linewidth=0.5, zorder=2)
            if y.max() > ymax:
                ymax = y.max()

        y_W = kde_eval(W, x_eval)
        ax.plot(x_eval, y_W, color="#d62728", linewidth=1.6,
                label=f"within (pooled)",
                zorder=5)

        y_A = kde_eval(A, x_eval)
        ax.plot(x_eval, y_A, color="#1f77b4", linewidth=1.4, linestyle="--",
                label=f"across (pooled)",
                zorder=5)

        y_R = kde_eval(R, x_eval)
        ax.plot(x_eval, y_R, color="0.3", linewidth=1.4, linestyle=":",
                label=f"random baseline",
                zorder=5)

        ax.text(0.98, 0.97,
                f"AUC(W,A) = {summary['AUC_W_vs_A']:.3f}",
                transform=ax.transAxes,
                ha="right", va="top",
                family="monospace", fontsize=8.0,
                bbox=dict(facecolor="white", edgecolor="0.6",
                          boxstyle="round,pad=0.3", linewidth=0.5, alpha=0.92))

        ax.set_title(f"{summary['arch']}  (K={summary['K']})")
        ax.set_xlabel("Tanimoto similarity")
        ax.set_xlim(0, args.xmax)
        ax.set_ylim(0, max(ymax, y_W.max(), y_A.max(), y_R.max()) * 1.1)
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    axes[0].set_ylabel("density")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"wrote {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
