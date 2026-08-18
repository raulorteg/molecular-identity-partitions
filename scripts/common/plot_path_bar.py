"""Shared bar plotter for the Fig S14 straight-line paths (panels a-c).

Consumes a JSON list of {alpha, smiles, ...} records, as produced by
gdss_interpolate.py, walk_hiervae.py and walk_molminer.py. Emits two PNGs:

  {label}_barplot.png         horizontal bar along alpha in [0, 1], segments
                              colored by distinct RDKit canonical SMILES, in
                              order of first appearance.
  {label}_segment_molgrid.png one molecule per distinct identity, badged with
                              its bar color and dominant alpha range(s).

Invoked by repro/figS14_paths.sh.
"""
import argparse
import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from rdkit import Chem
from rdkit.Chem import Draw



def _canonical_key(smi):
    """RDKit canonical SMILES; empty/unparseable input → empty string."""
    if not smi or not str(smi).strip():
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol) or ""
    except Exception:
        return ""


def compute_keys_and_colors(data):
    """Return (keys, distinct, key_to_color).

    `keys[i]` is the canonical-SMILES identity for record i; `distinct` is
    the de-duplicated identity list in order of first appearance along the
    path; `key_to_color` maps each distinct identity to its bar segment color.
    Centralizing this here so the bar plot and the per-segment molecule grid
    share the exact same color assignment.
    """
    keys = [_canonical_key(d["smiles"]) for d in data]
    distinct = list(dict.fromkeys(keys))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = list(prop_cycle.by_key().get(
        "color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    ))
    if len(distinct) > len(colors):
        colors = (colors * ((len(distinct) // len(colors)) + 1))[:len(distinct)]
    key_to_color = {k: colors[i] for i, k in enumerate(distinct)}
    return keys, distinct, key_to_color


def plot_bar(data, keys, key_to_color, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 1.0))
    for i in range(len(data) - 1):
        left = data[i]["alpha"]
        right = data[i + 1]["alpha"]
        ax.barh(0, right - left, left=left, height=2.0,
                color=key_to_color[keys[i]], align="edge", linewidth=0)

    ax.set_xlabel("Distance along path", fontsize=11)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, length=3, width=1)
    ax.set_aspect("auto")
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved", out_path)


def _key_alpha_intervals(data, keys):
    """For each distinct identity, list the (start, end) alpha intervals where
    it dominates. Adjacent records with the same identity merge into one
    interval; non-contiguous reappearances each get their own interval.
    Returns OrderedDict keyed by first-appearance order."""
    intervals = OrderedDict()
    if not data:
        return intervals
    cur_k = keys[0]
    cur_start = data[0]["alpha"]
    cur_end = data[0]["alpha"]
    for i in range(1, len(data)):
        if keys[i] == cur_k:
            cur_end = data[i]["alpha"]
        else:
            intervals.setdefault(cur_k, []).append((cur_start, cur_end))
            cur_k = keys[i]
            cur_start = data[i]["alpha"]
            cur_end = data[i]["alpha"]
    intervals.setdefault(cur_k, []).append((cur_start, cur_end))
    return intervals


def _format_alpha_label(intervals):
    """Format an identity's alpha-interval list as a short title string."""
    parts = []
    for (a0, a1) in intervals:
        if abs(a1 - a0) < 1e-9:
            parts.append("{:.3f}".format(a0))
        else:
            parts.append("[{:.3f}, {:.3f}]".format(a0, a1))
    return r"$\alpha$: " + ", ".join(parts)


def plot_segment_molgrid(data, keys, key_to_color, out_path,
                         max_cols=5, sub_img_size=(300, 220)):
    """One molecule per distinct identity along the path, in left-to-right
    top-to-bottom order of first appearance. Each panel: 2D structure with a
    colored badge top-left matching its bar segment, and a title below showing
    the alpha interval(s) where that identity dominates. The badge color and
    panel order match plot_bar exactly, so the grid pairs visually with the bar
    for manual recomposition."""
    intervals_by_key = _key_alpha_intervals(data, keys)
    # Distinct in path-order, dropping the invalid sentinel if present.
    distinct = [k for k in intervals_by_key.keys() if k != ""]
    n = len(distinct)
    if n == 0:
        print("plot_segment_molgrid: no valid identities — skipping", out_path)
        return

    n_cols = min(max_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    # Figure size: roughly proportional to per-panel pixel size, with extra
    # vertical room for the per-panel title.
    fig_w = n_cols * 2.4
    fig_h = n_rows * 2.4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    for idx, k in enumerate(distinct):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]
        mol = Chem.MolFromSmiles(k)
        if mol is None:
            mol = Chem.MolFromSmiles("C")
        img = Draw.MolToImage(mol, size=sub_img_size)
        ax.imshow(np.array(img))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # Colored badge top-left (matches bar segment color)
        badge_sz = 0.14
        ax.add_patch(Rectangle(
            (0.03, 1.0 - badge_sz - 0.03), badge_sz, badge_sz,
            transform=ax.transAxes,
            facecolor=key_to_color[k],
            edgecolor="white",
            linewidth=1.5,
            zorder=10,
            clip_on=False,
        ))
        ax.set_title(_format_alpha_label(intervals_by_key[k]),
                     fontsize=9, pad=4)

    # Hide unused cells
    for idx in range(n, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].axis("off")
        for sp in axes[row][col].spines.values():
            sp.set_visible(False)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path, "({} distinct identities)".format(n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True,
                   help="Input JSON: list of {alpha, smiles, ...}")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", required=True,
                   help="Output filename prefix, e.g. 'figS14b_hiervae'")
    args = p.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    # Shared identity/color computation: ensures the bar and the per-segment
    # molgrid agree exactly on which color belongs to which identity.
    keys, distinct, key_to_color = compute_keys_and_colors(data)

    plot_bar(data, keys, key_to_color,
             os.path.join(args.out_dir, f"{args.label}_barplot.png"))
    plot_segment_molgrid(data, keys, key_to_color,
                         os.path.join(args.out_dir, f"{args.label}_segment_molgrid.png"))


if __name__ == "__main__":
    main()
