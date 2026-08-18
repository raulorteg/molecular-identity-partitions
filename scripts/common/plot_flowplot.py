"""Flow plotter for Fig 2 and Figs S7-S9.

Renders a probabilistic walk as a stacked bar chart: one bar per alpha step,
segments sized by molecule frequency and ordered by global frequency (most
common at the bottom). With --grouped-gray, molecules totalling under 10 counts
collapse into one gray band.

Reads a walk CSV headed alpha, smiles, canonical_smiles (other columns ignored).

--convention defaults to "canonical", using the canonical_smiles column
verbatim. Any other convention buckets by equiv_key(smiles, convention) and
suffixes the output stem with _<convention>.

Invoked by repro/fig2_flows.sh and repro/figS{7,8,9}_*_flows_coarsen.sh.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import sys
from collections import Counter, defaultdict

import tempfile

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
from matplotlib.patches import Rectangle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.equivalence import (
    CONVENTIONS, INVALID_KEY, NO_SCAFFOLD_KEY,
    equiv_key, representative_smiles_for_class,
)

_palette = [plt.cm.tab20(i / 20) for i in range(20)] + \
           [plt.cm.tab20b(i / 20) for i in range(20)] + \
           [plt.cm.tab20c(i / 20) for i in range(20)]
random.seed(0)
random.shuffle(_palette)
PALETTE = _palette
REST_COLOR = (0.6, 0.6, 0.6)  # gray for molecules appearing < 10 times total
RARE_THRESHOLD = 10


def read_csv(csv_path: str, convention: str = "canonical"):
    """Return (data, class_to_rep_smiles).

    data: dict alpha (float) -> Counter of class_label (under `convention`).
    class_to_rep_smiles: dict class_label -> a canonical-SMILES representative
      for the molgrid panel (the first canonical_smiles encountered for that
      class). For canonical convention this maps each smi to itself.
    """
    data = defaultdict(list)
    class_to_rep_smiles: dict[str, str] = {}
    opener = gzip.open if csv_path.endswith(".gz") else open
    with opener(csv_path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alpha = round(float(row["alpha"]), 6)
            canon = (row.get("canonical_smiles") or "").strip()
            raw = (row.get("smiles") or "").strip()
            if convention == "canonical":
                # Fast path: trust the pre-baked canonical_smiles column.
                key = canon if canon else INVALID_KEY
            else:
                # Recompute from the raw smiles column.
                source = raw if raw else canon
                key = equiv_key(source, convention)
            data[alpha].append(key)
            if key not in class_to_rep_smiles:
                if convention == "canonical":
                    class_to_rep_smiles[key] = canon or INVALID_KEY
                else:
                    class_to_rep_smiles[key] = (
                        representative_smiles_for_class(raw or canon, convention)
                    )
    return {a: Counter(v) for a, v in sorted(data.items())}, class_to_rep_smiles


def save_molecule_grid(
    plot_order: list[str],
    mol_to_color: dict[str, tuple],
    out_path: str,
    top_n: int = 4,
    sub_img_size: tuple[int, int] = (440, 320),
    class_to_rep_smiles: dict[str, str] | None = None,
):
    """Save a grid of the top-N most common classes (RDKit 2D) with a colored square on each (flow-plot color). Returns True if saved.

    For canonical convention `plot_order` entries are themselves SMILES. For
    other conventions the entries are class labels (InChIKey, formula, ...) and
    `class_to_rep_smiles` must provide a SMILES representative per class.
    """
    classes_to_show = [m for m in plot_order if m not in (INVALID_KEY, "__invalid__")][:top_n]
    if not classes_to_show:
        return False

    mols = []
    colors = []
    for cls in classes_to_show:
        if class_to_rep_smiles is not None:
            rep = class_to_rep_smiles.get(cls, cls)
        else:
            rep = cls
        if rep in (INVALID_KEY, NO_SCAFFOLD_KEY, "__invalid__") or not rep:
            mol = Chem.MolFromSmiles("C")
        else:
            mol = Chem.MolFromSmiles(rep)
            if mol is None:
                mol = Chem.MolFromSmiles("C")  # placeholder
        mols.append(mol)
        colors.append(mol_to_color[cls])

    # single column; height matches the flow plot
    n_cols = 1
    n_rows = len(mols)
    fig_height = 2.25
    fig_width = 2.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if len(mols) == 1:
        axes = np.array([axes])
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    for idx, (mol, color) in enumerate(zip(mols, colors)):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, 0] if n_rows > 1 else axes[0]
        # render via a tempfile path; works across rdkit versions
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            Draw.MolToFile(mol, tmp_path,
                           size=(sub_img_size[0], sub_img_size[1]),
                           imageType="png")
            img = Image.open(tmp_path).convert("RGB")
            img.load()
        finally:
            os.unlink(tmp_path)
        ax.imshow(np.array(img))
        # colored square at top-left
        rect = Rectangle((0, 1 - 0.12), 0.12, 0.12, transform=ax.transAxes,
                          facecolor=color, edgecolor="none", zorder=10)
        ax.add_patch(rect)
        ax.set_axis_off()


    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--no-collapse", action="store_true",
                   help="Give every unique molecule a color (cycle tab10); no gray bucket.")
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of top molecules to color distinctly (default: 10). Ignored with --no-collapse.")
    p.add_argument("--out", default=None, help="Output PNG path (default: <csv>_walk.png)")
    p.add_argument("--grouped-gray", action="store_true",
                   help="Group molecules with total count < 10 into a single gray 'rest' segment.")
    p.add_argument("--mol-grid-n", type=int, default=4,
                   help="Number of top molecules to show in the RDKit molecule grid plot (default: 4).")
    p.add_argument("--convention", default="canonical", choices=list(CONVENTIONS),
                   help="Equivalence relation used to bucket walk samples (default: canonical).")
    args = p.parse_args()

    data, class_to_rep_smiles = read_csv(args.csv, args.convention)
    alphas = sorted(data.keys())
    n_steps = len(alphas)

    # global frequency across all alpha steps
    global_counts: Counter = Counter()
    for counter in data.values():
        global_counts.update(counter)

    if args.grouped_gray:
        rare_mols = {mol for mol, c in global_counts.items() if c < RARE_THRESHOLD}
        plot_order = [mol for mol, _ in global_counts.most_common() if mol not in rare_mols]
    else:
        rare_mols = set()
        plot_order = [mol for mol, _ in global_counts.most_common()]
    mol_to_color = {mol: PALETTE[i % len(PALETTE)] for i, mol in enumerate(plot_order)}

    # total samples per alpha (for normalization)
    totals = {a: sum(data[a].values()) for a in alphas}

    fig, ax = plt.subplots(figsize=(3.67, 2.25))

    bar_width = (alphas[-1] - alphas[0]) / n_steps * 1.005 if n_steps > 1 else 0.02

    for alpha in alphas:
        counter = data[alpha]
        total = totals[alpha]
        bottom = 0.0

        segments = [(mol, counter.get(mol, 0) / total) for mol in plot_order if counter.get(mol, 0) > 0]
        if rare_mols:
            rest_freq = sum(counter.get(mol, 0) for mol in rare_mols) / total
            if rest_freq > 0:
                segments.append(("__rest__", rest_freq))

        for mol_or_rest, freq in segments:
            color = REST_COLOR if mol_or_rest == "__rest__" else mol_to_color[mol_or_rest]
            # Rasterize the bar segments only: a coarse-convention walk can
            # draw ~10^6 sub-pixel rectangles. Axes/ticks/labels stay vector.
            bars = ax.bar(alpha, freq, width=bar_width, bottom=bottom,
                          color=color, linewidth=0, zorder=2)
            for patch in bars:
                patch.set_rasterized(True)
            bottom += freq

    ax.set_xlabel(r"$\alpha$", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_xlim(alphas[0] - bar_width, alphas[-1] + bar_width)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", labelsize=8)
    fig.tight_layout(pad=0)

    out_path = args.out if args.out else args.csv.rstrip("/") + "_walk.png"
    base_path = os.path.splitext(out_path)[0]
    if args.convention != "canonical":
        base_path = base_path + f"_{args.convention}"
    out_dir = os.path.dirname(base_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_kw = dict(bbox_inches=None, pad_inches=0)
    # dpi governs the rasterized bar layer; text stays vector
    fig.savefig(base_path + ".png", dpi=600, **save_kw)
    plt.close(fig)

    # Pie charts at alpha=0.0 and alpha=1.0 with same colors as flow plot
    alpha_start = 0.0 if 0.0 in data else alphas[0]
    alpha_end = 1.0 if 1.0 in data else alphas[-1]
    pie_suffix = "" if args.convention == "canonical" else f"_{args.convention}"
    pie_paths = []
    for alpha_val, name in [
        (alpha_start, f"pie_start{pie_suffix}.png"),
        (alpha_end, f"pie_end{pie_suffix}.png"),
    ]:
        counter = data[alpha_val]
        segments = [(mol, counter.get(mol, 0)) for mol in plot_order if counter.get(mol, 0) > 0]
        if rare_mols:
            rest_count = sum(counter.get(mol, 0) for mol in rare_mols)
            if rest_count > 0:
                segments.append(("__rest__", rest_count))
        if not segments:
            continue
        sizes = [c for _, c in segments]
        colors = [REST_COLOR if mol == "__rest__" else mol_to_color[mol] for mol, _ in segments]
        fig_pie, ax_pie = plt.subplots(figsize=(2.2, 2.2))
        ax_pie.pie(sizes, colors=colors, labels=None, autopct="")
        ax_pie.axis("equal")
        fig_pie.tight_layout(pad=0)
        pie_path = os.path.join(out_dir, name) if out_dir else name
        fig_pie.savefig(pie_path, dpi=600, **save_kw)
        plt.close(fig_pie)
        pie_paths.append(pie_path)

    n_unique = len(plot_order) + (len(rare_mols) if rare_mols else 0)
    n_total = sum(global_counts.values())
    print(f"Saved {base_path}.png ({n_steps} alpha steps, {n_total} total samples, {n_unique} classes under convention={args.convention})")
    if pie_paths:
        print(f"Saved pie charts: {', '.join(pie_paths)}")

    # Grid of top molecules with flow-plot colored squares
    molgrid_path = base_path + "_molgrid.png"
    if save_molecule_grid(plot_order, mol_to_color, molgrid_path, top_n=args.mol_grid_n,
                          class_to_rep_smiles=class_to_rep_smiles):
        print(f"Saved molecule grid: {molgrid_path}")


if __name__ == "__main__":
    main()