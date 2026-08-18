"""HierVAE tessellation evolution across training steps — Fig S11.

Draws the slice ladder, with the top-6 persistent identities as one enlarged
row above it. Identity is RDKit canonical SMILES. Only every --stride-th
checkpoint is shown (default 2), with the final checkpoint always pinned in,
and panels are packed into a grid with no incomplete rows (--cols, or an auto
near-square factorization).

Reads 2d-slices/tessellation_{step}.txt relative to the working directory,
headed z0, z1, smiles, inchi, inchikey_first.

Invoked by repro/figS11_hiervae_evolution.sh.
"""

import argparse
import csv
import os
import re
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

# constants
INVALID_KEY = ""
BLOB_DIR    = "2d-slices"
FILE_RE     = re.compile(r"tessellation_(\d+)\.txt$")

# font sizes (for a full-page figure)
FONT_SM = 9    # panel labels
FONT_LG = 13   # shared tessellation label

# figure geometry
FIG_WIDTH = 8.3  # inches — fits A4 and US Letter with margins
ROW_TITLE = 0.4  # budget for the shared tessellation title (inches)

TOP_COLORS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#CC79A7",  # pink
    "#D55E00",  # vermillion
    "#0072B2",  # deep blue
]
GRAY = "#AAAAAA"


# formatting
def fmt_step(s):
    """Compact thousands label: 5000 -> 5k, 105000 -> 105k, 240000 -> 240k.

    Every panel carries a label, so use a format that distinguishes adjacent
    multiples of 1000 (1.05e5 and 1.15e5 would both round to "1.1e5")."""
    if s >= 1000:
        return f"{int(round(s / 1000))}k"
    return str(int(s))


# I/O
def discover_files(directory):
    """Return sorted list of (step, path) for all matching files."""
    entries = []
    for fname in os.listdir(directory):
        m = FILE_RE.match(fname)
        if m:
            entries.append((int(m.group(1)), os.path.join(directory, fname)))
    entries.sort(key=lambda x: x[0])
    return entries


def canonical_smiles(smi):
    """Return RDKit canonical SMILES, or INVALID_KEY if parsing fails."""
    if not smi:
        return INVALID_KEY
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return INVALID_KEY
    return Chem.MolToSmiles(mol)


def read_file(path):
    z0, z1, key_list = [], [], []
    key_to_smiles = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            z0.append(float(row["z0"]))
            z1.append(float(row["z1"]))
            smi = (row.get("smiles") or "").strip()
            k = canonical_smiles(smi)
            key_list.append(k)
            if k != INVALID_KEY and k not in key_to_smiles:
                key_to_smiles[k] = k
    return np.array(z0), np.array(z1), key_list, key_to_smiles


# top-K persistent blobs
def find_top_persistent(per_step, top_k=6):
    appearance_count = Counter()
    territory_size   = Counter()
    for key_list, _ in per_step.values():
        for k in set(key_list):
            appearance_count[k] += 1
        for k in key_list:
            territory_size[k] += 1
    appearance_count.pop(INVALID_KEY, None)
    ranked = sorted(
        appearance_count.keys(),
        key=lambda k: (appearance_count[k], territory_size[k]),
        reverse=True,
    )
    return ranked[:top_k]


# grid choice
def choose_grid(n, cols=0):
    """Return (cols, rows, n_used) for a full rectangular grid of n panels.

    If cols>0 it is honored and trailing panels are trimmed to fill whole rows.
    Otherwise pick the near-square *landscape* factorization of n; if n has no
    such factorization (e.g. prime), fall back to ~sqrt(n) columns and trim.
    """
    if cols and cols > 0:
        n_used = (n // cols) * cols or n
        return cols, -(-n_used // cols), n_used
    divs = [c for c in range(1, n + 1) if n % c == 0 and c >= n // c]
    best = min(divs, key=lambda c: c - n // c)
    rows = n // best
    if rows >= 2 and best >= 2:
        return best, rows, n
    cols = max(2, int(round(n ** 0.5)))
    n_used = (n // cols) * cols
    return cols, n_used // cols, n_used


# plotting helpers
def mol_to_image(smi, size=(300, 300)):
    """Render SMILES to PIL Image via RDKit (tempfile path for rdkit compat)."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return Image.new("RGB", size, "white")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Draw.MolToFile(mol, tmp_path, size=size, imageType="png")
        img = Image.open(tmp_path).convert("RGB")
        img.load()  # force read before file is deleted
    finally:
        os.remove(tmp_path)
    return img


def draw_mol_panel(ax, smi, color):
    """Molecule drawing, no spines, with a colored square badge top-left."""
    img = mol_to_image(smi)
    ax.imshow(np.array(img))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    badge_sz = 0.18
    badge = mpatches.Rectangle(
        (0.03, 1.0 - badge_sz - 0.03), badge_sz, badge_sz,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        zorder=10,
        clip_on=False,
    )
    ax.add_patch(badge)


def draw_mol_row(fig, outer_ss, top_keys, key_to_color):
    """Single full-width row of the top-K persistent molecules, enlarged."""
    n = len(top_keys)
    inner = gridspec.GridSpecFromSubplotSpec(
        1, n, subplot_spec=outer_ss, wspace=0.05,
    )
    for i, k in enumerate(top_keys):
        ax = fig.add_subplot(inner[0, i])
        draw_mol_panel(ax, k, key_to_color[k])


def draw_blob_edges(ax, z0, z1, key_list):
    """Draw thin black lines at boundaries between cells of different identity."""
    uz0, uz1 = np.unique(z0), np.unique(z1)
    if len(uz0) < 2 or len(uz1) < 2:
        return
    keypt  = lambda a, b: (round(float(a), 10), round(float(b), 10))
    pt2key = {keypt(z0[i], z1[i]): key_list[i] for i in range(len(key_list))}
    dx = (uz0[-1] - uz0[0]) / (len(uz0) - 1)
    dy = (uz1[-1] - uz1[0]) / (len(uz1) - 1)
    kw = dict(color="black", linewidth=0.3, alpha=0.6, zorder=3)
    for i0 in range(len(uz0)):
        for i1 in range(len(uz1)):
            k = pt2key.get(keypt(uz0[i0], uz1[i1]))
            if k is None:
                continue
            if i0 + 1 < len(uz0):
                r = pt2key.get(keypt(uz0[i0 + 1], uz1[i1]))
                if r is not None and r != k:
                    x = (uz0[i0] + uz0[i0 + 1]) / 2
                    ax.plot([x, x], [uz1[i1] - dy / 2, uz1[i1] + dy / 2], **kw)
            if i1 + 1 < len(uz1):
                r = pt2key.get(keypt(uz0[i0], uz1[i1 + 1]))
                if r is not None and r != k:
                    y = (uz1[i1] + uz1[i1 + 1]) / 2
                    ax.plot([uz0[i0] - dx / 2, uz0[i0] + dx / 2], [y, y], **kw)


def draw_tessellation(ax, z0, z1, key_list, key_to_color, label=None):
    colors = [key_to_color.get(k, "black" if k == INVALID_KEY else GRAY)
              for k in key_list]
    ax.scatter(z0, z1, c=colors, s=2, alpha=0.9,
               rasterized=True, marker="s", linewidths=0, zorder=1)
    draw_blob_edges(ax, z0, z1, key_list)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label is not None:
        ax.text(0.04, 0.96, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=FONT_SM - 2, color="#333333",
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none",
                          alpha=0.7), zorder=5)


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=0,
                    help="Panels per row (0 = auto near-square, default 0)")
    ap.add_argument("--top",  type=int, default=6,
                    help="Top-K persistent blobs to color (default 6)")
    ap.add_argument("--stride", type=int, default=2,
                    help="Show every S-th checkpoint (default 2 = 1 of 2)")
    ap.add_argument("--no-labels", action="store_true",
                    help="Hide the small per-panel step labels")
    ap.add_argument("--out",  default="tessellation_evolution_mod.png",
                    help="Output figure path")
    args = ap.parse_args()

    # discover checkpoints
    file_list = discover_files(BLOB_DIR)
    if not file_list:
        print(f"No matching files found in {BLOB_DIR}/")
        return
    all_steps = [s for s, _ in file_list]
    print(f"Found {len(file_list)} checkpoints: steps {all_steps}")

    # subsample (1 of --stride), then pin the final checkpoint
    sel = file_list[::max(1, args.stride)]
    cols, rows, n_used = choose_grid(len(sel), args.cols)
    if n_used < len(sel):
        print(f"  dropping {len(sel) - n_used} earliest panel(s) for a full "
              f"{cols}x{rows} grid")
        sel = sel[len(sel) - n_used:]   # front-trim: keep the latest checkpoints
    if sel and sel[-1][0] != file_list[-1][0]:
        sel[-1] = file_list[-1]  # keep the training endpoint
    print(f"Showing {len(sel)} panels in a {cols}x{rows} grid: "
          f"steps {[s for s, _ in sel]}")

    # load selected files
    per_step = OrderedDict()
    coords   = OrderedDict()
    for step, path in sel:
        z0, z1, key_list, key_to_smiles = read_file(path)
        per_step[step] = (key_list, key_to_smiles)
        coords[step]   = (z0, z1)
        print(f"  step {step:>8} ({fmt_step(step)}): "
              f"{len(set(key_list))} unique blobs, {len(key_list)} grid points")

    # top-K persistent blobs (over the shown panels)
    top_keys     = find_top_persistent(per_step, top_k=args.top)
    key_to_color = {k: TOP_COLORS[i] for i, k in enumerate(top_keys)}
    print(f"\nTop {len(top_keys)} persistent blobs:")
    for k, c in key_to_color.items():
        n_steps = sum(1 for sl, _ in per_step.values() if k in set(sl))
        print(f"  {k[:40]}  color={c}  present in {n_steps}/{len(sel)} panels")

    # figure geometry
    tess_wspace = 0.04
    panel_w     = FIG_WIDTH / (cols + (cols - 1) * tess_wspace)
    ROW_TESS    = panel_w
    ROW_MOL     = FIG_WIDTH / len(top_keys)  # square-ish full-width mol row

    fig_h = ROW_MOL + ROW_TITLE + rows * ROW_TESS
    fig   = plt.figure(figsize=(FIG_WIDTH, fig_h))

    outer = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[ROW_MOL, ROW_TITLE, rows * ROW_TESS],
        hspace=0.0,
    )

    # row 0: full-width molecule strip
    draw_mol_row(fig, outer[0], top_keys, key_to_color)

    # row 1: shared italic title
    ax_title = fig.add_subplot(outer[1])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.0,
        "Evolution of the internal arrangement during training.",
        ha="center", va="bottom",
        fontsize=FONT_LG, fontstyle="italic",
        transform=ax_title.transAxes,
    )

    # row 2: rectangular tessellation grid
    tess_gs = gridspec.GridSpecFromSubplotSpec(
        rows, cols,
        subplot_spec=outer[2],
        wspace=tess_wspace, hspace=tess_wspace,
    )
    for idx, (step, _) in enumerate(sel):
        r, c = divmod(idx, cols)
        ax = fig.add_subplot(tess_gs[r, c])
        z0, z1      = coords[step]
        key_list, _ = per_step[step]
        label = None if args.no_labels else fmt_step(step)
        draw_tessellation(ax, z0, z1, key_list, key_to_color, label=label)

    out_dir = os.path.dirname(str(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.splitext(str(args.out))[0] + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
