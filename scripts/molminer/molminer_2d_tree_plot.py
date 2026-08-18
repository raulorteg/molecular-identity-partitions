"""Multi-panel (logP, qed) tessellation plus per-step identity curves — Fig 4
and the Fig S13 companion.

The default combined layout draws 2x4 tessellation panels (every
COMBINED_TESS_STRIDE steps, up to COMBINED_TESS_COUNT distinct steps) above a
plot of distinct identities at each decoding step, for both root lineages and
in the same colors as the tessellation. Cell fill comes from root identity,
edges from current-step identity, and black boundaries mark identity changes.
Combined figure width is 210 mm; --classic selects a single-row tessellation
at 170 mm.

Invoked by repro/fig4_branching.sh and repro/figS13_molminer_lineage.sh.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from rdkit import Chem

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
JSONL_PATH = SCRIPT_DIR / "best_molminer-2026-tess.pth_tree.jsonl"
OUT_DIR = SCRIPT_DIR.parent / "figures"
OUT_PATH_DEFAULT = OUT_DIR / "2d_tree_plot.png"
OUT_PATH_LINEAGES = OUT_DIR / "2d_tree_plot_lineages.png"

# LaTeX-style text width (classic mode).
A4_LINEWIDTH_IN = 170.0 / 25.4
# Full A4 portrait page width — combined tessellation + curve + tree figure.
A4_SHEET_WIDTH_MM = 210.0
A4_COMBINED_WIDTH_IN = A4_SHEET_WIDTH_MM / 25.4
PANEL_SQUARE_HEIGHT_FACTOR = 1.32
# Tight packing for the 4×2 tessellation block (subgridspec w/hspace).
TESS_SUBGRID_WSPACE = 0.03
TESS_SUBGRID_HSPACE = 0.055

# Combined layout: eight tessellation steps at this stride (0-based indices 0, 6, 12, …).
COMBINED_TESS_STRIDE = 6
COMBINED_TESS_COUNT = 8
# Classic single-row mode stride (panels every N steps + last).
STEP_STRIDE_CLASSIC = 8

INVALID_KEY = ""

SCATTER_KW = dict(s=8, alpha=0.92, rasterized=True, marker="s", zorder=1)
EDGE_KW = dict(color="black", linewidth=1.0, alpha=1.0, zorder=3)


def _load_lineage_module():
    """Load the lineage plotter by path (its name is not importable as-is)."""
    spec = importlib.util.spec_from_file_location(
        "molminer_lineage_plot", SCRIPT_DIR / "molminer_lineage_plot.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load molminer_lineage_plot.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["molminer_lineage_plot"] = mod
    spec.loader.exec_module(mod)
    return mod


def canonical_smiles_key(smiles: str) -> str:
    """Return the project-standard molecular-identity key: RDKit canonical SMILES.

    Returns INVALID_KEY ("") for empty/unparseable input.
    """
    if not smiles or not str(smiles).strip():
        return INVALID_KEY
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return INVALID_KEY
    try:
        return Chem.MolToSmiles(mol) or INVALID_KEY
    except Exception:
        return INVALID_KEY


def key_for_smiles(smi: str, cache: dict[str, str]) -> str:
    if smi in cache:
        return cache[smi]
    k = canonical_smiles_key(smi)
    cache[smi] = k
    return k


def compute_lineage_fill(
    padded: list[list[str]],
    failed: list[bool],
    smi_cache: dict[str, str],
    n_cells: int,
) -> tuple[list[int | None], list[str], str, str | None]:
    roots: list[str | None] = []
    for i in range(n_cells):
        if failed[i]:
            roots.append(None)
            continue
        smi = padded[i][0] if padded[i] else ""
        rk = key_for_smiles(smi, smi_cache)
        roots.append(rk if rk != INVALID_KEY else None)

    valid = [r for r in roots if r is not None]
    cnt = Counter(valid)
    sorted_keys = [k for k, _ in cnt.most_common()]

    if not sorted_keys:
        return [None] * n_cells, [], "", None

    k0 = sorted_keys[0]
    k1 = sorted_keys[1] if len(sorted_keys) > 1 else None

    key_to_bin: dict[str, int] = {k0: 0}
    if k1 is not None:
        key_to_bin[k1] = 1
        for k in sorted_keys[2:]:
            key_to_bin[k] = 0

    bins: list[int | None] = []
    for i in range(n_cells):
        r = roots[i]
        if r is None:
            bins.append(None)
        else:
            bins.append(key_to_bin.get(r, 0))

    return bins, sorted_keys, k0, k1


def pad_intermediates(inter: list[str], length: int) -> list[str]:
    if not inter:
        return [""] * length
    last = inter[-1]
    if len(inter) >= length:
        return list(inter[:length])
    return list(inter) + [last] * (length - len(inter))


def plot_one_step(
    ax,
    z0,
    z1,
    identity_key_list: list[str],
    fill_colors: list,
    step_label: str,
    draw_edges: bool,
) -> None:
    if draw_edges:
        uz0, uz1 = np.unique(z0), np.unique(z1)
        keypt = lambda a, b: (round(float(a), 10), round(float(b), 10))
        pt2key = {
            keypt(z0[i], z1[i]): identity_key_list[i] for i in range(len(identity_key_list))
        }
        dy = (uz1[-1] - uz1[0]) / (len(uz1) - 1) if len(uz1) > 1 else 0
        dx = (uz0[-1] - uz0[0]) / (len(uz0) - 1) if len(uz0) > 1 else 0
        for i0 in range(len(uz0)):
            for i1 in range(len(uz1)):
                k = pt2key.get(keypt(uz0[i0], uz1[i1]))
                if k is None:
                    continue
                if i0 + 1 < len(uz0):
                    r = pt2key.get(keypt(uz0[i0 + 1], uz1[i1]))
                    if r is not None and r != k:
                        x = (uz0[i0] + uz0[i0 + 1]) / 2
                        ax.plot([x, x], [uz1[i1] - dy / 2, uz1[i1] + dy / 2], **EDGE_KW)
                if i1 + 1 < len(uz1):
                    r = pt2key.get(keypt(uz0[i0], uz1[i1 + 1]))
                    if r is not None and r != k:
                        y = (uz1[i1] + uz1[i1 + 1]) / 2
                        ax.plot([uz0[i0] - dx / 2, uz0[i0] + dx / 2], [y, y], **EDGE_KW)

    ax.scatter(z0, z1, c=fill_colors, **SCATTER_KW)
    ax.set_xlim(float(z0.min()), float(z0.max()))
    ax.set_ylim(float(z1.min()), float(z1.max()))
    ax.set_box_aspect(1)
    ax.set_title(step_label, fontsize=6.5, pad=2)
    ax.tick_params(axis="both", which="major", labelsize=5.5)


def plot_combined_lineages_figure(
    *,
    out_path: pathlib.Path,
    records: list[dict],
    padded: list[list[str]],
    failed: list[bool],
    max_len: int,
    logp: np.ndarray,
    qed: np.ndarray,
    smi_cache: dict[str, str],
    palette: list,
    lineage_bins: list[int | None],
    lineage_k1: str | None,
    n_keys: int,
    x_label: str = "logP",
    y_label: str = "qed",
) -> None:
    lin = _load_lineage_module()

    raw = [
        min(COMBINED_TESS_STRIDE * i, max(0, max_len - 1))
        for i in range(COMBINED_TESS_COUNT)
    ]
    step_indices = list(dict.fromkeys(raw))

    idx0 = [
        i
        for i in range(len(records))
        if not failed[i] and lineage_bins[i] is not None and lineage_bins[i] == 0
    ]
    idx1 = [
        i
        for i in range(len(records))
        if not failed[i] and lineage_bins[i] is not None and lineage_bins[i] == 1
    ]

    fig_w = A4_COMBINED_WIDTH_IN
    # 9 cols — tess 4 + curve 4 + narrow right pad (curve not flush to edge).
    _WR_TOP = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.28]
    _top_units = sum(_WR_TOP)
    col_w = fig_w / _top_units
    row_tess = col_w * PANEL_SQUARE_HEIGHT_FACTOR
    fig_h = 2.0 * row_tess

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.set_constrained_layout_pads(
        w_pad=0.018, h_pad=0.02, wspace=0.04, hspace=0.1
    )

    gs_top = fig.add_gridspec(
        2,
        9,
        width_ratios=_WR_TOP,
        wspace=TESS_SUBGRID_WSPACE,
        hspace=TESS_SUBGRID_HSPACE,
    )

    tess_axes: list = []
    for k in range(8):
        r, c = divmod(k, 4)
        ax = fig.add_subplot(gs_top[r, c])
        tess_axes.append(ax)
        if k >= len(step_indices):
            ax.set_visible(False)
            continue
        t = step_indices[k]
        identity_key_list: list[str] = []
        for i in range(len(records)):
            if failed[i]:
                identity_key_list.append(INVALID_KEY)
            else:
                identity_key_list.append(key_for_smiles(padded[i][t], smi_cache))

        fill_colors = [
            "#bfbfbf"
            if lineage_bins[i] is None
            else (palette[0] if lineage_bins[i] == 0 else palette[1])
            for i in range(len(records))
        ]

        plot_one_step(
            ax,
            logp,
            qed,
            identity_key_list,
            fill_colors,
            step_label=f"Step {t + 1}",
            draw_edges=True,
        )

    for ax in tess_axes:
        if ax.get_visible():
            ax.label_outer()

    bottom_vis = [k for k in (4, 5, 6, 7) if tess_axes[k].get_visible()]
    top_vis = [k for k in (0, 1, 2, 3) if tess_axes[k].get_visible()]
    if bottom_vis:
        tess_axes[bottom_vis[len(bottom_vis) // 2]].set_xlabel(
            x_label, fontsize=8, color="0.25"
        )
    elif top_vis:
        tess_axes[top_vis[len(top_vis) // 2]].set_xlabel(x_label, fontsize=8, color="0.25")
    if tess_axes[4].get_visible():
        tess_axes[4].set_ylabel(y_label, fontsize=8, color="0.25")
    elif tess_axes[0].get_visible():
        tess_axes[0].set_ylabel(y_label, fontsize=8, color="0.25")

    # Columns 4–7: distinct identities per step (column 8 is empty padding).
    ax_curve = fig.add_subplot(gs_top[0:2, 4:8])
    ax_curve.set_facecolor("white")
    steps = np.arange(max_len, dtype=np.float64)
    c0, _ = lin.distinct_identities_at_step(
        idx0, padded, failed, smi_cache, max_len
    )
    c1, _ = lin.distinct_identities_at_step(
        idx1, padded, failed, smi_cache, max_len
    )
    if max_len > 0:
        if lineage_k1 is not None:
            ax_curve.fill_between(
                steps, c0, alpha=0.14, color=palette[0], linewidth=0, zorder=1
            )
            ax_curve.fill_between(
                steps, c1, alpha=0.14, color=palette[1], linewidth=0, zorder=1
            )
            ax_curve.plot(steps, c0, color=palette[0], lw=1.35, zorder=2, label="Lineage A")
            ax_curve.plot(steps, c1, color=palette[1], lw=1.35, zorder=2, label="Lineage B")
        else:
            ax_curve.fill_between(
                steps, c0, alpha=0.14, color=palette[0], linewidth=0, zorder=1
            )
            ax_curve.plot(
                steps, c0, color=palette[0], lw=1.35, zorder=2, label="Lineage A"
            )
        xmax = float(max(max_len - 1, 1))
        ax_curve.set_xlim(0.0, xmax)
        ystack = np.concatenate([c0, c1]) if lineage_k1 is not None else c0
        y_hi = float(np.max(ystack))
        if y_hi <= 0.0:
            y_hi = 1.0
        span = y_hi
        ax_curve.set_ylim(0.0, y_hi + 0.08 * max(span, 1.0))
    else:
        ax_curve.set_xlim(0.0, 1.0)
        ax_curve.set_ylim(0.0, 1.0)
    ax_curve.set_xlabel("Decoding step", fontsize=7.5, color="0.25")
    ax_curve.set_ylabel("Distinct identities at step", fontsize=7.5, color="0.25")
    ax_curve.tick_params(axis="both", labelsize=6.5, colors="0.35")
    ax_curve.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax_curve.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_curve.grid(True, alpha=0.4, linewidth=0.5, color="0.72")
    ax_curve.set_axisbelow(True)
    for spine in ax_curve.spines.values():
        spine.set_edgecolor("0.78")
        spine.set_linewidth(0.75)
    if lineage_k1 is not None:
        ax_curve.legend(
            loc="lower right",
            fontsize=8.5,
            framealpha=0.92,
            edgecolor="0.82",
        )

    fig.suptitle(
        f"Partitioning during autorregressive decoding ({x_label} vs {y_label} slice)",
        fontsize=9,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches=None, pad_inches=0.04)
    plt.close(fig)

    print(
        f"Saved {out_path} (combined: {len(step_indices)} tess steps, "
        f"{n_keys} identities)"
    )


def main_classic_single_row(
    *,
    out_path: pathlib.Path,
    lineages: bool,
    records: list[dict],
    padded: list[list[str]],
    failed: list[bool],
    max_len: int,
    logp: np.ndarray,
    qed: np.ndarray,
    smi_cache: dict[str, str],
    x_label: str = "logP",
    y_label: str = "qed",
) -> None:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    palette = list(
        prop_cycle.by_key().get(
            "color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        )
    )
    invalid_fill = "#bfbfbf"

    lineage_bins: list[int | None] | None = None
    lineage_k0: str | None = None
    lineage_k1: str | None = None

    if lineages:
        lineage_bins, _sorted_roots, lineage_k0, lineage_k1 = compute_lineage_fill(
            padded, failed, smi_cache, len(records)
        )
        seen_all: set[str] = set()
        for t in range(max_len):
            for i in range(len(records)):
                k = INVALID_KEY if failed[i] else key_for_smiles(padded[i][t], smi_cache)
                seen_all.add(k)
        n_keys = len(seen_all)
    else:
        all_keys_ordered: list[str] = []
        seen: set[str] = set()
        for t in range(max_len):
            for i in range(len(records)):
                if failed[i]:
                    k = INVALID_KEY
                else:
                    k = key_for_smiles(padded[i][t], smi_cache)
                if k not in seen:
                    seen.add(k)
                    all_keys_ordered.append(k)

        n_keys = len(all_keys_ordered)
        if n_keys > len(palette):
            palette = (palette * ((n_keys // len(palette)) + 1))[:n_keys]
        key_to_color = {k: palette[i] for i, k in enumerate(all_keys_ordered)}

    step_indices = list(range(0, max_len, STEP_STRIDE_CLASSIC))
    if max_len > 0 and step_indices[-1] != max_len - 1:
        step_indices.append(max_len - 1)
    n_panels = len(step_indices)

    panel_w_in = A4_LINEWIDTH_IN / max(n_panels, 1)
    fig_h = panel_w_in * PANEL_SQUARE_HEIGHT_FACTOR

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(A4_LINEWIDTH_IN, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.015, h_pad=0.025, wspace=0.055, hspace=0.0)
    axes_flat = axes.ravel()

    for plot_idx, t in enumerate(step_indices):
        ax = axes_flat[plot_idx]
        identity_key_list: list[str] = []
        for i in range(len(records)):
            if failed[i]:
                identity_key_list.append(INVALID_KEY)
            else:
                identity_key_list.append(key_for_smiles(padded[i][t], smi_cache))

        if lineages:
            fill_colors = [
                invalid_fill
                if lineage_bins[i] is None
                else (palette[0] if lineage_bins[i] == 0 else palette[1])
                for i in range(len(records))
            ]
        else:
            fill_colors = [key_to_color[k] for k in identity_key_list]

        plot_one_step(
            ax,
            logp,
            qed,
            identity_key_list,
            fill_colors,
            step_label=f"Step {t + 1}",
            draw_edges=True,
        )

    title = f"Tessellation by decoding step ({x_label} vs {y_label})"
    if lineages:
        title += ". Lineages"
    fig.suptitle(title, fontsize=9)
    fig.supxlabel(x_label, fontsize=8)
    fig.supylabel(y_label, fontsize=8)
    fig.align_labels()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    msg = (
        f"Saved {out_path} ({len(records)} points/panel, {max_len} steps max, "
        f"{n_panels} panels every {STEP_STRIDE_CLASSIC} steps, {n_keys} global identities)"
    )
    if lineages:
        extra = f" lineages: root0={lineage_k0!r}"
        if lineage_k1 is not None:
            extra += f" root1={lineage_k1!r}; rarer roots→lineage0"
        else:
            extra += " (single root lineage)"
        msg += ";" + extra
    print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D tree tessellation plots from JSONL.")
    parser.add_argument(
        "--lineages",
        action="store_true",
        help="Classic mode: color fills by root lineage (two lineages).",
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help="Single row of tessellation panels only (no identity curves).",
    )
    parser.add_argument(
        "--jsonl", type=pathlib.Path, default=None,
        help="Input lineage JSONL. If omitted, uses "
             "best_molminer-2026-tess.pth_tree.jsonl next to this script.",
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default=None,
        help="Output PNG path (default: figures/2d_tree_plot.png, or _lineages in classic mode).",
    )
    parser.add_argument(
        "--x_label", default=None,
        help="x-axis label (default: derived from the JSONL x_prop, else 'logP').",
    )
    parser.add_argument(
        "--y_label", default=None,
        help="y-axis label (default: derived from the JSONL y_prop, else 'qed').",
    )
    args = parser.parse_args()
    lineages = args.lineages
    classic = args.classic
    jsonl_path = args.jsonl if args.jsonl is not None else JSONL_PATH

    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise SystemExit(f"No records in {jsonl_path}")

    max_len = max(len(r.get("intermediates") or []) for r in records)
    if max_len == 0:
        raise SystemExit("All intermediates empty; nothing to plot.")

    # Generic axes: records from a non-logP/qed slice carry x_prop/y_prop and
    # store coords under those property names; records without them use logP/qed.
    x_key = records[0].get("x_prop", "logP")
    y_key = records[0].get("y_prop", "qed")
    x_label = args.x_label if args.x_label is not None else x_key
    y_label = args.y_label if args.y_label is not None else y_key
    logp = np.array([float(r[x_key]) for r in records], dtype=np.float64)
    qed = np.array([float(r[y_key]) for r in records], dtype=np.float64)
    failed = [bool(r.get("failed")) for r in records]

    padded: list[list[str]] = []
    for r in records:
        inter = r.get("intermediates") or []
        padded.append(pad_intermediates(inter, max_len))

    smi_cache: dict[str, str] = {}

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    palette = list(
        prop_cycle.by_key().get(
            "color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        )
    )

    if classic:
        if args.out is not None:
            out_path = args.out
        else:
            out_path = OUT_PATH_LINEAGES if lineages else OUT_PATH_DEFAULT
        main_classic_single_row(
            out_path=out_path,
            lineages=lineages,
            records=records,
            padded=padded,
            failed=failed,
            max_len=max_len,
            logp=logp,
            qed=qed,
            smi_cache=smi_cache,
            x_label=x_label,
            y_label=y_label,
        )
        return

    lineage_bins, _sorted_roots, _lineage_k0, lineage_k1 = compute_lineage_fill(
        padded, failed, smi_cache, len(records)
    )
    seen_all: set[str] = set()
    for t in range(max_len):
        for i in range(len(records)):
            k = INVALID_KEY if failed[i] else key_for_smiles(padded[i][t], smi_cache)
            seen_all.add(k)
    n_keys = len(seen_all)

    plot_combined_lineages_figure(
        out_path=args.out if args.out is not None else OUT_PATH_DEFAULT,
        records=records,
        padded=padded,
        failed=failed,
        max_len=max_len,
        logp=logp,
        qed=qed,
        smi_cache=smi_cache,
        palette=palette,
        lineage_bins=lineage_bins,
        lineage_k1=lineage_k1,
        n_keys=n_keys,
        x_label=x_label,
        y_label=y_label,
    )


if __name__ == "__main__":
    main()
