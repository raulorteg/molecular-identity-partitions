"""Lineage decoding DAG for Fig 4 and Fig S13.

Reads the tree-tessellation JSONL, bins roots into two lineages (same rules as
molminer_2d_tree_plot.py), builds a layered DAG over RDKit canonical SMILES,
and draws the structures and arrows.

Fixed at 297 mm wide; height is the minimum that fits the padded graph extent.

Invoked by repro/fig4_branching.sh and repro/figS13_molminer_lineage.sh.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR.parent / "figures" / "lineage.png"
JSONL_PATH = SCRIPT_DIR / "best_molminer-2026-tess.pth_tree.jsonl"

INVALID_KEY = ""


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

# Full paper width (A4 landscape long side, mm → in). Height is computed per graph.
PAPER_WIDTH_IN = 297.0 / 25.4
# Layout coordinates scaled as if drawn at this reference width (in).
LAYOUT_SCALE = PAPER_WIDTH_IN / 18.0

_BASE_LAYER_DX = 16.0
_BASE_LAYER_DY = 10.5
_BASE_SUBLINEAGE_GAP = 4.0
LAYER_DX = _BASE_LAYER_DX * LAYOUT_SCALE
LAYER_DY = _BASE_LAYER_DY * LAYOUT_SCALE
SUBLINEAGE_GAP = _BASE_SUBLINEAGE_GAP * LAYOUT_SCALE
CROSSING_MIN_ITERATIONS = 24
FIG_DPI = 600

# Molecule appearance, scaled by LAYOUT_SCALE in main(). MOL_IMG_* is the RDKit
# MolToImage raster in px; MOL_ZOOM_BASE goes to OffsetImage(zoom=...). Tune
# both with --mol-scale (default 1.2).
MOL_IMG_WIDTH = 200
MOL_IMG_HEIGHT = 148
MOL_IMG_MIN_WIDTH = 48
MOL_IMG_MIN_HEIGHT = 36
MOL_ZOOM_BASE = 0.30

# Figure height (in): guardrails around the aspect-predicted height.
FIG_HEIGHT_MIN_IN = 1.35
FIG_HEIGHT_MAX_IN = 210.0 / 25.4  # one A4 short side, tall DAGs only


def _padded_bounds_xy(
    pos: dict[str, np.ndarray], sc: float
) -> tuple[float, float, float, float] | None:
    """Axis-aligned bounds (xmin, xmax, ymin, ymax) in data space with layout padding."""
    if not pos:
        return None
    xs = [float(pos[n][0]) for n in pos]
    ys = [float(pos[n][1]) for n in pos]
    rx = max(xs) - min(xs)
    ry = max(ys) - min(ys)
    # asymmetric padding: tighter vertical, room for halos/arrows
    pad_x = max(4.0 * sc, rx * 0.055 + 6.5 * sc) + 2.0 * sc
    pad_y = max(2.5 * sc, ry * 0.032 + 2.8 * sc) + 1.1 * sc
    return (
        min(xs) - pad_x,
        max(xs) + pad_x,
        min(ys) - pad_y,
        max(ys) + pad_y,
    )


def molecule_render_params(
    layout_scale: float, mol_scale: float = 1.2
) -> tuple[tuple[int, int], float]:
    """Return ``(mol_size_px, offsetimage_zoom)`` for :func:`draw_lineage_on_axes`."""
    s = layout_scale * mol_scale
    mol_size = (
        max(MOL_IMG_MIN_WIDTH, int(MOL_IMG_WIDTH * s)),
        max(MOL_IMG_MIN_HEIGHT, int(MOL_IMG_HEIGHT * s)),
    )
    mol_zoom = MOL_ZOOM_BASE * s
    return mol_size, mol_zoom


def _figure_height_in_for_paper_width(pos: dict[str, np.ndarray], sc: float) -> float:
    """Minimum figure height (in) so axes aspect matches padded graph extent at ``PAPER_WIDTH_IN``."""
    b = _padded_bounds_xy(pos, sc)
    if b is None:
        return 3.0
    w = b[1] - b[0]
    h = b[3] - b[2]
    w = max(w, 1e-6)
    y = PAPER_WIDTH_IN * (h / w)
    return max(FIG_HEIGHT_MIN_IN, min(y, FIG_HEIGHT_MAX_IN))


def load_records(jsonl_path: pathlib.Path = JSONL_PATH) -> tuple[list[dict], list[list[str]], list[bool], int]:
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
        raise SystemExit("Empty intermediates.")
    failed = [bool(r.get("failed")) for r in records]
    padded = [
        pad_intermediates(r.get("intermediates") or [], max_len) for r in records
    ]
    return records, padded, failed, max_len


def distinct_identity_count(
    indices: list[int],
    padded: list[list[str]],
    failed: list[bool],
    smi_cache: dict[str, str],
) -> int:
    seen: set[str] = set()
    for i in indices:
        if failed[i]:
            continue
        for smi in padded[i]:
            k = key_for_smiles(smi, smi_cache)
            if k != INVALID_KEY:
                seen.add(k)
    return len(seen)


def distinct_identities_at_step(
    indices: list[int],
    padded: list[list[str]],
    failed: list[bool],
    smi_cache: dict[str, str],
    max_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step distinct canonical SMILES among ``indices`` (used by molminer_2d_tree_plot.py)."""
    if max_len <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    counts = np.zeros(max_len, dtype=np.float64)
    for s in range(max_len):
        seen: set[str] = set()
        for i in indices:
            if failed[i]:
                continue
            k = key_for_smiles(padded[i][s], smi_cache)
            if k != INVALID_KEY:
                seen.add(k)
        counts[s] = float(len(seen))
    steps = np.arange(max_len, dtype=np.float64)
    return counts, steps


def build_graph_for_lineage(
    indices: list[int],
    padded: list[list[str]],
    failed: list[bool],
    smi_cache: dict[str, str],
) -> tuple[nx.DiGraph, dict[str, str]]:
    G = nx.DiGraph()
    key_to_smi: dict[str, str] = {}

    def note(k: str, smi: str) -> None:
        if k == INVALID_KEY or not smi or not str(smi).strip():
            return
        if k not in key_to_smi:
            key_to_smi[k] = smi.strip()

    for i in indices:
        if failed[i]:
            continue
        inter = padded[i]
        for t in range(len(inter)):
            smi = inter[t]
            k = key_for_smiles(smi, smi_cache)
            if k != INVALID_KEY:
                G.add_node(k)
                note(k, smi)
        for t in range(len(inter) - 1):
            kp = key_for_smiles(inter[t], smi_cache)
            kc = key_for_smiles(inter[t + 1], smi_cache)
            if kp == INVALID_KEY or kc == INVALID_KEY or kp == kc:
                continue
            G.add_edge(kp, kc)

    return G, key_to_smi


def dag_levels(G: nx.DiGraph) -> dict[str, int]:
    nodes = list(G.nodes())
    if not nodes:
        return {}
    try:
        order = list(nx.topological_sort(G))
    except Exception:
        order = list(nodes)
    level: dict[str, int] = {}
    for n in order:
        preds = list(G.predecessors(n))
        level[n] = max((level[p] + 1 for p in preds), default=0)
    return level


def _layers_sorted(level: dict[str, int]) -> dict[int, list[str]]:
    layers: dict[int, list[str]] = {}
    for n, lv in level.items():
        layers.setdefault(lv, []).append(n)
    for lv in layers:
        layers[lv].sort()
    return layers


def _median_index(idxs: list[int]) -> float:
    if not idxs:
        return 0.0
    s = sorted(idxs)
    m = len(s)
    mid = m // 2
    if m % 2:
        return float(s[mid])
    return 0.5 * (s[mid - 1] + s[mid])


def minimize_crossings_median_sweeps(
    G: nx.DiGraph,
    level: dict[str, int],
    layers: dict[int, list[str]],
    *,
    iterations: int,
) -> dict[int, list[str]]:
    """Reorder nodes within each layer (median heuristic) to reduce edge crossings."""
    max_lv = max(level.values(), default=0)
    out: dict[int, list[str]] = {lv: list(ns) for lv, ns in layers.items()}
    for _ in range(iterations):
        # Down: layer lv uses median predecessor index in layer lv - 1.
        for lv in range(1, max_lv + 1):
            if lv not in out or not out[lv]:
                continue
            prev = out.get(lv - 1, [])
            ip = {p: i for i, p in enumerate(prev)}
            old_i = {n: i for i, n in enumerate(out[lv])}

            def key_down(n: str) -> tuple[float, str]:
                preds = [p for p in G.predecessors(n) if level.get(p) == lv - 1]
                idxs = [ip[p] for p in preds if p in ip]
                med = _median_index(idxs) if idxs else float(old_i[n])
                return (med, n)

            out[lv] = sorted(out[lv], key=key_down)
        # Up: layer lv uses median successor index in layer lv + 1.
        for lv in range(max_lv - 1, -1, -1):
            if lv not in out or not out[lv]:
                continue
            nxt = out.get(lv + 1, [])
            ic = {c: i for i, c in enumerate(nxt)}
            old_i = {n: i for i, n in enumerate(out[lv])}

            def key_up(n: str) -> tuple[float, str]:
                succs = [c for c in G.successors(n) if level.get(c) == lv + 1]
                idxs = [ic[c] for c in succs if c in ic]
                med = _median_index(idxs) if idxs else float(old_i[n])
                return (med, n)

            out[lv] = sorted(out[lv], key=key_up)
    return out


def sublineage_groups(G: nx.DiGraph) -> dict[str, int]:
    """Stable group id per node: weakly connected component × primary root branch.

    Primary root is the minimum root index (among DAG sources) along any path
    from a source, so merged nodes sit in one branch label. Used only for
    vertical spacing, not for edge routing.
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}
    wccs = sorted(nx.weakly_connected_components(G), key=lambda c: min(c))
    node_wcc: dict[str, int] = {}
    for wi, comp in enumerate(wccs):
        for n in comp:
            node_wcc[n] = wi
    roots = sorted(n for n in nodes if G.in_degree(n) == 0)
    root_idx = {r: i for i, r in enumerate(roots)}
    primary_root: dict[str, int] = {n: 0 for n in nodes}
    try:
        order = list(nx.topological_sort(G))
        for n in order:
            if n in root_idx:
                primary_root[n] = root_idx[n]
            else:
                preds = [p for p in G.predecessors(n) if p in primary_root]
                if preds:
                    primary_root[n] = min(primary_root[p] for p in preds)
    except Exception:
        pass
    pairs = sorted({(node_wcc[n], primary_root[n]) for n in nodes})
    pmap = {p: i for i, p in enumerate(pairs)}
    return {n: pmap[(node_wcc[n], primary_root[n])] for n in nodes}


def _group_layers_by_sublineage(
    layers: dict[int, list[str]],
    node_group: dict[str, int],
) -> dict[int, list[str]]:
    """Keep median order within each group; stack groups as contiguous blocks."""
    out: dict[int, list[str]] = {}
    for lv, ns in layers.items():
        idx = {n: i for i, n in enumerate(ns)}
        out[lv] = sorted(ns, key=lambda n: (node_group.get(n, 0), idx[n]))
    return out


def _pos_from_layers(
    layers: dict[int, list[str]],
    layer_dx: float,
    layer_dy: float,
    *,
    node_group: dict[str, int] | None = None,
    sublineage_gap: float = 0.0,
) -> dict[str, np.ndarray]:
    pos: dict[str, np.ndarray] = {}
    multi_group = (
        node_group is not None
        and sublineage_gap > 0.0
        and len({node_group.get(n, 0) for n in node_group}) > 1
    )
    for lv, ns in sorted(layers.items()):
        if not ns:
            continue
        x = float(lv) * layer_dx
        if not multi_group:
            w = len(ns)
            for j, n in enumerate(ns):
                y = (j - (w - 1) / 2.0) * layer_dy
                pos[n] = np.array([x, y], dtype=np.float64)
            continue
        ys: list[float] = []
        y = 0.0
        prev_g: int | None = None
        for j, n in enumerate(ns):
            g = node_group.get(n, 0) if node_group else 0
            if j > 0 and prev_g is not None and g != prev_g:
                y += sublineage_gap
            ys.append(y)
            y += layer_dy
            prev_g = g
        shift = float(np.mean(ys))
        for n, yy in zip(ns, ys):
            pos[n] = np.array([x, yy - shift], dtype=np.float64)
    return pos


def center_on_dag_roots(
    G: nx.DiGraph, pos: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not roots:
        return pos
    cx = float(np.mean([pos[r][0] for r in roots]))
    cy = float(np.mean([pos[r][1] for r in roots]))
    return {
        n: np.array([pos[n][0] - cx, pos[n][1] - cy], dtype=np.float64) for n in pos
    }


def layout_compact_tree(
    G: nx.DiGraph,
    *,
    layer_dx: float,
    layer_dy: float,
    crossing_iterations: int = CROSSING_MIN_ITERATIONS,
    sublineage_gap: float = SUBLINEAGE_GAP,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Layered layout: depth = column, median sweeps reduce crossings, sublineage gaps."""
    level = dag_levels(G)
    layers = _layers_sorted(level)
    layers = minimize_crossings_median_sweeps(
        G, level, layers, iterations=crossing_iterations
    )
    node_group = sublineage_groups(G)
    layers = _group_layers_by_sublineage(layers, node_group)
    pos = _pos_from_layers(
        layers,
        layer_dx,
        layer_dy,
        node_group=node_group,
        sublineage_gap=sublineage_gap,
    )
    pos = center_on_dag_roots(G, pos)
    return pos, level


def _rgba_transparent_white(img: Image.Image, threshold: int = 248) -> Image.Image:
    img = img.convert("RGBA")
    arr = np.asarray(img, dtype=np.uint8)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (r >= threshold) & (g >= threshold) & (b >= threshold)
    out = arr.copy()
    out[:, :, 3] = np.where(mask, 0, out[:, :, 3])
    return Image.fromarray(out, mode="RGBA")


def mol_with_2d_coords(
    smi: str, parent_mol: Chem.Mol | None = None
) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if parent_mol is not None:
        try:
            AllChem.GenerateDepictionMatching2DStructure(mol, parent_mol)
        except Exception:
            try:
                AllChem.Compute2DCoords(mol)
            except Exception:
                return None
    else:
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            return None
    return mol


def mol_to_pil(mol: Chem.Mol, size: tuple[int, int]) -> Image.Image | None:
    try:
        opts = Draw.MolDrawOptions()
        if hasattr(opts, "bgColor"):
            opts.bgColor = (1.0, 1.0, 1.0, 0.0)
        img = Draw.MolToImage(mol, size=size, options=opts)
    except Exception:
        try:
            img = Draw.MolToImage(mol, size=size)
        except Exception:
            return None
    return _rgba_transparent_white(img.convert("RGB") if img.mode != "RGBA" else img)


def apply_lineage_fill_panel(ax) -> None:
    """After layout, set data limits to fill the axes (equal aspect). Call before ``savefig``."""
    fig = ax.figure
    fig.canvas.draw()
    _expand_data_limits_for_equal_aspect(ax)


def _expand_data_limits_for_equal_aspect(ax) -> None:
    """Expand x/y limits so padded bounds fill the axes without cropping (uses ``_lineage_padded_bounds``)."""
    bounds = getattr(ax, "_lineage_padded_bounds", None)
    if bounds is None:
        return
    xmin, xmax, ymin, ymax = bounds
    bbox = ax.get_window_extent()
    if bbox.width <= 0 or bbox.height <= 0:
        return
    ar = bbox.width / bbox.height
    W, H = xmax - xmin, ymax - ymin
    if W <= 0 or H <= 0:
        return
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    # Minimal rectangle with aspect ar = width/height containing [W x H], centered on content.
    new_rh = max(H, W / ar)
    new_rw = new_rh * ar
    ax.set_xlim(xc - new_rw / 2.0, xc + new_rw / 2.0)
    ax.set_ylim(yc - new_rh / 2.0, yc + new_rh / 2.0)


def draw_lineage_on_axes(
    ax,
    G: nx.DiGraph,
    pos: dict[str, np.ndarray],
    node_level: dict[str, int],
    key_to_smi: dict[str, str],
    *,
    layout_scale: float,
    mol_size: tuple[int, int],
    mol_zoom: float,
    show_layers: bool = False,
) -> None:
    """Draw molecules, root halos, and arrows. Call :func:`apply_lineage_fill_panel` after layout before save."""
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    sc = layout_scale
    max_lv = max(node_level.values(), default=0)

    real_nodes = [n for n in G.nodes() if n in pos]
    pos_draw = {n: pos[n] for n in real_nodes}
    ax._lineage_padded_bounds = None  # type: ignore[attr-defined]
    b = _padded_bounds_xy(pos_draw, sc) if real_nodes else None
    if b is not None:
        ax._lineage_padded_bounds = b  # type: ignore[attr-defined]
        ax.set_xlim(b[0], b[1])
        ax.set_ylim(b[2], b[3])

    mol_cache: dict[str, Chem.Mol] = {}
    draw_nodes = sorted(real_nodes, key=lambda nn: (node_level.get(nn, 0), nn))
    pil_by_node: dict[str, Image.Image | None] = {}
    for n in draw_nodes:
        smi = key_to_smi.get(n, "")
        preds = sorted(G.predecessors(n))
        pm = mol_cache.get(preds[0]) if preds else None
        mol = mol_with_2d_coords(smi, pm) if smi else None
        if mol is not None:
            mol_cache[n] = mol
        pil_by_node[n] = mol_to_pil(mol, mol_size) if mol is not None else None

    if show_layers:
        for lv in range(max_lv + 1):
            layer_xs = [
                float(pos[n][0])
                for n in G.nodes()
                if n in pos and node_level.get(n, -1) == lv
            ]
            if not layer_xs:
                continue
            xv = float(np.median(layer_xs))
            ax.axvline(
                xv,
                color="0.78",
                linewidth=0.85,
                linestyle=(0, (3, 5)),
                zorder=0,
                alpha=0.85,
            )

    halo_r, halo_c = 4.1 * sc, (0.74, 0.84, 0.98, 0.52)
    for n in G.nodes():
        if G.in_degree(n) != 0 or n not in pos:
            continue
        x, y = float(pos[n][0]), float(pos[n][1])
        ax.add_patch(
            Circle((x, y), halo_r, facecolor=halo_c, edgecolor="none", zorder=2)
        )

    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0 = float(pos[u][0]), float(pos[u][1])
        x1, y1 = float(pos[v][0]), float(pos[v][1])
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            textcoords="data",
            zorder=0.75,
            arrowprops=dict(
                arrowstyle="-|>",
                color="0.72",
                alpha=0.38,
                lw=2.65 * sc,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=14 * sc,
                connectionstyle="arc3,rad=0",
            ),
        )

    for n in draw_nodes:
        pil = pil_by_node.get(n)
        if pil is None:
            ax.text(
                float(pos[n][0]),
                float(pos[n][1]),
                (n[:14] + "…") if len(n) > 14 else n,
                ha="center",
                va="center",
                fontsize=max(5.5, 7.0 * sc),
                color="0.35",
                zorder=6,
            )
            continue
        imagebox = OffsetImage(np.asarray(pil), zoom=mol_zoom)
        ax.add_artist(
            AnnotationBbox(
                imagebox,
                (float(pos[n][0]), float(pos[n][1])),
                frameon=False,
                pad=0,
                zorder=6,
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw lineage decoding graph (lineage.png).")
    ap.add_argument(
        "--show_layers",
        action="store_true",
        help="Draw faint vertical guides at each DAG layer.",
    )
    ap.add_argument(
        "--mol-scale",
        type=float,
        default=1.2,
        metavar="F",
        help=(
            "Scale molecule raster size and on-canvas zoom by F (default 1.2). "
            "Use --mol-scale 1 for smaller rendering."
        ),
    )
    ap.add_argument(
        "--jsonl", type=pathlib.Path, default=None,
        help="Input lineage JSONL. If omitted, uses "
             "best_molminer-2026-tess.pth_tree.jsonl next to this script.",
    )
    ap.add_argument(
        "--out", type=pathlib.Path, default=None,
        help="Output PNG path (default: figures/lineage.png).",
    )
    args = ap.parse_args()
    if args.mol_scale <= 0:
        raise SystemExit("--mol-scale must be positive.")

    out_path = args.out if args.out is not None else OUT_PATH
    records, padded, failed, _ = load_records(
        args.jsonl if args.jsonl is not None else JSONL_PATH
    )
    smi_cache: dict[str, str] = {}
    lineage_bins, _sorted_roots, _, k1 = compute_lineage_fill(
        padded, failed, smi_cache, len(records)
    )

    idx0 = [i for i, b in enumerate(lineage_bins) if b == 0]
    idx1 = [i for i, b in enumerate(lineage_bins) if b == 1]

    if k1 is None:
        chosen, indices = 0, idx0
        label = "single root lineage (all cells lineage 0)"
    else:
        c0 = distinct_identity_count(idx0, padded, failed, smi_cache)
        c1 = distinct_identity_count(idx1, padded, failed, smi_cache)
        if c0 < c1:
            chosen, indices = 0, idx0
        elif c1 < c0:
            chosen, indices = 1, idx1
        else:
            chosen, indices = 0, idx0
        label = f"lineage {chosen} (fewer distinct identities: {min(c0, c1)} vs {max(c0, c1)})"

    if not indices:
        raise SystemExit("No cells in selected lineage.")

    G, key_to_smi = build_graph_for_lineage(indices, padded, failed, smi_cache)
    if G.number_of_nodes() == 0:
        raise SystemExit("Empty graph for lineage.")

    pos, node_level = layout_compact_tree(G, layer_dx=LAYER_DX, layer_dy=LAYER_DY)

    sc = LAYOUT_SCALE
    mol_size, mol_zoom = molecule_render_params(sc, mol_scale=args.mol_scale)
    fig_h = _figure_height_in_for_paper_width(pos, sc)
    fig, ax = plt.subplots(1, 1, figsize=(PAPER_WIDTH_IN, fig_h))
    fig.patch.set_facecolor("white")
    draw_lineage_on_axes(
        ax,
        G,
        pos,
        node_level,
        key_to_smi,
        layout_scale=sc,
        mol_size=mol_size,
        mol_zoom=mol_zoom,
        show_layers=args.show_layers,
    )

    # Axes must fill the figure before fill-panel (aspect uses pixel bbox).
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    apply_lineage_fill_panel(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    n_n = G.number_of_nodes()
    e_n = G.number_of_edges()
    print(f"Saved {out_path} ({n_n} nodes, {e_n} edges) — {label}")


if __name__ == "__main__":
    main()
