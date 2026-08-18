"""Fig 5 (+ Tables S1/S2) — ball-tessellation evolution across training.

Turns the deterministic ball-convergence decodes
(<in_dir>/<ckpt>/ball_*_deterministic.csv) into the 3-panel evolution figure
(--fig) and the table (--table, .csv).

Completeness-gated (a checkpoint counts only when all K balls hold their full
N decode rows) and incrementally cached, so a re-run recomputes only new
checkpoints.

Method imported from scripts/cohesion/chemical_cohesiveness.py and
jaccard_by_convention.py: Morgan/ECFP r=2 2048-bit; N_SUB_PER_BALL per ball;
N_PAIRS_PER_BALL within + across pairs; N_PAIRS_R pooled-random pairs; 6
equivalence conventions for cross-ball Jaccard. W = pairs within a ball,
A = ball i vs the pool of others, R = pairs ignoring ball labels.

Figure panels: unique identities per epoch (boxplot); within-vs-across ECFP
Tanimoto (dodged boxplots) with best-checkpoint refs; within-vs-across KDE
ridgeline, one row per epoch. Table: one row per checkpoint, columns = the 6
Jaccard conventions + med(W), med(A), AUC(W,A), and n_unique. Distribution
summaries are written as separate median, Q1, and Q3 columns at full precision.

Caches under <in_dir>: cohesiveness_cache/*.npz and jaccard_cache/*.json.

Invoked by repro/fig5_convergence.sh.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import time
from collections import defaultdict

import numpy as np

# reuse the sampling/fingerprint helpers from the static probe
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cohesion"))
import chemical_cohesiveness as cc  # noqa: E402
import jaccard_by_convention as jbc  # noqa: E402
import csv as _csv
import json as _json  # noqa: E402

from rdkit import Chem, DataStructs, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

CHUNK_RE = re.compile(r"^ball_(\d+)_(\d+)_(\d+)_deterministic\.csv$")
EPOCH_RE = re.compile(r"[._](\d+)$")  # trailing step/epoch after '_' (MolMiner) or '.' (HierVAE model.ckpt.N)
KDE_STORE = 30_000  # cap stored pooled W/A samples per ckpt (KDE doesn't need 300k)


def epoch_of(tag):
    m = EPOCH_RE.search(tag)
    return int(m.group(1)) if m else None


def canon(smi):
    if not smi:
        return ""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else ""


# When set (by --balls), restrict every checkpoint to exactly this ball set, so
# the cross-checkpoint comparison is over an identical ball set. Applied at the
# single CSV-discovery choke point, so it governs ckpt_status, load_ckpt_sets
# and both metric passes.
KEEP_BALLS = None


def _chunks_by_ball(ckpt_dir: pathlib.Path):
    chunks_by_ball = defaultdict(list)
    for p in ckpt_dir.glob("ball_*_*_*_deterministic.csv"):
        m = CHUNK_RE.match(p.name)
        if m:
            ball = int(m.group(1))
            if KEEP_BALLS is not None and ball not in KEEP_BALLS:
                continue
            chunks_by_ball[ball].append(
                (int(m.group(2)), int(m.group(3)), p))
    return chunks_by_ball


def ckpt_status(ckpt_dir: pathlib.Path):
    """(n_balls, complete) — complete iff every chunk CSV holds its full E-S
    decode rows. Guards against reading a checkpoint that the sweep is still
    actively writing (partial CSVs read as spuriously low n_unique)."""
    cbb = _chunks_by_ball(ckpt_dir)
    if not cbb:
        return 0, False
    complete = True
    for ball, chunks in cbb.items():
        for s, e, p in chunks:
            with open(p) as f:
                rows = sum(1 for _ in f) - 1  # minus header
            if rows < (e - s):
                complete = False
    return len(cbb), complete


def load_ckpt_sets(ckpt_dir: pathlib.Path):
    """Return list of per-ball unique-canonical-SMILES sets, ordered by ball id."""
    chunks_by_ball = defaultdict(list)
    for ball, chunks in _chunks_by_ball(ckpt_dir).items():
        chunks_by_ball[ball] = [(s, p) for s, e, p in chunks]
    sets = []
    for ball in sorted(chunks_by_ball):
        uniq = set()
        for _, path in sorted(chunks_by_ball[ball]):
            with open(path) as f:
                for row in csv.DictReader(f):
                    c = canon(row["smiles"])
                    if c:
                        uniq.add(c)
        sets.append(uniq)
    return sets


def compute_war(fps_per_ball, rng):
    """Pooled W / A / R Tanimoto arrays + per-ball medians.

    Faithful re-implementation of the sampling block in
    chemical_cohesiveness.run_one_arch (same RNG call order, same constants),
    factored out so we can run it per checkpoint.
    """
    K = len(fps_per_ball)
    flat = np.array([(j, k) for j in range(K) for k in range(len(fps_per_ball[j]))],
                    dtype=int)
    M_total = len(flat)

    W_all, A_all = [], []
    med_W, med_A, n_unique = [], [], []
    for i in range(K):
        fps_i = fps_per_ball[i]
        n_i = len(fps_i)
        n_unique.append(n_i)

        w_pairs = cc.sample_within_pairs(n_i, cc.N_PAIRS_PER_BALL, rng)
        W_i = (cc.tanimoto_pairs(fps_i, fps_i, w_pairs.tolist())
               if len(w_pairs) else np.empty(0, dtype=np.float32))

        other_flat = flat[flat[:, 0] != i]
        if n_i and len(other_flat):
            a_idx = rng.integers(0, n_i, size=cc.N_PAIRS_PER_BALL)
            b_choice = rng.integers(0, len(other_flat), size=cc.N_PAIRS_PER_BALL)
            A_i = np.empty(cc.N_PAIRS_PER_BALL, dtype=np.float32)
            for p in range(cc.N_PAIRS_PER_BALL):
                bj, bk = other_flat[b_choice[p]]
                A_i[p] = DataStructs.TanimotoSimilarity(
                    fps_i[int(a_idx[p])], fps_per_ball[int(bj)][int(bk)])
        else:
            A_i = np.empty(0, dtype=np.float32)

        W_all.append(W_i)
        A_all.append(A_i)
        med_W.append(float(np.median(W_i)) if len(W_i) else np.nan)
        med_A.append(float(np.median(A_i)) if len(A_i) else np.nan)

    W = np.concatenate(W_all) if W_all else np.empty(0, dtype=np.float32)
    A = np.concatenate(A_all) if A_all else np.empty(0, dtype=np.float32)

    # R: random pairs from pooled subsample, drop self-pairs
    Ra = rng.integers(0, M_total, size=cc.N_PAIRS_R)
    Rb = rng.integers(0, M_total, size=cc.N_PAIRS_R)
    R = np.empty(cc.N_PAIRS_R, dtype=np.float32)
    for p in range(cc.N_PAIRS_R):
        if Ra[p] == Rb[p]:
            R[p] = 1.0
            continue
        ba, ma = flat[Ra[p]]
        bb, mb = flat[Rb[p]]
        R[p] = DataStructs.TanimotoSimilarity(
            fps_per_ball[int(ba)][int(ma)], fps_per_ball[int(bb)][int(mb)])
    R = R[R < 1.0 - 1e-9]

    return {
        "W": W, "A": A, "R": R,
        "med_W": np.array(med_W), "med_A": np.array(med_A),
        "n_unique": np.array(n_unique),
        "auc_WA": cc.auc_dominance(W, A),
    }


def compute_cohesiveness(in_dir, cache_dir, min_balls, force):
    """Build one W/A/R .npz per complete checkpoint (incremental, gated)."""
    fp_fn = cc.make_fp_fn("morgan")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dirs = sorted(d for d in in_dir.iterdir()
                       if d.is_dir() and d != cache_dir)
    for ckpt_dir in ckpt_dirs:
        tag = ckpt_dir.name
        K, complete = ckpt_status(ckpt_dir)
        if K == 0:
            continue  # not a checkpoint dir
        out = cache_dir / f"{tag}.npz"
        if out.exists() and not force:
            print(f"skip {tag} (cached)")
            continue
        if K < min_balls or not complete:
            print(f"skip {tag} ({K} balls, complete={complete}) — still writing")
            continue
        sets = load_ckpt_sets(ckpt_dir)
        print(f"\n=== {tag}  (K={K}) ===", flush=True)
        rng = np.random.default_rng(cc.SEED)  # same seed per ckpt -> comparable
        t0 = time.time()
        fps_per_ball, _ = cc.subsample_and_fingerprint(sets, cc.N_SUB_PER_BALL, rng, fp_fn)
        res = compute_war(fps_per_ball, rng)
        # subsample stored pooled arrays for KDE
        sub_rng = np.random.default_rng(cc.SEED + 7)
        def cap(a):
            return a if len(a) <= KDE_STORE else sub_rng.choice(a, KDE_STORE, replace=False)
        np.savez_compressed(
            out,
            W=cap(res["W"]), A=cap(res["A"]), R=cap(res["R"]),
            med_W=res["med_W"], med_A=res["med_A"], n_unique=res["n_unique"],
            auc_WA=res["auc_WA"], epoch=epoch_of(tag) if epoch_of(tag) is not None else -1,
        )
        print(f"  {tag}: med(W)={np.nanmedian(res['med_W']):.3f} "
              f"med(A)={np.nanmedian(res['med_A']):.3f} "
              f"med(R)={np.median(res['R']):.3f} AUC(W,A)={res['auc_WA']:.3f} "
              f"[{time.time()-t0:.1f}s]", flush=True)
    print("\ncompute done.")


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _sparse_labels(eps, min_frac=0.06):
    """Pick a non-overlapping subset of epoch labels for a LINEAR x-axis: the
    early checkpoints crowd together (e.g. 1..5), so label them sparsely while
    every checkpoint still gets a box. Always keeps the first and last.
    min_frac is the minimum spacing between labels as a fraction of the axis
    range (use a larger value for wide/scientific labels that take more room)."""
    eps_s = sorted({int(e) for e in eps})
    if len(eps_s) <= 1:
        return set(eps_s)
    rng = (eps_s[-1] - eps_s[0]) or 1
    shown, last = [eps_s[0]], eps_s[0]
    for e in eps_s[1:-1]:
        if (e - last) >= min_frac * rng:
            shown.append(e); last = e
    shown.append(eps_s[-1])
    return set(shown)


def _whisker_top(cols):
    """Largest within-1.5*IQR point across all boxes (the y a boxplot would
    show if fliers were hidden) -> use as the main-panel y-cap so a single
    outlier ball doesn't stretch the axis and flatten the early epochs."""
    tops = []
    for c in cols:
        c = np.asarray(c, float); c = c[np.isfinite(c)]
        if not len(c):
            continue
        q1, q3 = np.percentile(c, [25, 75]); iqr = q3 - q1
        inl = c[c <= q3 + 1.5 * iqr]
        tops.append(float(inl.max()) if len(inl) else float(c.max()))
    return max(tops) if tops else 1.0


def _box(ax, positions, cols, color, width):
    ax.boxplot(
        cols, positions=positions, widths=width, patch_artist=True,
        showfliers=True, manage_ticks=False,
        medianprops=dict(color=color, lw=1.3),
        boxprops=dict(facecolor="none", edgecolor=color, lw=0.9),
        whiskerprops=dict(color=color, lw=0.9),
        capprops=dict(color=color, lw=0.9),
        flierprops=dict(marker="o", ms=2.4, mfc=color, mec="none", alpha=0.7),
    )


def render_figure(cache_dir, out, ridge_epochs=None, x_label="epoch", x_sci=False):
    """3-panel evolution figure from the cohesiveness cache.

    ridge_epochs: optional set of epochs to RESTRICT panel 3 (the ridgeline) to,
    e.g. {1,2,3,4,5,20,50} to focus on the early checkpoints where the
    distribution changes most. Panels 1-2 always show every checkpoint.
    x_label: x-axis label for panels 1-2 (e.g. "epoch" for MolMiner, "steps"
    for HierVAE training steps).
    x_sci: format the x-tick labels in scientific notation and tilt them (for
    large step counts that would otherwise overlap).

    Fliers are clipped from the main panels so one outlier ball doesn't stretch
    the y-axis and flatten the early epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.stats import gaussian_kde

    files = sorted(cache_dir.glob("*.npz"))
    traj, best = [], []
    for f in files:
        d = np.load(f)
        rec = {k: d[k] for k in d.files}
        rec["tag"] = f.stem
        (traj if int(rec["epoch"]) >= 0 else best).append(rec)
    traj.sort(key=lambda r: int(r["epoch"]))
    if not traj:
        raise SystemExit("no trajectory checkpoints in cache")
    epochs = [int(r["epoch"]) for r in traj]

    WITHIN, ACROSS, RAND = "0.0", "#1f77b4", "0.55"
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "legend.frameon": False,
    })

    def style(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)

    fig = plt.figure(figsize=(11.5, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.32)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    # per-checkpoint local spacing; box widths scale with it, capped
    ep_arr = np.asarray(epochs, dtype=float)
    order = np.argsort(ep_arr)
    sorted_ep = ep_arr[order]
    local_gap = np.full(len(ep_arr), 1.0)
    if len(sorted_ep) > 1:
        d = np.diff(sorted_ep)
        g = np.empty_like(sorted_ep)
        g[0], g[-1] = d[0], d[-1]
        g[1:-1] = np.minimum(d[:-1], d[1:])
        local_gap[order] = g
    span = float(np.min(local_gap))
    bw = max(0.6, 0.34 * span)          # reference (widest) box width
    w0 = np.minimum(bw * 1.4, 0.55 * local_gap)   # panel 1 (single box)
    w1 = np.minimum(bw,       0.28 * local_gap)   # panel 2 (dodged pair)
    off = np.minimum(bw * 0.62, 0.30 * local_gap)  # panel 2 dodge offset (per box)

    # ---- Panel 1: unique identities (boxplot per epoch)
    nu_cols = [r["n_unique"][~np.isnan(r["n_unique"])] for r in traj]
    _box(ax0, epochs, nu_cols, WITHIN, w0)
    ax0.plot(epochs, [np.median(c) for c in nu_cols], "-", color=WITHIN, lw=1.4)
    best_meds = []
    for r in best:
        m = float(np.median(r["n_unique"])); best_meds.append(m)
        ax0.axhline(m, color=RAND, ls=(0, (4, 3)), lw=1.0, label="best (val)")
    # clip fliers: cap y at the max whisker, keeping the best(val) line in view
    ax0.set_ylim(0, 1.06 * max([_whisker_top(nu_cols)] + best_meds))
    ax0.set_title("Unique identities")
    ax0.set_ylabel("")
    _labels = _sparse_labels(epochs, 0.13 if x_sci else 0.06)

    def _sci(e):
        m, _, ex = ("%.1e" % e).partition("e")
        m = m.rstrip("0").rstrip(".")
        return rf"${m}\times10^{{{int(ex)}}}$"

    def _xaxis(ax):
        ax.set_xlabel(x_label)
        ax.set_xticks(epochs)
        labs = [(_sci(e) if x_sci else str(int(e))) if int(e) in _labels else ""
                for e in epochs]
        if x_sci:
            ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=7)
        else:
            ax.set_xticklabels(labs)

    _xaxis(ax0)
    style(ax0)
    if best:  # only models with a best(val) checkpoint have a panel-1 legend
        ax0.legend(fontsize=7, loc="upper left")

    # ---- Panel 2: pairwise similarity within vs across (dodged) + random ref
    w_cols = [r["med_W"][~np.isnan(r["med_W"])] for r in traj]
    a_cols = [r["med_A"][~np.isnan(r["med_A"])] for r in traj]
    _box(ax1, [e - o for e, o in zip(epochs, off)], w_cols, WITHIN, w1)
    _box(ax1, [e + o for e, o in zip(epochs, off)], a_cols, ACROSS, w1)
    ax1.plot(epochs, [np.median(c) for c in w_cols], "-", color=WITHIN, lw=1.4)
    ax1.plot(epochs, [np.median(c) for c in a_cols], "-", color=ACROSS, lw=1.4)
    # best (val) checkpoint as within/across reference lines
    for r in best:
        ax1.axhline(float(np.nanmedian(r["med_W"])), color=WITHIN, ls=(0, (4, 3)), lw=1.0)
        ax1.axhline(float(np.nanmedian(r["med_A"])), color=ACROSS, ls=(0, (4, 3)), lw=1.0)
    ax1.set_ylim(0, 1.06 * _whisker_top(w_cols + a_cols))
    ax1.set_title("Pairwise similarity")
    ax1.set_ylabel("ECFP Tanimoto")
    _xaxis(ax1)
    style(ax1)
    handles = [
        Patch(facecolor="none", edgecolor=WITHIN, label="within ball"),
        Patch(facecolor="none", edgecolor=ACROSS, label="across balls"),
    ]
    if best:  # HierVAE has no best(val) checkpoint -> drop that entry
        handles.append(plt.Line2D([], [], color=RAND, ls=(0, (4, 3)), lw=1.0,
                                  label="best (val)"))
    # column-wise legend in an opaque white box
    ax1.legend(handles=handles, fontsize=7, loc="upper right",
               frameon=True, facecolor="white", edgecolor="0.8", framealpha=1.0)

    # ---- Panel 3: ridgeline of within vs across distance KDE, one row per epoch
    ridge = traj if not ridge_epochs else [r for r in traj
                                            if int(r["epoch"]) in ridge_epochs]
    if not ridge:
        ridge = traj  # requested epochs not found -> fall back to all
    grid = np.linspace(0.0, 1.0, 240)
    n_rows = len(ridge)
    row_gap = 1.0
    ymax = 0.0
    for i, r in enumerate(ridge):
        base = (n_rows - 1 - i) * row_gap   # earliest epoch at top
        for arr, color, fill in ((r["W"], WITHIN, "0.15"),
                                  (r["A"], ACROSS, ACROSS)):
            arr = arr[np.isfinite(arr)]
            if len(arr) < 5:
                continue
            kde = gaussian_kde(arr)
            dens = kde(grid)
            dens = dens / dens.max() * 0.9   # normalize each ridge to row height
            ymax = max(ymax, base + dens.max())
            ax2.fill_between(grid, base, base + dens, color=fill, alpha=0.30, lw=0)
            ax2.plot(grid, base + dens, color=color, lw=1.0)
        row_lab = _sci(r["epoch"]) if x_sci else f"ep {int(r['epoch'])}"
        ax2.text(-0.02, base + 0.05, row_lab,
                 ha="right", va="bottom", fontsize=7, transform=ax2.get_yaxis_transform())
    ax2.set_yticks([])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.1, ymax + 0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("ECFP Tanimoto")
    ax2.set_title(f"Within vs across, by {x_label}")
    ax2.legend(handles=[
        Patch(facecolor="0.15", edgecolor=WITHIN, alpha=0.5, label="within"),
        Patch(facecolor=ACROSS, edgecolor=ACROSS, alpha=0.5, label="across"),
    ], fontsize=7, loc="upper right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
# Jaccard identity-overlap across balls, per equivalence convention, per epoch
# --------------------------------------------------------------------------- #
def compute_jaccard(in_dir, cache_dir, min_balls, force):
    """Per checkpoint, re-key each ball's canonical-SMILES set under the 6
    equivalence conventions and cache the median cross-ball pairwise Jaccard
    (reusing jaccard_by_convention.rekey / .pairwise_jaccard verbatim) as one
    JSON per ckpt (incremental, completeness-gated)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dirs = sorted(d for d in in_dir.iterdir()
                       if d.is_dir() and d != cache_dir)
    for ckpt_dir in ckpt_dirs:
        tag = ckpt_dir.name
        K, complete = ckpt_status(ckpt_dir)
        if K == 0:
            continue
        out = cache_dir / f"{tag}.json"
        if out.exists() and not force:
            print(f"skip {tag} (cached)")
            continue
        if K < min_balls or not complete:
            print(f"skip {tag} ({K} balls, complete={complete}) — still writing")
            continue
        sets = load_ckpt_sets(ckpt_dir)
        rec = {"tag": tag, "epoch": epoch_of(tag) if epoch_of(tag) is not None else -1,
               "K": len(sets), "conventions": {}}
        for conv in jbc.CONVENTIONS:
            rekeyed, tot_in, tot_drop = [], 0, 0
            for s in sets:
                cls, dropped = jbc.rekey(s, conv)
                rekeyed.append(cls)
                tot_in += len(s); tot_drop += dropped
            J = jbc.pairwise_jaccard(rekeyed)
            rec["conventions"][conv] = {
                "median_J": float(np.median(J)), "mean_J": float(np.mean(J)),
                "q25_J": float(np.percentile(J, 25)), "q75_J": float(np.percentile(J, 75)),
                "universe_N": len(set().union(*rekeyed)) if rekeyed else 0,
                "median_ball_classes": float(np.median([len(s) for s in rekeyed])),
                "dropped_frac": (tot_drop / tot_in) if tot_in else 0.0,
            }
        with open(out, "w") as f:
            _json.dump(rec, f, indent=2)
        print(f"  {tag}: " + "  ".join(
            f"{c}={rec['conventions'][c]['median_J']:.4f}" for c in jbc.CONVENTIONS))


def write_table(jac_cache, coh_cache, table, model_name="MolMiner", label=None):
    """Combined paper table (.csv): one row per checkpoint, columns = the 6
    Jaccard conventions (from jac_cache) plus the within-ball cohesiveness
    metrics med(W)/med(A)/AUC/n_unique (from coh_cache).

    Each quantity that is a distribution across balls is written as three
    columns -- <name>_median, <name>_q25, <name>_q75 -- the same distributions
    the figure boxes show. AUC is a single per-checkpoint scalar. Values are
    written at full precision; round at presentation time, not here.
    """
    recs = [_json.load(open(f)) for f in sorted(jac_cache.glob("*.json"))]
    traj = sorted([r for r in recs if int(r["epoch"]) >= 0], key=lambda r: int(r["epoch"]))
    best = [r for r in recs if int(r["epoch"]) < 0]
    if not traj:
        raise SystemExit("no trajectory checkpoints in jaccard cache")
    convs = list(jbc.CONVENTIONS)

    coh = {}
    for f in sorted(coh_cache.glob("*.npz")):
        z = np.load(f)
        coh[f.stem] = {k: z[k] for k in z.files}

    def coh_stats(tag, key):
        """median/q25/q75 across balls, or three blanks when absent."""
        z = coh.get(tag)
        if z is None or key not in z:
            return ["", "", ""]
        a = np.asarray(z[key], float)
        return [float(np.nanmedian(a)),
                float(np.nanpercentile(a, 25)),
                float(np.nanpercentile(a, 75))]

    header = ["model", "epoch"]
    for c in convs:
        header += [f"jaccard_{c}_median", f"jaccard_{c}_q25", f"jaccard_{c}_q75"]
    for m in ("med_W", "med_A", "n_unique"):
        header += [f"{m}_median", f"{m}_q25", f"{m}_q75"]
    header += ["auc_WA"]

    rows = [(str(int(r["epoch"])), r) for r in traj] + [("best", r) for r in best]

    csv_path = table.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for epoch_label, r in rows:
            tag = r["tag"]
            row = [model_name, epoch_label]
            for c in convs:
                cv = r["conventions"][c]
                row += [cv["median_J"], cv["q25_J"], cv["q75_J"]]
            for m in ("med_W", "med_A", "n_unique"):
                row += coh_stats(tag, m)
            z = coh.get(tag)
            row += ["" if (z is None or "auc_WA" not in z) else float(z["auc_WA"])]
            w.writerow(row)
    print(f"wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Ball-tessellation evolution: emit the 3-panel figure + "
                    "the combined CSV table from the ball-convergence decodes.")
    ap.add_argument("--in_dir", required=True, type=pathlib.Path,
                    help="sweep output root (contains <ckpt>/ subdirs)")
    ap.add_argument("--fig", required=True, type=pathlib.Path,
                    help="3-panel evolution figure (.png written alongside)")
    ap.add_argument("--table", required=True, type=pathlib.Path,
                    help="combined table path base (.csv written here)")
    ap.add_argument("--coh_cache", type=pathlib.Path, default=None,
                    help="cohesiveness npz cache (default: <in_dir>/cohesiveness_cache)")
    ap.add_argument("--jac_cache", type=pathlib.Path, default=None,
                    help="jaccard json cache (default: <in_dir>/jaccard_cache)")
    ap.add_argument("--min_balls", type=int, default=30)
    ap.add_argument("--balls", type=str, default=None,
                    help="comma-separated ball ids to RESTRICT to (uniform across "
                         "all checkpoints). Use to exclude early-epoch polymer-runaway "
                         "balls so the cross-checkpoint comparison is over one fixed set.")
    ap.add_argument("--force", action="store_true",
                    help="recompute caches even if present")
    ap.add_argument("--model_name", type=str, default="MolMiner",
                    help="model name written in the first CSV column (e.g. HierVAE)")
    ap.add_argument("--table_label", type=str, default="tab:ball_convergence",
                    help="Unused table-label field.")
    ap.add_argument("--ridge_epochs", type=str, default=None,
                    help="comma-separated epochs to RESTRICT panel 3 (ridgeline) to, "
                         "e.g. '1,2,3,4,5,20,50'. Default: show all checkpoints.")
    ap.add_argument("--x_label", type=str, default="epoch",
                    help="x-axis label for panels 1-2 (e.g. 'steps' for HierVAE)")
    ap.add_argument("--x_sci", action="store_true",
                    help="format x-tick labels in scientific notation and tilt them "
                         "(for large step counts that would overlap)")
    args = ap.parse_args()
    ridge_epochs = ({int(x) for x in args.ridge_epochs.split(",") if x.strip()}
                    if args.ridge_epochs else None)
    if args.balls:
        global KEEP_BALLS
        KEEP_BALLS = {int(x) for x in args.balls.split(",") if x.strip() != ""}
        print(f"[balls] restricting to {len(KEEP_BALLS)} balls: {sorted(KEEP_BALLS)}")
    coh_cache = args.coh_cache or (args.in_dir / "cohesiveness_cache")
    jac_cache = args.jac_cache or (args.in_dir / "jaccard_cache")

    # 1-2: build/refresh the two caches (incremental, completeness-gated)
    compute_cohesiveness(args.in_dir, coh_cache, args.min_balls, args.force)
    compute_jaccard(args.in_dir, jac_cache, args.min_balls, args.force)
    # 3-4: the only two shipped artifacts
    render_figure(coh_cache, args.fig, ridge_epochs, args.x_label, args.x_sci)
    write_table(jac_cache, coh_cache, args.table, args.model_name, args.table_label)


if __name__ == "__main__":
    main()
