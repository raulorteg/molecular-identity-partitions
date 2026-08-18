"""Emit the SI CSVs from the cached cohesiveness outputs.

Writes, under data/chemical_cohesiveness/:
  CROSS_ARCH_SUMMARY.csv   one row per architecture
  <arch>/per_ball.csv      one row per ball: median(W_i), median(A_i), AUC, |S_i|

and, under figures/reproducibility/fig3_cohesiveness/:
  identity_overlap_summary.csv       one row per architecture
  identity_overlap_pairs_<arch>.csv  one row per ball pair (i, j, d, O, J,
                                     |S_i|, |S_j|)

Invoked by repro/tab1_jaccard.sh.
"""

from __future__ import annotations

import csv
import json
import pathlib
import pickle

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# arch -> (display_label, chemcoh_dir, identity_npz, identity_dir_for_sizes)
ARCHS = [
    ("MolMiner",
     "data/chemical_cohesiveness/molminer",
     "figures/fig3_cohesiveness/column_molminer_cohesiveness.npz",
     "data/molminer/identity_sets"),
    ("HierVAE",
     "data/chemical_cohesiveness/hiervae",
     "figures/fig3_cohesiveness/column_hiervae_cohesiveness.npz",
     "data/hiervae/identity_sets_pooled"),
    ("GDSS",
     "data/chemical_cohesiveness/gdss",
     "figures/fig3_cohesiveness/column_gdss_cohesiveness.npz",
     "data/gdss/identity_sets"),
]


def load_set_sizes(identity_dir: pathlib.Path) -> list[int]:
    pkls = sorted(identity_dir.glob("ball_*.pkl"))
    sizes = []
    for p in pkls:
        with open(p, "rb") as f:
            S = pickle.load(f)
        sizes.append(len(S))
    return sizes


def write_cross_arch_summary_csv(out_path: pathlib.Path):
    rows = []
    for label, chem_dir, _, _ in ARCHS:
        chem_dir = REPO_ROOT / chem_dir
        s_path = chem_dir / "summary.json"
        if not s_path.is_file():
            continue
        s = json.load(open(s_path))
        pwd_path = chem_dir / "pairwise_tanimoto_distance.json"
        pwd = json.load(open(pwd_path)) if pwd_path.is_file() else {}
        rows.append({
            "architecture": label,
            "K": s["K"],
            "median_W": round(s["median_W"], 4),
            "median_A": round(s["median_A"], 4),
            "median_R": round(s["median_R"], 4),
            "AUC_W_vs_A": round(s["AUC_W_vs_A"], 4),
            "AUC_W_vs_R": round(s["AUC_W_vs_R"], 4),
            "AUC_A_vs_R": round(s["AUC_A_vs_R"], 4),
            "R2_d_vs_chemD": (round(pwd["R2_distance_vs_chem_distance"], 4)
                              if "R2_distance_vs_chem_distance" in pwd else ""),
            "Spearman_rho_d_vs_T": (round(pwd["spearman_rho_d_vs_sim"], 4)
                                    if "spearman_rho_d_vs_sim" in pwd else ""),
            "Spearman_p_alt_less": (f"{pwd['spearman_p_alt_less']:.2e}"
                                    if "spearman_p_alt_less" in pwd else ""),
            "per_ball_AUC_median": round(s["per_ball_AUC_median"], 4),
            "per_ball_AUC_min": round(s["per_ball_AUC_min"], 4),
            "per_ball_AUC_max": round(s["per_ball_AUC_max"], 4),
            "n_balls_AUC_gt_0p5": s["n_balls_AUC_gt_0p5"],
        })
    if not rows:
        print("no chemical_cohesiveness summaries found")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")


def write_per_ball_csv(label: str, chem_dir: pathlib.Path,
                       identity_dir: pathlib.Path):
    summary_path = chem_dir / "summary.json"
    if not summary_path.is_file():
        return
    W_pb = np.load(chem_dir / "W_per_ball.npy")
    A_pb = np.load(chem_dir / "A_per_ball.npy")
    auc_pb = np.load(chem_dir / "per_ball_AUC.npy")
    sizes = load_set_sizes(identity_dir)
    K = len(sizes)
    out_path = chem_dir / "per_ball.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ball_id", "architecture", "n_unique_identities",
            "median_within_tanimoto",
            "median_across_tanimoto",
            "per_ball_AUC_W_vs_A",
        ])
        for k in range(K):
            wi = W_pb[k][np.isfinite(W_pb[k])]
            ai = A_pb[k][np.isfinite(A_pb[k])]
            w.writerow([
                k, label, sizes[k],
                f"{float(np.median(wi)):.4f}" if len(wi) else "",
                f"{float(np.median(ai)):.4f}" if len(ai) else "",
                f"{float(auc_pb[k]):.4f}",
            ])
    print(f"wrote {out_path}")


def write_identity_overlap_summary(out_path: pathlib.Path):
    rows = []
    for label, _, npz_rel, _ in ARCHS:
        # prefer a freshly reproduced npz, fall back to the shipped one
        npz_path = REPO_ROOT / npz_rel.replace('figures/', 'figures/reproducibility/', 1)
        if not npz_path.is_file():
            npz_path = REPO_ROOT / npz_rel
        if not npz_path.is_file():
            continue
        npz = np.load(npz_path)
        d = npz["d"]; J = npz["J"]; O = npz["O"]
        K = int(npz["K"]); N = int(npz["N"])
        n_pairs = len(J)
        n_with_O = int((O > 0).sum())
        rho_dJ, p_dJ = spearmanr(d, J, alternative="less")
        rows.append({
            "architecture": label,
            "K": K,
            "n_pairs": n_pairs,
            "universe_size": N,
            "f_singleton": round(float(npz["f_singleton"]), 6),
            "n_pairs_with_O_gt_0": n_with_O,
            "pct_pairs_with_O_gt_0": round(100 * n_with_O / n_pairs, 2),
            "max_pair_overlap_O": int(O.max()) if n_pairs else 0,
            "max_pair_Jaccard": round(float(J.max()), 6) if n_pairs else 0.0,
            "spearman_rho_d_J": round(float(rho_dJ), 4) if np.isfinite(rho_dJ) else None,
            "spearman_p_one_sided": round(float(p_dJ), 6) if np.isfinite(p_dJ) else None,
        })
    if not rows:
        print("no identity-overlap .npz outputs found")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")


def write_per_pair_csv(label: str, npz_rel: str, identity_dir: str):
    npz_path = REPO_ROOT / npz_rel
    if not npz_path.is_file():
        return
    npz = np.load(npz_path)
    import re
    short = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    out_path = REPO_ROOT / "figures/reproducibility/fig3_cohesiveness" / f"identity_overlap_pairs_{short}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ball_i", "ball_j", "distance",
            "n_unique_i", "n_unique_j",
            "overlap_O", "jaccard_J",
        ])
        for k in range(len(npz["d"])):
            w.writerow([
                int(npz["i"][k]), int(npz["j"][k]),
                f"{float(npz['d'][k]):.4f}",
                int(npz["Ki"][k]), int(npz["Kj"][k]),
                int(npz["O"][k]),
                f"{float(npz['J'][k]):.6f}",
            ])
    print(f"wrote {out_path}")


def main():
    out_chem = REPO_ROOT / "data/chemical_cohesiveness/CROSS_ARCH_SUMMARY.csv"
    write_cross_arch_summary_csv(out_chem)

    for label, chem_dir, _, ids_dir in ARCHS:
        chem_dir = REPO_ROOT / chem_dir
        ids_dir = REPO_ROOT / ids_dir
        if (chem_dir / "summary.json").is_file():
            write_per_ball_csv(label, chem_dir, ids_dir)

    out_idov = REPO_ROOT / "figures/reproducibility/fig3_cohesiveness/identity_overlap_summary.csv"
    write_identity_overlap_summary(out_idov)

    for label, _, npz_rel, ids_dir in ARCHS:
        if (REPO_ROOT / npz_rel).is_file():
            write_per_pair_csv(label, npz_rel, ids_dir)


if __name__ == "__main__":
    main()
