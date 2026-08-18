"""Build the chemical-cohesiveness table (ECFP + MACCS) as CSV.

Reads, per architecture:
  morgan (ECFP r=2 2048b): data/chemical_cohesiveness/<arch>/{summary,pairwise_tanimoto_distance}.json
  maccs  (166-bit):        data/chemical_cohesiveness/<arch>/maccs/{summary,pairwise_tanimoto_distance}.json

Tanimoto similarity throughout; distance = 1 - similarity. Spreads go in
separate _q25/_q75 columns. Writes one row per (fingerprint, architecture) to
data/chemical_cohesiveness/CHEMICAL_COHESION_TWIN_TABLES.csv.

Invoked by repro/tab2_metric_scaling.sh.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("data/chemical_cohesiveness")
ARCHS = [("MolMiner", "molminer"), ("HierVAE", "hiervae"), ("GDSS", "gdss")]
FPS = [("ECFP (Morgan r=2, 2048-bit)", ""), ("MACCS (166-bit)", "maccs")]

HEADER = [
    "fingerprint", "architecture", "K",
    "med_W", "W_q25", "W_q75",
    "med_A", "A_q25", "A_q75",
    "med_R", "R_q25", "R_q75",
    "AUC_W_A", "AUC_W_A_q25", "AUC_W_A_q75",
    "AUC_W_R", "AUC_W_R_q25", "AUC_W_R_q75",
    "R2_euclidean", "rho_euclidean", "p_euclidean",
    "R2_cosine", "rho_cosine", "p_cosine",
]


def row_for(key, sub):
    base = ROOT / key / sub if sub else ROOT / key
    sp = base / "summary.json"
    pp = base / "pairwise_tanimoto_distance.json"
    if not sp.is_file():
        return None
    s = json.load(open(sp))
    pw = json.load(open(pp)) if pp.is_file() else {}
    return [
        s["K"],
        s["median_W"], s["W_q25"], s["W_q75"],
        s["median_A"], s["A_q25"], s["A_q75"],
        s["median_R"], s["R_q25"], s["R_q75"],
        s["AUC_W_vs_A"], s["per_ball_AUC_q25"], s["per_ball_AUC_q75"],
        s["AUC_W_vs_R"], s["per_ball_AUC_WR_q25"], s["per_ball_AUC_WR_q75"],
        pw.get("R2_distance_vs_chem_distance", ""),
        pw.get("spearman_rho_d_vs_sim", ""),
        pw.get("spearman_p_alt_less", ""),
        pw.get("R2_cosdist_vs_chem_distance", ""),
        pw.get("spearman_rho_cos_vs_sim", ""),
        pw.get("spearman_p_cos_alt_greater", ""),
    ]


def main():
    rows = []
    for fp_label, sub in FPS:
        for arch_label, key in ARCHS:
            r = row_for(key, sub)
            if r is None:
                print(f"  skipped {fp_label} / {arch_label}: no summary.json")
                continue
            rows.append([fp_label, arch_label] + r)

    out_csv = ROOT / "CHEMICAL_COHESION_TWIN_TABLES.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
