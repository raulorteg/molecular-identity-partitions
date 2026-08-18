"""GDSS-only comparison: analysis under the model's native RDKit (2020.09.1,
tess-gdss) vs the cross-architecture RDKit (2024.x, tess-molminer).

The newer RDKit's stricter sanitization rejects ~13% of GDSS decodes.

Reads data/chemical_cohesiveness/gdss/ (cross-arch) against
data/chemical_cohesiveness/gdss_gdssrdkit/ (native), and writes
GDSS_RDKIT_COMPARISON_{jaccard,chemistry}.csv beside them: one row per variant
for the identity-overlap table, one row per (fingerprint, variant) for the
chemistry table.

Invoked by repro/tab2_metric_scaling.sh.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("data/chemical_cohesiveness")
NEW = ROOT / "gdss"                 # cross-arch RDKit 2024.x
NATIVE = ROOT / "gdss_gdssrdkit"    # native RDKit 2020.09.1
VARIANTS = [("RDKit 2024.x (cross-arch)", NEW),
            ("RDKit 2020.09.1 (native)", NATIVE)]
CONV = ["canonical", "inchikey", "murcko", "murcko_generic", "formula", "composition"]
FPS = [("ECFP (Morgan r=2, 2048-bit)", ""), ("MACCS (166-bit)", "maccs")]


def load(p):
    return json.load(open(p)) if Path(p).is_file() else None


def write_jaccard_csv():
    """One row per RDKit variant: canonical universe size + median J per convention."""
    rows = []
    universes = {}
    for vlabel, vdir in VARIANTS:
        j = load(vdir / "jaccard_by_convention.json")
        if j is None:
            print(f"  skipped {vlabel}: no jaccard_by_convention.json")
            continue
        uni = j["conventions"]["canonical"]["universe_N"]
        universes[vlabel] = uni
        rows.append([vlabel, uni] + [j["conventions"][c]["median_J"] for c in CONV])

    out = ROOT / "GDSS_RDKIT_COMPARISON_jaccard.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rdkit", "canonical_universe_N"] + [f"medJ_{c}" for c in CONV])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    if len(universes) == 2:
        n_new = universes[VARIANTS[0][0]]
        n_nat = universes[VARIANTS[1][0]]
        print(f"  canonical identity universe: native={n_nat:,} vs cross-arch={n_new:,} "
              f"(native retains {100 * (n_nat - n_new) / n_nat:+.1f}% more)")


def write_chemistry_csv():
    """One row per (fingerprint, RDKit variant): W/A/R, AUCs, and both metric probes."""
    rows = []
    for fp_label, sub in FPS:
        for vlabel, vdir in VARIANTS:
            base = vdir / sub if sub else vdir
            s = load(base / "summary.json")
            if s is None:
                print(f"  skipped {fp_label} / {vlabel}: no summary.json")
                continue
            pw = load(base / "pairwise_tanimoto_distance.json") or {}
            rows.append([
                fp_label, vlabel,
                s["median_W"], s["median_A"], s["median_R"],
                s["AUC_W_vs_A"], s["AUC_W_vs_R"],
                pw.get("R2_distance_vs_chem_distance", ""),
                pw.get("spearman_rho_d_vs_sim", ""),
                pw.get("spearman_p_alt_less", ""),
                pw.get("R2_cosdist_vs_chem_distance", ""),
                pw.get("spearman_rho_cos_vs_sim", ""),
                pw.get("spearman_p_cos_alt_greater", ""),
            ])

    out = ROOT / "GDSS_RDKIT_COMPARISON_chemistry.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fingerprint", "rdkit", "med_W", "med_A", "med_R",
                    "AUC_W_A", "AUC_W_R",
                    "R2_euclidean", "rho_euclidean", "p_euclidean",
                    "R2_cosine", "rho_cosine", "p_cosine"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


def main():
    write_jaccard_csv()
    write_chemistry_csv()


if __name__ == "__main__":
    main()
