"""Build Table 1: median Jaccard identity-overlap by equivalence convention.

Reads the per-architecture jaccard_by_convention.json and summary.json, and
writes the combined table as .csv: median pairwise J(i,j) over all K(K-1)/2
ball pairs under each of the 6 conventions, plus ECFP AUC(W,A).

Invoked by repro/tab1_jaccard.sh.
"""
import csv
import json
from pathlib import Path

ROOT = Path("data/chemical_cohesiveness")
CONV = ["canonical", "inchikey", "murcko", "murcko_generic", "formula", "composition"]
ARCHS = [("MolMiner", "molminer"), ("HierVAE", "hiervae"), ("GDSS", "gdss")]


def main():
    rows = []
    for label, key in ARCHS:
        jp = ROOT / key / "jaccard_by_convention.json"
        sp = ROOT / key / "summary.json"
        if not jp.is_file():
            continue
        j = json.load(open(jp))
        auc = json.load(open(sp))["AUC_W_vs_A"] if sp.is_file() else float("nan")
        rows.append((label, j, auc))
    if not rows:
        raise SystemExit("no jaccard_by_convention.json found; run jaccard_by_convention.py first")

    with open(ROOT / "JACCARD_BY_CONVENTION.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "K"] + [f"medJ_{c}" for c in CONV] + ["ECFP_AUC_W_A"])
        for label, j, auc in rows:
            w.writerow([label, j["K"]]
                       + [f"{j['conventions'][c]['median_J']:.6f}" for c in CONV]
                       + [f"{auc:.6f}"])
    print(f"wrote {ROOT}/JACCARD_BY_CONVENTION.csv")


if __name__ == "__main__":
    main()
