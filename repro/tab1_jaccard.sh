#!/usr/bin/env bash
# Table 1 — median Jaccard identity-overlap by equivalence convention.
#   Per-architecture pairwise Jaccard under 6 conventions (canonical -> composition),
#   plus the SI identity-overlap CSVs.
#
# Analyze (default): compute from the cached identity sets + shipped
#   cohesiveness outputs. No GPU. See data/fig3_cohesiveness/SOURCE.md to re-create it.
#
# Run repro/fig3_cohesiveness.sh first — the SI per-pair CSVs read the identity-
# overlap .npz it writes under figures/fig3_cohesiveness/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

activate_env tess-molminer

# --- per-architecture Jaccard-by-convention JSON (from cached identity sets) ---
python scripts/cohesion/jaccard_by_convention.py --arch "MolMiner" \
    --identity_dir data/molminer/identity_sets       --out_json data/chemical_cohesiveness/molminer/jaccard_by_convention.json
python scripts/cohesion/jaccard_by_convention.py --arch "HierVAE" \
    --identity_dir data/hiervae/identity_sets_pooled --out_json data/chemical_cohesiveness/hiervae/jaccard_by_convention.json
python scripts/cohesion/jaccard_by_convention.py --arch "GDSS" \
    --identity_dir data/gdss/identity_sets           --out_json data/chemical_cohesiveness/gdss/jaccard_by_convention.json

# --- combined Table 1 (cross-arch) ---
python scripts/cohesion/build_jaccard_table.py

# --- SI supporting CSVs (per-ball, per-pair, identity-overlap summary) ---
python scripts/cohesion/export_cohesion_csvs.py

echo "Table 1: data/chemical_cohesiveness/JACCARD_BY_CONVENTION.csv"
