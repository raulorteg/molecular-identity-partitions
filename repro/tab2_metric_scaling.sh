#!/usr/bin/env bash
# Table 2 — chemical-cohesiveness metric scaling (within/across/random Tanimoto,
#   AUC, R^2, Spearman rho), with the ECFP primary + MACCS robustness twin, and
#   the GDSS native-vs-cross RDKit comparison.
#
# Analyze (default): compute from the shipped cohesiveness outputs under
#   data/chemical_cohesiveness/. No GPU. See data/fig3_cohesiveness/SOURCE.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/repro/_env.sh"
cd "$REPO_ROOT"

activate_env tess-molminer

# --- ECFP primary + MACCS robustness twin table ---
python scripts/cohesion/build_cohesion_twin_tables.py

# --- GDSS native (RDKit 2020.09.1) vs cross-arch (RDKit 2024.x) comparison ---
python scripts/cohesion/build_gdss_rdkit_comparison.py

# --- the R^2 / rho stats behind the metric-scaling columns ---
# Writes pairwise_tanimoto_distance.{csv,json}, which build_cohesion_twin_tables.py
# reads for Table 2. ECFP primary, then the MACCS twin; both cover all three
# architectures. Pass --diagnostic_png for the scatter, which is not in the paper.
python scripts/cohesion/pairwise_tanimoto_distance.py
python scripts/cohesion/pairwise_tanimoto_distance.py --fingerprint maccs

# GDSS-native variants: the identity sets are the same, but the SMILES are
# re-fingerprinted under rdkit 2020.09.1, which accepts the ~13% of GDSS
# decodes newer rdkit rejects. Hence tess-gdss.
activate_env tess-gdss
python scripts/cohesion/pairwise_tanimoto_distance.py \
    --label GDSS-native \
    --identity_dir data/gdss/identity_sets \
    --chemcoh_dir  data/chemical_cohesiveness/gdss_gdssrdkit
python scripts/cohesion/pairwise_tanimoto_distance.py --fingerprint maccs \
    --label GDSS-native \
    --identity_dir data/gdss/identity_sets \
    --chemcoh_dir  data/chemical_cohesiveness/gdss_gdssrdkit/maccs

echo "Table 2: data/chemical_cohesiveness/CHEMICAL_COHESION_TWIN_TABLES.csv"
echo "         data/chemical_cohesiveness/GDSS_RDKIT_COMPARISON_{jaccard,chemistry}.csv"
echo "         data/chemical_cohesiveness/<arch>/pairwise_tanimoto_distance.csv"
