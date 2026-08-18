"""Emit per-class cell-count CSV for a single Fig 1a/1b/1c section.

For each convention, groups the decoded SMILES of every grid cell into its
class label and writes a long-format CSV, one row per (convention, class),
sorted by descending count within each convention block.

Auto-detects CSV (header z0/z1/smiles, or a numeric first column) vs JSON
(a list of {z_0, z_1, smiles}). The CSVs are gitignored, not shipped.

Invoked by repro/fig2_flows.sh, repro/figS4_molminer_coarsen.sh and
repro/figS5_hiervae_coarsen.sh.
"""
import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.equivalence import CONVENTIONS, equiv_key

DEFAULT_CONVENTIONS = (
    'canonical', 'inchikey', 'murcko', 'murcko_generic', 'formula', 'composition',
)


def load_smiles(data_path):
    """Return list of raw SMILES strings (one per grid cell)."""
    if data_path.lower().endswith('.json'):
        with open(data_path) as f:
            data = json.load(f)
        return [(d.get('smiles') or '') for d in data]
    opener = gzip.open if data_path.lower().endswith('.gz') else open
    with opener(data_path, 'rt', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'smiles' not in reader.fieldnames:
            raise ValueError(f'{data_path}: expected a CSV with a `smiles` column')
        return [(row.get('smiles') or '') for row in reader]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to the grid CSV or JSON.')
    parser.add_argument('--out_csv', required=True, help='Path to write the per-class CSV.')
    parser.add_argument('--conventions', nargs='+', default=list(DEFAULT_CONVENTIONS),
                        choices=list(CONVENTIONS),
                        help='Equivalence conventions to include (default: canonical + 5 alts).')
    args = parser.parse_args()

    smiles_list = load_smiles(args.data)
    n_total = len(smiles_list)
    if n_total == 0:
        raise SystemExit(f'No rows found in {args.data}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['convention', 'class_label', 'n_cells', 'frac_cells'])
        for conv in args.conventions:
            counter = Counter(equiv_key(s, conv) for s in smiles_list)
            for label, n in counter.most_common():
                w.writerow([conv, label, n, f'{n / n_total:.6f}'])

    n_conv = len(args.conventions)
    print(f'Wrote {args.out_csv} ({n_total} cells, {n_conv} conventions)')


if __name__ == '__main__':
    main()
