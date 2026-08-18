"""MolMiner 2D-section plotter for Fig 1a, Figs S1 and S4.

Reads a slice CSV (header <axis_x>,<axis_y>,smiles,inchi,inchi_key_firstblock;
the first two column names double as axis labels), colors each point by the
equivalence class of its decoded molecule, and writes the tessellation PNG plus
a representative molecule grid to <out_dir>/<label>{,_molecules}.png.

--convention selects the equivalence relation (default "canonical"; see
scripts/common/equivalence.py). Any non-canonical value suffixes the output
stem with "_<convention>".

Invoked by repro/fig1_sections.sh, repro/figS1_molminer_slices.sh and
repro/figS4_molminer_coarsen.sh.
"""
import argparse
import csv
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.equivalence import (
    CONVENTIONS, INVALID_KEY, NO_SCAFFOLD_KEY,
    equiv_key, palette, representative_smiles_for_class,
)

MAX_GRID_MOLS_ALT = 24  # cap for non-canonical conventions; canonical shows all


def plot_molecule_grid(class_keys, class_to_rep_smiles, out_path):
    """Render one molecule per class. class_keys is the list to render (already capped)."""
    mols, legends = [], []
    for k in class_keys:
        rep = class_to_rep_smiles.get(k, k)
        if rep in (INVALID_KEY, NO_SCAFFOLD_KEY) or not rep:
            mol = Chem.MolFromSmiles("C")
            smi_display = "N/A" if rep == INVALID_KEY else "(no ring scaffold)"
            label = f"{k}\n{smi_display}"
        else:
            mol = Chem.MolFromSmiles(rep)
            if mol is None:
                mol = Chem.MolFromSmiles("C")
            smi_display = rep if len(rep) <= 50 else rep[:47] + "..."
            label = smi_display if k == rep else f"{k}\n{smi_display}"
        mols.append(mol)
        legends.append(label)
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 200), legends=legends)
    img.save(out_path)
    print("Saved {}".format(out_path))


def read_csv_rows(csv_path, convention):
    z0, z1, class_keys, raw_smiles = [], [], [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or len(reader.fieldnames) < 3:
            raise ValueError(f'CSV {csv_path} must have at least 3 columns: <x>,<y>,smiles,...')
        col_x, col_y = reader.fieldnames[0], reader.fieldnames[1]
        for row in reader:
            z0.append(float(row[col_x]))
            z1.append(float(row[col_y]))
            smi = row.get('smiles', '') or ''
            class_keys.append(equiv_key(smi, convention))
            raw_smiles.append(smi)
    return np.array(z0), np.array(z1), class_keys, raw_smiles, col_x, col_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to CSV with logP,qed,smiles,...')
    parser.add_argument('--out_dir', default=None,
                        help='If set, write outputs into this directory.')
    parser.add_argument('--label', default=None,
                        help='Filename stem under --out_dir (default: CSV basename).')
    parser.add_argument('--convention', default='canonical', choices=list(CONVENTIONS),
                        help='Equivalence relation used to label cells (default: canonical).')
    args = parser.parse_args()

    z0, z1, class_keys, raw_smiles, col_x, col_y = read_csv_rows(args.csv, args.convention)

    distinct = list(dict.fromkeys(class_keys))
    n_classes = len(distinct)

    # map each class to a representative SMILES, from the first matching row
    class_to_rep_smiles = {}
    for k, smi in zip(class_keys, raw_smiles):
        if k not in class_to_rep_smiles:
            class_to_rep_smiles[k] = representative_smiles_for_class(smi, args.convention)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        base_stem = args.label or os.path.splitext(os.path.basename(args.csv))[0]
        stem = base_stem if args.convention == 'canonical' else f'{base_stem}_{args.convention}'
        out_png = os.path.join(args.out_dir, stem + '.png')
        out_molgrid = os.path.join(args.out_dir, stem + '_molecules.png')
    else:
        suffix = '' if args.convention == 'canonical' else f'_{args.convention}'
        out_png = args.csv.rstrip('/') + suffix + '.png'
        out_molgrid = f'grid2d_molecules{suffix}.png'

    if args.convention == 'canonical':
        grid_keys = distinct
    else:
        # top-N by cells covered; ties broken by first appearance
        counts = Counter(class_keys)
        appearance = {k: i for i, k in enumerate(distinct)}
        grid_keys = sorted(distinct, key=lambda k: (-counts[k], appearance[k]))[:MAX_GRID_MOLS_ALT]
    plot_molecule_grid(grid_keys, class_to_rep_smiles, out_path=out_molgrid)

    colors = palette(n_classes)
    class_to_color = {k: colors[i] for i, k in enumerate(distinct)}
    point_colors = [class_to_color[k] for k in class_keys]

    fig, ax = plt.subplots(1, 1, figsize=(2.2, 2.2))

    uz0, uz1 = np.unique(z0), np.unique(z1)
    keypt  = lambda a, b: (round(float(a), 10), round(float(b), 10))
    pt2key = {keypt(z0[i], z1[i]): class_keys[i] for i in range(len(class_keys))}

    kw = dict(color='black', linewidth=1, alpha=1, zorder=3)
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
                    ax.plot([x, x], [uz1[i1] - dy / 2, uz1[i1] + dy / 2], **kw)
            if i1 + 1 < len(uz1):
                r = pt2key.get(keypt(uz0[i0], uz1[i1 + 1]))
                if r is not None and r != k:
                    y = (uz1[i1] + uz1[i1 + 1]) / 2
                    ax.plot([uz0[i0] - dx / 2, uz0[i0] + dx / 2], [y, y], **kw)

    # s=18 looks right at figsize=(8,8), dpi=600; scale by (2.2/8)^2
    ax.scatter(z0, z1, c=point_colors, s=18 * (2.2 / 8) ** 2, alpha=0.9,
               rasterized=True, marker='s', zorder=1)

    ax.set_xlabel(col_x, fontsize=10)
    ax.set_ylabel(col_y, fontsize=10)
    ax.set_xlim(z0.min(), z0.max())
    ax.set_ylim(z1.min(), z1.max())
    ax.set_xticks(np.linspace(z0.min(), z0.max(), 5).round(1))
    ax.set_yticks(np.linspace(z1.min(), z1.max(), 5).round(2))
    ax.tick_params(axis='both', which='major', labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('Saved {} ({} points, {} classes under convention={})'.format(
        out_png, len(class_keys), n_classes, args.convention))


if __name__ == '__main__':
    main()