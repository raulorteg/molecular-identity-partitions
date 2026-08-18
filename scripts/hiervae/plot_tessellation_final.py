"""HierVAE 2D-section plotter for Fig 1b, Figs S2 and S5.

Reads a slice CSV (header z0,z1,smiles,inchi,inchikey_first), colors each point
by the equivalence class of its decoded molecule, and writes the tessellation
PNG plus a representative molecule grid to <out_dir>/<label>{,_molecules}.png.

--convention selects the equivalence relation (default "canonical"; see
scripts/common/equivalence.py). Any non-canonical value suffixes the output
stem with "_<convention>".

Invoked by repro/fig1_sections.sh, repro/figS2_hiervae_slices.sh and
repro/figS5_hiervae_coarsen.sh.
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

MAX_GRID_MOLS_ALT = 24


def plot_molecule_grid(keys, key_to_smiles, out_png, max_mols=None):
    """Plot one representative per class. If max_mols is set, cap at that count."""
    if max_mols is not None:
        keys = keys[:max_mols]
    mols, legends = [], []
    for k in keys:
        rep = key_to_smiles.get(k, "")
        if not rep or rep in (INVALID_KEY, NO_SCAFFOLD_KEY):
            mol = Chem.MolFromSmiles("C")
            smi_display = "N/A" if rep != NO_SCAFFOLD_KEY else "(no ring scaffold)"
            label = f"{k}\n{smi_display}"
        else:
            mol = Chem.MolFromSmiles(rep)
            if mol is None:
                mol = Chem.MolFromSmiles("C")
            smi_display = rep if len(rep) <= 50 else rep[:47] + "..."
            label = f"{k}\n{smi_display}"
        mols.append(mol)
        legends.append(label)

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(300, 200),
        legends=legends,
    )
    img.save(out_png)
    print("Saved", out_png)


def read_csv_rows(csv_path, convention):
    z0, z1, smiles_list, key_list = [], [], [], []
    key_to_smiles = {}

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            z0.append(float(row['z0']))
            z1.append(float(row['z1']))

            smiles = row.get('smiles', '')
            smiles = (smiles if smiles is not None else '').strip()

            k = equiv_key(smiles, convention)

            smiles_list.append(smiles if smiles else "")
            key_list.append(k)

            if k not in key_to_smiles and smiles:
                key_to_smiles[k] = representative_smiles_for_class(smiles, convention)

    return np.array(z0), np.array(z1), smiles_list, key_list, key_to_smiles


def axis_labels_from_meta(csv_path):
    """Read slice_meta.json next to the CSV and return (xlabel, ylabel) like
    $z_{10}$ etc. Returns ($z_0$, $z_1$) defaults if no sidecar found."""
    import json as _json
    meta_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)),
                             'slice_meta.json')
    if not os.path.exists(meta_path):
        return r'$z_0$', r'$z_1$'
    try:
        with open(meta_path) as f:
            meta = _json.load(f)
    except Exception:
        return r'$z_0$', r'$z_1$'
    a, b = meta.get('axis_a'), meta.get('axis_b')
    if a is None or b is None:
        return r'$z_0$', r'$z_1$'
    return r'$z_{' + str(a) + r'}$', r'$z_{' + str(b) + r'}$'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to CSV with z0,z1,smiles,inchi,inchikey_first')
    parser.add_argument('--out_dir', default=None,
                        help='If set, write outputs into this directory.')
    parser.add_argument('--label', default=None,
                        help='Filename stem under --out_dir. Default: hiervae-tess-<step>.')
    parser.add_argument('--convention', default='canonical', choices=list(CONVENTIONS),
                        help='Equivalence relation used to label cells (default: canonical).')
    args = parser.parse_args()

    z0, z1, smiles_list, key_list, key_to_smiles = read_csv_rows(args.csv, args.convention)
    xlabel, ylabel = axis_labels_from_meta(args.csv)

    distinct_keys = list(dict.fromkeys(key_list))
    n_keys = len(distinct_keys)

    # default stem when --label is not given
    default_stem = "hiervae-tess-" + args.csv.split("/")[-1].replace('.txt', '').split("_")[-1]
    base_stem = args.label or default_stem
    stem = base_stem if args.convention == 'canonical' else f'{base_stem}_{args.convention}'

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_png = os.path.join(args.out_dir, stem + '.png')
        out_molgrid = os.path.join(args.out_dir, stem + '_molecules.png')
    else:
        out_png = stem + '.png'
        out_molgrid = 'grid2d_molecules.png'

    if args.convention == 'canonical':
        grid_keys = distinct_keys
        cap = None
    else:
        counts = Counter(key_list)
        appearance = {k: i for i, k in enumerate(distinct_keys)}
        grid_keys = sorted(distinct_keys, key=lambda k: (-counts[k], appearance[k]))
        cap = MAX_GRID_MOLS_ALT
    plot_molecule_grid(grid_keys, key_to_smiles, out_png=out_molgrid, max_mols=cap)

    colors = palette(n_keys)
    key_to_color = {k: colors[i] for i, k in enumerate(distinct_keys)}
    point_colors = [key_to_color[k] for k in key_list]

    fig, ax = plt.subplots(1, 1, figsize=(2.2, 2.2))

    uz0, uz1 = np.unique(z0), np.unique(z1)
    keypt = lambda a, b: (round(float(a), 10), round(float(b), 10))
    pt2key = {keypt(z0[i], z1[i]): key_list[i] for i in range(len(key_list))}

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

    ax.scatter(z0, z1, c=point_colors, s=18 * (2.2 / 8) ** 2, alpha=0.9,
               rasterized=True, marker='s', zorder=1)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(z0.min(), z0.max())
    ax.set_ylim(z1.min(), z1.max())
    ax.set_xticks(np.linspace(z0.min(), z0.max(), 5).round(1))
    ax.set_yticks(np.linspace(z1.min(), z1.max(), 5).round(2))
    ax.tick_params(axis='both', which='major', labelsize=8)
    fig.tight_layout()

    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {out_png} ({len(key_list)} points, {n_keys} classes under convention={args.convention})")


if __name__ == '__main__':
    main()