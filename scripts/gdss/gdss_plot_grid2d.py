"""Plot a GDSS 2D slice: parse grid_log.json, color each point by the
equivalence class of its decoded molecule, scatter z_0 vs z_1, save.

--convention selects the equivalence relation (default "canonical"); any other
value suffixes the output stem with "_<convention>". See
scripts/common/equivalence.py.

Reads grid_log.json and optional slice_meta.json from --in_dir, else from
alongside this script. Writes <out_dir>/<label>{,_molecules}.png, defaulting
--out_dir to --in_dir and <label> to the input dir's basename.

Invoked by repro/fig1_sections.sh, repro/figS3_gdss_slices.sh and
repro/figS6_gdss_coarsen.sh.
"""
import argparse
import json
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from common.equivalence import (  # noqa: E402
    CONVENTIONS, INVALID_KEY, NO_SCAFFOLD_KEY,
    equiv_key, palette, representative_smiles_for_class,
)

MAX_GRID_MOLS_ALT = 24


def plot_molecule_grid(class_keys, class_to_rep_smiles, out_path):
    """Render one molecule per class. class_keys is already capped/ordered."""
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

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(300, 200),
        legends=legends,
    )
    img.save(out_path)
    print("Saved {}".format(out_path))


def axis_labels_from_meta(meta):
    """Return (xlabel, ylabel) derived from slice_meta.json, or defaults."""
    if not meta:
        return r'$z_0$', r'$z_1$'
    tensor = meta.get('slice_tensor', 'x')
    a = meta.get('axis_a')
    b = meta.get('axis_b')
    if a is None or b is None:
        return r'$z_0$', r'$z_1$'
    sym = 'z_x' if tensor == 'x' else 'z_{\\mathrm{adj}}'
    return (r'$' + sym + '[' + ','.join(str(i) for i in a) + ']$',
            r'$' + sym + '[' + ','.join(str(i) for i in b) + ']$')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default=None,
                        help='Directory containing grid_log.json (and optionally slice_meta.json).')
    parser.add_argument('--out_dir', default=None,
                        help='If set, write outputs into this directory.')
    parser.add_argument('--label', default=None,
                        help='Filename stem (default: basename of --in_dir, or "gdss-tessel").')
    parser.add_argument('--convention', default='canonical', choices=list(CONVENTIONS),
                        help='Equivalence relation used to label cells (default: canonical).')
    args = parser.parse_args()

    in_dir = args.in_dir if args.in_dir else SCRIPT_DIR
    log_path = os.path.join(in_dir, 'grid_log.json')
    meta_path = os.path.join(in_dir, 'slice_meta.json')

    out_dir = args.out_dir or (in_dir if args.in_dir else SCRIPT_DIR)
    if args.label:
        base_label = args.label
    elif args.in_dir:
        base_label = os.path.basename(os.path.abspath(in_dir))
    else:
        base_label = 'gdss-tessel'
    label = base_label if args.convention == 'canonical' else f'{base_label}_{args.convention}'

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, label + '.png')
    out_molgrid = os.path.join(out_dir, label + '_molecules.png')

    with open(log_path) as f:
        data = json.load(f)

    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    xlabel, ylabel = axis_labels_from_meta(meta)

    z0 = np.array([d['z_0'] for d in data])
    z1 = np.array([d['z_1'] for d in data])

    raw_smiles = [d.get('smiles') or '' for d in data]
    class_keys = [equiv_key(s, args.convention) for s in raw_smiles]

    distinct = list(dict.fromkeys(class_keys))
    n_classes = len(distinct)

    class_to_rep_smiles = {}
    for k, smi in zip(class_keys, raw_smiles):
        if k not in class_to_rep_smiles:
            class_to_rep_smiles[k] = representative_smiles_for_class(smi, args.convention)

    if args.convention == 'canonical':
        grid_keys = distinct
    else:
        counts = Counter(class_keys)
        appearance = {k: i for i, k in enumerate(distinct)}
        grid_keys = sorted(distinct, key=lambda k: (-counts[k], appearance[k]))[:MAX_GRID_MOLS_ALT]
    plot_molecule_grid(grid_keys, class_to_rep_smiles, out_path=out_molgrid)

    colors = palette(n_classes)
    class_to_color = {k: colors[i] for i, k in enumerate(distinct)}
    point_colors = [class_to_color[k] for k in class_keys]

    fig, ax = plt.subplots(1, 1, figsize=(2.2, 2.2))

    uz0, uz1 = np.unique(z0), np.unique(z1)
    key = lambda a, b: (round(a, 10), round(b, 10))
    pt2key = {key(z0[i], z1[i]): class_keys[i] for i in range(len(data))}
    kw = dict(color='black', linewidth=1, alpha=1, zorder=1)
    dy = (uz1[-1] - uz1[0]) / (len(uz1) - 1) if len(uz1) > 1 else 0
    dx = (uz0[-1] - uz0[0]) / (len(uz0) - 1) if len(uz0) > 1 else 0
    for i0 in range(len(uz0)):
        for i1 in range(len(uz1)):
            s = pt2key.get(key(uz0[i0], uz1[i1]))
            if s is None:
                continue
            if i0 + 1 < len(uz0):
                r = pt2key.get(key(uz0[i0 + 1], uz1[i1]))
                if r is not None and r != s:
                    x = (uz0[i0] + uz0[i0 + 1]) / 2
                    ax.plot([x, x], [uz1[i1] - dy / 2, uz1[i1] + dy / 2], **kw)
            if i1 + 1 < len(uz1):
                r = pt2key.get(key(uz0[i0], uz1[i1 + 1]))
                if r is not None and r != s:
                    y = (uz1[i1] + uz1[i1 + 1]) / 2
                    ax.plot([uz0[i0] - dx / 2, uz0[i0] + dx / 2], [y, y], **kw)

    ax.scatter(z0, z1, c=point_colors, s=1, alpha=0.8, rasterized=True, marker='s', zorder=1)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(axis='both', which='major', labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('Saved {} ({} points, {} classes under convention={})'.format(
        out_png, len(data), n_classes, args.convention))


if __name__ == '__main__':
    main()