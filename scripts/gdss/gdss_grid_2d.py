"""GDSS 2D grid probe: one base z_init (fixed seed), two scalar entries varied
over a grid, each point decoded with deterministic diffusion. Logs z_0, z_1
and smiles for the colored slice plot.

The perturbed entries come from --slice_tensor + --axis_a + --axis_b; the
defaults (x, 0,0 and 0,1) reproduce the Fig 1c probe.

  --slice_tensor x    axis indices are "node,feat" into z_x
  --slice_tensor adj  axis indices are "i,j" into z_adj; each perturbation is
                      mirrored to z_adj[j,i] to keep the matrix symmetric.

--append reuses the base from grid_base_z.npz to continue an experiment.

Invoked by repro/figS3_gdss_slices.sh; run under PYTHONPATH=$GDSS_ROOT.
"""
from __future__ import print_function

import argparse
import json
import os
import sys

# GDSS imports need the upstream source on sys.path and its dir as cwd.
_gdss_root = os.environ.get('GDSS_ROOT') or os.path.expanduser('~/src/GDSS')
if not os.path.isdir(_gdss_root):
    raise RuntimeError(
        'GDSS_ROOT not set or invalid: {}. Source repro/_env.sh '
        'or export GDSS_ROOT manually.'.format(_gdss_root))
if _gdss_root not in sys.path:
    sys.path.insert(0, _gdss_root)
os.chdir(_gdss_root)

import numpy as np
import torch

from parsers.config import get_config
from utils.loader import (
    load_ckpt,
    load_device,
    load_model_from_ckpt,
    load_seed,
    load_sde,
)
from solver import get_pc_sampler
from utils.graph_utils import mask_x, mask_adjs, quantize_mol
from utils.mol_utils import gen_mol
from rdkit import Chem
from tqdm import tqdm


SEED = 42
GRID_DEFAULT_N = 50
LOG_KEY_PRECISION = 10  # for dedupe when appending

# 24 active nodes, padded to max_node_num; near the ZINC250k atom-count median.
N_ACTIVE_NODES = 24


def main():
    parser = argparse.ArgumentParser(description='GDSS 2D grid over (z_0, z_1)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: ../data/gdss/gdss_grid2d)')
    parser.add_argument('--config', type=str, default='sample_zinc250k')
    parser.add_argument('--n', type=int, default=GRID_DEFAULT_N,
                        help='Grid size N: NxN points')
    parser.add_argument('--z0_min', type=float, default=-1.0, help='z_0 grid lower bound')
    parser.add_argument('--z0_max', type=float, default=0.0, help='z_0 grid upper bound')
    parser.add_argument('--z1_min', type=float, default=-1.0, help='z_1 grid lower bound')
    parser.add_argument('--z1_max', type=float, default=0.0, help='z_1 grid upper bound')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing grid_log.json; reuse base from grid_base_z.npz')
    parser.add_argument('--slice_tensor', type=str, default='x', choices=['x', 'adj'],
                        help='Which tensor to perturb: x (node feats) or adj (adjacency).')
    parser.add_argument('--axis_a', type=str, default='0,0',
                        help='First slice axis indices. x mode: "node,feat". adj mode: "i,j".')
    parser.add_argument('--axis_b', type=str, default='0,1',
                        help='Second slice axis indices. x mode: "node,feat". adj mode: "i,j".')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Stop after this many grid rows (benchmark aid). Default: full N rows.')
    args = parser.parse_args()

    def _parse_pair(s, name):
        try:
            a, b = s.split(',')
            return int(a), int(b)
        except Exception:
            raise SystemExit('Invalid --{}: {!r}. Expected "i,j".'.format(name, s))

    ax_a = _parse_pair(args.axis_a, 'axis_a')
    ax_b = _parse_pair(args.axis_b, 'axis_b')

    # slice tag for the default out_dir; the (x, 0,0, 0,1) probe keeps 'gdss_grid2d/'
    if args.slice_tensor == 'x' and ax_a == (0, 0) and ax_b == (0, 1):
        slice_tag = ''
    elif args.slice_tensor == 'x':
        slice_tag = '_x_a{}f{}_a{}f{}'.format(ax_a[0], ax_a[1], ax_b[0], ax_b[1])
    else:
        pa = tuple(sorted(ax_a))
        pb = tuple(sorted(ax_b))
        slice_tag = '_adj_{}-{}_{}-{}'.format(pa[0], pa[1], pb[0], pb[1])

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.normpath(os.path.join(os.getcwd(), '..', 'data', 'gdss', 'gdss_grid2d' + slice_tag))
    os.makedirs(out_dir, exist_ok=True)

    N = args.n
    grid_z0 = np.linspace(args.z0_min, args.z0_max, N)
    grid_z1 = np.linspace(args.z1_min, args.z1_max, N)
    device = load_device()
    device_id = device if isinstance(device, str) else device[0]

    load_seed(SEED)
    config = get_config(args.config, SEED)
    ckpt_dict = load_ckpt(config, device)
    configt = ckpt_dict['config']
    config.sampler.corrector = 'None'
    config.sample.probability_flow = True

    sde_x = load_sde(configt.sde.x)
    sde_adj = load_sde(configt.sde.adj)
    max_node_num = configt.data.max_node_num
    max_feat_num = configt.data.max_feat_num

    if args.slice_tensor == 'x':
        for label, (n, f) in (('axis_a', ax_a), ('axis_b', ax_b)):
            if not (0 <= n < N_ACTIVE_NODES):
                raise SystemExit('{}: node {} outside active range [0, {})'.format(
                    label, n, N_ACTIVE_NODES))
            if not (0 <= f < max_feat_num):
                raise SystemExit('{}: feat {} outside [0, {})'.format(label, f, max_feat_num))
        if ax_a == ax_b:
            raise SystemExit('axis_a and axis_b refer to the same scalar; need distinct cells.')
    else:  # adj
        for label, (i, j) in (('axis_a', ax_a), ('axis_b', ax_b)):
            for v in (i, j):
                if not (0 <= v < N_ACTIVE_NODES):
                    raise SystemExit('{}: node {} outside active range [0, {})'.format(
                        label, v, N_ACTIVE_NODES))
            if i == j:
                raise SystemExit('{}: diagonal not perturbable (i == j == {}).'.format(label, i))
        if tuple(sorted(ax_a)) == tuple(sorted(ax_b)):
            raise SystemExit('axis_a and axis_b refer to the same unordered pair.')

    # slice metadata sidecar; the plot scripts read it for axis labels
    slice_meta = {
        'slice_tensor': args.slice_tensor,
        'axis_a': list(ax_a),
        'axis_b': list(ax_b),
        'n': N,
        'z0_range': [args.z0_min, args.z0_max],
        'z1_range': [args.z1_min, args.z1_max],
        'seed': SEED,
        'n_active_nodes': N_ACTIVE_NODES,
        'max_node_num': max_node_num,
        'max_feat_num': max_feat_num,
        'atomic_num_list': [6, 7, 8, 9, 15, 16, 17, 35, 53, 0],
    }
    with open(os.path.join(out_dir, 'slice_meta.json'), 'w') as f:
        json.dump(slice_meta, f, indent=2)

    sampling_fn = get_pc_sampler(
        sde_x=sde_x, sde_adj=sde_adj,
        shape_x=(N, max_node_num, max_feat_num),
        shape_adj=(N, max_node_num, max_node_num),
        predictor=config.sampler.predictor,
        corrector=config.sampler.corrector,
        snr=config.sampler.snr,
        scale_eps=config.sampler.scale_eps,
        n_steps=config.sampler.n_steps,
        probability_flow=config.sample.probability_flow,
        continuous=True,
        denoise=config.sample.noise_removal,
        eps=config.sample.eps,
        device=device_id,
    )
    model_x = load_model_from_ckpt(
        ckpt_dict['params_x'], ckpt_dict['x_state_dict'], device
    )
    model_adj = load_model_from_ckpt(
        ckpt_dict['params_adj'], ckpt_dict['adj_state_dict'], device
    )

    # constant node mask, same for all grid points
    if N_ACTIVE_NODES > max_node_num:
        raise ValueError('N_ACTIVE_NODES={} exceeds max_node_num={}'.format(
            N_ACTIVE_NODES, max_node_num))
    flags_one = torch.zeros(1, max_node_num, dtype=torch.float32,
                            device=device_id)
    flags_one[:, :N_ACTIVE_NODES] = 1.0
    flags_batch = flags_one.repeat(N, 1)

    # one base z_init: from seed, or loaded from file with --append
    base_path = os.path.join(out_dir, 'grid_base_z.npz')
    if args.append and os.path.isfile(base_path):
        loaded = np.load(base_path)
        base_z_x = torch.tensor(loaded['x'], dtype=torch.float32, device=device_id)
        base_z_adj = torch.tensor(loaded['adj'], dtype=torch.float32, device=device_id)
        print('Append: reusing base z_init from {}'.format(base_path))
    else:
        load_seed(SEED)
        base_z_x = sde_x.prior_sampling((1, max_node_num, max_feat_num)).to(device_id)
        base_z_adj = sde_adj.prior_sampling_sym((1, max_node_num, max_node_num)).to(device_id)
        base_z_x = mask_x(base_z_x, flags_one)
        base_z_adj = mask_adjs(base_z_adj, flags_one)
        np.savez(
            base_path,
            x=base_z_x.detach().cpu().numpy(),
            adj=base_z_adj.detach().cpu().numpy(),
            seed=SEED,
        )
        print('Saved base z_init (seed={}) to {}'.format(SEED, base_path))

    log_path = os.path.join(out_dir, 'grid_log.json')
    existing_entries = []
    existing_keys = set()
    if args.append and os.path.isfile(log_path):
        with open(log_path) as f:
            existing_entries = json.load(f)
        for e in existing_entries:
            k = (round(e['z_0'], LOG_KEY_PRECISION), round(e['z_1'], LOG_KEY_PRECISION))
            existing_keys.add(k)
        print('Append mode: loaded {} existing entries'.format(len(existing_entries)))

    new_entries = []

    # for each z_0, batch over all z_1
    total_rows = N if args.max_rows is None else min(N, args.max_rows)
    grid_z1_tensor = torch.tensor(grid_z1, dtype=base_z_x.dtype, device=device_id)
    for i0, z_0 in tqdm(enumerate(grid_z0[:total_rows]), total=total_rows, desc='Grid rows'):
        # copy the base, then perturb per the slice spec
        z_x_batch = base_z_x.repeat(N, 1, 1).clone()
        z_adj_batch = base_z_adj.repeat(N, 1, 1).clone()
        if args.slice_tensor == 'x':
            n_a, f_a = ax_a
            n_b, f_b = ax_b
            z_x_batch[:, n_a, f_a] = z_0
            z_x_batch[:, n_b, f_b] = grid_z1_tensor
        else:  # adj — perturb both (i,j) and (j,i) to preserve symmetry
            ai, aj = ax_a
            bi, bj = ax_b
            z_adj_batch[:, ai, aj] = z_0
            z_adj_batch[:, aj, ai] = z_0
            z_adj_batch[:, bi, bj] = grid_z1_tensor
            z_adj_batch[:, bj, bi] = grid_z1_tensor

        load_seed(SEED)
        x_out, adj_out, _ = sampling_fn(
            model_x, model_adj, flags_batch,
            initial_x=z_x_batch,
            initial_adj=z_adj_batch,
        )

        adj_out = adj_out.detach().cpu()
        x_out = x_out.detach().cpu()
        samples_int = quantize_mol(adj_out)
        samples_int = samples_int - 1
        samples_int[samples_int == -1] = 3
        adj_onehot = torch.nn.functional.one_hot(
            torch.tensor(samples_int), num_classes=4
        ).permute(0, 3, 1, 2)
        x_bin = torch.where(x_out > 0.5, 1, 0)
        x_feat = torch.concat([x_bin, 1 - x_bin.sum(dim=-1, keepdim=True)], dim=-1)

        for i1 in range(N):
            z_1 = float(grid_z1[i1])
            key = (round(z_0, LOG_KEY_PRECISION), round(z_1, LOG_KEY_PRECISION))
            if args.append and key in existing_keys:
                continue
            mols, _ = gen_mol(
                x_feat[i1:i1+1], adj_onehot[i1:i1+1], configt.data.data
            )
            smi = None
            if mols and len(mols) > 0:
                mol = mols[0]
                try:
                    if hasattr(mol, 'GetNumAtoms'):
                        smi = Chem.MolToSmiles(mol)
                        if not smi:
                            smi = None
                except Exception:
                    pass
            new_entries.append({'z_0': z_0, 'z_1': z_1, 'smiles': smi})
            if args.append:
                existing_keys.add(key)

    all_entries = existing_entries + new_entries
    with open(log_path, 'w') as f:
        json.dump(all_entries, f, indent=2)
    print('Saved {} total entries ({} new) to {}'.format(
        len(all_entries), len(new_entries), log_path
    ))
    print('Done. Output in', out_dir)


if __name__ == '__main__':
    main()
