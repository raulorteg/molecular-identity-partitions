"""GDSS deterministic straight-line path (Fig S14c probe): linear interpolation
between two noise samples with a fixed node mask and deterministic diffusion.
Logs alpha, z_interpolated and decoded SMILES for each of NUM_POINTS points.

Output feeds scripts/common/plot_path_bar.py.

Re-create only; run under PYTHONPATH=$GDSS_ROOT.
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


NUM_POINTS = 200   # alpha grid over [0, 1]
SEED = 42

# 24 active nodes, padded to max_node_num; near the ZINC250k atom-count median.
N_ACTIVE_NODES = 24


def main():
    parser = argparse.ArgumentParser(description='GDSS noise interpolation probe')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: ../data/gdss/gdss_interpolation)')
    parser.add_argument('--config', type=str, default='sample_zinc250k')
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.normpath(os.path.join(os.getcwd(), '..', 'data', 'gdss', 'gdss_interpolation'))
    os.makedirs(out_dir, exist_ok=True)

    config = get_config(args.config, SEED)
    device = load_device()
    device_id = device if isinstance(device, str) else device[0]

    load_seed(SEED)
    ckpt_dict = load_ckpt(config, device)
    configt = ckpt_dict['config']   # training config
    # override for the deterministic probability-flow ODE
    config.sampler.corrector = 'None'
    config.sample.probability_flow = True

    sde_x = load_sde(configt.sde.x)
    sde_adj = load_sde(configt.sde.adj)
    max_node_num = configt.data.max_node_num
    max_feat_num = configt.data.max_feat_num
    shape_x = (NUM_POINTS, max_node_num, max_feat_num)
    shape_adj = (NUM_POINTS, max_node_num, max_node_num)

    sampling_fn = get_pc_sampler(
        sde_x=sde_x, sde_adj=sde_adj, shape_x=shape_x, shape_adj=shape_adj,
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

    # Constant node mask: leading N_ACTIVE_NODES slots = 1, rest = 0.
    if N_ACTIVE_NODES > max_node_num:
        raise ValueError('N_ACTIVE_NODES={} exceeds max_node_num={}'.format(
            N_ACTIVE_NODES, max_node_num))
    flags_one = torch.zeros(1, max_node_num, dtype=torch.float32,
                            device=device_id)
    flags_one[:, :N_ACTIVE_NODES] = 1.0
    flags = flags_one.repeat(NUM_POINTS, 1)

    load_seed(SEED)
    z1_x = sde_x.prior_sampling((1, max_node_num, max_feat_num)).to(device_id)
    z1_adj = sde_adj.prior_sampling_sym((1, max_node_num, max_node_num)).to(device_id)
    z1_x = mask_x(z1_x, flags_one)
    z1_adj = mask_adjs(z1_adj, flags_one)

    z2_x = sde_x.prior_sampling((1, max_node_num, max_feat_num)).to(device_id)
    z2_adj = sde_adj.prior_sampling_sym((1, max_node_num, max_node_num)).to(device_id)
    z2_x = mask_x(z2_x, flags_one)
    z2_adj = mask_adjs(z2_adj, flags_one)

    alphas = np.linspace(0.0, 1.0, NUM_POINTS).tolist()
    z_x = torch.stack([
        (1 - a) * z1_x + a * z2_x for a in alphas
    ], dim=0).squeeze(1)
    z_adj = torch.stack([
        (1 - a) * z1_adj + a * z2_adj for a in alphas
    ], dim=0).squeeze(1)

    z_x_np = z_x.detach().cpu().numpy()
    z_adj_np = z_adj.detach().cpu().numpy()
    np.savez(
        os.path.join(out_dir, 'z_interpolated.npz'),
        x=z_x_np,
        adj=z_adj_np,
        alphas=np.array(alphas),
    )

    # Deterministic decode (same seed; only initial noise differs)
    load_seed(SEED)
    x_out, adj_out, _ = sampling_fn(
        model_x, model_adj, flags,
        initial_x=z_x,
        initial_adj=z_adj,
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

    # decode per sample so the alpha index is preserved; None on failure
    gen_smiles = []
    for i in range(NUM_POINTS):
        mols, _ = gen_mol(
            x_feat[i:i+1], adj_onehot[i:i+1], configt.data.data
        )
        if mols and len(mols) > 0:
            mol = mols[0]
            try:
                if hasattr(mol, 'GetNumAtoms'):  # single ROMol
                    smi = Chem.MolToSmiles(mol)
                    gen_smiles.append(smi if smi else None)
                else:
                    gen_smiles.append(None)
            except Exception:
                gen_smiles.append(None)
        else:
            gen_smiles.append(None)

    log_entries = []
    for i, a in enumerate(alphas):
        log_entries.append({
            'alpha': a,
            'z_interpolated': 'z_interpolated.npz (index {})'.format(i),
            'smiles': gen_smiles[i] if i < len(gen_smiles) else None,
        })

    with open(os.path.join(out_dir, 'interpolation_log.json'), 'w') as f:
        json.dump(log_entries, f, indent=2)

    with open(os.path.join(out_dir, 'interpolation_log.txt'), 'w') as f:
        for e in log_entries:
            smi_str = e['smiles'] if e['smiles'] is not None else 'None'
            f.write('alpha={:.4f}  z_interpolated={}  smiles={}\n'.format(
                e['alpha'], e['z_interpolated'], smi_str
            ))

    print('Done. Output in', out_dir)
    print('interpolation_log.json / .txt: alpha, z_interpolated, smiles')
    print('z_interpolated.npz: x, adj, alphas (full batch)')


if __name__ == '__main__':
    main()
