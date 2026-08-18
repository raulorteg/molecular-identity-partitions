"""GDSS ball decoder for Fig 3.

Decodes precomputed ball samples (flat 492-d vectors from
gdss_precompute_balls.py) through GDSS's PC sampler in --batch_size chunks.

Each flat vector is inflated back into the structured prior: z_x (38, 9) with
active block z_x[:24] := flat[:216].reshape(24, 9), and z_adj (38, 38)
symmetric with the i<j upper triangle in [0,24) := flat[216:492], mirrored.
Padding nodes 24-37 and the adj diagonal stay zero.

probability_flow=False, so each batch draws fresh SDE noise. The RNG is seeded
once per process from (seed, ball_id) and not reset per batch.

Writes <mci_dir>/results/ball_{id:02d}_{start}_{end}.csv with columns
ball_id, idx, smiles.

Re-create only; see data/gdss/SOURCE.md.
"""
from __future__ import print_function

import argparse
import csv
import os
import pathlib
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
from tqdm import tqdm

from parsers.config import get_config
from utils.loader import (
    load_ckpt, load_device, load_model_from_ckpt, load_seed, load_sde,
)
from solver import get_pc_sampler
from utils.graph_utils import mask_x, mask_adjs, quantize_mol
from utils.mol_utils import gen_mol
from rdkit import Chem


SEED_DEFAULT = 42
N_ACTIVE_NODES = 24


def flat_to_structured(flat, max_node_num, max_feat_num, n_active):
    """flat: (B, d) numpy -> (z_x, z_adj) torch tensors on CPU.

    z_x: (B, max_node_num, max_feat_num), active block from first n_active*F entries.
    z_adj: (B, max_node_num, max_node_num) symmetric, off-diag upper from the rest.
    """
    B = flat.shape[0]
    d_x = n_active * max_feat_num
    z_x = np.zeros((B, max_node_num, max_feat_num), dtype=np.float32)
    z_adj = np.zeros((B, max_node_num, max_node_num), dtype=np.float32)
    z_x[:, :n_active, :max_feat_num] = flat[:, :d_x].reshape(B, n_active, max_feat_num)
    iu, ju = np.triu_indices(n_active, k=1)
    z_adj[:, iu, ju] = flat[:, d_x:]
    z_adj[:, ju, iu] = flat[:, d_x:]
    return torch.from_numpy(z_x), torch.from_numpy(z_adj)


def main():
    p = argparse.ArgumentParser(
        description='Decode GDSS ball points through PC sampler (batched).'
    )
    p.add_argument('--ball_id', type=int, required=True)
    p.add_argument('--idx_start', type=int, required=True)
    p.add_argument('--idx_end', type=int, required=True)
    p.add_argument('--mci_dir', type=pathlib.Path, required=True)
    p.add_argument('--batch_size', type=int, default=1024,
                   help='Score-net batch size (default 1024; bench higher if VRAM allows).')
    p.add_argument('--config', type=str, default='sample_zinc250k')
    p.add_argument('--seed', type=int, default=SEED_DEFAULT)
    args = p.parse_args()

    results_dir = args.mci_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / 'ball_{:02d}_{}_{}.csv'.format(
        args.ball_id, args.idx_start, args.idx_end)

    ball_path = args.mci_dir / 'ball_{:02d}.npy'.format(args.ball_id)
    ball = np.load(str(ball_path))
    vectors = ball[args.idx_start:args.idx_end]
    total = len(vectors)
    print('Decoding {} samples from {} [{}:{}]'.format(
        total, ball_path.name, args.idx_start, args.idx_end))

    device = load_device()
    device_id = device if isinstance(device, str) else device[0]
    # per-process seed; not reset inside the per-batch loop below
    process_seed = (args.seed * 1_000_003 + args.ball_id * 1009) & 0x7FFF_FFFF
    load_seed(process_seed)
    config = get_config(args.config, args.seed)
    ckpt_dict = load_ckpt(config, device)
    configt = ckpt_dict['config']
    config.sampler.corrector = 'None'
    # SDE reverse process, not probability-flow ODE: fresh noise per sample
    config.sample.probability_flow = False

    sde_x = load_sde(configt.sde.x)
    sde_adj = load_sde(configt.sde.adj)
    max_node_num = configt.data.max_node_num
    max_feat_num = configt.data.max_feat_num

    B = args.batch_size
    sampling_fn = get_pc_sampler(
        sde_x=sde_x, sde_adj=sde_adj,
        shape_x=(B, max_node_num, max_feat_num),
        shape_adj=(B, max_node_num, max_node_num),
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
        ckpt_dict['params_x'], ckpt_dict['x_state_dict'], device)
    model_adj = load_model_from_ckpt(
        ckpt_dict['params_adj'], ckpt_dict['adj_state_dict'], device)

    # Constant node mask matches gdss_grid_2d.py
    flags_one = torch.zeros(1, max_node_num, dtype=torch.float32, device=device_id)
    flags_one[:, :N_ACTIVE_NODES] = 1.0
    flags_batch = flags_one.repeat(B, 1)

    # loop over batches, padding the tail
    with open(str(out_path), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball_id', 'idx', 'smiles'])

        for batch_start in tqdm(range(0, total, B),
                                desc='ball {:02d}'.format(args.ball_id)):
            batch_end = min(batch_start + B, total)
            actual = batch_end - batch_start

            z_x_np = np.zeros((B, max_node_num, max_feat_num), dtype=np.float32)
            z_adj_np = np.zeros((B, max_node_num, max_node_num), dtype=np.float32)
            z_x_part, z_adj_part = flat_to_structured(
                vectors[batch_start:batch_end].astype(np.float32),
                max_node_num, max_feat_num, N_ACTIVE_NODES,
            )
            z_x_np[:actual] = z_x_part.numpy()
            z_adj_np[:actual] = z_adj_part.numpy()

            z_x_batch = torch.from_numpy(z_x_np).to(device_id)
            z_adj_batch = torch.from_numpy(z_adj_np).to(device_id)
            z_x_batch = mask_x(z_x_batch, flags_one)
            z_adj_batch = mask_adjs(z_adj_batch, flags_one)

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

            for j in range(actual):
                global_idx = args.idx_start + batch_start + j
                mols, _ = gen_mol(
                    x_feat[j:j+1], adj_onehot[j:j+1], configt.data.data
                )
                smi = ''
                if mols and len(mols) > 0:
                    mol = mols[0]
                    try:
                        if hasattr(mol, 'GetNumAtoms'):
                            s = Chem.MolToSmiles(mol)
                            if s:
                                smi = s
                    except Exception:
                        pass
                writer.writerow([args.ball_id, global_idx, smi])

    print('Done ->', out_path)


if __name__ == '__main__':
    main()
