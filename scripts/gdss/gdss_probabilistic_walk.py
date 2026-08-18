"""GDSS probabilistic latent walk — the Fig S9 probe. The walk data lives in
data/fig2_flows/.

Uses probability_flow=False (SDE reverse process) and re-runs the sampler from
the same z_interp(alpha) without re-seeding, giving N samples per alpha step.

Writes alpha, sample_idx, smiles, canonical_smiles; plot_flowplot.py reads only
alpha and canonical_smiles.

Re-create only; see data/fig2_flows/SOURCE.md.
"""
from __future__ import print_function

import argparse
import csv
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# GDSS imports need the upstream source on sys.path and its dir as cwd.
_gdss_root = os.environ.get('GDSS_ROOT') or os.path.expanduser('~/src/GDSS')
if not os.path.isdir(_gdss_root):
    raise RuntimeError(
        'GDSS_ROOT not set or invalid: {}. Source repro/_env.sh '
        'or export GDSS_ROOT manually.'.format(_gdss_root))
if _gdss_root not in sys.path:
    sys.path.insert(0, _gdss_root)
os.chdir(_gdss_root)

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


def to_canonical(smi):
    if not smi:
        return ''
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ''
        return Chem.MolToSmiles(mol)
    except Exception:
        return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='sample_zinc250k')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--alpha_steps', type=int, default=50)
    ap.add_argument('--alpha_min', type=float, default=0.0,
                    help="Lower bound of the alpha sweep (default 0.0). "
                         "Combined with --alpha_max, lets you zoom into a "
                         "sub-range of the original z1->z2 interpolation. "
                         "Example: --alpha_min 0 --alpha_max 0.01 walks "
                         "only the first 1%% of the path, at full resolution.")
    ap.add_argument('--alpha_max', type=float, default=1.0,
                    help="Upper bound of the alpha sweep (default 1.0). See "
                         "--alpha_min.")
    ap.add_argument('--n_samples', type=int, default=80)
    ap.add_argument('--decode_seed', type=int, default=None,
                    help="Seed for the stochastic SDE-reverse decode loop. "
                         "If None (default), uses --seed. When "
                         "sharding across GPUs, pass --seed 42 to every shard "
                         "(so they share endpoints z1, z2) but a distinct "
                         "--decode_seed per shard so noise sequences in the "
                         "decode are independent across shards.")
    ap.add_argument('--out', type=str, required=True,
                    help='Output CSV path.')
    ap.add_argument('--corrector', type=str, default='None',
                    help="Langevin corrector or 'None'. Combined with "
                         "probability_flow=False, an SDE-only reverse with "
                         "no corrector is already stochastic.")
    ap.add_argument('--n_active_nodes', type=int, default=24,
                    help="Number of active nodes in the constant flag mask. "
                         "Default 24 ~= median of ZINC250k atom-count "
                         "distribution (mean 23.1, median 23, p10/p90 17/29 "
                         "on a 5000-row sample). Held constant across all "
                         "alphas so the walk varies content, not size.")
    ap.add_argument('--batch_per_call', type=int, default=None,
                    help="GPU batch size for each PC-sampler call. If None "
                         "(default), uses n_samples (one call per alpha). For "
                         "n_samples too big to fit in GPU memory, set this "
                         "smaller; the probe will issue ceil(n_samples / "
                         "batch_per_call) sequential sampler calls per alpha "
                         "and concatenate their outputs.")
    ap.add_argument('--alpha_start_idx', type=int, default=None,
                    help="Inclusive start index into np.linspace(0,1,alpha_steps). "
                         "Use with --alpha_end_idx to split a walk across "
                         "multiple GPU processes (each owns a contiguous alpha "
                         "range, writes its own CSV; concatenate afterwards).")
    ap.add_argument('--alpha_end_idx', type=int, default=None,
                    help="Exclusive end index into the alpha grid. See "
                         "--alpha_start_idx.")
    args = ap.parse_args()

    batch_per_call = args.batch_per_call or args.n_samples

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    config = get_config(args.config, args.seed)
    device = load_device()
    device_id = device if isinstance(device, str) else device[0]

    load_seed(args.seed)
    ckpt_dict = load_ckpt(config, device)
    configt = ckpt_dict['config']
    config.sampler.corrector = args.corrector
    config.sample.probability_flow = False  # SDE reverse → stochastic per call

    sde_x = load_sde(configt.sde.x)
    sde_adj = load_sde(configt.sde.adj)
    max_node_num = configt.data.max_node_num
    max_feat_num = configt.data.max_feat_num

    # PC-sampler shapes are pinned to batch_per_call
    shape_x = (batch_per_call, max_node_num, max_feat_num)
    shape_adj = (batch_per_call, max_node_num, max_node_num)

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
        ckpt_dict['params_x'], ckpt_dict['x_state_dict'], device)
    model_adj = load_model_from_ckpt(
        ckpt_dict['params_adj'], ckpt_dict['adj_state_dict'], device)

    # constant node mask: --n_active_nodes leading slots = 1, rest = 0
    if args.n_active_nodes > max_node_num:
        raise ValueError(
            'n_active_nodes={} exceeds max_node_num={}'.format(
                args.n_active_nodes, max_node_num))
    flags_one = torch.zeros(1, max_node_num, dtype=torch.float32,
                            device=device_id)
    flags_one[:, :args.n_active_nodes] = 1.0
    flags = flags_one.repeat(batch_per_call, 1)

    # two anchor noise samples; same seed as gdss_interpolate.py gives the same z1, z2
    load_seed(args.seed)
    z1_x = sde_x.prior_sampling((1, max_node_num, max_feat_num)).to(device_id)
    z1_adj = sde_adj.prior_sampling_sym((1, max_node_num, max_node_num)).to(device_id)
    z1_x = mask_x(z1_x, flags_one)
    z1_adj = mask_adjs(z1_adj, flags_one)
    z2_x = sde_x.prior_sampling((1, max_node_num, max_feat_num)).to(device_id)
    z2_adj = sde_adj.prior_sampling_sym((1, max_node_num, max_node_num)).to(device_id)
    z2_x = mask_x(z2_x, flags_one)
    z2_adj = mask_adjs(z2_adj, flags_one)

    if args.alpha_max <= args.alpha_min:
        raise ValueError('alpha_max ({}) must be > alpha_min ({})'.format(
            args.alpha_max, args.alpha_min))
    full_alphas = np.linspace(args.alpha_min, args.alpha_max,
                              args.alpha_steps).tolist()
    start_idx = 0 if args.alpha_start_idx is None else args.alpha_start_idx
    end_idx = args.alpha_steps if args.alpha_end_idx is None else args.alpha_end_idx
    alphas = full_alphas[start_idx:end_idx]
    if not alphas:
        raise ValueError('Empty alpha slice [{}:{}]'.format(start_idx, end_idx))

    # inner-batch count per alpha; a ragged last call still runs the full batch
    inner_calls = (args.n_samples + batch_per_call - 1) // batch_per_call

    # do NOT re-seed inside the alpha loop; that would collapse to determinism
    decode_seed = args.decode_seed if args.decode_seed is not None else args.seed
    load_seed(decode_seed)

    rows_written = 0
    with open(out_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha', 'sample_idx', 'smiles', 'canonical_smiles'])

        for a in tqdm(alphas, desc='alphas'):
            zx_one = ((1.0 - a) * z1_x + a * z2_x)
            zadj_one = ((1.0 - a) * z1_adj + a * z2_adj)
            zx = zx_one.repeat(batch_per_call, 1, 1)
            zadj = zadj_one.repeat(batch_per_call, 1, 1)

            sample_idx = 0
            for _call in range(inner_calls):
                x_out, adj_out, _ = sampling_fn(
                    model_x, model_adj, flags,
                    initial_x=zx,
                    initial_adj=zadj,
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
                x_feat = torch.concat(
                    [x_bin, 1 - x_bin.sum(dim=-1, keepdim=True)], dim=-1)

                for i in range(batch_per_call):
                    if sample_idx >= args.n_samples:
                        break
                    smi = ''
                    try:
                        mols, _ = gen_mol(
                            x_feat[i:i + 1], adj_onehot[i:i + 1], configt.data.data)
                        if mols and len(mols) > 0:
                            mol = mols[0]
                            if hasattr(mol, 'GetNumAtoms'):
                                smi = Chem.MolToSmiles(mol) or ''
                    except Exception:
                        smi = ''
                    writer.writerow([float(a), sample_idx, smi, to_canonical(smi)])
                    sample_idx += 1
                    rows_written += 1

    print('Wrote {} rows ({} alphas in slice [{}:{}], {} samples/alpha) -> {}'.format(
        rows_written, len(alphas), start_idx, end_idx,
        args.n_samples, out_path))


if __name__ == '__main__':
    main()
