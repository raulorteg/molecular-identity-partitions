"""Shared helpers for the HierVAE probes: architecture flags, seed reset,
model construction, InChI computation.

The architecture defaults (hidden_size=250, latent_size=32, depthT=15,
depthG=15, rnn_type=LSTM) match the checkpoints under checkpoints/hiervae/.
Override at the CLI for a checkpoint from a different training config.
"""
import random

import numpy as np
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import inchi

from hgraph import HierVAE, PairVocab, common_atom_vocab

rdkit.RDLogger.logger().setLevel(rdkit.RDLogger.CRITICAL)


def add_hiervae_args(parser):
    """Add the HierVAE I/O + architecture flags to a parser."""
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--model', required=True)

    parser.add_argument('--seed', type=int, default=7)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--depthG', type=int, default=15)
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    return parser


def set_all_seeds(seed):
    """Reset every RNG (python, numpy, torch CPU/CUDA) and force cudnn determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_hiervae(args, force_cpu=False):
    """Parse vocab, build HierVAE, load state dict, set eval. Returns (model, use_cuda).

    Pass force_cpu=True to keep the model on CPU even when CUDA is available
    (tessellation_map.py runs CPU-only because its 40k greedy decodes are
    RDKit-bottlenecked, not GPU-bottlenecked)."""
    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
    args.vocab = PairVocab(vocab)

    use_cuda = (not force_cpu) and torch.cuda.is_available()
    model = HierVAE(args).cuda() if use_cuda else HierVAE(args).cpu()
    state = torch.load(args.model, map_location=None if use_cuda else torch.device('cpu'))
    model.load_state_dict(state[0])
    model.eval()
    return model, use_cuda


def compute_inchi(smiles):
    """Return (inchi_string, inchikey_first_block) for a SMILES string.

    Empty strings on failure (None/blank SMILES, unparseable, or InChI errors)."""
    if not smiles:
        return "", ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", ""
    try:
        inchi_str = inchi.MolToInchi(mol)
        key = inchi.MolToInchiKey(mol)
        return inchi_str, (key.split('-')[0] if key else "")
    except Exception:
        return "", ""
