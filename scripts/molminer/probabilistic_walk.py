"""MolMiner probabilistic walk — the Fig 2 flow probe.

Walks between two points in the (logP, qed) slice with all other properties
fixed, sampling N molecules per alpha step with no fixed seed.

Writes one row per sample: alpha, logP, qed, smiles, canonical_smiles;
plot_flowplot.py reads only alpha and canonical_smiles.

Re-create only; see data/fig2_flows/SOURCE.md.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import rdkit.Chem as Chem
from tqdm import tqdm

sys.path.append("..")

from molminer.generator import MolecularGenerator


def to_canonical(smiles: str) -> str:
    if not smiles or not str(smiles).strip():
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Probabilistic latent walk between two (logP, qed) points."
    )
    p.add_argument("--ckpt_molminer", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_starter", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_gmm", required=True, type=pathlib.Path)
    p.add_argument("--stats_path", required=True, type=pathlib.Path)
    p.add_argument("--vocab_fragments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_attachments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_anchors", required=True, type=pathlib.Path)
    p.add_argument("--device", default="cpu")
    # walk endpoints in (logP, qed)
    p.add_argument("--logP_start", type=float, default=1.5)
    p.add_argument("--logP_end",   type=float, default=2.75)
    p.add_argument("--qed_start",  type=float, default=0.75)
    p.add_argument("--qed_end",    type=float, default=0.70)
    # walk resolution and samples per step
    p.add_argument("--alpha_steps", type=int, default=50,   help="Number of points along the walk")
    p.add_argument("--n_samples",   type=int, default=100,  help="Number of samples per alpha step (no fixed seed)")
    # fixed properties
    p.add_argument("--SAS",                type=float, default=2.5)
    p.add_argument("--FractionCSP3",       type=float, default=0.3)
    p.add_argument("--molWt",              type=float, default=350.0)
    p.add_argument("--TPSA",               type=float, default=70.0)
    p.add_argument("--MR",                 type=float, default=100.0)
    p.add_argument("--hbd",                type=float, default=2.0)
    p.add_argument("--hba",                type=float, default=5.0)
    p.add_argument("--num_rings",          type=float, default=2.0)
    p.add_argument("--num_rotable_bonds",  type=float, default=4.0)
    p.add_argument("--num_quiral_centers", type=float, default=0.0)
    p.add_argument("--out_dir", type=pathlib.Path, default=None)
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir) if args.out_dir is not None else pathlib.Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.ckpt_molminer.name + "_walk.csv")

    gen = MolecularGenerator(
        ckpt_molminer=args.ckpt_molminer,
        ckpt_starter=args.ckpt_starter,
        ckpt_gmm=args.ckpt_gmm,
        stats_path=args.stats_path,
        vocab_fragments=args.vocab_fragments,
        vocab_attachments=args.vocab_attachments,
        vocab_anchors=args.vocab_anchors,
        device=args.device,
    )

    alphas   = np.linspace(0.0, 1.0, args.alpha_steps)
    logP_vals = (1 - alphas) * args.logP_start + alphas * args.logP_end
    qed_vals  = (1 - alphas) * args.qed_start  + alphas * args.qed_end

    rows = []
    for alpha, logP, qed in tqdm(zip(alphas, logP_vals, qed_vals), total=len(alphas), desc="alpha"):
        # no set_seed here — intentionally non-deterministic
        samples = gen.sample(
            logP=float(logP),
            qed=float(qed),
            SAS=args.SAS,
            FractionCSP3=args.FractionCSP3,
            molWt=args.molWt,
            TPSA=args.TPSA,
            MR=args.MR,
            hbd=args.hbd,
            hba=args.hba,
            num_rings=args.num_rings,
            num_rotable_bonds=args.num_rotable_bonds,
            num_quiral_centers=args.num_quiral_centers,
            num_samples=args.n_samples,
            greedy=True,
            weighted=True,
        )
        for smi in samples:
            canonical = to_canonical(smi) if smi else ""
            rows.append((round(float(alpha), 6), round(float(logP), 6), round(float(qed), 6), smi or "", canonical))

    append_mode = out_path.exists()
    with open(out_path, "a" if append_mode else "w", newline="") as f:
        w = csv.writer(f)
        if not append_mode:
            w.writerow(["alpha", "logP", "qed", "smiles", "canonical_smiles"])
        w.writerows(rows)

    print(f"{'Appended' if append_mode else 'Wrote'} {len(rows)} rows to {out_path}")
    print(f"({args.alpha_steps} steps x {args.n_samples} samples = {args.alpha_steps * args.n_samples} total)")


if __name__ == "__main__":
    main()