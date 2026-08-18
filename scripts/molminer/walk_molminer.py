"""Deterministic straight-line walk between two (logP, qed) points in
MolMiner's conditioning slice (Fig S14a probe). All other properties are held
fixed. One greedy decode per alpha-step, with the seed reset before each call.

Output format matches data/gdss/gdss_interpolation/interpolation_log.json so
scripts/common/plot_path_bar.py consumes it directly:
    [{"alpha": float, "smiles": str, "inchi": str, "inchikey_first": str,
      "latent_ref": "logP=X.XXXX qed=X.XXXX"}, ...]

Re-create only; see data/fig1_sections/SOURCE.md.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import rdkit.Chem as Chem
from tqdm import tqdm

sys.path.append("..")

from molminer.generator import MolecularGenerator
from molminer.utils import set_seed


def smiles_to_inchi_info(smiles: str) -> tuple[str, str]:
    if not smiles or not str(smiles).strip():
        return "", ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", ""
    try:
        inchi = Chem.MolToInchi(mol)
        key = Chem.MolToInchiKey(mol)
        return inchi, (key.split("-")[0] if key else "")
    except Exception:
        return "", ""


def main() -> None:
    p = argparse.ArgumentParser(
        description="Deterministic latent walk between two (logP, qed) points in MolMiner."
    )
    p.add_argument("--ckpt_molminer", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_starter", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_gmm", required=True, type=pathlib.Path)
    p.add_argument("--stats_path", required=True, type=pathlib.Path)
    p.add_argument("--vocab_fragments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_attachments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_anchors", required=True, type=pathlib.Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="Output JSON path")

    p.add_argument("--seed", type=int, default=42,
                   help="Reset before every gen.sample call so the walk is deterministic")
    p.add_argument("--alpha_steps", type=int, default=200)

    # walk endpoints in (logP, qed) — same as probabilistic_walk.py (Fig 2) for visual consistency
    p.add_argument("--logP_start", type=float, default=1.5)
    p.add_argument("--logP_end",   type=float, default=2.75)
    p.add_argument("--qed_start",  type=float, default=0.75)
    p.add_argument("--qed_end",    type=float, default=0.70)

    # fixed properties (match probabilistic_walk.py and molminer_2d_slice.py)
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

    args = p.parse_args()

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

    alphas = np.linspace(0.0, 1.0, args.alpha_steps)
    logP_vals = (1 - alphas) * args.logP_start + alphas * args.logP_end
    qed_vals  = (1 - alphas) * args.qed_start  + alphas * args.qed_end

    records = []
    for alpha, logP, qed in tqdm(zip(alphas, logP_vals, qed_vals),
                                 total=len(alphas), desc="alpha"):
        set_seed(args.seed)
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
            num_samples=1,
            greedy=True,
            weighted=True,
            seed=args.seed,
        )
        smiles = samples[0] if samples else ""
        inchi_str, ikey = smiles_to_inchi_info(smiles)
        records.append({
            "alpha": float(alpha),
            "smiles": smiles or "",
            "inchi": inchi_str,
            "inchikey_first": ikey,
            "latent_ref": f"logP={float(logP):.4f} qed={float(qed):.4f}",
        })

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} alpha-points to {out_path}")


if __name__ == "__main__":
    main()
