"""2D tessellation over a pair of continuous MolMiner properties: for each
(axis_a_val, axis_b_val) with all other properties fixed and greedy decoding,
record the generated SMILES and InChI info.

Default axis_a/axis_b = (logP, qed) is the Fig 1a probe. Any other axis pair
renames the first two CSV columns to match the swept properties; a
slice_meta.json sidecar always records the slice spec.

The 7 continuous properties available as slice axes:
  logP, qed, SAS, FractionCSP3, molWt, TPSA, MR
Discrete properties (hbd, hba, num_rings, num_rotable_bonds, num_quiral_centers)
are always held at their point defaults.

Writes one CSV per run into --out_dir, named after the --ckpt_molminer file and
headed <axis_a>,<axis_b>,smiles,inchi,inchi_key_firstblock.

Invoked by repro/figS1_molminer_slices.sh; also re-creates the Fig 1a and Fig 6
slices (see data/fig1_sections/SOURCE.md and data/fig6_molminer_evolution/SOURCE.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import rdkit.Chem as Chem
from tqdm import tqdm

sys.path.append("..")

from molminer.generator import MolecularGenerator
from molminer.utils import set_seed


CONT_PROPS = ['logP', 'qed', 'SAS', 'FractionCSP3', 'molWt', 'TPSA', 'MR']
DISCRETE_PROPS = ['hbd', 'hba', 'num_rings', 'num_rotable_bonds', 'num_quiral_centers']


def smiles_to_inchi_info(smiles: str) -> tuple[str, str]:
    """Return (inchi, inchi_key_first_block). Empty strings if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", ""
    try:
        inchi = Chem.MolToInchi(mol)
        inchi_key = Chem.MolToInchiKey(mol)
        first_block = inchi_key.split("-")[0]
        return inchi, first_block
    except Exception:
        return "", ""


def main() -> None:
    p = argparse.ArgumentParser(
        description="2D (logP, qed) tessellation with fixed other props, greedy decode."
    )
    p.add_argument("--ckpt_molminer", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_starter", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_gmm", required=True, type=pathlib.Path)
    p.add_argument("--stats_path", required=True, type=pathlib.Path)
    p.add_argument("--vocab_fragments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_attachments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_anchors", required=True, type=pathlib.Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42, help="Random seed; reset before each (axis_a, axis_b) sample.")
    # Axis selectors (default reproduces paper Fig 1a)
    p.add_argument("--axis_a", default='logP', choices=CONT_PROPS,
                   help="First swept property name. Default: logP.")
    p.add_argument("--axis_b", default='qed', choices=CONT_PROPS,
                   help="Second swept property name. Default: qed.")
    # Per-continuous-property point defaults (used when the property is NOT swept)
    p.add_argument("--logP", type=float, default=2.0)
    p.add_argument("--qed", type=float, default=0.7)
    p.add_argument("--SAS", type=float, default=2.5)
    p.add_argument("--FractionCSP3", type=float, default=0.3)
    p.add_argument("--molWt", type=float, default=350.0)
    p.add_argument("--TPSA", type=float, default=70.0)
    p.add_argument("--MR", type=float, default=100.0)
    # Per-continuous-property sweep ranges (used when the property IS swept)
    p.add_argument("--logP_min", type=float, default=1.0)
    p.add_argument("--logP_max", type=float, default=3.0)
    p.add_argument("--logP_steps", type=int, default=5)
    p.add_argument("--qed_min", type=float, default=0.6)
    p.add_argument("--qed_max", type=float, default=0.8)
    p.add_argument("--qed_steps", type=int, default=5)
    p.add_argument("--SAS_min", type=float, default=2.0)
    p.add_argument("--SAS_max", type=float, default=4.0)
    p.add_argument("--SAS_steps", type=int, default=5)
    p.add_argument("--FractionCSP3_min", type=float, default=0.2)
    p.add_argument("--FractionCSP3_max", type=float, default=0.6)
    p.add_argument("--FractionCSP3_steps", type=int, default=5)
    p.add_argument("--molWt_min", type=float, default=250.0)
    p.add_argument("--molWt_max", type=float, default=450.0)
    p.add_argument("--molWt_steps", type=int, default=5)
    p.add_argument("--TPSA_min", type=float, default=40.0)
    p.add_argument("--TPSA_max", type=float, default=100.0)
    p.add_argument("--TPSA_steps", type=int, default=5)
    p.add_argument("--MR_min", type=float, default=60.0)
    p.add_argument("--MR_max", type=float, default=130.0)
    p.add_argument("--MR_steps", type=int, default=5)
    # Discrete property point defaults (never swept)
    p.add_argument("--hbd", type=float, default=2.0)
    p.add_argument("--hba", type=float, default=5.0)
    p.add_argument("--num_rings", type=float, default=2.0)
    p.add_argument("--num_rotable_bonds", type=float, default=4.0)
    p.add_argument("--num_quiral_centers", type=float, default=0.0)
    p.add_argument("--out_dir", type=pathlib.Path, default=None, help="Output dir (default: cwd)")
    args = p.parse_args()

    if args.axis_a == args.axis_b:
        raise SystemExit(f"axis_a and axis_b must differ (both = {args.axis_a!r})")

    # Output file: same name as molminer checkpoint + ".txt"
    out_dir = pathlib.Path(args.out_dir) if args.out_dir is not None else pathlib.Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.ckpt_molminer.name + ".txt")

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

    a, b = args.axis_a, args.axis_b
    a_vals = np.linspace(
        getattr(args, f"{a}_min"), getattr(args, f"{a}_max"), getattr(args, f"{a}_steps")
    )
    b_vals = np.linspace(
        getattr(args, f"{b}_min"), getattr(args, f"{b}_max"), getattr(args, f"{b}_steps")
    )

    def build_kwargs(a_val: float, b_val: float) -> dict:
        kw = {}
        for prop in CONT_PROPS:
            if prop == a:
                kw[prop] = float(a_val)
            elif prop == b:
                kw[prop] = float(b_val)
            else:
                kw[prop] = float(getattr(args, prop))
        for prop in DISCRETE_PROPS:
            kw[prop] = float(getattr(args, prop))
        return kw

    # Sidecar metadata — records the slice spec independent of CSV column names.
    slice_meta = {
        "axis_a": a,
        "axis_b": b,
        "a_range": [float(a_vals[0]), float(a_vals[-1])],
        "b_range": [float(b_vals[0]), float(b_vals[-1])],
        "a_steps": len(a_vals),
        "b_steps": len(b_vals),
        "seed": args.seed,
        "fixed_props": {
            prop: float(getattr(args, prop))
            for prop in CONT_PROPS + DISCRETE_PROPS
            if prop not in (a, b)
        },
        "device": args.device,
        "ckpt_molminer": str(args.ckpt_molminer),
    }
    with open(out_dir / "slice_meta.json", "w") as f:
        json.dump(slice_meta, f, indent=2)

    rows = []
    for a_val in tqdm(a_vals, desc=a):
        for b_val in b_vals:
            set_seed(args.seed)
            samples = gen.sample(
                **build_kwargs(a_val, b_val),
                num_samples=1,
                greedy=True,
                weighted=True,
                seed=args.seed,
            )
            smiles = samples[0] if samples else ""
            inchi, inchi_key_first = smiles_to_inchi_info(smiles) if smiles else ("", "")
            rows.append((float(a_val), float(b_val), smiles, inchi, inchi_key_first))

    append_mode = out_path.exists()
    with open(out_path, "a" if append_mode else "w", newline="") as f:
        w = csv.writer(f)
        if not append_mode:
            # first two columns named after the swept properties
            w.writerow([a, b, "smiles", "inchi", "inchi_key_firstblock"])
        for a_val, b_val, smiles, inchi, inchi_key_first in rows:
            w.writerow([a_val, b_val, smiles, inchi, inchi_key_first])

    print(f"{'Appended' if append_mode else 'Wrote'} {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
