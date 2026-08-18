"""(logP, qed) tessellation that keeps each cell's generation trajectory — the
input behind Fig 4 and Fig S13.

Like molminer_2d_slice.py, but each grid point is decoded with a greedy rollout
that records every intermediate canonical SMILES from
MolecularGenerator.sample_with_intermediate_smiles, not just the final molecule.

Writes JSON Lines, one object per (logP, qed): coordinates, final_smiles, a
failed flag, message, and intermediates. Optionally also a .npz with stacked
coordinates and a parallel object array of intermediate lists.

Re-create only; see data/fig4_branching/SOURCE.md.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from tqdm import tqdm

sys.path.append("..")

from molminer.generator import MolecularGenerator
from molminer.utils import set_seed


def main() -> None:
    p = argparse.ArgumentParser(
        description="2D (logP, qed) grid with full intermediate SMILES trajectories."
    )
    p.add_argument("--ckpt_molminer", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_starter", required=True, type=pathlib.Path)
    p.add_argument("--ckpt_gmm", required=True, type=pathlib.Path)
    p.add_argument("--stats_path", required=True, type=pathlib.Path)
    p.add_argument("--vocab_fragments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_attachments", required=True, type=pathlib.Path)
    p.add_argument("--vocab_anchors", required=True, type=pathlib.Path)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed; reset before each (logP, qed) sample.",
    )
    p.add_argument("--logP_min", type=float, default=1.0)
    p.add_argument("--logP_max", type=float, default=3.0)
    p.add_argument("--logP_steps", type=int, default=5)
    p.add_argument("--qed_min", type=float, default=0.6)
    p.add_argument("--qed_max", type=float, default=0.8)
    p.add_argument("--qed_steps", type=int, default=5)
    # 2D-axis selection. With --x_prop/--y_prop both unset, grids (logP, qed)
    # from the --logP_*/--qed_* args above. With them set, any two of the 12
    # conditioning properties become the axes and the rest are held at their
    # scalar args; each record then carries x_prop/y_prop.
    PROP_CHOICES = [
        "logP", "qed", "SAS", "FractionCSP3", "molWt", "TPSA", "MR",
        "hbd", "hba", "num_rings", "num_rotable_bonds", "num_quiral_centers",
    ]
    p.add_argument("--x_prop", choices=PROP_CHOICES, default=None,
                   help="Property for the x grid axis (default: logP).")
    p.add_argument("--y_prop", choices=PROP_CHOICES, default=None,
                   help="Property for the y grid axis (default: qed).")
    p.add_argument("--x_min", type=float, default=None)
    p.add_argument("--x_max", type=float, default=None)
    p.add_argument("--x_steps", type=int, default=None)
    p.add_argument("--y_min", type=float, default=None)
    p.add_argument("--y_max", type=float, default=None)
    p.add_argument("--y_steps", type=int, default=None)
    # Shard the x axis by index for CPU fan-out: the full x linspace is computed
    # once, then sliced [x_start_idx:x_end_idx), so shard values stay identical
    # to an unsharded run.
    p.add_argument("--x_start_idx", type=int, default=None)
    p.add_argument("--x_end_idx", type=int, default=None)
    # Scalar logP/qed, used only when they are NOT a grid axis (held fixed).
    p.add_argument("--logP", type=float, default=2.0)
    p.add_argument("--qed", type=float, default=0.7)
    p.add_argument("--SAS", type=float, default=2.5)
    p.add_argument("--FractionCSP3", type=float, default=0.3)
    p.add_argument("--molWt", type=float, default=350.0)
    p.add_argument("--TPSA", type=float, default=70.0)
    p.add_argument("--MR", type=float, default=100.0)
    p.add_argument("--hbd", type=float, default=2.0)
    p.add_argument("--hba", type=float, default=5.0)
    p.add_argument("--num_rings", type=float, default=2.0)
    p.add_argument("--num_rotable_bonds", type=float, default=4.0)
    p.add_argument("--num_quiral_centers", type=float, default=0.0)
    p.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory (default: cwd).",
    )
    p.add_argument(
        "--out_jsonl",
        type=pathlib.Path,
        default=None,
        help="Output JSONL path (default: <out_dir>/<ckpt>_tree.jsonl).",
    )
    p.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save <stem>_tree.npz with logP, qed, intermediates (object array).",
    )
    args = p.parse_args()

    out_dir = (
        pathlib.Path(args.out_dir) if args.out_dir is not None else pathlib.Path.cwd()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = (
        args.out_jsonl
        if args.out_jsonl is not None
        else out_dir / (args.ckpt_molminer.name + "_tree.jsonl")
    )

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

    # resolve the two grid axes
    generic = args.x_prop is not None or args.y_prop is not None
    if generic:
        if args.x_prop is None or args.y_prop is None:
            p.error("--x_prop and --y_prop must be set together.")
        if args.x_prop == args.y_prop:
            p.error("--x_prop and --y_prop must differ.")
        for ax in ("x", "y"):
            if any(getattr(args, f"{ax}_{k}") is None for k in ("min", "max", "steps")):
                p.error(f"--{ax}_min/--{ax}_max/--{ax}_steps are required with --{ax}_prop.")
        x_prop, y_prop = args.x_prop, args.y_prop
        x_vals = np.linspace(args.x_min, args.x_max, args.x_steps)
        y_vals = np.linspace(args.y_min, args.y_max, args.y_steps)
    else:
        x_prop, y_prop = "logP", "qed"
        x_vals = np.linspace(args.logP_min, args.logP_max, args.logP_steps)
        y_vals = np.linspace(args.qed_min, args.qed_max, args.qed_steps)

    # Optional index-shard of the (full) x axis for CPU fan-out.
    if args.x_start_idx is not None or args.x_end_idx is not None:
        s = 0 if args.x_start_idx is None else args.x_start_idx
        e = len(x_vals) if args.x_end_idx is None else args.x_end_idx
        x_vals = x_vals[s:e]
        if len(x_vals) == 0:
            raise SystemExit(f"Empty x shard [{s}:{e}] of {x_prop} grid.")

    # Fixed scalar context for every non-axis property.
    fixed_props = {
        "logP": args.logP, "qed": args.qed, "SAS": args.SAS,
        "FractionCSP3": args.FractionCSP3, "molWt": args.molWt, "TPSA": args.TPSA,
        "MR": args.MR, "hbd": args.hbd, "hba": args.hba,
        "num_rings": args.num_rings, "num_rotable_bonds": args.num_rotable_bonds,
        "num_quiral_centers": args.num_quiral_centers,
    }

    coord_x: list[float] = []
    coord_y: list[float] = []
    intermediates_list: list[list[str]] = []
    finals: list[str | None] = []
    failed_flags: list[bool] = []

    append_mode = jsonl_path.exists()
    mode = "a" if append_mode else "w"

    n_written = 0
    with open(jsonl_path, mode, encoding="utf-8") as fj:
        for xv in tqdm(x_vals, desc=x_prop):
            for yv in y_vals:
                set_seed(args.seed)
                props = dict(fixed_props)
                props[x_prop] = float(xv)
                props[y_prop] = float(yv)
                batch = gen.sample_with_intermediate_smiles(
                    **props,
                    num_samples=1,
                    greedy=True,
                    weighted=True,
                    seed=args.seed,
                )
                final_smi, failed, msg, inter = batch[0]
                if generic:
                    # named-axis keys; the plotters key off x_prop
                    record = {
                        "x_prop": x_prop,
                        "y_prop": y_prop,
                        x_prop: float(xv),
                        y_prop: float(yv),
                        "final_smiles": final_smi,
                        "failed": bool(failed),
                        "message": msg,
                        "intermediates": inter,
                    }
                else:
                    # (logP, qed) schema
                    record = {
                        "logP": float(xv),
                        "qed": float(yv),
                        "final_smiles": final_smi,
                        "failed": bool(failed),
                        "message": msg,
                        "intermediates": inter,
                    }
                fj.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
                coord_x.append(float(xv))
                coord_y.append(float(yv))
                intermediates_list.append(inter)
                finals.append(final_smi)
                failed_flags.append(bool(failed))

    print(
        f"{'Appended' if append_mode else 'Wrote'} {n_written} records to {jsonl_path}"
    )

    if args.save_npz:
        if append_mode:
            print(
                "Warning: --save_npz skipped because output JSONL was opened in append "
                "mode (arrays would not match full file). Run on a fresh file for .npz.",
                flush=True,
            )
        elif n_written > 0:
            npz_path = jsonl_path.with_suffix(".npz")
            np.savez(
                npz_path,
                **{
                    x_prop: np.array(coord_x, dtype=np.float64),
                    y_prop: np.array(coord_y, dtype=np.float64),
                },
                intermediates=np.array(intermediates_list, dtype=object),
                final_smiles=np.array(finals, dtype=object),
                failed=np.array(failed_flags, dtype=bool),
            )
            print(f"Wrote {npz_path}", flush=True)


if __name__ == "__main__":
    main()
