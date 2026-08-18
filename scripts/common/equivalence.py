"""Equivalence relations over SMILES, shared by every section, flow and
cohesion probe.

Each convention maps a SMILES string to an equivalence-class label. Two grid
cells share a color iff their labels are equal; borders are drawn wherever
adjacent cells disagree.

The six conventions, coarser as you go down. The code name is what every
--convention flag and CSV column uses:

  code name       paper name      meaning
  canonical       SMILES          canonical isomeric SMILES (baseline)
  inchikey        InChIKey-14     InChIKey first block (tautomers + some stereo merged)
  murcko          Murcko          Bemis-Murcko scaffold SMILES (ring system + linkers)
  murcko_generic  Murcko-generic  Murcko scaffold with all atoms->C, all bonds->single
  formula         Formula         molecular formula (atom counts only)
  composition     Elements        sorted set of heavy-atom element symbols (no H, no counts)

Invalid / unparseable / empty SMILES always map to INVALID_KEY regardless of
convention.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401  (ensures InChI/scaffold deps load)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


INVALID_KEY = "__invalid__"
NO_SCAFFOLD_KEY = "__no_scaffold__"

CONVENTIONS = (
    "canonical",
    "inchikey",
    "murcko",
    "murcko_generic",
    "formula",
    "composition",
)


def _parse(smiles):
    if not smiles or not str(smiles).strip():
        return None
    return Chem.MolFromSmiles(smiles)


def equiv_key(smiles, convention="canonical"):
    """Return the equivalence class label for a SMILES under the given convention.

    Always returns INVALID_KEY for empty/unparseable input, or when RDKit
    raises during downstream operations (e.g. MakeScaffoldGeneric raising
    AtomValenceException on a pathological HierVAE/GDSS decode).
    """
    mol = _parse(smiles)
    if mol is None:
        return INVALID_KEY

    try:
        if convention == "canonical":
            return Chem.MolToSmiles(mol)
        if convention == "inchikey":
            ikey = Chem.MolToInchiKey(mol)
            if not ikey:
                return INVALID_KEY
            return ikey.split("-")[0]
        if convention == "formula":
            return CalcMolFormula(mol)
        if convention == "composition":
            elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
            return ",".join(elements) if elements else INVALID_KEY
        if convention == "murcko":
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumAtoms() == 0:
                return NO_SCAFFOLD_KEY
            return Chem.MolToSmiles(scaffold)
        if convention == "murcko_generic":
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumAtoms() == 0:
                return NO_SCAFFOLD_KEY
            generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            return Chem.MolToSmiles(generic)
    except (Chem.AtomValenceException, Chem.KekulizeException,
            Chem.AtomKekulizeException, ValueError, RuntimeError):
        return INVALID_KEY

    raise ValueError(
        f"unknown convention {convention!r}; expected one of {CONVENTIONS}"
    )


def representative_smiles_for_class(smiles, convention="canonical"):
    """Return a canonical-SMILES representative for a class.

    For canonical the label is already a SMILES. For murcko and
    murcko_generic the label is the scaffold SMILES (still a valid molecule).
    For inchikey/formula/composition we return the input SMILES canonicalized
    so the molecule-grid PNG has something chemically meaningful to render.
    """
    if convention in ("canonical", "murcko", "murcko_generic"):
        # the equiv_key is itself a SMILES, or the sentinel
        return equiv_key(smiles, convention)
    mol = _parse(smiles)
    if mol is None:
        return INVALID_KEY
    return Chem.MolToSmiles(mol)


# palette

def palette(n_classes):
    """Return n matplotlib color specs from TAB10, cycled with `% 10`.

    All views (canonical and alternate) share this palette. The tessellation
    borders are drawn between cells of different classes, so identity is
    carried by the black boundaries — same-color non-adjacent cells are
    unambiguously distinct because the borders separate them.
    """
    return [f"C{i % 10}" for i in range(n_classes)]
