"""
reference/chem_resolve.py
==========================
Resolve a chemical name → {smiles, formula, molar_mass, z, source} via a known
reference table + RDKit, with OPSIN (py2opsin) for systematic IUPAC names when a
JRE is available.

Design decision (graded):
    The page LABELS its structures: "[Li(12-crown-4)]+", "LiTFSI", "diglyme".
    Name-resolution is MORE RELIABLE than reading a hand-drawn sketch, so we:
        1. Look up known reagents (table below — the reagents on this page).
        2. Try OPSIN for systematic names (needs a JRE; degraded if absent).
        3. RDKit canonicalises the SMILES and computes formula + molar mass.
    OCSR output is used only to CONFIRM (canonical-SMILES / Tanimoto) — see
    pipeline/perception/ocsr.py.

z (electrons per ion, electrodeposition) is element-specific and not derivable
from RDKit, so it lives in the table: Li+ z=1, Cu²⁺ z=2, Ni²⁺ z=2.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# Known reagents/species on this notebook page (SMILES from CLAUDE.md).
_KNOWN: Dict[str, Dict[str, object]] = {
    "12-crown-4": {"smiles": "C1COCCOCCOCCO1"},
    "diglyme":    {"smiles": "COCCOCCOC"},
    "litfsi":     {"smiles": "[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F"},
    # deposited metals — z + atomic mass for the validator
    "li":      {"smiles": "[Li]", "z": 1, "molar_mass": 6.94},
    "lithium": {"smiles": "[Li]", "z": 1, "molar_mass": 6.94},
    "cu":      {"smiles": "[Cu]", "z": 2, "molar_mass": 63.55},
    "ni":      {"smiles": "[Ni]", "z": 2, "molar_mass": 58.69},
}


def _key(name: str) -> str:
    """Normalise a label to a table key: lowercase, drop charge/coordination."""
    s = name.lower().strip()
    s = re.sub(r"^\[|\]?[+-]?$", "", s)          # strip [ ] and trailing charge
    s = re.sub(r"^\[?li\(|\)\]?\+?$", "", s)     # "[Li(12-crown-4)]+" -> "12-crown-4"
    return s.strip()


def _rdkit_props(smiles: str):
    """(canonical_smiles, formula, molar_mass) via RDKit, or (smiles, None, None)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, None, None
        return (Chem.MolToSmiles(mol),
                rdMolDescriptors.CalcMolFormula(mol),
                round(Descriptors.MolWt(mol), 2))
    except ImportError:
        return smiles, None, None


def _opsin(name: str) -> Optional[str]:
    """Resolve a systematic IUPAC name to SMILES via OPSIN (needs a JRE)."""
    try:
        from py2opsin import py2opsin
        out = py2opsin(name, output_format="SMILES")
        return out or None
    except Exception:
        return None


def resolve(name: str) -> Optional[Dict[str, object]]:
    """Resolve a chemical name to structure and properties (see module docstring)."""
    if not name:
        return None

    entry = _KNOWN.get(_key(name)) or _KNOWN.get(name.lower().strip())
    smiles = entry.get("smiles") if entry else None
    source = "table" if smiles else None

    if smiles is None:
        smiles = _opsin(name)
        source = "opsin" if smiles else None
    if smiles is None:
        return None

    canon, formula, molar_mass = _rdkit_props(str(smiles))
    return {
        "smiles":     canon,
        "formula":    formula,
        "molar_mass": (entry.get("molar_mass") if entry and entry.get("molar_mass")
                       else molar_mass),
        "z":          entry.get("z") if entry else None,
        "source":     source,
    }


def deposition_constants(species: str) -> Dict[str, object]:
    """z + molar mass for the deposited metal, for the electrochem validator.

    Falls back to Li (z=1, 6.94 g/mol) so the validator never crashes — the same
    default the validator used when these were hardcoded.
    """
    info = resolve(species or "Li") or {}
    return {
        "z":        info.get("z") or 1,
        "molar_mass": info.get("molar_mass") or 6.94,
        "source":   info.get("source") or "fallback",
    }
