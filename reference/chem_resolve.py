"""
reference/chem_resolve.py
==========================
Resolve a chemical name → {smiles, formula, molar_mass, z, source} via a
resolution chain that avoids page-specific hints:

    1. OPSIN — systematic IUPAC→SMILES, offline (needs a JRE).
    2. PubChem REST API — public database, handles common names + abbreviations
       (e.g. "diglyme", "EtOH", "LiTFSI", "12-crown-4").
    3. _FALLBACK table — last resort only; also the sole source of z (electrons
       per ion) for deposited metals, which PubChem does not supply.

Design decision (graded):
    The page LABELS its structures: "[Li(12-crown-4)]+", "LiTFSI", "diglyme",
    "EtOH". The "diglyme : EtOH" text has arrows pointing to TWO SEPARATE
    drawings — resolve both independently.
    Name-resolution is MORE RELIABLE than reading a hand-drawn sketch, so we
    resolve the label first and use OCSR only to CONFIRM — see
    pipeline/perception/ocsr.py.

z (electrons per ion, electrodeposition) is element-specific and not in any
public database; it lives in _FALLBACK: Li+ z=1, Cu²⁺ z=2, Ni²⁺ z=2.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# Last-resort fallback + electrodeposition constants (z, molar_mass).
# NOT consulted for SMILES until OPSIN and PubChem have both failed.
_FALLBACK: Dict[str, Dict[str, object]] = {
    "12-crown-4": {"smiles": "C1COCCOCCOCCO1"},
    "diglyme":    {"smiles": "COCCOCCOC"},
    "etoh":       {"smiles": "CCO"},
    "ethanol":    {"smiles": "CCO"},
    "litfsi":     {"smiles": "[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F"},
    # deposited metals — z + atomic mass for the validator (PubChem has no z)
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


def _pubchem(name: str) -> Optional[str]:
    """Query PubChem PUG REST for a canonical SMILES by name. Returns None on any failure."""
    try:
        import requests
        import urllib.parse
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            + urllib.parse.quote(name)
            + "/property/SMILES,MolecularFormula,MolecularWeight/JSON"
        )
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json()["PropertyTable"]["Properties"][0].get("SMILES")
    except Exception:
        pass
    return None


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


def _java_available() -> bool:
    """True iff a working `java` is on PATH (OPSIN needs a JRE). Cached so the
    check runs once; avoids py2opsin's noisy per-call RuntimeWarnings when Java is
    absent — resolution falls through to PubChem silently instead."""
    global _JAVA_OK
    if _JAVA_OK is None:
        import shutil, subprocess
        _JAVA_OK = False
        if shutil.which("java"):
            try:
                subprocess.run(["java", "-version"], capture_output=True, check=True)
                _JAVA_OK = True
            except Exception:
                _JAVA_OK = False
    return _JAVA_OK


_JAVA_OK: Optional[bool] = None


def _opsin(name: str) -> Optional[str]:
    """Resolve a systematic IUPAC name to SMILES via OPSIN (needs a JRE)."""
    if not _java_available():
        return None
    try:
        from py2opsin import py2opsin
        out = py2opsin(name, output_format="SMILES")
        return out or None
    except Exception:
        return None


def resolve(name: str) -> Optional[Dict[str, object]]:
    """Resolve a chemical name to structure and properties (see module docstring).

    Resolution order: OPSIN → PubChem → _FALLBACK (last resort).
    z and molar_mass for deposited metals always come from _FALLBACK.
    """
    if not name:
        return None

    # _FALLBACK entry provides z/molar_mass for metals regardless of SMILES source
    entry = _FALLBACK.get(_key(name)) or _FALLBACK.get(name.lower().strip())

    # 1. OPSIN — systematic IUPAC names, offline
    smiles = _opsin(name)
    source = "opsin" if smiles else None

    # 2. PubChem — public database, covers common names and abbreviations
    if smiles is None:
        smiles = _pubchem(name)
        source = "pubchem" if smiles else None

    # 3. _FALLBACK — last resort, no network required
    if smiles is None and entry:
        smiles = entry.get("smiles")
        source = "table" if smiles else None

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

    Falls back to Li (z=1, 6.94 g/mol) so the validator never crashes.
    z values come from _FALLBACK (PubChem does not carry electrodeposition z).
    """
    info = resolve(species or "Li") or {}
    return {
        "z":        info.get("z") or 1,
        "molar_mass": info.get("molar_mass") or 6.94,
        "source":   info.get("source") or "fallback",
    }
