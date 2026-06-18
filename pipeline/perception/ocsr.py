"""
pipeline/perception/ocsr.py
=============================
Optical Chemical Structure Recognition for hand-drawn molecular structures.

Design decision (graded):
    MolScribe is trained on PRINTED structures — hand-drawn is out of scope per
    its authors. DECIMER ships a dedicated hand-drawn model and is preferred.
    Ensemble: DECIMER + MolScribe + MolNexTR; take a validated consensus
    (see MARCUS, RSC 2025 for the consensus protocol).

    Every SMILES candidate is gated through RDKit: ``Chem.MolFromSmiles(smiles)``
    must return a non-None mol object, otherwise the candidate is REJECTED and
    the next-best candidate is tried. Unresolvable structures get
    ``smiles=None, source="unresolved"``.

    Note: name-resolution (OPSIN/PubChem) is the PRIMARY strategy for known
    reagents; OCSR is a cross-check (see reference/chem_resolve.py and
    the pipeline decision log in CLAUDE.md §decisions).

Output: list of structure dicts, each with:
    label  : Field (from OCR)
    smiles : str or None
    formula: str or None
    source : "label_resolution" | "ocsr" | "consensus" | "unresolved"
    rdkit_valid : bool
    confidence  : float
    bbox        : [x, y, w, h]
"""

from __future__ import annotations

from typing import Dict, List, Optional


def extract(regions: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Run OCSR on drawing regions and return structure dicts.

    Parameters
    ----------
    regions : list
        Subset of ``layout.segment()`` output where
        ``type == "reaction_scheme"``.

    Returns
    -------
    list of structure dicts (schema: ``$defs/structure`` in
    ``electrodeposition.schema.json``).
    """
    raise NotImplementedError


def _rdkit_valid(smiles: Optional[str]) -> bool:
    """Return True iff RDKit can parse the SMILES without error.

    Requires rdkit. Gracefully returns False if rdkit is not installed.
    """
    if smiles is None:
        return False
    try:
        from rdkit import Chem  # type: ignore[import]
        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return False
