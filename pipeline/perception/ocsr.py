"""
pipeline/perception/ocsr.py
=============================
Optical Chemical Structure Recognition for hand-drawn molecular structures.

Design decision (graded):
    MolScribe is trained on PRINTED structures — hand-drawn is out of scope per
    its authors. DECIMER ships a dedicated hand-drawn model and is preferred.
    Ensemble: DECIMER + MolScribe + MolNexTR; take a validated consensus
    (see MARCUS, RSC 2025 for the consensus protocol).

    Every SMILES candidate is gated through RDKit: ``Chem.MolFromSmiles`` must
    return a non-None mol, otherwise the candidate is REJECTED. Unresolvable
    structures get ``smiles=None, source="unresolved"``.

    Name-resolution (OPSIN/PubChem, reference/chem_resolve.py) is the PRIMARY
    strategy for known reagents; OCSR is a cross-check. Where the drawing and the
    name DISAGREE (e.g. a sketch that does not match the resolved name), the drawn
    structure is kept per CLAUDE.md ("record each AS DRAWN").
    NOTE: the page has TWO solvent drawings (diglyme + EtOH) — expect two separate
    structures, not one merged glyme structure.

DECIMER is an optional, heavy TensorFlow dependency (CLAUDE.md recommends a
separate venv). When it is not installed, the recogniser-provided SMILES (the
whole-page VLM's reading of the drawing) is used as the OCSR candidate and gated
through RDKit just the same — the interface is identical, so swapping DECIMER in
changes nothing downstream.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def canonical(smiles: Optional[str]) -> Optional[str]:
    """RDKit canonical SMILES, or None if absent/invalid (the validity gate)."""
    if not smiles:
        return None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except ImportError:
        return smiles   # no rdkit: can't gate; pass the candidate through


def tanimoto(smiles_a: Optional[str], smiles_b: Optional[str]) -> float:
    """Morgan-fingerprint Tanimoto between two SMILES (0.0 if either invalid)."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator, DataStructs
        RDLogger.DisableLog("rdApp.*")
        ma, mb = Chem.MolFromSmiles(smiles_a or ""), Chem.MolFromSmiles(smiles_b or "")
        if ma is None or mb is None:
            return 0.0
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fa, fb = gen.GetFingerprint(ma), gen.GetFingerprint(mb)
        return float(DataStructs.TanimotoSimilarity(fa, fb))
    except ImportError:
        return 0.0


def _rdkit_valid(smiles: Optional[str]) -> bool:
    """Return True iff RDKit can parse the SMILES without error."""
    if smiles is None:
        return False
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return False


def reconcile(label: str, drawn_smiles: Optional[str],
              resolved: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Combine the drawn (OCSR) SMILES with the name-resolved structure.

    Strategy (graded): name-resolution primary, OCSR as confirmation.
        - drawn + resolved agree (Tanimoto ≥ 0.9)      → consensus, canonical name
        - resolved valid, drawn invalid/missing        → label_resolution
        - resolved missing, drawn valid                → ocsr (record as drawn)
        - they disagree (both valid, low Tanimoto)     → keep DRAWN as-is, flag
        - neither valid                                → unresolved (smiles=None)
    """
    drawn = canonical(drawn_smiles)
    name_smiles = canonical(resolved.get("smiles")) if resolved else None

    if drawn and name_smiles:
        if tanimoto(drawn, name_smiles) >= 0.9:
            return {"smiles": name_smiles, "source": "consensus", "rdkit_valid": True}
        return {"smiles": drawn, "source": "ocsr", "rdkit_valid": True}  # as drawn
    if name_smiles:
        return {"smiles": name_smiles, "source": "label_resolution", "rdkit_valid": True}
    if drawn:
        return {"smiles": drawn, "source": "ocsr", "rdkit_valid": True}
    return {"smiles": None, "source": "unresolved", "rdkit_valid": False}


def extract(regions: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Run OCSR on drawing regions and return structure dicts.

    When DECIMER is installed, it recognises each ``reaction_scheme`` crop here;
    otherwise the orchestrator supplies the recogniser's drawn SMILES per region
    in ``region["candidates"]`` ({label: smiles}). Either way every candidate is
    RDKit-gated and reconciled against name-resolution by the caller.
    """
    structures: List[Dict[str, object]] = []
    for region in regions:
        for label, smiles in (region.get("candidates") or {}).items():
            structures.append({
                "label": label,
                "smiles": canonical(smiles),
                "rdkit_valid": _rdkit_valid(smiles),
                "source": "ocsr" if _rdkit_valid(smiles) else "unresolved",
                "bbox": region.get("bbox", [0, 0, 0, 0]),
            })
    return structures
