"""
pipeline/correct.py
====================
Optional bounded correction loop — the only "agentic" part of the pipeline.

Triggered ONLY when the domain validator flags an inconsistency in a specific
field. Scope is strictly limited to the ONE failing region; the rest of the
extraction is never re-run.

Protocol (max N=2 attempts):
    1. Re-extract the failing region with an ALTERNATE strategy:
           text region  → swap text-OCR  ↔ math-OCR
           math region  → try different binarisation (Sauvola window size)
           structure    → try the second OCSR model in the ensemble
    2. Re-run the domain validator on the corrected field only.
    3. On success → return updated fields dict.
       After N attempts without success → set confidence=0.0, flag for human.

Design constraints:
    - Max N is an argument; caller enforces the cap.
    - Bounded retries + single-region scope make this evaluable and reproducible.
    - The core pipeline MUST NOT become an open-ended agent loop.
"""

from __future__ import annotations

from typing import Dict


def correct(
    fields: Dict[str, Dict],
    failing_key: str,
    region: Dict[str, object],
    *,
    max_attempts: int = 2,
) -> Dict[str, Dict]:
    """Attempt to correct a single failing field via alternate extraction.

    Parameters
    ----------
    fields : dict
        The full fields dict from the perception layer.
    failing_key : str
        The field name flagged by the domain validator (e.g. ``"Q_C"``).
    region : dict
        The layout region dict (bbox + image crop) for the failing field.
    max_attempts : int
        Hard cap on re-extraction attempts. Default 2.

    Returns
    -------
    Updated ``fields`` dict. If correction succeeds, the failing field is
    updated with the new value and ``provenance = "derived"`` (re-extracted).
    If all attempts fail, the field gets ``confidence = 0.0`` and a
    ``"human_review_required": True`` annotation.
    """
    raise NotImplementedError
