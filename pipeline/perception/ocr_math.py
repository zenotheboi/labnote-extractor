"""
pipeline/perception/ocr_math.py
=================================
LaTeX-OCR for the calculation block (equations, superscripts, E-notation).

Design decision (graded):
    Generic handwriting OCR mangles superscripts (cm² → cm2) and scientific
    notation (1.5E-4 → 1.5E-9 or "1.5 E-4"). A dedicated math-OCR engine
    (pix2tex / Mathpix) preserves structure because it was trained on formulae.
    The LaTeX output is then parsed back into Python numeric fields for the
    validator to consume.

LaTeX → numeric parsing:
    ``\\frac{a}{b}`` → a/b
    ``a \\times 10^{-4}`` → a * 1e-4
    Superscript units (cm^{2}, mA/cm^{2}) → stored in field["unit"] as
    "cm²", "mA/cm²" after Unicode normalisation (see normalize.py).

Output: ``fields`` dict fragment with provenance "ocr_math".
"""

from __future__ import annotations

from typing import Dict, List


def extract(regions: List[Dict[str, object]]) -> Dict[str, Dict]:
    """Run LaTeX-OCR on math regions and return numeric field fragments.

    Parameters
    ----------
    regions : list
        Subset of ``layout.segment()`` output where ``type == "calc_block"``.

    Returns
    -------
    dict mapping field name → Field envelope.
        Every field has ``provenance == "ocr_math"``; numeric values are
        Python ``float``; raw LaTeX is stored in ``field["raw"]``.
    """
    raise NotImplementedError
