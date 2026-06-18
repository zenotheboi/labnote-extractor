"""
pipeline/perception/normalize.py
==================================
Symbol / unit post-processing and strikethrough / edit detection.

Runs AFTER ocr_text and ocr_math; updates fields in-place.

Tasks:
1. Unit normalisation:
       bare "C" after a temperature value → "°C"
       "2theta" / "2Θ" → "2θ"
       "cm2" → "cm²",  "mA/cm2" → "mA/cm²"
       "Li+" stays "Li+"; "e-" stays "e-"
2. Scientific-notation repair:
       "1.5 E-4" / "1.5e -4" → 1.5e-4 (float)
3. Strikethrough detection:
       The page has edits ("8%" struck → "5 mol%"; "diglyme" written twice).
       Record FINAL value; store the struck text in ``field["raw"]`` so the
       correction is auditable.

Output: the same ``fields`` dict with updated values; provenance of any updated
    field is set to "normalize".
"""

from __future__ import annotations

from typing import Dict


def run(fields: Dict[str, Dict]) -> Dict[str, Dict]:
    """Apply symbol normalisation and strikethrough detection to all fields.

    Parameters
    ----------
    fields : dict
        Accumulated Field envelopes from ocr_text, ocr_math, and ocsr.

    Returns
    -------
    The same dict, with updated ``value``, ``unit``, and ``provenance``
    entries where normalisation was applied.
    """
    raise NotImplementedError
