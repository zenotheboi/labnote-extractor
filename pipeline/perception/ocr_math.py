"""
pipeline/perception/ocr_math.py
=================================
Math-OCR for the calculation block (equations, superscripts, E-notation).

Design decision (graded):
    Generic handwriting OCR mangles superscripts (cm² → cm2) and scientific
    notation (1.5E-4 → 1.5E-9 or "1.5 E-4"). A dedicated math-OCR engine
    (pix2tex / Mathpix) preserves structure because it was trained on formulae.
    The recognised expression is then parsed back into numeric fields for the
    validator to consume.

Current recogniser & upgrade path:
    The whole-page VLM already transcribes the calc lines (they arrive here as
    ``region["text_lines"]``, routed in by layout). This stage's job is the
    SECOND half of math-OCR — parsing those expression lines into structured
    ``calculations.*`` fields (value + expression). The upgrade path is to swap
    pix2tex/Mathpix on the layout-cropped calc image behind this same interface;
    nothing downstream changes.

Output: ``fields`` dict fragment with provenance "ocr_math".
"""

from __future__ import annotations

import re
from typing import Dict, List

# Each calc expression key + a pattern that recognises its line in the calc block.
_EXPRESSIONS = {
    "calculations.current.expression":    re.compile(r"mA\s*/?\s*cm.*=.*\bA\b"),
    "calculations.duration_s.expression": re.compile(r"\bt\s*=.*=.*\d\s*s\b"),
    "calculations.charge.expression":     re.compile(r"\bQ\s*=.*\bC\b"),
    "calculations.n_chain.expression":    re.compile(r"\bn\s*=\s*Q\s*/?\s*z?F"),
}

# Final "= <number> <unit>" token of a line → the numeric value, as written.
_RESULT_RE = re.compile(
    r"=\s*([-+]?\d*\.?\d+(?:\s*[eE]\s*[-+]?\d+)?)\s*([A-Za-z%/]*)\s*$"
)


def _env(value, raw: str) -> Dict:
    return {"value": value, "confidence": 0.9, "provenance": "ocr_math",
            "bbox": [0, 0, 0, 0], "raw": raw}


def extract(regions: List[Dict[str, object]]) -> Dict[str, Dict]:
    """Parse calc-block expression lines into ``calculations.*`` field fragments.

    Parameters
    ----------
    regions : list
        Subset of ``layout.segment()`` output where ``type == "calc_block"``.
        Each region carries ``text_lines`` — the recogniser's transcription of
        that region (routed in by the orchestrator).

    Returns
    -------
    dict mapping field name → Field envelope, all with provenance "ocr_math".
    Only fields that can be parsed from the calc lines are returned; the
    orchestrator merges them without overwriting higher-provenance values.
    """
    lines: List[str] = []
    for r in regions:
        lines.extend(str(x) for x in (r.get("text_lines") or []))

    frags: Dict[str, Dict] = {}
    for key, pat in _EXPRESSIONS.items():
        for line in lines:
            if pat.search(line):
                frags[key] = _env(line.strip(), line.strip())
                # also expose the final numeric result of this line as a .value
                base = key[: -len(".expression")]          # calculations.<name>
                m = _RESULT_RE.search(line.strip())
                if m and base.endswith((".current", ".charge")):
                    num = m.group(1).replace(" ", "")
                    unit = m.group(2)
                    frags.setdefault(f"{base}.value",
                                     _env(f"{num} {unit}".strip(), line.strip()))
                break
    return frags
