"""
pipeline/correct.py
====================
Bounded correction loop — the only "agentic" part of the pipeline.

Triggered ONLY when the domain validator flags an inconsistency in a specific
field. Scope is strictly limited to the flagged field(s); the rest of the
extraction is never re-run.

Protocol (max N=2 attempts):
    1. Re-extract the failing field with an ALTERNATE strategy. Here the
       recogniser's ``calculations.<x>.value`` read is replaced by re-parsing the
       final number out of the corresponding EQUATION line
       (``calculations.<x>.expression``) — a genuinely different source.
       (Upgrade path: swap text-OCR ↔ math-OCR, a second OCSR model, or a
       different binarisation — all behind this same call.)
    2. Re-run the domain validator on the corrected fields.
    3. Stop on success, or after N attempts leave the field flagged
       (ok=False persists in the report → the human-review signal).

Design constraints:
    - Max N is an argument; the loop enforces the cap.
    - Bounded retries + single-field scope make this evaluable and reproducible.
    - The core pipeline MUST NOT become an open-ended agent loop.
"""

from __future__ import annotations

import re
from typing import Callable, Dict

# validator check-name → the flat fields key it compares
_CHECK_KEY = {"I": "I_A", "t": "t_s", "Q": "Q_C", "n": "n_mol", "mass": "mass_g"}

# final "= <number> <unit>" token of an equation line
_RESULT_RE = re.compile(r"=\s*([-+]?\d*\.?\d+(?:\s*[eE]\s*[-+]?\d+)?)\s*[A-Za-z%/·]*\s*$")


def _parse_final_number(text: str):
    m = _RESULT_RE.search(text.strip())
    if not m:
        return None
    try:
        return float(m.group(1).replace(" ", ""))
    except ValueError:
        return None


def correct(
    fields: Dict[str, Dict],
    failing_key: str,
    region: Dict[str, object],
    *,
    max_attempts: int = 2,
) -> Dict[str, Dict]:
    """Re-extract a single failing field via an alternate strategy.

    Parses the value from the equation line(s) carried in
    ``region["text_lines"]`` (a different source than the recogniser's ``.value``
    read) and writes it back with ``provenance = "derived"``. Returns ``fields``.
    """
    for line in region.get("text_lines", []):
        alt = _parse_final_number(str(line))
        if alt is not None:
            fields[failing_key] = {"value": alt, "confidence": 0.5,
                                   "provenance": "derived"}
            break
    return fields


def apply(report, fields: Dict[str, Dict],
          alt_regions: Dict[str, Dict],
          revalidate: Callable[[Dict], object],
          *, max_attempts: int = 2):
    """Run the bounded correction loop and return the (possibly improved) report.

    Parameters
    ----------
    report : ValidationReport
        The first validator pass; ``checks`` with ``ok is False`` are flagged.
    fields : dict
        The validator's input fields (mutated in place for flagged keys only).
    alt_regions : dict
        Map flagged-field key → region dict carrying ``text_lines`` (the
        alternate source). e.g. ``{"Q_C": {"text_lines": ["Q = ... = 0.81 C"]}}``.
    revalidate : callable
        ``fields -> ValidationReport`` (normally registry.dispatch bound to the
        experiment type).
    """
    for _ in range(max_attempts):
        flagged = [c for c in report.checks if c.ok is False]
        if not flagged:
            break
        for c in flagged:
            key = _CHECK_KEY.get(c.name)
            if key and key in fields and key in alt_regions:
                correct(fields, key, alt_regions[key], max_attempts=max_attempts)
        report = revalidate(fields)
    return report
