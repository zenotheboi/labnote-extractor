"""
pipeline/perception/ocr_text.py
=================================
Handwriting OCR for prose, header, table cells, and margin notes.

Design decision (graded):
    Ensemble TWO engines (e.g. Azure Read + Google Vision, or TrOCR + one
    cloud API) and reconcile disagreements. A single engine is not trusted
    because handwriting recognition error rates are 5–15% even on clean scans;
    an ensemble with majority vote reduces random errors and makes systematic
    errors visible.

Reconciliation strategy:
    - Agreement → use the shared text, average the confidence scores.
    - Disagreement → emit the higher-confidence result, set confidence to the
      lower of the two scores (conservative), mark provenance "ocr_text".
    - Unreadable by both → value=None, confidence=0.0, flag for human review.

Output: ``fields`` dict fragment; keys are semantic field names inferred from
    region type and position (e.g. "date", "project", "additive_name").
    Each value follows the Field envelope: {value, confidence, provenance,
    bbox, raw}.
"""

from __future__ import annotations

from typing import Dict, List


def extract(regions: List[Dict[str, object]]) -> Dict[str, Dict]:
    """Run handwriting OCR on text regions and return field fragments.

    Parameters
    ----------
    regions : list
        Subset of ``layout.segment()`` output where ``type`` is NOT
        ``"reaction_scheme"`` (i.e. not a drawing region).

    Returns
    -------
    dict mapping field name → Field envelope.
        Every field has ``provenance == "ocr_text"``.
    """
    raise NotImplementedError
