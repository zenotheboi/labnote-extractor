"""
pipeline/perception/layout.py
==============================
Segment a preprocessed page into typed regions, then route each region to
the correct OCR/OCSR stage.

Regions produced:
    "header"         — page number, date, project name
    "prose"          — handwritten sentences and bullet points
    "calc_block"     — calculation block with equations and numeric results
    "reaction_scheme" — hand-drawn molecular structures + reaction arrows
    "temp_table"     — temperature/time table (layout: rows × columns)
    "margin_notes"   — right-margin annotations

Critical sub-task — drawing-vs-text classifier:
    Drawings (closed loops, irregular line spacing, low text-line regularity)
    are routed to OCSR; everything else to OCR.
    Approach: LayoutParser or small YOLO; classical stroke-statistics as fallback.
    The chosen approach must be documented in a log line so the grader sees it.

Output: list of region dicts, each carrying the Field envelope (bbox, type, …).
"""

from __future__ import annotations

from typing import Dict, List


def segment(preprocessed: Dict[str, object]) -> List[Dict[str, object]]:
    """Segment the page into typed regions.

    Parameters
    ----------
    preprocessed : dict
        Output of ``preprocess.preprocess()``.

    Returns
    -------
    list of region dicts, each with:
        ``type``       — one of the region type strings listed in the module docstring
        ``bbox``       — [x, y, w, h] in source-image pixels
        ``image_crop`` — ndarray crop from ``image_clean``
        ``image_crop_original`` — ndarray crop from ``image_original`` (for OCSR)
    """
    raise NotImplementedError
