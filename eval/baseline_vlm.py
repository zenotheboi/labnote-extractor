"""
eval/baseline_vlm.py
======================
Single-VLM baseline — one prompt, one model call, no decomposition.

Purpose: establish the score a naive approach achieves so we can quantify
the lift our pipeline provides. This is the headline number for the demo video.

Model: GPT-4o or Claude Sonnet (configurable). Single prompt asks for all
fields at once. Output is JSON-parsed and run through score.py.

WARNING: do not use this baseline's output as ground truth (circular — see
CLAUDE.md §ground-truth-warning).
"""

from __future__ import annotations

from typing import Dict, Optional


def run_baseline(image_path: str, model: str = "claude-sonnet-4-6") -> Dict:
    """Send a single page image to a VLM and return the raw extraction dict.

    Parameters
    ----------
    image_path : str
        Path to the page image.
    model : str
        Model identifier. Defaults to claude-sonnet-4-6.

    Returns
    -------
    dict — raw extraction (NOT schema-validated; may have missing/extra fields).
    """
    raise NotImplementedError
