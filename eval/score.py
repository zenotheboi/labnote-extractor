"""
eval/score.py
==============
Compare pipeline output (or baseline output) against the reference labels and
report evaluation metrics.

NOTE: the reference labels in eval/draft_labels/ are a DRAFT (VLM-drafted, not
yet human-verified). Scores against them are indicative only — see the
ground-truth warning in CLAUDE.md. They become trustworthy once a human who can
read the hand + check the chemistry verifies them.

Metrics:
    CER  — Character Error Rate  (for text fields: prose, labels, notes)
    WER  — Word Error Rate       (for multi-word fields)
    exact_match — for numeric fields (within a ±3% tolerance)
    tanimoto     — for SMILES fields (RDKit canonical-SMILES + Morgan Tanimoto)
    schema_pass  — bool, did the output validate against the JSON schema?

Usage:
    python -m eval.score --pred pipeline_output.json --gt eval/draft_labels/page57.draft.json
"""

from __future__ import annotations

from typing import Dict


def score(pred: Dict, gt: Dict, schema_path: str) -> Dict[str, float]:
    """Compute all metrics comparing ``pred`` against ``gt``.

    Parameters
    ----------
    pred : dict
        Pipeline or baseline extraction output.
    gt : dict
        Reference labels (eval/draft_labels/*.draft.json) — VLM-drafted DRAFT,
        not yet human-verified.
    schema_path : str
        Path to the experiment-type JSON schema to check schema_pass.

    Returns
    -------
    dict with keys: "cer", "wer", "exact_match", "tanimoto", "schema_pass".
    """
    raise NotImplementedError


def cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate (Levenshtein / len(reference))."""
    raise NotImplementedError


def wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate (word-level Levenshtein / len(reference.split()))."""
    raise NotImplementedError


def smiles_tanimoto(smiles_a: str, smiles_b: str) -> float:
    """Morgan-fingerprint Tanimoto similarity between two SMILES strings.

    Requires rdkit. Returns 0.0 if either SMILES is invalid.
    """
    raise NotImplementedError
