"""
tests/test_correct.py
======================
The bounded correction loop: when the domain validator flags a misread numeric
field, correct.apply() re-extracts that ONE field from its alternate source (the
equation line) and re-validates — recovering within N=2 attempts.
"""

import pipeline.semantics.electrodeposition  # noqa: F401 — registers the plugin
from pipeline.registry import dispatch
from pipeline import correct


def _fields(q_c):
    """Page-57 validator fields with a tunable charge value."""
    return {
        "J_mA_cm2": {"value": 0.50}, "area_cm2": {"value": 0.3},
        "t_min": {"value": 90}, "z": {"value": 1}, "M_g_mol": {"value": 6.94},
        "I_A": {"value": 1.5e-4}, "Q_C": {"value": q_c},
        "n_mol": {"value": 8.4e-6}, "mass_g": {"value": 5.8e-5},
    }


def test_misread_charge_is_flagged_then_corrected():
    fields = _fields(0.31)                       # OCR misread of 0.81 C
    revalidate = lambda f: dispatch("electrodeposition", f)

    report = revalidate(fields)
    assert not report.all_passed                 # Q flagged

    alt = {"Q_C": {"text_lines": ["Q = 1.5E-4 A · 5400 s = 0.81 A·s = 0.81 C"]}}
    report = correct.apply(report, fields, alt, revalidate)

    assert report.all_passed                     # recovered from the equation line
    assert abs(fields["Q_C"]["value"] - 0.81) < 1e-9


def test_correct_loop_is_a_noop_when_nothing_flagged():
    fields = _fields(0.81)                        # correct as written
    revalidate = lambda f: dispatch("electrodeposition", f)
    report = revalidate(fields)
    assert report.all_passed
    report = correct.apply(report, fields, {}, revalidate)
    assert report.all_passed                      # unchanged
