"""
tests/test_electrodeposition.py
=================================
Regression tests for the electrodeposition validator.

Fixture data: page 57, run 240604-B1 — exactly as written in the notebook.
Derived from the __main__ block originally in _seed/electrochem_validator.py.
"""

from pipeline.semantics.electrodeposition import validate_deposition


# --- fixture data ---

PAGE_INPUTS = {
    "J_mA_cm2": 0.50,
    "area_cm2": 0.3,
    "t_min": 90,
    "z": 1,          # 1 e- = 1 Li+  (stated on the page)
    "M_g_mol": 6.94,
}

PAGE_RESULTS = {
    "I_A":    1.5e-4,   # "1.5E-4 A"
    "t_s":    5400,     # "5400 s"
    "Q_C":    0.81,     # "0.81 C"
    "n_mol":  8.4e-6,   # "8.4 E-6 mol"
    "mass_g": 5.8e-5,   # "5.8E-5 g"
}


def test_page57_all_checks_pass():
    """Page as written — all five steps should pass."""
    report = validate_deposition(PAGE_INPUTS, PAGE_RESULTS)
    print("\n" + str(report))
    assert report.all_passed, "Expected all checks to pass on the correct page data."


def test_ocr_misread_Q_flagged():
    """Simulated OCR misread: Q = 0.31 instead of 0.81.

    The validator must detect the inconsistency and set all_passed = False.
    """
    corrupted = dict(PAGE_RESULTS, Q_C=0.31)
    report = validate_deposition(PAGE_INPUTS, corrupted)
    print("\n" + str(report))
    assert not report.all_passed, "Expected all_passed=False when Q is misread."


def test_no_extracted_values_passes():
    """When the OCR layer provided nothing to compare, all_passed should be True
    (nothing failed — there was nothing to check)."""
    report = validate_deposition(PAGE_INPUTS, {})
    assert report.all_passed


def test_derived_values_are_correct():
    """Spot-check the physics re-derivation numerically."""
    report = validate_deposition(PAGE_INPUTS, {})
    by_name = {c.name: c for c in report.checks}

    assert abs(by_name["I"].derived - 1.5e-4) < 1e-10
    assert abs(by_name["t"].derived - 5400.0) < 1e-10
    assert abs(by_name["Q"].derived - 0.81)   < 1e-6
