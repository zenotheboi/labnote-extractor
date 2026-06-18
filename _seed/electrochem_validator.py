"""
electrochem_validator.py
========================
Cross-checks the electrochemistry calculation on a lab-notebook page by
RE-DERIVING it from first principles and comparing every intermediate step
against the value the OCR layer read off the handwriting.

WHY THIS EXISTS (this is the differentiator)
--------------------------------------------
OCR and vision-language models silently misread digits:
    "5400 s" -> "5900 s",  "0.81 C" -> "0.31 C",  "1.5E-4" -> "1.5E-9".
A generic extractor has no way to notice it was wrong.

But an electrodeposition calculation is a CLOSED CHAIN:
    J  ->  I  ->  Q  ->  n  ->  mass
If we recompute that chain from the primary inputs and a step disagrees with
what was written on the page, we have (1) caught the extraction error and
(2) localized exactly which value is bad. This converts domain knowledge into
an automatic validator — something no off-the-shelf model does.

No third-party dependencies. Runs anywhere.
"""

from dataclasses import dataclass
from typing import Optional

# --- physical constants ---
FARADAY = 96485.0          # C / mol of electrons
M_LI_DEFAULT = 6.94        # g / mol  (Li molar mass, as written on the page)


@dataclass
class StepCheck:
    name: str
    derived: float          # what physics says the value should be
    extracted: Optional[float]  # what OCR read off the page (None = not written)
    unit: str
    rel_error: Optional[float]
    ok: Optional[bool]      # None = nothing to compare against

    def line(self) -> str:
        d = f"{self.derived:.4g} {self.unit}"
        if self.extracted is None:
            return f"  [ -- ] {self.name:8s} derived {d}  (not written on page)"
        flag = "PASS" if self.ok else "FLAG"
        e = f"{self.extracted:.4g} {self.unit}"
        return (f"  [{flag}] {self.name:8s} derived {d:>16s} | "
                f"page {e:>16s} | rel err {self.rel_error*100:5.2f}%")


@dataclass
class ValidationReport:
    checks: list
    all_passed: bool

    def __str__(self) -> str:
        head = "ELECTROCHEMISTRY SELF-CONSISTENCY CHECK\n" + "-" * 70
        body = "\n".join(c.line() for c in self.checks)
        verdict = "ALL CHECKS CONSISTENT" if self.all_passed \
            else ">>> INCONSISTENCY DETECTED — flag for human review <<<"
        return f"{head}\n{body}\n" + "-" * 70 + f"\n{verdict}"


def validate_deposition(inputs: dict,
                        extracted: Optional[dict] = None,
                        rel_tol: float = 0.03) -> ValidationReport:
    """
    inputs : primary measured quantities read from the page
        J_mA_cm2 : current density (mA/cm^2)
        area_cm2 : electrode area (cm^2)
        t_min    : deposition time (minutes)
        z        : electrons transferred per ion (1 for Li+)   [default 1]
        M_g_mol  : molar mass of deposited species (g/mol)     [default 6.94]

    extracted : the intermediate RESULTS the chemist wrote down, as read by OCR.
        Any subset of {I_A, t_s, Q_C, n_mol, mass_g}. Each one present is
        cross-checked against the re-derived value.

    rel_tol : relative error allowed before a step is flagged. Default 3% is
        loose enough for 2-significant-figure handwriting, tight enough to
        catch a transposed or dropped digit.
    """
    extracted = extracted or {}
    z = inputs.get("z", 1)
    M = inputs.get("M_g_mol", M_LI_DEFAULT)

    # --- re-derive the full chain from primary inputs ---
    I_A   = (inputs["J_mA_cm2"] / 1000.0) * inputs["area_cm2"]   # mA/cm^2 -> A
    t_s   = inputs["t_min"] * 60.0
    Q_C   = I_A * t_s
    n_mol = Q_C / (z * FARADAY)
    mass  = n_mol * M

    derived = {
        "I":    (I_A,   "A"),
        "t":    (t_s,   "s"),
        "Q":    (Q_C,   "C"),
        "n":    (n_mol, "mol"),
        "mass": (mass,  "g"),
    }
    key_map = {"I": "I_A", "t": "t_s", "Q": "Q_C", "n": "n_mol", "mass": "mass_g"}

    checks = []
    for name, (dval, unit) in derived.items():
        page_val = extracted.get(key_map[name])
        if page_val is None:
            checks.append(StepCheck(name, dval, None, unit, None, None))
            continue
        rel = abs(dval - page_val) / abs(dval) if dval else float("inf")
        checks.append(StepCheck(name, dval, page_val, unit, rel, rel <= rel_tol))

    comparable = [c for c in checks if c.ok is not None]
    all_passed = all(c.ok for c in comparable) if comparable else True
    return ValidationReport(checks, all_passed)


if __name__ == "__main__":
    # Page 57, run 240604-B1 — exactly as written in the notebook
    page_inputs = {
        "J_mA_cm2": 0.50,
        "area_cm2": 0.3,
        "t_min": 90,
        "z": 1,                 # 1 e- = 1 Li+  (stated on the page)
        "M_g_mol": 6.94,
    }
    page_results = {
        "I_A":    1.5e-4,       # "1.5E-4 A"
        "t_s":    5400,         # "5400 s"
        "Q_C":    0.81,         # "0.81 C"
        "n_mol":  8.4e-6,       # "8.4 E-6 mol"
        "mass_g": 5.8e-5,       # "5.8E-5 g"
    }
    print("CASE 1 — page as written (should all pass):")
    print(validate_deposition(page_inputs, page_results))

    print("\nCASE 2 — simulate an OCR misread of Q (0.81 -> 0.31):")
    corrupted = dict(page_results, Q_C=0.31)
    print(validate_deposition(page_inputs, corrupted))
