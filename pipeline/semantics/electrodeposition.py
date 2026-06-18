"""
pipeline/semantics/electrodeposition.py
========================================
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

Plugin registration
-------------------
The ``@plugin`` decorator at the bottom of this module self-registers this
family with ``pipeline.registry`` under the key ``"electrodeposition"``.

No third-party dependencies. Runs anywhere.
"""

from dataclasses import dataclass
from typing import Optional

from pipeline.registry import plugin, value_of

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


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

@plugin("electrodeposition",
        schema="schema/electrodeposition.schema.json",
        required=["J_mA_cm2", "area_cm2", "t_min"])
def electrodeposition_validator(fields: dict):
    """Adapter: maps the unified ``fields`` dict into validate_deposition()."""
    inputs = {
        "J_mA_cm2": value_of(fields, "J_mA_cm2"),
        "area_cm2": value_of(fields, "area_cm2"),
        "t_min":    value_of(fields, "t_min"),
        # z and molar mass SHOULD come from reference/chem_resolve (PubChem);
        # fall back to Li defaults only if the resolver didn't supply them.
        "z":        value_of(fields, "z", 1),
        "M_g_mol":  value_of(fields, "M_g_mol", 6.94),
    }
    extracted = {
        k: value_of(fields, k)
        for k in ("I_A", "t_s", "Q_C", "n_mol", "mass_g")
        if value_of(fields, k) is not None
    }
    return validate_deposition(inputs, extracted)
