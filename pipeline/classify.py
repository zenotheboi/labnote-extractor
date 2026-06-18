"""
classify.py
===========
Decide which experiment a notebook page describes, so the right schema +
validator can be dispatched. This is the piece that turns "works on page 57"
into "works on the notebook".

Rule-based v1 on purpose: transparent, debuggable, zero training data, and easy
to defend in an interview. Upgrade path once you have labeled pages: swap the
keyword scorer for an embedding-similarity or small-LLM classifier behind the
SAME `classify()` signature. Nothing downstream changes.
"""

from dataclasses import dataclass

# Keyword signatures per experiment family (lowercased substring match).
SIGNATURES = {
    "electrodeposition": [
        "electrodeposition", "deposition", "plating", "v vs", "ag/agcl",
        "rde", "current density", "ma/cm", "rpm", "working elec",
        "counter elec", "ce:", "potential", "coulomb", "faraday",
    ],
    "spectroscopy": [
        "absorbance", "uv-vis", "uv vis", "wavelength", " nm", "beer",
        "molar absorptivity", "epsilon", "transmittance", "baseline", "λ",
    ],
    "synthesis": [
        "yield", "reflux", "equiv", "mmol", "purified", "column", "tlc",
        "recrystall", "workup", "rotovap", "product", "stir overnight",
    ],
    "battery_cycling": [
        "cycle", "charge/discharge", "capacity", "mah/g", "coulombic efficiency",
        "c-rate", "soc", "voltage window", "galvanostatic cycling",
    ],
}


@dataclass
class Classification:
    experiment_type: str   # "unknown" when there isn't enough signal
    confidence: float      # share of total keyword signal won by the top type
    scores: dict           # per-type hit counts — keep for logging/provenance


def classify(text: str, min_hits: int = 2) -> Classification:
    """
    text     : the concatenated plain text from the perception layer.
    min_hits : guard so one stray keyword can't trigger a confident label.
    """
    t = text.lower()
    scores = {
        etype: sum(t.count(kw) for kw in kws)
        for etype, kws in SIGNATURES.items()
    }
    total = sum(scores.values())
    top_type = max(scores, key=scores.get)
    top = scores[top_type]

    if top < min_hits or total == 0:
        return Classification("unknown", 0.0, scores)
    return Classification(top_type, top / total, scores)


if __name__ == "__main__":
    page57 = ("Project: Li electrodeposition - glyme electrolytes. "
              "Deposition run 240604-B1. Apply -0.45 V vs Ag/AgCl, 90 min, "
              "w = 1600 rpm. J = 0.50 mA/cm2. Working elec: glassy C RDE. "
              "CE: Li foil. ref: Ag/AgCl.")
    print("page 57 ->", classify(page57))

    other = "Absorbance at 540 nm, baseline corrected, Beer-Lambert fit."
    print("uv-vis   ->", classify(other))

    junk = "Lab cleaned. Ordered more pipette tips."
    print("no signal->", classify(junk))
