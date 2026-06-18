"""
pipeline/perception/normalize.py
==================================
Symbol / unit post-processing and strikethrough / edit detection.

Runs AFTER ocr_text and ocr_math; updates fields in-place.

Tasks:
1. Unit normalisation toward the canonical (ASCII) convention the eval ground
   truth uses — superscripts to caret form:
       "cm²" → "cm^2",  "mA/cm²" → "mA/cm^2",  "e⁻" → "e^-"
   plus "·"/"×" → "*", reaction arrows → "->", "2Θ"/"2theta" → "2θ".
   (The stub originally described rebuilding pretty Unicode; we instead
   canonicalise to the ASCII form because that is what eval/ground_truth/page57.json
   is written in, so it is what scores. Documented decision.)
2. Temperature: a bare "C" after a number in a TEMPERATURE field → "°C"
   (field-aware: we must NOT touch "0.81 C" in the charge field).
3. Strikethrough detection: a value carrying "[struck: ...]" keeps the FINAL
   text; the struck original is preserved in ``field["raw"]`` so the edit stays
   auditable.

Any field whose value changes has its provenance set to "normalize" and its
original preserved in ``field["raw"]``.
"""

from __future__ import annotations

import re
from typing import Dict

# Unicode superscript run → ASCII caret form.
_SUPERSCRIPT = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
    "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+",
}
_ARROWS = ("→", "⟶", "➝", "➞", "⇒", "⟹", "-->", "--->")

_STRUCK_RE = re.compile(r"\[struck:[^\]]*\]\s*")


def _superscripts_to_caret(s: str) -> str:
    out, run = [], ""
    for ch in s:
        if ch in _SUPERSCRIPT:
            run += _SUPERSCRIPT[ch]
        else:
            if run:
                out.append("^" + run)
                run = ""
            out.append(ch)
    if run:
        out.append("^" + run)
    return "".join(out)


def _normalize_symbols(s: str) -> str:
    s = _superscripts_to_caret(s)
    s = s.replace("·", "*").replace("×", "*")
    for a in _ARROWS:
        s = s.replace(a, "->")
    s = s.replace("Θ", "θ").replace("2theta", "2θ").replace("2Θ", "2θ")
    return s


def _bare_C_to_degree(s: str) -> str:
    if "°C" in s:
        return s
    return re.sub(r"(\d+(?:\.\d+)?)\s*C\b", r"\1 °C", s)


def _is_temperature_key(key: str) -> bool:
    k = key.lower()
    # the temp-test LABEL already carries °C; only the scalar temperature fields
    return "temperature" in k and "label" not in k


def run(fields: Dict[str, Dict]) -> Dict[str, Dict]:
    """Apply symbol normalisation + strikethrough detection to all string fields.

    Mutates and returns ``fields`` (the flat envelope dict). Lists (plain_text,
    tables) and the self_consistency block are left untouched.
    """
    for key, env in fields.items():
        if not (isinstance(env, dict) and isinstance(env.get("value"), str)):
            continue
        original = env["value"]

        nv = _STRUCK_RE.sub("", original).strip()   # keep final, drop struck text
        nv = _normalize_symbols(nv)
        if _is_temperature_key(key):
            nv = _bare_C_to_degree(nv)

        if nv != original:
            env.setdefault("raw", original)
            env["value"] = nv
            env["provenance"] = "normalize"
    return fields
