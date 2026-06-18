"""
main.py — Phase 1: preprocess + single VLM call (whole page, no layout splitting).

Full call order (eventual):
    perception:  preprocess → layout → ocr_text + ocr_math + ocsr → normalize
    dispatch:    classify → registry.dispatch (→ domain validator)
    emit:        assemble schema-valid JSON, validate, write output

Phase 2 adds the trust layer on top of Phase 1:
    preprocess → VLM whole-page → fill schema shape
        → re-derive J→I→Q→n→mass and write calculations.self_consistency
        → STRICT schema validation (fail loudly)
        → JSON.

Import the semantics plugins so they self-register before dispatch is called.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path

import jsonschema

# Trigger plugin self-registration
import pipeline.semantics.electrodeposition  # noqa: F401
from pipeline.semantics.electrodeposition import validate_deposition

_SCHEMA_PATH = Path(__file__).parent / "schema" / "electrodeposition.schema.json"


def run(image_path: str, output_path: str | None = None) -> dict:
    """Run the Phase-1 pipeline: preprocess + single-VLM extraction.

    Parameters
    ----------
    image_path : str
        Path to the input scan (JPEG / PNG / TIFF).
    output_path : str, optional
        If given, write the JSON to this path.

    Returns
    -------
    dict — extracted result shaped to the electrodeposition schema.
    """
    from pipeline.perception.preprocess import preprocess
    preprocess(image_path)  # deskew + binarize (artefacts used by later phases)

    result = _vlm_extract(image_path)

    # --- Phase 2: trust layer ---
    _attach_self_consistency(result)   # re-derive the electrochem chain

    # Write the artefact BEFORE validating, so a schema failure still leaves an
    # inspectable file on disk (the exception below is the loud signal).
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))

    _validate_schema(result)           # strict — raises on any nonconformance

    return result


# ---------------------------------------------------------------------------
# Phase 2 — trust layer: re-derive the electrochemistry, cross-check the page
# ---------------------------------------------------------------------------

def _attach_self_consistency(result: dict) -> None:
    """Re-derive J→I→Q→n→mass and record the check under calculations.

    Degrades gracefully: if the primary inputs (current density, electrode area,
    deposition time) aren't all present, no self_consistency block is written
    rather than crashing — perception output still stands.
    """
    runs = result.get("runs") or []
    calc = result.get("calculations")
    if not runs or not isinstance(calc, dict):
        return

    run0 = runs[0]
    J     = _field_float(run0, "current_density")   # mA/cm²
    area  = _field_float(run0, "electrode_area")    # cm²
    t_min = _field_float(run0, "duration")          # minutes
    if None in (J, area, t_min):
        return

    inputs = {
        "J_mA_cm2": J,
        "area_cm2": area,
        "t_min":    t_min,
        # z and molar mass come from reference/chem_resolve in Phase 4;
        # fall back to Li defaults until then.
        "z":        1,
        "M_g_mol":  6.94,
    }

    # The intermediate RESULTS the chemist wrote, as read off the page. Only
    # include a value if it was actually extracted (don't fabricate t_s).
    written = {
        "I_A":    _field_float(calc, "current"),
        "Q_C":    _field_float(calc, "charge"),
        "n_mol":  _field_float(calc, "moles_deposited"),
        "mass_g": _field_float(calc, "mass_deposited"),
    }
    extracted = {k: v for k, v in written.items() if v is not None}

    report = validate_deposition(inputs, extracted)
    calc["self_consistency"] = _report_to_dict(report)


def _report_to_dict(report) -> dict:
    """Serialise a ValidationReport into the schema's selfConsistency shape."""
    return {
        "all_passed": bool(report.all_passed),
        "checks": [
            {
                "name":      c.name,
                "derived":   round(float(c.derived), 12),
                "extracted": c.extracted,
                "unit":      c.unit,
                "rel_error": (round(float(c.rel_error), 6)
                              if c.rel_error is not None else None),
                "ok":        c.ok,
            }
            for c in report.checks
        ],
    }


def _field_float(obj: dict, key: str):
    """Pull obj[key]['value'] (the field envelope) and coerce it to float."""
    field = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(field, dict):
        return None
    return _to_float(field.get("value"))


def _to_float(value):
    """Extract the leading numeric literal from a number or unit-bearing string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if m:
            return float(m.group(0))
    return None


def _validate_schema(result: dict) -> None:
    """Validate ``result`` against the electrodeposition schema. Fail loudly."""
    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.validate(instance=result, schema=schema)


def _load_api_key() -> str:
    """Return ANTHROPIC_API_KEY from env or a .env file in the project root."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set.\n"
        "  Option 1: export ANTHROPIC_API_KEY=sk-ant-...\n"
        "  Option 2: create a .env file with ANTHROPIC_API_KEY=sk-ant-..."
    )


def _vlm_extract(image_path: str) -> dict:
    """Send the whole page to the VLM and parse out the schema-shaped JSON."""
    import anthropic

    api_key = _load_api_key()

    suffix = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }.get(suffix, "image/jpeg")

    with open(image_path, "rb") as fh:
        img_b64 = base64.standard_b64encode(fh.read()).decode()

    client = anthropic.Anthropic(api_key=api_key)  # noqa: S106
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": _EXTRACTION_PROMPT},
            ],
        }],
    )

    raw = next(
        (block.text for block in response.content if hasattr(block, "text")),
        "",
    ).strip()
    return _parse_json_response(raw)


def _parse_json_response(text: str) -> dict:
    """Try to extract a JSON object from the model's text response."""
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. ```json … ``` fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First { … } block in the response
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Fallback — return raw text wrapped in an error envelope
    return {
        "schema_version": "electrodeposition/1.0",
        "experiment_type": "electrodeposition",
        "_parse_error": "VLM did not return valid JSON",
        "_raw": text[:2000],
    }


_EXTRACTION_PROMPT = """\
You are extracting structured data from a handwritten chemistry lab-notebook page.

Read EVERY piece of text and every number on the page carefully.

Return a JSON object shaped exactly like this (omit keys you cannot find; \
do NOT invent values):

{
  "schema_version": "electrodeposition/1.0",
  "experiment_type": "electrodeposition",
  "page":   {"value": "<page number>", "confidence": 0.9, "provenance": "ocr_text", "bbox": [0,0,0,0]},
  "date":   {"value": "<date as written>", "confidence": 0.9, "provenance": "ocr_text", "bbox": [0,0,0,0]},
  "project":{"value": "<project name>", "confidence": 0.9, "provenance": "ocr_text", "bbox": [0,0,0,0]},
  "goal":   {"value": "<goal sentence>", "confidence": 0.85, "provenance": "ocr_text", "bbox": [0,0,0,0]},
  "electrolyte": {
    "salt":          {"value": "<salt name>",         "confidence": 0.9, "provenance": "ocr_text", "bbox": [0,0,0,0]},
    "concentration": {"value": "<conc with unit>",    "confidence": 0.9, "provenance": "ocr_text", "bbox": [0,0,0,0]},
    "solvent":       {"value": "<solvent name(s)>",   "confidence": 0.85,"provenance": "ocr_text", "bbox": [0,0,0,0]},
    "additive": {
      "name":  {"value": "<additive name>",  "confidence": 0.85, "provenance": "ocr_text", "bbox": [0,0,0,0]},
      "amount":{"value": "<amount + unit>",  "confidence": 0.85, "provenance": "ocr_text", "bbox": [0,0,0,0]}
    }
  },
  "cell": {
    "working_electrode":   {"value": "<material/description>", "confidence": 0.85, "provenance": "ocr_text", "bbox": [0,0,0,0]},
    "counter_electrode":   {"value": "<material/description>", "confidence": 0.85, "provenance": "ocr_text", "bbox": [0,0,0,0]},
    "reference_electrode": {"value": "<material/description>", "confidence": 0.85, "provenance": "ocr_text", "bbox": [0,0,0,0]}
  },
  "runs": [
    {
      "id":             {"value": "<run id>",              "confidence": 0.9,  "provenance": "ocr_text", "bbox": [0,0,0,0]},
      "potential":      {"value": "<voltage with unit>",   "confidence": 0.9,  "provenance": "ocr_text", "bbox": [0,0,0,0]},
      "reference":      {"value": "<reference electrode>", "confidence": 0.85, "provenance": "ocr_text", "bbox": [0,0,0,0]},
      "duration":       {"value": "<time with unit>",      "confidence": 0.9,  "provenance": "ocr_text", "bbox": [0,0,0,0]},
      "rotation_rate":  {"value": "<rpm>",                 "confidence": 0.85, "provenance": "ocr_text", "bbox": [0,0,0,0]},
      "current_density":{"value": "<mA/cm² value>",       "confidence": 0.9,  "provenance": "ocr_text", "bbox": [0,0,0,0]},
      "electrode_area": {"value": "<cm² value>",           "confidence": 0.85, "provenance": "ocr_text", "bbox": [0,0,0,0]}
    }
  ],
  "calculations": {
    "current":         {"value": <number in A>,   "unit": "A",   "confidence": 0.9,  "provenance": "ocr_math", "bbox": [0,0,0,0]},
    "charge":          {"value": <number in C>,   "unit": "C",   "confidence": 0.9,  "provenance": "ocr_math", "bbox": [0,0,0,0]},
    "moles_deposited": {"value": <number in mol>, "unit": "mol", "confidence": 0.9,  "provenance": "ocr_math", "bbox": [0,0,0,0]},
    "mass_deposited":  {"value": <number in g>,   "unit": "g",   "confidence": 0.9,  "provenance": "ocr_math", "bbox": [0,0,0,0]}
  },
  "structures": [
    {"label": {"value": "<molecule name>", "confidence": 0.85, "provenance": "ocr_text", "bbox": [0,0,0,0]},
     "smiles": null, "formula": null, "source": "unresolved", "rdkit_valid": false, "confidence": 0.5}
  ],
  "results": {
    "film_observation": {"value": "<observation>", "confidence": 0.8, "provenance": "ocr_text", "bbox": [0,0,0,0]}
  }
}

Rules:
- Use the exact JSON field names and envelope shape shown above.
- For every {"value":…, "confidence":…, "provenance":…, "bbox":…} field, supply \
all four keys.
- bbox is [0,0,0,0] for Phase 1 (no layout segmentation yet).
- Do NOT add fields not present in the template; the schema rejects them.
- Return ONLY the JSON object — no preamble, no explanation.
"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [output.json]")
        sys.exit(1)
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        out = run(sys.argv[1], out_path)
    except jsonschema.ValidationError as exc:
        loc = " → ".join(str(p) for p in exc.absolute_path) or "<root>"
        print("SCHEMA VALIDATION FAILED (output is NOT schema-valid)",
              file=sys.stderr)
        print(f"  at: {loc}", file=sys.stderr)
        print(f"  {exc.message}", file=sys.stderr)
        if out_path:
            print(f"  (the offending output was still written to {out_path} "
                  "for inspection)", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(out, indent=2, default=str))
