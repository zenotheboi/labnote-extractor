"""
main.py — pipeline orchestrator.

Output contract (since the eval harness landed): a FLAT dotted-key envelope dict.
Keys mirror the ground-truth namespace in eval/ground_truth/page57.json
(e.g. "electrolyte.salt", "runs[0].id", "calculations.current.value",
"structures.12-crown-4"), so eval/score.py can flatten + compare directly. Each
scalar is a {value, confidence, provenance, bbox} envelope; plain-text and tables
are lists; the validator's report sits at "calculations.self_consistency".

Phase capability so far:
    Phase 1  preprocess (rough) + single whole-page VLM call
    Phase 2  + re-derive J->I->Q->n->mass into calculations.self_consistency
             + strict schema validation (flat structural schema)

Every run writes its output to BOTH the --output file (default output.json) AND
eval/runs/phase{N}_{model}.json — never overwriting a previous phase's file, so
phases can be compared across the run history.
"""

from __future__ import annotations

import argparse
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

_ROOT = Path(__file__).parent
_SCHEMA_PATH = _ROOT / "schema" / "electrodeposition.schema.json"
_RUNS_DIR = _ROOT / "eval" / "runs"

MODEL = "claude-opus-4-8"
CURRENT_PHASE = 2   # bump as each build-order phase lands


def _model_slug(model: str = MODEL) -> str:
    return model.replace("claude-", "")


def run(image_path: str,
        output_path: str | None = "output.json",
        phase: int = CURRENT_PHASE,
        model: str = MODEL) -> dict:
    """Run the pipeline on a scanned page and emit flat-key JSON.

    Writes the result to ``output_path`` AND to
    ``eval/runs/phase{phase}_{model_slug}.json`` (kept for cross-phase scoring).
    """
    from pipeline.perception.preprocess import preprocess
    preprocess(image_path)  # deskew + binarize (artefacts used by later phases)

    result = _vlm_extract(image_path, model)

    # --- Phase 2: trust layer ---
    _attach_self_consistency(result)   # re-derive the electrochem chain

    # Write artefacts BEFORE validating, so a schema failure still leaves
    # inspectable files on disk (the exception below is the loud signal).
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = _RUNS_DIR / f"phase{phase}_{_model_slug(model)}.json"
    blob = json.dumps(result, indent=2)
    if output_path:
        Path(output_path).write_text(blob)
    run_path.write_text(blob)

    _validate_schema(result)           # strict — raises on any nonconformance

    return result


# ---------------------------------------------------------------------------
# VLM recognition — single whole-page call, returns the flat-key shape
# ---------------------------------------------------------------------------

def _vlm_extract(image_path: str, model: str = MODEL) -> dict:
    """Send the whole page to the VLM and parse out the flat-key JSON."""
    import anthropic

    api_key = _load_api_key()

    suffix = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".tiff": "image/tiff", ".tif": "image/tiff",
    }.get(suffix, "image/jpeg")

    with open(image_path, "rb") as fh:
        img_b64 = base64.standard_b64encode(fh.read()).decode()

    client = anthropic.Anthropic(api_key=api_key)  # noqa: S106
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": _EXTRACTION_PROMPT},
            ],
        }],
    )
    raw = next((b.text for b in response.content if hasattr(b, "text")), "").strip()
    return _parse_json_response(raw)


def _load_api_key() -> str:
    """Return ANTHROPIC_API_KEY from env or a .env file in the project root."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env_path = _ROOT / ".env"
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


def _parse_json_response(text: str) -> dict:
    """Extract a JSON object from the model's text response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"_parse_error": {"value": "VLM did not return valid JSON",
                             "confidence": 0.0, "provenance": "ocr_text",
                             "bbox": [0, 0, 0, 0]},
            "_raw": [text[:2000]]}


# ---------------------------------------------------------------------------
# Phase 2 — trust layer: re-derive the electrochemistry, cross-check the page
# ---------------------------------------------------------------------------

def _attach_self_consistency(result: dict) -> None:
    """Re-derive J->I->Q->n->mass and record it at calculations.self_consistency.

    Degrades gracefully: if current density, electrode area, or deposition time
    aren't all present, no self_consistency block is written rather than crashing.
    """
    J     = _flat_float(result, "runs[0].current_density")    # mA/cm²
    area  = _flat_float(result, "cell.working_electrode_area")  # cm²
    t_min = _flat_float(result, "runs[0].duration")           # minutes
    if None in (J, area, t_min):
        return

    inputs = {
        "J_mA_cm2": J, "area_cm2": area, "t_min": t_min,
        # z and molar mass come from reference/chem_resolve in Phase 4;
        # fall back to Li defaults until then.
        "z": 1, "M_g_mol": 6.94,
    }
    written = {
        "I_A":    _flat_float(result, "calculations.current.value"),
        "Q_C":    _flat_float(result, "calculations.charge.value"),
        "n_mol":  _flat_float(result, "calculations.moles_deposited.value"),
        "mass_g": _flat_float(result, "calculations.mass_deposited.value"),
    }
    extracted = {k: v for k, v in written.items() if v is not None}

    report = validate_deposition(inputs, extracted)
    result["calculations.self_consistency"] = _report_to_dict(report)


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


def _flat_float(result: dict, key: str):
    """Pull result[key]['value'] (a flat envelope) and coerce it to float."""
    field = result.get(key)
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
    """Validate ``result`` against the flat electrodeposition schema. Fail loudly."""
    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.validate(instance=result, schema=schema)


# ---------------------------------------------------------------------------
# Extraction prompt — asks the VLM for the flat-key envelope shape directly
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = r"""
You are extracting structured data from a handwritten chemistry lab-notebook page
(an Li-electrodeposition experiment). Read EVERY line and number carefully.

Return ONE JSON object using the EXACT flat dotted keys below. Omit any key you
cannot read; do NOT invent values. Do NOT add keys outside this list.

SCALAR fields — each value is an envelope:
  {"value": <number|string>, "confidence": <0..1>, "provenance": "ocr_text", "bbox": [0,0,0,0]}
Use these scalar keys:
  "page", "continued_from", "date", "project", "goal",
  "electrolyte.salt", "electrolyte.concentration", "electrolyte.solvent",
  "electrolyte.solvent_ratio", "electrolyte.volume",
  "electrolyte.additive.name", "electrolyte.additive.amount",
  "procedure.stir_time",
  "conditions.temperature", "conditions.atmosphere", "conditions.water_content",
  "cell.working_electrode", "cell.working_electrode_area",
  "cell.counter_electrode", "cell.reference_electrode",
  "runs[0].id", "runs[0].potential", "runs[0].duration",
  "runs[0].rotation_rate", "runs[0].current_density",
  "reaction.equation", "reaction.conditions",
  "results.film_observation", "results.temperature_test.label"

CALCULATION fields (use provenance "ocr_math"). For every ".value" key, give the
quantity AS WRITTEN on the page INCLUDING its unit, e.g. "1.5E-4 A", "0.81 C",
"8.4E-6 mol", "5.8E-5 g", "5400 s". For ".expression" / ".annotation" keys, copy
the handwritten line verbatim. Keys:
  "calculations.current.value", "calculations.current.expression",
  "calculations.duration_s", "calculations.charge.value",
  "calculations.charge.expression", "calculations.moles_deposited.value",
  "calculations.mass_deposited.value", "calculations.n_chain.expression",
  "calculations.n_chain.annotation"

MOLECULES drawn on the page — key by the drawn label, value is the SMILES you
read from the DRAWING (record it as drawn, even if it looks unusual). Envelope
with provenance "ocsr". Also give the molecular formula where you can:
  "structures.12-crown-4", "structures.LiTFSI", "structures.diglyme",
  "structures.12-crown-4.formula", "structures.LiTFSI.formula"

LISTS — plain arrays, NOT envelopes:
  "plain_text": ["<every line of the page, verbatim, top to bottom; mark struck
                  text as [struck: ...]>", ...]
  "procedure.steps": ["<each procedure step as a line>", ...]
  "results.xrd": [{"two_theta": "2θ = 2.1°", "note": "main peak, low intensity"}, ...]
  "results.temperature_test.readings": [{"time_min": 0, "temp_C": 22.4}, ...]

Rules:
- Every scalar/molecule field MUST carry all four envelope keys.
- bbox is [0,0,0,0] for now (no layout segmentation yet).
- Return ONLY the JSON object — no preamble, no explanation, no markdown fence.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the labnote-extractor pipeline.")
    ap.add_argument("image_path")
    ap.add_argument("output_path", nargs="?", default="output.json")
    ap.add_argument("--phase", type=int, default=CURRENT_PHASE,
                    help="build-order phase number (labels the eval/runs/ file)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    try:
        out = run(args.image_path, args.output_path, args.phase, args.model)
    except jsonschema.ValidationError as exc:
        loc = " → ".join(str(p) for p in exc.absolute_path) or "<root>"
        print("SCHEMA VALIDATION FAILED (output is NOT schema-valid)", file=sys.stderr)
        print(f"  at: {loc}", file=sys.stderr)
        print(f"  {exc.message}", file=sys.stderr)
        print("  (the offending output was still written for inspection)", file=sys.stderr)
        sys.exit(2)

    run_file = _RUNS_DIR / f"phase{args.phase}_{_model_slug(args.model)}.json"
    print(json.dumps(out, indent=2, default=str))
    print(f"\n-> wrote {args.output_path} and {run_file}", file=sys.stderr)
