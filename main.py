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
from pipeline.perception.layout import segment as _layout_segment
from pipeline.perception.ocr_math import extract as _ocr_math_extract
from pipeline.perception.normalize import run as _normalize
from pipeline.perception import ocsr
from reference import chem_resolve

_ROOT = Path(__file__).parent
_SCHEMA_PATH = _ROOT / "schema" / "electrodeposition.schema.json"
_RUNS_DIR = _ROOT / "eval" / "runs"
_CACHE_DIR = _ROOT / "eval" / "cache"   # raw recogniser output, keyed by image

MODEL = "claude-opus-4-8"
CURRENT_PHASE = 4   # bump as each build-order phase lands


def _model_slug(model: str = MODEL) -> str:
    return model.replace("claude-", "")


def run(image_path: str,
        output_path: str | None = "output.json",
        phase: int = CURRENT_PHASE,
        model: str = MODEL,
        refresh: bool = False) -> dict:
    """Run the pipeline on a scanned page and emit flat-key JSON.

    The recogniser (VLM) output is cached per image, so only the FIRST run on a
    page calls the VLM; later runs re-read the cache and re-run just the
    deterministic wrapper (layout/ocr_math/normalize/validator) — no API call,
    no API key needed. Pass ``refresh=True`` to force a fresh VLM call.

    Writes the result to ``output_path`` AND to
    ``eval/runs/phase{phase}_{model_slug}.json`` (kept for cross-phase scoring).
    """
    from pipeline.perception.preprocess import preprocess
    preprocessed = preprocess(image_path)   # deskew + binarize

    regions = _layout_segment(preprocessed)        # Phase 3: layout segmentation
    result = _recognize(image_path, model, refresh)  # whole-page recogniser (cached)

    # --- Phase 3: accuracy routing ---
    _route_calc_block(result, regions)  # route calc lines -> ocr_math parser
    _normalize(result)                  # symbol/unit + strikethrough post-proc

    # --- Phase 4: chemistry ---
    _resolve_structures(result)         # name-resolution + RDKit-gated OCSR

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

def _recognize(image_path: str, model: str = MODEL, refresh: bool = False) -> dict:
    """Return the raw recogniser output for an image, using a per-image cache.

    Cache hit  -> load from eval/cache/ (no VLM call, no API key required).
    Cache miss -> call the VLM, save the raw output, return it.
    ``refresh=True`` forces a fresh VLM call and overwrites the cache.

    The cache holds the recogniser output ONLY — before any wrapper stage — so
    the deterministic wrapper re-runs cleanly on every iteration.
    """
    cache_path = _CACHE_DIR / f"{Path(image_path).stem}.{_model_slug(model)}.recognizer.json"
    if cache_path.exists() and not refresh:
        print(f"[recognizer] cache hit: {cache_path} (no VLM call)", file=sys.stderr)
        return json.loads(cache_path.read_text())

    result = _vlm_extract(image_path, model)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2))
    print(f"[recognizer] VLM call made; cached -> {cache_path}", file=sys.stderr)
    return result


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
# Phase 3 — accuracy routing: send the calc block to the math parser
# ---------------------------------------------------------------------------

def _route_calc_block(result: dict, regions: list) -> None:
    """Route the recogniser's calc-block lines to ocr_math and merge the parsed
    fields. Layout supplies region geometry; the per-line text comes from the
    whole-page recogniser (plain_text). Parsed fields fill GAPS only — they never
    overwrite a value the recogniser already provided.
    """
    calc_lines = [ln for ln in (result.get("plain_text") or []) if "=" in str(ln)]
    calc_regions = [r for r in regions if r.get("type") == "calc_block"]
    if not calc_regions:
        calc_regions = [{"type": "calc_block", "bbox": [0, 0, 0, 0]}]
    for r in calc_regions:
        r["text_lines"] = calc_lines

    for key, env in _ocr_math_extract(calc_regions).items():
        result.setdefault(key, env)


# ---------------------------------------------------------------------------
# Phase 4 — chemistry: name-resolution-first, RDKit-gated OCSR cross-check
# ---------------------------------------------------------------------------

_METALS = ("Li", "Cu", "Ni", "Na", "Zn", "Mg", "K")


def _resolve_structures(result: dict) -> None:
    """For each drawn-molecule field, reconcile the recogniser's SMILES with
    name-resolution (chem_resolve) and gate through RDKit. Canonical SMILES is
    stored back in the field; the original is kept in ``raw``.
    """
    for key in list(result.keys()):
        if not key.startswith("structures.") or key.endswith(".formula"):
            continue
        env = result.get(key)
        if not (isinstance(env, dict) and isinstance(env.get("value"), str)):
            continue

        label = key[len("structures."):]
        drawn = env["value"]
        rec = ocsr.reconcile(label, drawn, chem_resolve.resolve(label))
        if rec["smiles"]:
            if rec["smiles"] != drawn:
                env.setdefault("raw", drawn)
            env["value"] = rec["smiles"]
            env["provenance"] = ("chem_resolve"
                                 if rec["source"] in ("consensus", "label_resolution")
                                 else "ocsr")


def _deposited_species(result: dict) -> str:
    """Identify the deposited metal from the salt / reaction (default Li)."""
    text = " ".join(
        str(result.get(k, {}).get("value", ""))
        for k in ("electrolyte.salt", "reaction.equation")
        if isinstance(result.get(k), dict)
    )
    for metal in _METALS:
        if metal.lower() in text.lower():
            return metal
    return "Li"


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

    # z and molar mass come from reference/chem_resolve (de-hardcoded), keyed
    # off the deposited metal; chem_resolve falls back to Li if unresolved.
    consts = chem_resolve.deposition_constants(_deposited_species(result))
    inputs = {
        "J_mA_cm2": J, "area_cm2": area, "t_min": t_min,
        "z": consts["z"], "M_g_mol": consts["molar_mass"],
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
    ap.add_argument("--refresh", action="store_true",
                    help="force a fresh VLM call (ignore + overwrite the cache)")
    args = ap.parse_args()

    try:
        out = run(args.image_path, args.output_path, args.phase, args.model,
                  args.refresh)
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
