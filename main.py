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
import logging
import os
import re
import sys
from pathlib import Path

import jsonschema

# Trigger plugin self-registration
import pipeline.semantics.electrodeposition  # noqa: F401
from pipeline.perception.layout import segment as _layout_segment
from pipeline.perception.ocr_math import extract as _ocr_math_extract
from pipeline.perception.normalize import run as _normalize
from pipeline.perception import ocsr
from pipeline.classify import classify
from pipeline.registry import dispatch, DegradedResult
from pipeline import correct
from reference import chem_resolve

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
_SCHEMA_PATH = _ROOT / "schema" / "electrodeposition.schema.json"
_RUNS_DIR = _ROOT / "eval" / "runs"
_CACHE_DIR = _ROOT / "eval" / "cache"   # raw recogniser output, keyed by image

MODEL = "claude-opus-4-8"
RESCAN_MODEL = "claude-opus-4-7"   # cheaper second-pass model
CURRENT_PHASE = 7   # bump as each build-order phase lands


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
    _resolve_structures(result)                          # name-resolution + RDKit-gated OCSR
    # _fill_solvent_structures(result)                   # DISABLED: text-inference fallback pre-empted
    #                                                      the rescan; off so the rescan is the sole
    #                                                      recovery path (clean test of its value).
    _rescan_missing_structures(image_path, result)       # second-pass: targeted VLM rescan

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


def _fill_solvent_structures(result: dict) -> None:
    """Text-inference fallback: add structure keys for solvent components the VLM grouped.

    ASSUMPTION: every chemical name in electrolyte.solvent has a corresponding
    drawing on the page. This is often true in lab notebooks but NOT guaranteed —
    a page may list solvents as plain text without structural drawings.
    Provenance is 'text_inferred' to flag that no drawing was confirmed by the VLM.
    The _rescan_missing_structures() second pass is the principled replacement;
    this fallback fires first so rescan only needs to check what's truly unresolved.
    """
    solvent_env = result.get("electrolyte.solvent")
    if not isinstance(solvent_env, dict):
        return
    solvent_str = str(solvent_env.get("value", ""))
    if not solvent_str:
        return

    import re as _re
    for label in _re.split(r"[:/+,]", solvent_str):
        label = label.strip()
        if not label:
            continue
        key = f"structures.{label}"
        if key in result:
            continue
        resolved = chem_resolve.resolve(label)
        if resolved and resolved.get("smiles"):
            result[key] = {
                "value":      resolved["smiles"],
                "confidence": 0.6,
                "provenance": "text_inferred",   # no drawing confirmed; name only
                "bbox":       [0, 0, 0, 0],
            }


_RESCAN_PROMPT = """\
A first-pass model has already extracted these molecular structure drawings from this page:
  {detected}

Your ONLY task: look carefully at the page image and identify any molecular structure
drawings that have a label NOT in the list above. Do NOT re-extract ones already found.

Return a JSON array. Each element must have "label" (text written next to the drawing)
and "smiles" (SMILES as drawn). Return [] if nothing is missing.

Example: [{{"label": "EtOH", "smiles": "CCO"}}]
Return ONLY the JSON array, no explanation.
"""


def _rescan_missing_structures(image_path: str, result: dict,
                                model: str = RESCAN_MODEL) -> None:
    """Second-pass VLM scan: find drawings the first pass missed.

    Sends the image with the already-detected labels so the model only needs to
    report genuinely new drawings. Results are reconciled via chem_resolve +
    ocsr.reconcile and stored with provenance='rescan'. No caching — this pass
    is cheap and depends on what the first pass found.
    """
    detected = [
        key[len("structures."):]
        for key in result
        if key.startswith("structures.") and not key.endswith(".formula")
    ]
    if not detected:
        return

    import anthropic
    api_key = _load_api_key()
    suffix = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".tiff": "image/tiff", ".tif": "image/tiff",
    }.get(suffix, "image/jpeg")
    with open(image_path, "rb") as fh:
        img_b64 = base64.standard_b64encode(fh.read()).decode()

    prompt = _RESCAN_PROMPT.format(detected=", ".join(detected))
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    raw = next((b.text for b in response.content if hasattr(b, "text")), "").strip()

    try:
        arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
        missed = json.loads(raw if raw.startswith("[") else (arr_match.group(0) if arr_match else "[]"))
    except Exception:
        log.warning("[rescan] could not parse response: %s", raw[:200])
        return

    # Duplicate suppression against already-detected structures, by BOTH:
    #  - canonical SMILES (catches identical structures), and
    #  - normalised label (catches coordination/charge variants the VLM may draw
    #    differently across runs, e.g. "[Li(12-crown-4)]+" vs "12-crown-4", whose
    #    SMILES differ only by a .[Li+] fragment and so dodge a SMILES-only check).
    structure_items = [
        (k, v) for k, v in result.items()
        if k.startswith("structures.") and not k.endswith(".formula")
        and isinstance(v, dict)
    ]
    existing_smiles = {ocsr.canonical(v.get("value", "")) for _, v in structure_items} - {None}
    existing_labels = {chem_resolve._key(k[len("structures."):]) for k, _ in structure_items}

    for item in (missed or []):
        label = str(item.get("label", "")).strip()
        drawn = str(item.get("smiles", "")).strip()
        if not label:
            continue
        key = f"structures.{label}"
        if key in result:
            continue
        # Skip coordination/charge variants of a structure we already have
        if chem_resolve._key(label) in existing_labels:
            log.info("[rescan] skipping %s — same core label as existing entry", label)
            continue
        rec = ocsr.reconcile(label, drawn, chem_resolve.resolve(label))
        if not rec["smiles"]:
            continue
        # Skip if this is a duplicate of a structure we already have
        if ocsr.canonical(rec["smiles"]) in existing_smiles:
            log.info("[rescan] skipping %s — same canonical SMILES as existing entry", label)
            continue
        result[key] = {
            "value":      rec["smiles"],
            "confidence": 0.65,
            "provenance": "rescan",
            "bbox":       [0, 0, 0, 0],
        }
        log.info("[rescan] added %s = %s (source: %s)", key, rec["smiles"], rec["source"])


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

    # Build the inter-stage `fields` dict (registry envelope: {value: ...}).
    fields = {
        "J_mA_cm2": {"value": J}, "area_cm2": {"value": area},
        "t_min": {"value": t_min},
        "z": {"value": consts["z"]}, "M_g_mol": {"value": consts["molar_mass"]},
    }
    for key, src in (("I_A", "calculations.current.value"),
                     ("Q_C", "calculations.charge.value"),
                     ("n_mol", "calculations.moles_deposited.value"),
                     ("mass_g", "calculations.mass_deposited.value")):
        v = _flat_float(result, src)
        if v is not None:
            fields[key] = {"value": v}

    # Phase 5: classify the page -> registry.dispatch -> domain validator.
    # (Pluggable: adding an experiment family is one @plugin, no change here.)
    cls = classify(_classify_text(result))
    log.info("classify: %s (confidence %.2f) scores=%s",
             cls.experiment_type, cls.confidence, cls.scores)

    report = dispatch(cls.experiment_type, fields)
    if isinstance(report, DegradedResult):
        log.info("dispatch degraded: %s", report)   # graceful: skip domain check
        return

    # Phase 5: bounded correction loop (N=2) on any validator-flagged field.
    report = correct.apply(
        report, fields, _alt_sources(result),
        revalidate=lambda f: dispatch(cls.experiment_type, f))
    result["calculations.self_consistency"] = _report_to_dict(report)


def _classify_text(result: dict) -> str:
    """Concatenate all extracted prose for the keyword classifier."""
    parts = [env["value"] for env in result.values()
             if isinstance(env, dict) and isinstance(env.get("value"), str)]
    parts += [str(x) for x in (result.get("plain_text") or [])]
    return " ".join(parts)


def _alt_sources(result: dict) -> dict:
    """Map each validator field to its alternate source (the equation line), for
    the correction loop to re-parse if the recogniser's .value is flagged."""
    pairs = {
        "I_A":    "calculations.current.expression",
        "Q_C":    "calculations.charge.expression",
        "n_mol":  "calculations.n_chain.expression",
        "mass_g": "calculations.n_chain.expression",
    }
    out = {}
    for key, src in pairs.items():
        env = result.get(src)
        if isinstance(env, dict) and env.get("value"):
            out[key] = {"text_lines": [env["value"]]}
    return out


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
  "structures.EtOH",
  "structures.12-crown-4.formula", "structures.LiTFSI.formula"
NOTE: the page shows a "diglyme : EtOH" label with arrows pointing to TWO
SEPARATE molecular drawings. Extract diglyme and EtOH as two distinct structure
keys — do NOT merge them into one entry.

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

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s",
                        stream=sys.stderr)

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
