"""
main.py — Phase 1: preprocess + single VLM call (whole page, no layout splitting).

Full call order (eventual):
    perception:  preprocess → layout → ocr_text + ocr_math + ocsr → normalize
    dispatch:    classify → registry.dispatch (→ domain validator)
    emit:        assemble schema-valid JSON, validate, write output

Phase 1 short-circuits to: preprocess → VLM whole-page → fill schema shape → JSON.
Strict schema validation is wired in Phase 2.

Import the semantics plugins so they self-register before dispatch is called.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path

# Trigger plugin self-registration
import pipeline.semantics.electrodeposition  # noqa: F401


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

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))

    return result


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
    out = run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(out, indent=2, default=str))
