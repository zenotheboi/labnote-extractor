"""
eval/baseline_vlm.py
======================
Single-VLM baseline — one prompt, one model call, no decomposition.

Purpose: establish the score a naive approach achieves so we can quantify
the lift our pipeline provides. This is the headline number for the demo video.

Model: claude-opus-4-8 — the SAME model the pipeline uses, on purpose. Holding the
model constant means the baseline-vs-pipeline gap isolates the PIPELINE's lift,
not a model difference. Single prompt asks for all fields at once — no layout
routing, no chem_resolve, no validator, no second pass: a naive generic prompt.
Whatever the VLM returns cold is the score.

Compare against TWO reference points (both opus-4-8):
  - this naive baseline (generic prompt, single call)
  - "phase 1" = the pipeline's recognizer cache (ENGINEERED prompt, single call),
    score eval/cache/page57.opus-4-8.recognizer.json directly.
The two gaps separate prompt-engineering lift from pipeline-architecture lift.

Usage:
    python eval/baseline_vlm.py data/page57.jpg
    python eval/score.py eval/runs/baseline_sonnet-4-6.json eval/ground_truth/page57.json

WARNING: do not use this baseline's output as ground truth (circular — see
CLAUDE.md §ground-truth-warning).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_RUNS_DIR = _ROOT / "eval" / "runs"

_BASELINE_PROMPT = """
You are reading a scanned handwritten chemistry lab-notebook page.
Extract every piece of information you can see into a single flat JSON object.

Use dotted keys that reflect the structure of the data, for example:
  "electrolyte.salt", "electrolyte.solvent", "electrolyte.concentration"
  "cell.working_electrode", "cell.counter_electrode"
  "runs[0].current_density", "runs[0].duration"
  "calculations.charge.value", "calculations.moles_deposited.value"
  "structures.<label>"  for any molecular structure drawings (value = SMILES)
  "results.observations", "results.xrd"
  "plain_text"  for verbatim text lines as an array

Each scalar field value should be:
  {"value": <number|string>, "confidence": <0..1>, "provenance": "ocr_text", "bbox": [0,0,0,0]}

Return ONLY the JSON object. No explanation, no markdown fences.
"""


def run_baseline(image_path: str, model: str = "claude-opus-4-8") -> dict:
    """Send a single page image to a VLM and return the raw extraction dict.

    Parameters
    ----------
    image_path : str
        Path to the page image.
    model : str
        Model identifier. Defaults to claude-opus-4-8.

    Returns
    -------
    dict — raw extraction (NOT schema-validated; may have missing/extra fields).
    """
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        env_path = _ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    suffix = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".tiff": "image/tiff", ".tif": "image/tiff",
    }.get(suffix, "image/jpeg")

    with open(image_path, "rb") as fh:
        img_b64 = base64.standard_b64encode(fh.read()).decode()

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": _BASELINE_PROMPT},
            ],
        }],
    )
    raw = next((b.text for b in response.content if hasattr(b, "text")), "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"_parse_error": raw[:500]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-VLM baseline extraction")
    parser.add_argument("image", help="Path to page image")
    parser.add_argument("--model", default="claude-opus-4-8")
    args = parser.parse_args()

    print(f"[baseline] calling {args.model} (single pass, no pipeline)...", file=sys.stderr)
    result = run_baseline(args.image, model=args.model)

    slug = args.model.replace("claude-", "")
    out_path = _RUNS_DIR / f"baseline_{slug}.json"
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[baseline] wrote {out_path}", file=sys.stderr)
    print(json.dumps(result, indent=2))
