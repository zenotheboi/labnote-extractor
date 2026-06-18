"""
eval/report.py
==============
Render the submission results table from eval/scores.csv.

Reads the per-phase metrics, curates them into an ACCURACY LADDER (only the steps
that actually moved a scored metric, in order), and writes eval/results.md plus a
plaintext copy to stdout. Computes the headline lift (final - baseline).

Why a curated ladder, not one row per phase: several pipeline phases (the trust
layer, chemistry-robustness, scalability) add real capability that the four scored
metrics do not capture — listing them as flat rows implies they did nothing. They
are surfaced instead by the `validator` column and the footnotes, while the metric
ladder shows only where the numbers moved.

Usage:  python eval/report.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

_HERE = Path(__file__).parent
_CSV = _HERE / "scores.csv"
_RUNS = _HERE / "runs"
_OUT = _HERE / "results.md"

# The accuracy ladder: (csv-phase-id, human label, run-file stem or None).
# Only steps that moved a scored metric appear here, in order. The cumulative
# pipeline means each row includes everything above it.
_LADDER = [
    ("baseline_naive", "Baseline — single VLM call, naive prompt", "baseline_opus-4-8"),
    ("phase1",         "+ Engineered extraction prompt",           None),
    ("3",              "+ Symbol/unit normalization + routing",    "phase3_opus-4-8"),
    ("7",              "+ Structure recovery (second-pass rescan)  [FINAL]", "phase7_opus-4-8"),
]

# Phases that add capability the four metrics don't score (shown as footnotes).
_NON_METRIC_NOTES = [
    "Phase 2 — trust layer: Faraday self-consistency re-derivation + strict schema "
    "validation. Shown in the `self-consistency` column (checks reconciled / "
    "applicable, plus flagged count); not a text/scalar/SMILES/table metric. On "
    "this page 4 of 4 applicable checks reconcile (I, Q, n, mass); `t` has no "
    "independent page value to compare, so it is not applicable, not a failure.",
    "Phase 4 — chemistry robustness: name-resolution (PubChem/OPSIN) + RDKit canonical "
    "gate. Hardens SMILES correctness; SMILES count unchanged on this page.",
    "Phase 5 — scalability: classify -> registry dispatch -> bounded correction loop. "
    "Generalizes the pipeline to other experiment types; no single-page metric change.",
]

_METRIC_COLS = ["text_sim", "scalar", "smiles", "table", "missing_count"]
_METRIC_HDR = {"text_sim": "text", "scalar": "scalar", "smiles": "SMILES",
               "table": "table", "missing_count": "missing"}


def _load_scores() -> dict:
    rows = {}
    with _CSV.open() as fh:
        for row in csv.DictReader(fh):
            rows[row["phase"]] = row
    return rows


def _validator_status(run_stem: str | None) -> str:
    """Granular self-consistency status: 'P/A ✓ · F flagged', where P = checks
    reconciled, A = applicable checks (those with a comparable page value), and
    F = checks flagged as inconsistent. '—' when no validator ran on this step.
    """
    if not run_stem:
        return "—"
    path = _RUNS / f"{run_stem}.json"
    if not path.exists():
        return "—"
    try:
        sc = json.loads(path.read_text()).get("calculations.self_consistency")
        if not isinstance(sc, dict):
            return "—"
        checks = sc.get("checks", [])
        applicable = [c for c in checks if c.get("ok") is not None]
        passed = [c for c in applicable if c.get("ok") is True]
        flagged = [c for c in checks if c.get("ok") is False]
        if not checks:
            return "—"
        mark = "✓" if not flagged else "✗"
        return f"{len(passed)}/{len(applicable)} {mark} · {len(flagged)} flagged"
    except Exception:
        return "—"


def _fmt(col: str, val: str) -> str:
    if val == "" or val is None:
        return "n/a"
    return val if col == "missing_count" else f"{val}%"


def _lift_row(scores: dict) -> dict:
    base = scores["baseline_naive"]
    final = scores["7"]
    out = {}
    for col in _METRIC_COLS:
        try:
            d = int(final[col]) - int(base[col])
        except (ValueError, KeyError):
            out[col] = "n/a"; continue
        sign = "+" if d > 0 else ""
        # for missing_count, fewer is better -> show the raw delta
        out[col] = f"{sign}{d}" + ("" if col == "missing_count" else "pp")
    return out


def build() -> str:
    scores = _load_scores()
    lines = []
    lines.append("# Results\n")
    lines.append("Single page (`data/page57.jpg`), all runs on **claude-opus-4-8**, "
                 "scored against the frozen human-verified ground truth "
                 "(`eval/ground_truth/page57.json`). Baseline and pipeline use the "
                 "**same model**, so the lift isolates the wrapper, not a model swap.\n")

    # header
    cols = ["Step"] + [_METRIC_HDR[c] for c in _METRIC_COLS] + ["self-consistency"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for pid, label, stem in _LADDER:
        row = scores.get(pid)
        if not row:
            continue
        cells = [label] + [_fmt(c, row.get(c, "")) for c in _METRIC_COLS]
        cells.append(_validator_status(stem))
        lines.append("| " + " | ".join(cells) + " |")

    # lift row
    lift = _lift_row(scores)
    lift_cells = ["**Lift (final − baseline)**"] + [f"**{lift[c]}**" for c in _METRIC_COLS] + [""]
    lines.append("| " + " | ".join(lift_cells) + " |")

    lines.append("\n**Headline:** the wrapper lifts a same-model single call by "
                 f"**{lift['scalar']} scalar**, **{lift['smiles']} SMILES**, and "
                 f"**{lift['table']} table** accuracy, and drives missing fields "
                 f"from {scores['baseline_naive']['missing_count']} to "
                 f"{scores['7']['missing_count']}.\n")

    lines.append("### Capability not captured by these four metrics\n")
    for note in _NON_METRIC_NOTES:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    md = build()
    _OUT.write_text(md)
    print(md)
    print(f"\n[report] wrote {_OUT}")


if __name__ == "__main__":
    main()
