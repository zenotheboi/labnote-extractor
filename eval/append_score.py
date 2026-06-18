"""
append_score.py — run score.py (unmodified) on a phase's run file, append the
metrics to eval/scores.csv, and print the score line + the diff vs the previous
row in the CSV.

Usage:
    python eval/append_score.py <phase> <model> <run_output.json> <ground_truth.json>

Example:
    python eval/append_score.py 2 opus-4-8 \
        eval/runs/phase2_opus-4-8.json eval/ground_truth/page57.json

CSV columns: phase,model,text_sim,scalar,smiles,table,missing_count
score.py's logic is NOT touched; we parse its stdout.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_SCORE = _HERE / "score.py"
_CSV = _HERE / "scores.csv"
_HEADER = ["phase", "model", "text_sim", "scalar", "smiles", "table", "missing_count"]


def _run_score(run_path: str, truth_path: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(_SCORE), run_path, truth_path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"score.py failed (exit {proc.returncode})")
    return proc.stdout


def _parse(stdout: str) -> dict:
    """Pull the four metrics + missing count out of score.py's printed lines."""
    def pct(label: str):
        # leading-% form (text) OR 'a/b (P%)' form OR 'n/a'
        m = re.search(rf"{label}\s*:\s*(\d+)%", stdout)
        if m:
            return int(m.group(1))
        m = re.search(rf"{label}\s*:\s*\d+/\d+ \((\d+)%\)", stdout)
        if m:
            return int(m.group(1))
        return ""   # n/a

    miss = re.search(r"missing keys\s*:\s*(\d+)", stdout)
    return {
        "text_sim": pct("text similarity"),
        "scalar":   pct("scalar accuracy"),
        "smiles":   pct("SMILES match"),
        "table":    pct("table cells"),
        "missing_count": int(miss.group(1)) if miss else "",
    }


def _previous_row() -> dict | None:
    if not _CSV.exists():
        return None
    rows = list(csv.DictReader(_CSV.open()))
    return rows[-1] if rows else None


def _append(phase: str, model: str, metrics: dict) -> None:
    new = _CSV.exists()
    with _CSV.open("a", newline="") as fh:
        w = csv.writer(fh)
        if not new:
            w.writerow(_HEADER)
        w.writerow([phase, model, metrics["text_sim"], metrics["scalar"],
                    metrics["smiles"], metrics["table"], metrics["missing_count"]])


def _fmt(v):
    return "n/a" if v == "" else f"{v}"


def _diff_line(prev: dict | None, metrics: dict) -> str:
    if not prev:
        return "(no previous row to diff against — this is the first entry)"
    parts = []
    for key in ("text_sim", "scalar", "smiles", "table", "missing_count"):
        try:
            now = int(metrics[key]); was = int(prev[key])
            d = now - was
            sign = "+" if d > 0 else ""
            parts.append(f"{key} {was}->{now} ({sign}{d})")
        except (ValueError, KeyError, TypeError):
            parts.append(f"{key} {prev.get(key,'?')}->{_fmt(metrics[key])}")
    return "diff vs phase " + str(prev.get("phase", "?")) + ": " + ", ".join(parts)


def main(phase: str, model: str, run_path: str, truth_path: str) -> None:
    prev = _previous_row()                 # capture BEFORE appending
    stdout = _run_score(run_path, truth_path)
    metrics = _parse(stdout)

    print(stdout.rstrip())
    print()
    print(f"scores.csv row: {phase},{model},{metrics['text_sim']},"
          f"{metrics['scalar']},{metrics['smiles']},{metrics['table']},"
          f"{metrics['missing_count']}")
    print(_diff_line(prev, metrics))

    _append(phase, model, metrics)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
