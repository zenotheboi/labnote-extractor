"""
score.py — compare pipeline output against flat ground truth.

Usage:  python score.py output.json ground_truth.json

- Scalars: normalized string/number match.
- Molecules (keys under "structures."): RDKit canonical-SMILES match if rdkit
  is installed, else string match.
- Tables (ground-truth values that are lists): row-wise cell comparison.
Reports three metrics: scalar accuracy, SMILES match, table cell accuracy,
plus missing keys (recall misses).
"""

import json, sys, re
from difflib import SequenceMatcher

try:
    from rdkit import Chem
    def canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
except Exception:
    canon = lambda s: None  # rdkit absent -> fall back to string compare


def norm(v):
    """Loose normalization so '5 mol%' == '5mol %', '0.50' == '0.5',
    and 'A·s' == 'A*s' (· × unified to *) for expression fields."""
    s = str(v).lower().strip().replace(" ", "").replace("·", "*").replace("×", "*")
    try:
        return str(float(s))
    except ValueError:
        return s


def flatten(obj, prefix=""):
    """Pipeline output (rich envelopes) -> flat dotted keys -> scalar values.
    Pulls '.value' out of field envelopes; recurses dicts and lists."""
    flat = {}
    if isinstance(obj, dict):
        if "value" in obj and "confidence" in obj:   # a field envelope
            return {prefix: obj["value"]}
        for k, v in obj.items():
            flat.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        flat[prefix] = obj                            # keep lists whole (tables)
    else:
        flat[prefix] = obj
    return flat


def score_lines(truth_lines, out_val):
    """Plain-text: for each truth line, best fuzzy match among output lines.
    Returns (sum_of_best_ratios, n_lines) -> reported as avg similarity %."""
    if isinstance(out_val, str):
        out_lines = out_val.splitlines()
    elif isinstance(out_val, list):
        out_lines = out_val
    else:
        out_lines = []
    out_norm = [norm(x) for x in out_lines]
    total = 0.0
    for tl in truth_lines:
        tn = norm(tl)
        total += max((SequenceMatcher(None, tn, on).ratio() for on in out_norm),
                     default=0.0)
    return total, len(truth_lines)


def score_table(truth_rows, out_rows):
    """Row-wise: match by first key, compare cells. Returns (correct, total)."""
    if not isinstance(out_rows, list):
        return 0, sum(len(r) for r in truth_rows)
    correct = total = 0
    for i, trow in enumerate(truth_rows):
        orow = out_rows[i] if i < len(out_rows) else {}
        for k, v in trow.items():
            if k == "uncertain":
                continue
            total += 1
            if isinstance(orow, dict) and norm(orow.get(k)) == norm(v):
                correct += 1
    return correct, total


def main(out_path, truth_path):
    out = flatten(json.load(open(out_path)))
    truth = {k: v for k, v in json.load(open(truth_path)).items()
             if not k.startswith("_")}

    scal_ok = scal_tot = smi_ok = smi_tot = tbl_ok = tbl_tot = 0
    txt_sum = 0.0; txt_n = 0
    missing = []

    for key, tval in truth.items():
        if isinstance(tval, list):
            if tval and isinstance(tval[0], str):     # plain-text lines (Level 1)
                s, n = score_lines(tval, out.get(key))
                txt_sum += s; txt_n += n
            else:                                     # table rows
                c, t = score_table(tval, out.get(key))
                tbl_ok += c; tbl_tot += t
            if key not in out: missing.append(key)
            continue
        if key not in out:
            missing.append(key)
            if key.startswith("structures."): smi_tot += 1
            else: scal_tot += 1
            continue
        if key.startswith("structures."):             # molecule field
            smi_tot += 1
            a, b = canon(str(tval)), canon(str(out[key]))
            if a and b and a == b: smi_ok += 1
            elif norm(tval) == norm(out[key]): smi_ok += 1
        else:                                          # scalar field
            scal_tot += 1
            if norm(tval) == norm(out[key]): scal_ok += 1

    def pct(a, b): return f"{a}/{b} ({100*a/b:.0f}%)" if b else "n/a"
    print("text similarity :", f"{100*txt_sum/txt_n:.0f}% ({txt_n} lines)" if txt_n else "n/a")
    print("scalar accuracy :", pct(scal_ok, scal_tot))
    print("SMILES match    :", pct(smi_ok, smi_tot))
    print("table cells     :", pct(tbl_ok, tbl_tot))
    print("missing keys    :", len(missing), missing if missing else "")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
