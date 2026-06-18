# Labnote Extractor

Parses a scanned handwritten chemistry lab-notebook page into structured,
schema-validated JSON — **without trusting a single VLM call**. A fallible reader
is wrapped in preprocessing, content-type routing, chemistry/physics validation,
and a strict schema, so the output is trustworthy and less model-dependent.

```bash
python main.py data/page57.jpg          # → output.json (flat, machine-readable)
```

## Problem

The target is one page of a Li-electrodeposition experiment, handwritten: prose
notes, a calculation block (`J → I → Q → n → mass`), drawn molecular structures
(`12-crown-4`, `LiTFSI`, diglyme, EtOH), a temperature table, XRD peaks, and
struck-through edits. Off-the-shelf tools do poorly here:

- **Standard OCR** mangles scientific notation (`1.5E-4`), superscripts (`cm²`),
  and symbols (`°C`, `2θ`), and can't touch hand-drawn structures.
- **A bare VLM call** reads the prose reasonably but silently misreads digits,
  groups distinct drawings, returns invalid SMILES, and gives you no provenance
  and no way to know *which* fields to distrust.

The hard content — notation, structures, the numeric chain — is exactly where the
evaluation points are, and exactly where a single call is least reliable.

## Architecture

```
perception (page-agnostic)                semantics (experiment-specific)
──────────────────────────                ───────────────────────────────
preprocess → layout → ocr_text   ┐
                    → ocr_math    ├─→ fields → classify → registry.dispatch
                    → ocsr        ┘                            │
                    → normalize                                ▼
   chem_resolve + rescan ─────────────────────────→  domain validator
   (name → SMILES; second pass               (re-derives the physics,
    recovers missed drawings)                 cross-checks OCR values)
                                                               │
                                                               ▼
                                                    schema-validated JSON
```

One line each:

- **preprocess** — deskew + binarize the scan before the model sees it.
- **layout** — segment the page into regions (header, prose, calc block, drawings, table).
- **ocr_text** — read handwritten prose, table cells, margin notes.
- **ocr_math** — parse the calculation block into structured `value` + `expression` fields.
- **ocsr** — read drawn structures to SMILES, gated through RDKit for validity.
- **normalize** — restore symbols/units (`°C`, `2θ`, `cm²`, `1.5E-4`) and detect strikethrough.
- **chem_resolve** — resolve printed labels to canonical SMILES via OPSIN/PubChem (primary), OCSR confirms.
- **rescan** — targeted second pass (cheaper model) that recovers drawings the first pass missed, e.g. EtOH.
- **classify** — infer the experiment type from the extracted text.
- **registry.dispatch** — route to the matching experiment plugin (schema + validator).
- **domain validator** — re-derive `J → I → Q → n → mass` and cross-check every step against the page.
- **schema** — strict JSON-schema validation; hallucinated fields are rejected, output fails loudly.

The **validation layer** is the differentiator: a misread digit breaks the Faraday
chain and is localized automatically, and the validator *flags* disagreement for a
human rather than silently overwriting what the chemist wrote. See
[`DESIGN.md`](DESIGN.md) for the full rationale.

## Results

Single page (`data/page57.jpg`), all runs on **claude-opus-4-8**, scored against
the frozen human-verified ground truth (`eval/ground_truth/page57.json`). Baseline
and pipeline use the **same model**, so the lift isolates the wrapper, not a model
swap. Regenerate with `python eval/report.py`.

| Step | text | scalar | SMILES | table | missing | self-consistency |
|---|---|---|---|---|---|---|
| Baseline — single VLM call, naive prompt | 85% | 38% | 0% | 0% | 28 | — |
| + Engineered extraction prompt | 97% | 69% | 67% | 100% | 2 | — |
| + Symbol/unit normalization + routing | 97% | 82% | 67% | 100% | 1 | 4/4 ✓ · 0 flagged |
| + Structure recovery (second-pass rescan)  [FINAL] | 97% | 82% | 83% | 100% | 0 | 4/4 ✓ · 0 flagged |
| **Lift (final − baseline)** | **+12pp** | **+44pp** | **+83pp** | **+100pp** | **−28** |  |

**Headline:** the wrapper lifts a same-model single call by **+44pp scalar**,
**+83pp SMILES**, and **+100pp table** accuracy, and drives missing fields from
**28 → 0**.

Only the steps that moved a scored metric are listed above. Three other pipeline
layers add capability these four metrics don't measure: the **trust layer**
(Faraday self-consistency + schema — the `self-consistency` column), **chemistry
robustness** (RDKit-gated name resolution), and **scalability** (classify →
dispatch → bounded correction loop for other experiment types). Full breakdown in
[`eval/results.md`](eval/results.md).

### Evaluation levels reached

| Level | Evidence | |
|---|---|---|
| 1 — Text | 97% text similarity, 100% table cells | ✅ |
| 2 — Symbols | 82% scalar (°C, 2θ, cm², `1.5E-4`) | ✅ |
| 3 — Chemistry | 83% SMILES (5/6), formulas, reagents, concentrations | ✅ |
| 4 — Experiment | goal / conditions / procedure / results in schema, Faraday validator confirms the physics | ✅ |

## Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # or put it in a .env file
python main.py data/page57.jpg             # → output.json
```

The first run calls the VLM once and caches the raw output under `eval/cache/`;
later runs re-run only the deterministic wrapper (no API call). The structure
rescan makes one extra call on a cheaper model.

### Dependencies that need manual steps

- **RDKit** (SMILES validity gate, canonicalization, Tanimoto) — `pip install rdkit`,
  or `conda install -c conda-forge rdkit`. Without it, SMILES pass through ungated.
- **OPSIN** (`py2opsin`, IUPAC name → SMILES) — **requires a Java runtime (JRE ≥ 8)**.
  Without Java it's skipped and resolution falls back to PubChem.
- **DECIMER** (hand-drawn OCSR model) — heavy TensorFlow deps; install in a separate
  venv (`pip install decimer`). Optional: when absent, the VLM's drawn SMILES is
  RDKit-gated instead. The interface is identical, so adding DECIMER changes nothing
  downstream.

```bash
pytest          # validator + correction-loop regression tests
```

## Repo layout

```
labnote-extractor/
├── main.py                     orchestrator (the one runnable entry point)
├── pipeline/
│   ├── perception/             page-agnostic CV/OCR (preprocess → normalize, ocsr)
│   ├── classify.py             extracted text → experiment type
│   ├── registry.py             experiment type → {schema, validator, fields}
│   ├── correct.py              bounded correction loop (validator-triggered)
│   └── semantics/              one plugin per experiment family
├── reference/
│   ├── chem_resolve.py         label → OPSIN/PubChem → SMILES, z, molar_mass
│   └── constants.py            FARADAY, AVOGADRO, R_GAS
├── schema/                     base envelope + per-experiment JSON schemas
├── eval/
│   ├── ground_truth/page57.json    frozen, human-verified answer key
│   ├── baseline_vlm.py             single-VLM baseline (same model)
│   ├── score.py                    metrics vs ground truth
│   ├── report.py                   renders results.md from scores.csv
│   └── results.md                  generated results table
└── tests/                      validator + correction-loop regression tests
```

## Design decisions (summary)

Full reasoning in [`DESIGN.md`](DESIGN.md). In brief:

- **Decompose, don't one-shot.** Route each content type to the right handler so
  failures are localized and recoverable.
- **Validate against chemistry, not just shape.** Re-derive `J → I → Q → n → mass`,
  RDKit-gate every SMILES, enforce a strict schema. The validator **flags**, never
  auto-corrects — a faithfully transcribed real error is data, not noise.
- **Name-resolution first for structures.** Resolve the printed label to a canonical
  SMILES (deterministic lookup, not vector-RAG); use the drawing to confirm.
- **Perception shared, semantics pluggable.** Adding an experiment type is one
  plugin; the core and perception layers never change.
- **The VLM stays in the loop.** The wrapper is additive — it's on both sides of the
  baseline comparison, so the lift measures the wrapper, not the model.

### Assumptions & future work

- Structure detection depends on the VLM seeing the drawing; the rescan mitigates
  this, but the principled fix is a dedicated structure-region detector
  (YOLO / DECIMER detection mode).
- SMILES accuracy benefits from labeled, PubChem-resolvable reagents; novel/unlabeled
  structures would need DECIMER for cold OCSR.
- Numbers are reported on a single page; the architecture is page-agnostic but
  broader accuracy is unmeasured until more ground truth exists.
- Upgrade path (no training in scope): ground truth as the evaluation backbone now,
  then fine-tuning data for the recognisers, with a human-in-the-loop correction
  flywheel — never self-training.

## License

MIT — see [`LICENSE`](LICENSE). © 2026 Wunan Tang.
