# Labnote Extractor

Parses scanned handwritten chemistry lab-notebook pages into structured,
schema-validated JSON — without trusting a single VLM call.

## Architecture

```
perception (page-agnostic)            semantics (experiment-specific)
──────────────────────────            ────────────────────────────────
preprocess → layout → ocr_text   ┐
                    → ocr_math   ├─→ fields dict ─→ classify ─→ registry.dispatch
                    → ocsr       ┘                              │
                    → normalize                                 ▼
                                                     domain validator
                                                     (re-derives the physics,
                                                      cross-checks OCR values)
                                                          │
                                                          ▼
                                                 schema-validated JSON
```

The key differentiator is the **validation layer**: for an electrodeposition
page the `J → I → Q → n → mass` chain is re-derived from first principles and
every step is cross-checked against the value OCR read off the page. A
misread digit breaks the chain and localises the error automatically.

## Quick start

```bash
pip install -e .
pytest          # validator regression tests pass immediately
python main.py data/Example_Page.jpg output.json   # full pipeline (needs OCR deps)
```

## Install heavy dependencies

```bash
# RDKit (required for OCSR validity gate)
conda install -c conda-forge rdkit

# DECIMER hand-drawn OCSR model
pip install decimer

# OPSIN name resolver (requires Java ≥ 8)
pip install py2opsin
```

## Evaluation

```bash
python -m eval.score \
    --pred pipeline_output.json \
    --gt   eval/draft_labels/page57.draft.json
```

> **No verified ground truth yet**: `eval/draft_labels/page57.draft.json` was
> VLM-drafted and is a **draft**, not a trusted reference. The folder is named
> `draft_labels/` (not `ground_truth/`) on purpose — a human who can read the
> handwriting AND verify the chemistry must validate it before any score means
> anything. See `CLAUDE.md §eval`.

## Repo layout

```
labnote-extractor/
├── main.py                     orchestrator
├── pipeline/
│   ├── perception/             page-agnostic OCR/CV (preprocess→normalize)
│   ├── classify.py             text → experiment type
│   ├── registry.py             experiment type → {schema, validator, fields}
│   ├── correct.py              bounded correction loop (optional)
│   └── semantics/              one plugin per experiment family
│       ├── electrodeposition.py  validate_deposition + @plugin
│       ├── spectroscopy.py       stub
│       └── synthesis.py          stub
├── reference/
│   ├── chem_resolve.py         name → OPSIN/PubChem → SMILES, z, molar_mass
│   └── constants.py            FARADAY, AVOGADRO, R_GAS
├── schema/
│   ├── base.schema.json        shared field envelope
│   ├── electrodeposition.schema.json
│   └── spectroscopy.schema.json  stub
├── eval/
│   ├── draft_labels/page57.draft.json   VLM-drafted DRAFT (not verified)
│   ├── baseline_vlm.py
│   └── score.py
└── tests/
    └── test_electrodeposition.py
```

## Adding a new experiment type

1. Write `pipeline/semantics/<family>.py` with a `@plugin`-decorated validator.
2. Add a `schema/<family>.schema.json`.
3. Import the module in `main.py` so the decorator runs.

Nothing else changes. The perception layer and core pipeline are untouched.
