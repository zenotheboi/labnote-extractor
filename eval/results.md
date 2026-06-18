# Results

Single page (`data/page57.jpg`), all runs on **claude-opus-4-8**, scored against the frozen human-verified ground truth (`eval/ground_truth/page57.json`). Baseline and pipeline use the **same model**, so the lift isolates the wrapper, not a model swap.

| Step | text | scalar | SMILES | table | missing | self-consistency |
|---|---|---|---|---|---|---|
| Baseline — single VLM call, naive prompt | 85% | 38% | 0% | 0% | 28 | — |
| + Engineered extraction prompt | 97% | 69% | 67% | 100% | 2 | — |
| + Symbol/unit normalization + routing | 97% | 82% | 67% | 100% | 1 | 4/4 ✓ · 0 flagged |
| + Structure recovery (second-pass rescan)  [FINAL] | 97% | 82% | 83% | 100% | 0 | 4/4 ✓ · 0 flagged |
| **Lift (final − baseline)** | **+12pp** | **+44pp** | **+83pp** | **+100pp** | **-28** |  |

**Headline:** the wrapper lifts a same-model single call by **+44pp scalar**, **+83pp SMILES**, and **+100pp table** accuracy, and drives missing fields from 28 to 0.

### Capability not captured by these four metrics

- Phase 2 — trust layer: Faraday self-consistency re-derivation + strict schema validation. Shown in the `self-consistency` column (checks reconciled / applicable, plus flagged count); not a text/scalar/SMILES/table metric. On this page 4 of 4 applicable checks reconcile (I, Q, n, mass); `t` has no independent page value to compare, so it is not applicable, not a failure.
- Phase 4 — chemistry robustness: name-resolution (PubChem/OPSIN) + RDKit canonical gate. Hardens SMILES correctness; SMILES count unchanged on this page.
- Phase 5 — scalability: classify -> registry dispatch -> bounded correction loop. Generalizes the pipeline to other experiment types; no single-page metric change.
