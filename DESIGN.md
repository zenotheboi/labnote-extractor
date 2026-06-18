# DESIGN.md — Design decisions & rationale

The *why* behind the pipeline, written for reviewers. `README.md` is the quick
tour; this document is the reasoning, each decision as
**decision → why → what I considered and rejected**.

The thesis: the assignment notes that off-the-shelf OCR and VLMs do poorly on a
page like this, so the value isn't a magic model — it's the engineering judgment
wrapped around a fallible reader to make its output **trustworthy and less
model-dependent**. The sections below follow the pipeline's actual data flow:

```
preprocess → layout → {ocr_text, ocr_math, ocsr, normalize}
           → chem_resolve + rescan → classify → dispatch → validate → schema
```

> **Implementation honesty (read first).** The *current* recogniser for prose,
> layout, math lines, and drawn structures is a single strong VLM (claude-opus-4-8),
> wrapped by deterministic stages. Several "specialist" stages below
> (`ocr_math`/pix2tex, `ocr_text` ensemble, `ocsr`/DECIMER) are **documented as
> stable interfaces with a VLM-backed implementation today and a named drop-in
> upgrade path** — not as already-wired separate engines. Where a stage is an
> upgrade path rather than a running engine, it says so. This is the "verified
> wrapper" framing, stated plainly so claims match the code.

---

## 1. Layout-first decomposition, not one-shot prompting
**Decision.** Segment the page into regions (header, prose, calculation block,
drawn structures, table, margin notes) and handle each on its own terms, rather
than feeding the whole page to one call and hoping.
**Why.** Different content types fail in different ways and need different
post-processing. A single blob gives you no provenance and no place to intervene
when one region is wrong.
**Rejected.** One-shot VLM prompt — that's the naive baseline the assignment says
everyone submits; no provenance, no per-region control, no recovery path.

## 2. Route each content type to the right handler (VLM-backed today, upgrade path documented)
**Decision.** Prose → handwriting OCR/VLM; scientific notation → math parsing
(`ocr_math`, pix2tex as the upgrade); molecules → OCSR (`ocsr`, DECIMER as the
upgrade); tables → table parsing. Each is a stable interface so the engine behind
it can be swapped without touching anything downstream.
**Why.** Generic OCR mangles `1.5E-4`, superscripts, and `cm²`; bare VLMs are weak
on hand-drawn structures. A specialist beats a generalist on its home turf, and an
interface lets you upgrade region-by-region once the wrapper proves where the
errors are.
**Reality check.** Today the VLM reads all regions; `ocr_math` parses its calc
lines into structured fields with regex, and `ocsr` RDKit-gates its drawn SMILES.
pix2tex / DECIMER / a two-engine handwriting ensemble are the documented swaps,
not yet wired.
**Rejected.** Trusting one model for everything with no seam to upgrade — fast to
write, but it permanently fuses the easy and hard content together.

## 3. The image is read once; improvement is upstream prep + downstream verification
**Decision.** Most of the lift is **not** from re-reading the JPG. It comes from
(a) preprocessing the input *before* the model sees it (deskew, binarize),
(b) routing hard regions, and (c) verifying the *output* against chemistry and
logic (Faraday re-derivation, RDKit gate, schema) — none of which re-examines the
image.
**Why.** Worth stating precisely so the system isn't mistaken for iterative vision
passes. Two stages *do* re-read the image, both narrowly and on demand: the
bounded correction loop (§10) on a single validation-failed field, and the
structure rescan (§5) for drawings the first pass missed. Everything else is
upstream prep and downstream verification.
**Rejected.** "Just call the model again, harder" as a general strategy — costly,
non-reproducible, and it doesn't fix the failure modes that aren't about reading.

## 4. Name-resolution-first for chemistry; OCSR confirms; RDKit gates; grounding is deterministic
**Decision.** Read the printed *label* (`12-crown-4`, `LiTFSI`), resolve it to a
canonical structure via OPSIN/PubChem (`chem_resolve`), and use OCSR on the drawing
only to *confirm*. Gate every SMILES through RDKit — if it doesn't parse, reject
it. This grounding is a structured database lookup, not vector-RAG into an LLM.
**Why.** Resolving a known name is reliable; reading a hand sketch is not. Use the
reliable signal as primary and the unreliable one as a cross-check. A database
lookup returns the *correct* canonical SMILES; RAG-into-an-LLM returns a guess
conditioned on retrieved text, which can still hallucinate.
**Rejected.** OCSR-as-primary — sketches are noisy and you'd inherit their errors
as truth. Vector-RAG for structures — wrong tool; deterministic lookup is strictly
better for canonical structures. (Real RAG would only help with a corpus of prior
pages for handwriting disambiguation — out of scope.)

## 5. Structure recovery: record as-drawn, and a second-pass rescan for missed drawings
**Decision.** Record each structure **as drawn** (fidelity to the page), keyed by
its label. When the first pass misses a drawing entirely, a targeted **second-pass
rescan** (a cheaper model, claude-opus-4-7) is given the already-detected labels
and asked only for drawings *not* in that list; new hits are reconciled via
`chem_resolve` and RDKit-gated.
**Why.** The first pass grouped the `diglyme : EtOH` solvent sketch and missed
EtOH as a separate structure. The rescan looks at the image and *confirms* a second
drawing exists before adding it — recovering EtOH (SMILES 5/6 → 6/6) with honest
provenance (`rescan`), rather than inferring a drawing from text.
**Rejected.** Inferring the missing structure from the solvent *text* — it asserts
a drawing exists from a label alone (a `text_inferred` fallback exists but is
disabled by default for exactly this reason). The principled fix is a dedicated
structure-region detector (YOLO / DECIMER detection mode) that finds drawings
independently of what the text pass reported — the documented next step.

## 6. Numeric self-consistency validation (the differentiator)
**Decision.** Re-derive the electrochemistry chain `J → I → Q → n → mass` from the
primary inputs and cross-check each step against the value OCR read off the page.
**Why.** OCR silently misreads digits; a clean-looking wrong number corrupts the
result with nothing to catch it. A closed physical chain catches and *localizes*
the bad digit — domain knowledge turned into an automatic validator.
**Rejected.** Trusting extracted numbers as-is — the most common silent-failure
mode in document extraction.

## 7. The validator flags; it never auto-corrects
**Decision.** On a mismatch, mark the field low-confidence / consistency-failed,
record both the page value and the derived value side by side, and escalate to a
human. Never overwrite the page's value with the computed one.
**Why.** The validator detects *disagreement*; it cannot *adjudicate truth*. A
mismatch has three indistinguishable causes — OCR misread, a real chemist error,
or an assumption the validator doesn't model (e.g. <100% current efficiency).
Auto-"correcting" would falsify the notebook. A faithfully-transcribed real error
is *data*, not noise.
**Rejected.** Auto-correction — it would require knowing ground-truth reality (it
doesn't) and would destroy the record's fidelity.

## 8. Validation in three tiers, most of it not page-specific
**Decision.** Layer validation by scope: **Universal** (every page) — unit
consistency, range sanity, RDKit validity, schema conformance; **Per-experiment-
type** (dispatched) — the Faraday chain for electrodeposition, Beer-Lambert for
spectroscopy, capacity/CE for cycling, yield for synthesis; **None** — unknown
type degrades gracefully, skips domain checks, keeps the extraction.
**Why.** The `J → I → Q → n → mass` chain is **electrodeposition-specific, not
page-specific** — it works for any Li/Cu/Ni plating page but is meaningless on a
UV-vis page. Scoping validation by tier makes that explicit.
**Rejected.** A single monolithic validator — either too narrow or falsely applied
to the wrong experiment type.

## 9. Perception is shared; semantics is pluggable (scalability)
**Decision.** Perception layers (preprocess, layout, OCR, OCSR, normalize) are
page-agnostic and run on any page. Only the semantic layer (schema + validator) is
experiment-specific, organized as one plugin per experiment family behind a
`classify → registry → dispatch` flow. Constants (z, molar mass) come from
`chem_resolve` (PubChem), not hardcoded.
**Why.** This is what turns "works on page 57" into "works on the notebook."
Adding an experiment type = adding one plugin; the core never changes.
**Rejected.** Hardcoding per page — the `__main__` fixtures in the seed files are
*tests*, not the data path; the data path is upstream extraction.

## 10. Bounded correction loop — the only open-ended-looking part, deliberately bounded
**Decision.** Trigger *only* on a validator failure, scope to the one failing
field, try an alternate extraction (different binarization; swap OCR↔math-OCR;
second OCSR model; re-parse from the equation line), re-check, and after N=2
attempts emit low-confidence + a human flag. The core pipeline stays a
deterministic DAG.
**Why.** Deterministic pipelines are reproducible and evaluable — exactly what's
being graded. A bounded, failure-triggered loop adds self-correction without
sacrificing that. Knowing when *not* to reach for an open-ended agent is the point.
**Rejected.** An open-ended agent loop as the core — hard to evaluate,
non-reproducible, buzzword-driven.

## 11. The VLM stays in the pipeline — the wrapper is additive, not a substitute
**Decision.** Keep a strong VLM as the prose/layout reader; wrap it in
preprocessing + routing + validation + schema; route *around* it only where it's
weak (structures → OCSR, numbers → validator).
**Why.** Beating the bare-VLM baseline proves the wrapper *adds value on top of*
the VLM, not that it *replaces* it. The VLM is on both sides of the comparison, so
the lift measures the wrapper's contribution. "Beyond a single AI call" means don't
*blindly trust* one call — not don't use AI.
**Rejected.** Removing the VLM — nothing would be left to read the prose; the
"baseline-beating pipeline" would be an empty frame.

## 12. Evaluation: page-faithful ground truth, four levels, a same-model baseline ladder
**Decision.**
- **Ground truth records only what is on the page**, verbatim, each value in its
  own field — no analysis, no resolved interpretations in the answer key. Authored
  from the *image* by a human; a VLM-drafted file is a draft to verify, never
  trusted (scoring a model against a corrected copy of itself is circular). Two
  hard-won examples: a looked-up `diglyme` SMILES was removed because the *name* is
  text, not a drawing; and the solvent sketch was finally read as **two** drawings
  (diglyme + EtOH). Where a label and a drawing disagree, both are recorded —
  the *disagreement* is surfaced by the data, not pre-resolved in a note.
- **Each of the four levels is measured separately.** The page is stored in two
  representations: `plain_text` (verbatim, whole page) for Level 1 transcription,
  and parsed fields for Level 3/4 comprehension. `.value` grades the answer;
  `.expression` grades the as-written working.
- **The baseline is a single VLM call on the same model as the pipeline**
  (claude-opus-4-8), plus the engineered-prompt single call as a second reference
  point. Same model on both sides so the lift is attributable to the wrapper, not
  a model swap.
**Why.** Extracting `stir_time: 20 min` proves comprehension but not transcription;
the assignment grades both. Holding the model constant is what makes the lift
honest.
**Rejected.** Building ground truth by patching the pipeline's own JSON (circular);
storing only parsed values (collapses L1 and L3); a weaker-model baseline (inflates
the lift by conflating model with architecture). *An OCR-only (Tesseract) rung was
considered as a third, weakest reference point but is **not implemented** — listed
here as future work, not as a result.*

## 13. Model choice is a config variable, and the wrapper narrows the model gap
**Decision.** Make the model a single config variable. Develop on a cheaper model,
run final numbers on the strongest, and decide expensive-vs-cheap empirically by
scoring on *this page* against ground truth.
**Why.** For document parsing on one chemist's handwriting, general benchmarks are
only directional — your page *is* the benchmark. And the wrapper *narrows* the
expensive-vs-cheap gap: the validator and schema catch a weaker model's mistakes,
so "strong vs cheap, bare vs wrapped" is a real robustness result.
**Rejected.** Picking the leaderboard-topping model and asserting it's best for
this task; mixing models across baseline and pipeline (contaminates the lift).

## 14. No training by design, with a clear upgrade path
**Decision.** Ship a verified wrapper around pretrained components; no training in
scope for the deadline.
**Why.** Training from scratch is millions of dollars and pointless; fine-tuning
captures ~all the benefit. The ground-truth database upgrades the system in two
stages — first as the *evaluation backbone* that makes every change measurable,
then as *fine-tuning data* for the recognisers (TrOCR on this lab's hand, DECIMER
on its structures), with a **human-in-the-loop correction flywheel** as the path
to a system that genuinely improves.
**Rejected.** Claiming a trained model without data; *self*-training where the
model labels its own data (errors compound). A human stays in the labeling path so
it learns from truth, not from itself.

---

## Assumptions & known gaps (stated plainly)
- **Structure detection depends on the VLM seeing the drawing.** The rescan (§5)
  mitigates but doesn't eliminate this; the real fix is a dedicated structure-region
  detector. A drawing with no label or unusual notation can still slip through.
- **Name-resolution shines for *labeled* reagents.** SMILES accuracy on this page
  benefits from all structures being labeled and resolvable in PubChem; a page with
  novel/unlabeled intermediates would lean on cold OCSR, which needs DECIMER.
- **Single page, single hand.** Numbers are reported on `page57.jpg`. The
  architecture is page-agnostic, but accuracy on other hands/pages is unmeasured
  until more ground truth exists.
- **Several specialist engines are interfaces, not yet wired** (pix2tex, DECIMER,
  handwriting ensemble, Tesseract rung) — see the honesty note at the top.
