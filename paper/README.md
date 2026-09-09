# Manuscript workspace

This directory contains the evidence-linked writing workspace for the anonymous
NeurIPS Evaluations & Datasets submission.

## Current writing artifacts

The current local V11 manuscript integrates resource-matched contrasts from the
frozen V9 results, known-truth evaluator diagnostics, and a source-linked claim
repair. See `reviews/evaluation_argument_evidence_v10_zh.md` for the original
claim-to-evidence argument. The V11 archive is a review-ready local artifact;
an externally accessible anonymous code URL must still be supplied at submission.

- `manuscript_zh.md`: Chinese mother manuscript with inline frozen-claim IDs.
- `supplement_zh.md`: Chinese supplementary-methods and evidence-boundary draft.
- `chinese_reading/OpenAffect-EEG-zh.pdf`: local Chinese reading edition, rebuilt
  together with the English manuscript by `scripts/build_paper_bundle.py`.
- `benchmark_data_card_zh.md`: benchmark-level data card.
- `references.bib`: verified manuscript bibliography.
- `citation_audit.tsv`: persistent identifier, verification status, and intended
  support for every bibliography entry.
- `neurips2026/main.tex`: anonymous English main manuscript in the official
  NeurIPS 2026 Evaluations & Datasets format.
- `neurips2026/supplement.tex`: English supplementary methods and evidence
  boundary.
- `neurips2026/checklist.tex`: completed official paper checklist.
- `generated/`: hash-derived tables, figures, claim ledger, manifests, and
  manuscript audit reports.
- `generated/seedv_case_study.json`: source-safe aggregate-only record for an
  authorized SEED-V protocol-audit negative control.
- `generated/music_labram_cross_dataset_audit_v1.json`: source-safe aggregate
  record for the independent MusicEEG pretrained-versus-random LaBraM audit.
- `generated/identity_exposure_v1/`: fixed-support participant-calibration by
  repeated-stimulus response surfaces, crossed intervals, support audits, and
  deployment summaries. It contains aggregate results only, not source EEG.

The anonymous English manuscript uses the official E&D template. The paired
build enforces the nine-page main-content limit and the claim, citation, source,
anonymity, and figure gates rather than assuming the previous page count still
holds after a revision.

## Build the evidence package

```bash
python scripts/build_manuscript_assets.py \
  configs/paper_artifacts.yaml \
  configs/manuscript_claims.yaml \
  "${OPENAFFECT_DATA_ROOT}/derived/paper/manuscript_neurips_ed_v2" \
  --data-root "${OPENAFFECT_DATA_ROOT}"
```

The core registry verifies 56 hash-locked inputs and resolves 104 numeric
claims. The response-surface extension verifies four additional aggregate
tables and appends 44 claims (`C105`--`C148`), for 148 manuscript claims in
total. It writes deterministic manuscript assets, including benchmark
scope, external validation, representation, protocol, identity, support-matched,
subspace-intervention, semantic-prior ablation, and data-quality tables plus the
claim ledger and output manifest.

## Claim policy

Result numbers in manuscript prose must reference an entry from
`configs/manuscript_claims.yaml`. Each selector resolves to exactly one frozen
table cell and declares an expected value, tolerance, display precision, source
asset, and allowed wording. The build fails on a missing or ambiguous selector,
hash mismatch, or numeric drift.

Method constants are traced to their implementation or versioned configuration
in the supplementary mother draft. Frozen result values are injected into the
English manuscript through generated LaTeX macros in
`generated/claim_values.tex`, avoiding manual transcription.

Both submission build entry points rerun
`scripts/append_identity_exposure_claims.py` before LaTeX compilation. The
extension checks exact expected values and source-table hashes and fails on
selector ambiguity or numerical drift.

The submission build also regenerates `generated/submission_closure_v11/` from
hash-verified simulation and source-linked workflow-audit records. This keeps
the evaluator stress tests and five-fold EEGain claim repair coupled to their
recorded sources.

## Manuscript audit

```bash
python scripts/audit_manuscript.py \
  paper/manuscript_zh.md \
  paper/references.bib \
  paper/generated/claim_ledger.tsv \
  paper/citation_audit.tsv \
  --source-root . \
  --output paper/generated/manuscript_audit_zh.json
```

The audit rejects unresolved citation keys, citations without a verified source
record, unknown claim IDs, duplicate evidence keys, numeric prose without an
explicit claim/source mapping, source paths that do not exist, email addresses,
local home or data paths, Windows user paths, and local hostnames.

## Build the anonymous submission draft

```bash
python scripts/build_neurips_submission.py
```

This command audits the English main text and supplement, checks anonymous E&D
source settings, compiles the official LaTeX document, rejects undefined
citations/references and overfull boxes, and enforces the nine-page content
limit. During figure development it reports missing final figures without
failing. The release gate is stricter:

```bash
python scripts/build_neurips_submission.py --require-final-figures
```

## Build the paired Chinese reading PDF

The English PDF is the submission artifact. For every manuscript revision, use
the paired build command below rather than building the English document alone.
It first applies the anonymous English submission gate, then compiles a Chinese
reading edition from `manuscript_zh.md` and `supplement_zh.md` with the same
five approved figures:

```bash
python scripts/build_paper_bundle.py --require-final-figures
```

The local Chinese PDF is not part of the anonymous NeurIPS upload or public
release archive; it is a synchronized review artifact for the research team.

## Figure approval workflow

Figures are decided one scientific question at a time. For every new figure,
the workflow is:

1. identify the claim and comparison structure;
2. generate three genuinely different chart-type candidates suited to that
   question;
3. obtain the user's choice for that figure only;
4. render the selected design in vector PDF and high-resolution PNG;
5. inspect color, grayscale, text fit, and final-column dimensions.

Every final scientific figure must invoke the task-appropriate figure skill and
pass an independent visualization audit. Export is rejected when final-size
text falls below the recorded threshold, any text bounding boxes overlap or
leave the canvas, or the grayscale and whole-page manuscript renders have not
been produced and inspected. Each plotting script writes a machine-readable QA
record next to the review artifacts.

The selected Figure 1 validates its source boundary against the published
dataset--protocol matrix rather than a hand-coded exception. The compact main
figure shows the evaluation contract; the full applicability matrix remains in
Supplementary Table 4:

```bash
python scripts/plot_figure1_contract.py
```

Typography and accessibility can remain consistent across the paper, but one
figure's chart type does not constrain the next figure. The current main paper
uses F1, the identity-exposure response surface as F2, and F3; earlier protocol,
intervention, and external-condition views remain in the supplement. Selected
artwork must retain the dimensions recorded in
`figure_plan_zh.json` or trigger a fresh page-limit audit.

After the user approves an individual candidate, synchronize only that approved
candidate with the submission and release locations. The command rejects an
unregistered candidate, missing PDF/PNG pair, and a path outside the project;
it records the candidate and hashes in
`paper/generated/final_figure_selection.json`.

```bash
python scripts/finalize_figure_selection.py \
  --select F1=B2 --select F2=B2 --select F3=A \
  --select F4=A --select F5=C
```

The command copies each approved vector PDF and 300-DPI PNG to both
`paper/neurips2026/figures/` for LaTeX compilation and `paper/final_figures/`
for the allowlisted anonymous release. The release policy also requires the
five compiled-paper PDF/PNG pairs, so an incomplete Git-tracked source state
cannot produce a deceptively successful archive. This does not replace user
selection or the required whole-page and grayscale review.

## Interpretation boundary

The benchmark predicts harmonized self-report. It does not claim access to an
error-free private emotional state. The public package does not redistribute
raw EEG, source DE or feature arrays, model checkpoints, or restricted movie,
music, and image stimuli. A resumed dependency-graph validation is not described
as a cold raw-data rerun, and a clean environment on the same server is not
described as independent external reproduction.
