# Outcome-blind AMIGOS confirmation protocol (v14)

Status: locked before receipt or inspection of AMIGOS recordings or labels  
Lock date: 2026-09-14 (Asia/Shanghai)  
Parent manuscript commit: `90ec1ce59bfb9858fe79fa9ad2eec155f60995a4`  
Machine-readable companion: `configs/confirmatory_amigos_v14.yaml`

## Purpose and confirmatory status

This extension tests whether matching non-EEG label resources changes the
attribution of apparent EEG gains in a dataset that was not used to design,
tune, or diagnose OpenAffect-EEG. At lock time, no AMIGOS directory was present
in the local workspace or in the project data root on the compute host. Public
dataset descriptions and published aggregate results were used only to assess
eligibility; no AMIGOS trial labels, EEG arrays, or model outcomes were read.

The confirmation is outcome-blind, not hypothesis-naive. Existing results on
other datasets are known. Any AMIGOS result, including a positive matched EEG
increment or substantial heterogeneity, will be retained and reported.

## Dataset selection before outcome access

Candidate datasets were required to provide:

1. EEG recorded during affect elicitation;
2. participant-specific valence and arousal ratings;
3. stable stimulus identities repeated across participants;
4. at least 25 potentially eligible participants and 12 repeated stimuli;
5. research access to trial-level EEG and labels under a documented licence;
6. no prior outcome analysis by this project.

When more than one dataset met these conditions, the tie-breakers were, in
order: a larger participant count, a dataset not used in the closest matched
personalisation study, and a design distinct from both existing primary tasks.
AMIGOS was selected because its individual short-video arm reports 40
participants viewing the same 16 clips with EEG and participant self-ratings.
DEAP was not selected because it is central to the closest identified study;
DREAMER has fewer than 25 participants; MAHNOB-HCI has fewer participants than
AMIGOS. These are design choices, not quality judgements about the datasets.

Access requires the AMIGOS EULA and credentials issued by the data owner. The
authors will not scrape, bypass, redistribute, or publish restricted data.

## Analysis population and outcome-blind QC

Only the individual-viewing, short-video arm is eligible. A trial is eligible
when it has a stable participant ID, a stable short-video ID, finite
participant-reported valence and arousal, and a readable EEG recording with the
documented montage. Group-viewing and long-video trials are excluded.

After authorised download, structural QC may inspect file names, shapes,
sampling rates, channel labels, missingness indicators, and signal quality. The
valence and arousal values must remain sealed during this step. Before labels
are unsealed, an amendment will freeze:

- the exact source version and file hashes;
- the eligible participant and stimulus IDs;
- channel mapping and physical units;
- segment length and deterministic artefact thresholds;
- any exclusions caused by corrupt or absent recordings.

No exclusion may depend on a label value, an EEG/no-EEG score, an effect sign,
or model performance. Unexpected source defects may trigger a dated amendment;
the original rule and reason will remain visible.

## Fixed-support design

The resource grid is fixed at participant-label doses `0, 1, 2, 4, 8` and
same-stimulus other-participant doses `0, 1, 2, 4, 8`. Every EEG predictor is
paired with an EEG-free predictor receiving exactly the same supplied labels.
Population-training row count is held constant by replacement rather than by
adding calibration rows.

Five participant folds and four stimulus folds are formed deterministically
from hashed identities. Their 20 cross-products cover every eligible
participant-by-stimulus trial exactly once as out-of-fold test support. For a
test cross-product, population training excludes its participants and stimuli;
participant calibration uses only the same participant's non-test stimuli, and
stimulus calibration uses only other participants' labels for the same test
stimulus. Validation excludes all test identities.

The exact assignment seed is `20260914`. Identity ordering, shortfalls at a
requested dose, and replacement rows are recorded. A dose cell with inadequate
resources is marked unavailable rather than silently using a smaller dose.

## Predictors and training repetitions

All predictors receive the same folds and resource assignments.

- EEG-free matched comparator: population prior plus the prespecified
  participant and stimulus calibration components.
- Band-power Ridge: deterministic sanity baseline with alphas
  `0.1, 1, 10, 100`, selected on validation data.
- EEGNet: standard Braindecode EEGNet with the two already released learning
  rates (`1e-3`, `1e-4`), validation selection, and five training seeds.
- LaBraM-PEFT: official pretrained checkpoint with the final four transformer
  blocks, final normalisation, and regression head trainable; five training
  seeds. The head learning rate is `1e-3`, backbone learning rate is `1e-5`,
  AdamW weight decay is `0.01`, maximum epochs is 40, and patience is 8.

Training seeds are `2026091401` through `2026091405`. Model selection uses
validation MAE averaged across valence and arousal, with lower parameter count
as the deterministic tie-breaker. Test labels are never used for checkpoint,
epoch, hyperparameter, preprocessing, or model selection.

A label-permutation negative control and a synthetic EEG-signal positive
control must pass before real test effects are interpreted. Failure of a model
or seed is reported; it is not removed based on its direction or magnitude.

## Estimands and inference

For resource cell `(p, s)`, the matched increment is

`Delta(p,s) = CCC(Y, EEG+matched labels) - CCC(Y, matched labels only)`.

The single primary estimand is the uniform mean of `Delta(p,s)` across all 25
cells, averaged across valence and arousal, using pooled out-of-fold
predictions. Model-specific and target-specific estimates are secondary.

The primary uncertainty procedure combines:

1. complete participant-by-stimulus crossed resampling of out-of-fold trials;
2. resampling across the five prespecified neural training seeds;
3. five independently hashed fold rotations, each requiring a complete model
   refit and validation selection.

The resulting interval is unconditional with respect to the prespecified seed
and fold-rotation distributions. It is not claimed to cover arbitrary
architectures or tuning procedures. At least 2,000 analysis bootstrap draws are
used. Cellwise results use a max-absolute-deviation simultaneous 95% band over
the 25 cells and are otherwise descriptive.

## Prespecified practical-equivalence rule

The primary smallest effect size of interest is an absolute CCC increment of
`0.05`. Practical equivalence is declared only if the two-sided 95% interval
for the primary estimand lies wholly inside `[-0.05, +0.05]`. A stricter
`[-0.025, +0.025]` interval is reported as sensitivity. These thresholds are
decision conventions, not established clinical constants. They were fixed
before AMIGOS outcome access and will not be retroactively applied as
confirmatory thresholds to the earlier datasets.

If the interval overlaps either equivalence boundary, the result is
inconclusive. An interval containing zero is never described as evidence of
equivalence by itself.

## Reporting and stopping rules

There is no efficacy, futility, or sign-based early stopping. The full 5x5 grid
is run for the prespecified models once data QC and controls pass. All planned
models, seeds, exclusions, unavailable cells, failures, and deviations are
reported. Heterogeneity across targets, doses, and datasets is retained even
when it weakens the existing narrative.

## Conditions that can invalidate confirmation

The analysis becomes exploratory if labels or model outcomes are inspected
before the structural-QC amendment is locked, if eligibility rules change in
response to results, if fewer than 25 participants or 12 common short-video
stimuli remain, or if the matched comparator cannot be constructed without
using test labels outside the declared calibration resources.

