# Outcome-blind AMIGOS confirmation protocol (v14)

Status: superseded outcome-blind on 2026-09-14 before receipt or inspection of
AMIGOS recordings or labels; retained as an audit trail
Lock date: 2026-09-14 (Asia/Shanghai)
Parent manuscript commit: `90ec1ce59bfb9858fe79fa9ad2eec155f60995a4`
Machine-readable companion: `configs/confirmatory_amigos_v14.yaml`

The replacement and its access-only rationale are recorded in
`docs/confirmatory_dataset_amendment_v14.md`. No AMIGOS outcome was observed,
so this is a pre-outcome feasibility amendment rather than a response to a
scientific result.

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

After authorised download, structural QC may inspect file names, MAT variable
names and shapes, sampling-rate metadata, channel metadata, missingness
indicators, and EEG scale, but not valence or arousal values. The expected files
are `Data_Preprocessed_P01.mat` through `P40.mat`, with `joined_data` and
`labels_selfassessment`; only the first 16 trials are considered.

The preprocessing rule was frozen before data receipt. For each short trial,
the first 640 samples (the documented five-second baseline at 128 Hz) are
removed. The central 3,840 remaining samples (30 seconds) are selected and
polyphase-resampled by 25/32 to 3,000 samples at 100 Hz. The first 14 signal
columns are mapped, in order, to AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8,
FC6, F4, F8, and AF4. A trial is excluded for a missing/malformed variable,
missing signal or label, fewer than 14 channels, fewer than 3,840 post-baseline
samples, non-finite selected EEG, or non-finite/out-of-range valence or arousal.
Only participants retaining all 16 short trials enter the common-stimulus
analysis; missing trials are not imputed.

Because the public paper specifies acquisition resolution but not the numerical
unit stored in the preprocessed MAT arrays, units are resolved without labels:
a median selected-window channel SD in `[0.1, 1000]` is treated as microvolts;
one in `[1e-7, 1e-3]` is treated as volts and multiplied by `1e6`. Any other
scale stops ingestion until source documentation is obtained. File hashes,
observed shapes, the scale decision, participant exclusions, and signal-only QC
summaries are locked before labels are unsealed.

After unsealing, valence and arousal ratings must be finite and in `[1, 9]` and
are mapped linearly to `[-1, 1]` by `(rating - 5) / 4`. The deterministic Ridge
view contains per-channel log10 Welch band power at 100 Hz using two-second
windows and bands 2--4, 4--8, 8--13, 13--30, and 30--45 Hz, with `1e-12` added
before the logarithm. These transformations were fixed without viewing label
values.

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

The base assignment seed is `20260914`. Resource sampling uses
`20260914 + global_split_index`, where the global split index is 0--99 in
fold-rotation order and participant-block then stimulus-block order. Identity
ordering, shortfalls at a requested dose, and replacement rows are recorded. A
dose cell with inadequate resources is marked unavailable rather than silently
using a smaller dose.

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
control run over the full design before real test effects are interpreted. In
the negative control, residual targets are permuted independently within the
training and validation partitions, preserving their values and sizes; a
primary CCC increment above `+0.10` triggers a leakage/debugging warning. In
the positive control, target valence and arousal are deliberately encoded as
100-microvolt 10-Hz and 15-Hz sinusoids in the first two EEG channels; a primary
CCC increment below `+0.10` triggers a sensitivity/optimization warning. These
fixed thresholds are diagnostic engineering checks, not hypothesis tests or
empirical AMIGOS findings. All control estimates and warnings are retained.
Failure of a model or seed is reported; it is not removed based on its direction
or magnitude.

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
`0.05`. The two one-sided tests use alpha 0.05 per side: practical equivalence
is declared only if the 90% interval for the primary estimand lies wholly inside
`[-0.05, +0.05]`. A stricter `[-0.025, +0.025]` margin is reported as
sensitivity. The effect estimate also receives a 95% interval, which is not
mislabeled as the standard TOST decision interval. These thresholds are
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
