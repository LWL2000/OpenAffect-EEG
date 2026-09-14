# Outcome-blind FACED confirmation protocol (v14)

Status: locked before trial-level outcomes or EEG samples were accessed
Lock date: 2026-09-14 (Asia/Shanghai)
Parent repository commit: `1832b6bfb198de11236aa30c514a420dd4644421`
Machine-readable companion: `configs/confirmatory_faced_v14.yaml`

The outcome-blind header preflight subsequently exposed two acquisition-cohort
channel schemas. The dated, pre-outcome harmonisation in
`docs/confirmatory_faced_structural_amendment_v14.md` supersedes only the source
channel selection and tensor channel count below.

## Question, units, and outcome-blind status

The study tests whether matching participant- and stimulus-label resources
changes the attributed increment from EEG on a dataset that did not contribute
trial-level data to OpenAffect-EEG's design, tuning, or diagnosis. The independent
units are participants and video identities; repeated participant-video trials
are not treated as independent replicates. Outcomes are participant-specific
Valence and Arousal ratings.

Before this lock, only public descriptions, the BIDS schema, directory/file
metadata, channel metadata, and the names and descriptions of event columns were
read. No event row, self-rating value, EEG sample, FACED prediction, or FACED
effect estimate was inspected. Existing published FACED literature is known.
All results, including a positive EEG increment or dataset heterogeneity, will
be retained.

## Pinned source and eligibility

The sole confirmatory source is NEMAR dataset `nm000112`, version `v1.1.3`, DOI
`10.82901/nemar.nm000112`, under CC-BY-4.0. Files are accepted only from the
versioned path `https://data.nemar.org/nm000112/v1.1.3/`; a later `latest`
version is not silently substituted. Manifest and used-file SHA-256 values are
archived with the run.

A participant is eligible when the BDF is readable by MNE without repair, the
declared cohort channel schema and sampling rate are available, and exactly one valid
presentation is recoverable for every `video_index` 1--28. A trial requires a
finite onset/duration, at least 30 seconds of signal ending at video offset,
finite participant-specific Valence and Arousal, and no non-finite selected EEG
sample. Participants missing any of the 28 common stimuli are excluded from the
primary complete-support analysis. No missing label is imputed.

Structural QC first reads file hashes, BDF headers, channel names, sampling
rates, recording bounds, event column names, event counts, video-index
multiplicity, and rating missingness flags without retaining or summarising
rating values. A QC report and its hash are written before outcomes are unsealed.
If fewer than 25 complete participants remain, this dataset cannot supply the
prespecified confirmation and no replacement is selected after inspecting its
outcomes.

## Frozen signal and target processing

For every eligible presentation, select the last 30.0 seconds ending at the
video offset. Load the harmonised 30 scalp channels in the amended order, convert MNE's volt
output to microvolts, subtract the instantaneous average across channels, apply
a fourth-order zero-phase Butterworth bandpass at 0.5--45 Hz using a two-second
signal pad on each side where recording bounds allow, and resample to 100 Hz
with polyphase anti-alias filtering. The retained tensor is 32 by 3,000.

The retained tensor is 30 by 3,000. The signal-only scale check requires the median within-trial channel standard
deviation after conversion to fall in `[0.1, 1000]` microvolts. A nonconforming
file stops before label access and requires a dated, source-supported amendment;
it is not automatically rescaled based on model performance. There is no
outcome-dependent artefact rejection, ICA component selection, clipping, or
trial deletion.

The documented 0--7 Valence and Arousal ratings are mapped to `[-1, 1]` by
`(rating - 3.5) / 3.5`. Values outside `[0, 7]` invalidate the affected
participant under the complete-support rule. The deterministic Ridge view uses
per-channel log10 Welch power at 100 Hz with two-second windows and bands 2--4,
4--8, 8--13, 13--30, and 30--45 Hz, adding `1e-12` before the logarithm.

## Fixed-support design

The resource grid is participant-label doses `0, 1, 2, 4, 8` crossed with
same-stimulus other-participant doses `0, 1, 2, 4, 8`. Every EEG predictor is
paired with an EEG-free predictor receiving exactly the same labels. Population
training row count remains constant by replacement.

Five participant folds and four stimulus folds are formed from hashed identities.
All 20 cross-products cover each eligible participant-video trial once as test
support. For each split, population training excludes test participants and
stimuli; participant calibration uses that participant's non-test stimuli;
stimulus calibration uses other participants for that test stimulus; validation
contains neither test identity. Five independently hashed fold rotations require
complete refitting. Base assignment seed is `20260914`; training seeds are
`2026091401` through `2026091405`.

## Models, controls, estimand, and inference

The predictors are the matched EEG-free comparator, band-power Ridge with
validation-selected alpha in `0.1, 1, 10, 100`, validation-selected EEGNet, and
LaBraM with the last four transformer blocks, final normalisation, and regression
head trainable. EEGNet and LaBraM each use the five frozen training seeds. Test
labels never select preprocessing, checkpoints, epochs, or hyperparameters.

The full design receives the frozen within-partition label-permutation negative
control and synthetic 10/15-Hz EEG positive control. Failures and warnings are
reported rather than removed.

For cell `(p,s)`, the estimand is
`Delta(p,s) = CCC(EEG + matched labels) - CCC(matched labels only)` on identical
out-of-fold trials. The primary estimand is the uniform mean over 25 cells and
the two targets. Its interval crosses participants and stimuli and resamples the
five training seeds and five full-refit rotations with at least 2,000 draws.
Cellwise results use a simultaneous max-absolute-deviation 95% band.

Practical equivalence uses the pre-outcome absolute CCC margin 0.05 with a 90%
TOST decision interval; 0.025 is a sensitivity margin, and a separate 95%
estimation interval is reported. An interval merely containing zero is not
called equivalence. These margins apply confirmatorily only to FACED.

There is no sign-based early stopping. All eligible trials, grid cells, models,
seeds, rotations, exclusions, failures, and contradictory findings are reported.
