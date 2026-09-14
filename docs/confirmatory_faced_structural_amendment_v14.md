# FACED structural amendment: common 30-channel montage (v14)

Status: locked after header-only preflight and before EEG samples or rating values
were read  
Amendment date: 2026-09-14 (Asia/Shanghai)  
Parent protocol commit: `98b931f9a47acc586ef0c39c3f25297c8615f20a`

The first complete remote structural preflight revealed two source channel
schemas. Earlier subjects use the legacy temporal names T3, T4, T5, and T6 and
include A1/A2. Later subjects use the corresponding modern temporal names T7,
T8, P7, and P8 and include HEOR/HEOL instead of A1/A2. This observation came
from BDF headers only. No EEG sample or Valence/Arousal value was read or
retained.

The confirmation now uses the fixed intersection of 30 scalp channels. T3, T4,
T5, and T6 are renamed T7, T8, P7, and P8, respectively; A1/A2 and HEOR/HEOL
are excluded. The canonical order is:

`Fp1, Fp2, Fz, F3, F4, F7, F8, FC1, FC2, FC5, FC6, Cz, C3, C4, T7, T8,
CP1, CP2, CP5, CP6, Pz, P3, P4, P7, P8, PO3, PO4, Oz, O1, O2`.

This rule was chosen solely to create the same scalp montage in both acquisition
cohorts. It is applied to every participant before any outcome is opened and
does not depend on signal quality, a model score, or an effect direction. The
frozen tensor shape is therefore 30 by 3,000 rather than 32 by 3,000. All other
processing, folds, resource doses, models, estimands, and equivalence margins
remain unchanged.

The repeated preflight with retry handling found all 123 participants
structurally complete: 61 used the legacy schema and 62 used the modern schema;
68 BDF headers reported 1000 Hz and 55 reported 250 Hz. The corresponding
machine-readable, outcome-free summary is
`docs/faced_remote_preflight_summary_v14.json`. These observed sampling-rate
counts differ from the dataset README overview and will therefore be reported
from the pinned file headers.

The full local signal-only QC then passed all 123 participants and all 3,444
fixed windows. Its outcome-free summary and full-report hash are recorded in
`docs/faced_signal_qc_summary_v14.json`. Six initial checks differed by one
sample because onset and duration decimals were rounded independently. The
implementation now maps the video end to its nearest sample and takes exactly
30 seconds backwards for every trial, which directly implements the frozen
end-anchored window.

For filtering, the two-second pads are deterministic reflections of the fixed
within-video window. No post-video rating-period sample is allowed to enter the
filter padding. This clarification was fixed before target values were read or
any model was fitted.
