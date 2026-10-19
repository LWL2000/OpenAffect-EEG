# Model-assisted literature-resource audit (v14)

This directory contains the publishable, non-full-text outputs of two
independently locked model-assisted coding passes over the same 24-paper
access-conditioned cohort. It does not contain the source PDFs or extracted
full text.

Both coders classified 24/24 papers as lacking an eligible learned no-EEG
comparator. Raw agreement is 100% for `no_eeg_comparator` and `overall_parity`;
Cohen's kappa is undefined for these fields because both coders used a single
constant category. Coder A created 60 protocol units and coder B created 57.
The post-lock crosswalk resolves three factual unitization differences and
retains one irreducible granularity choice rather than manufacturing agreement.

These files are model-generated sensitivity evidence. They do not satisfy the
preregistered requirement for two independent human coders, and the cohort is
conditioned on validated public full-text access. The exact 95% interval
[0.858, 1.000] is descriptive for this cohort and is not an unbiased estimate
of field-wide prevalence.

The two lock files hash the immutable source coding tables. Paper-level
agreement is calculated before adjudication. `ADJUDICATION_REPORT.md` and the
two adjudication CSVs document the post-lock decisions without modifying either
coder's original files.
