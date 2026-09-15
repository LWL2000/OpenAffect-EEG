# Evidence status for the first five upgrades (v14)

Checked 2026-09-15. A computation launch or prepared form is not counted as a
completed scientific result.

| Upgrade | Verified evidence | Remaining completion condition |
|---|---|---|
| Untouched FACED confirmation | FACED v1.1.3; 123 participants; 3,444 eligible trials; no exclusions; 100 crossed splits; 2,500 resource cells; 430,500 Ridge OOF rows; 50-candidate EEGNet smoke complete | Finish and report every prespecified EEGNet/LaBraM observed and control run |
| Strong models and five seeds | ds005540 and ds006850 LaBraM final-four-block adaptation complete: 125 fits and five seeds per dataset, with no failed-fit files | Finish the FACED five-seed EEGNet and LaBraM sequence |
| Training uncertainty | Joint participant/stimulus, executed-seed, and supplied-rotation bootstrap code validated; same-data LaBraM analyses complete | Apply it to all completed FACED neural predictions and retain failure/model-selection histories |
| Practical equivalence | CCC margins 0.05 and 0.025 were fixed before FACED label access; 90% decision interval is implemented | Report the FACED decision. Same-data LaBraM is inconclusive at both margins and cannot substitute for FACED |
| Field audit | Original sampling flow retained; 24-paper public-access cohort with four papers per year validated; 24 PDF/text hashes and evidence packets created; two independent AI sensitivity codings completed and hash-locked | Harmonize the post-lock evaluation units, then have two real people independently code and lock both CSV pairs; compare agreement and adjudicate disagreements |

## Same-data LaBraM results

These are sensitivity evidence on the two existing datasets, not confirmation
cohorts and not a field-wide conclusion.

| Dataset | Uniform-grid mean matched EEG CCC increment | 95% interval | 90% equivalence interval | Decision at ±0.05 / ±0.025 |
|---|---:|---:|---:|---|
| ds005540 | -0.04127 | [-0.06902, -0.01120] | [-0.06549, -0.01628] | inconclusive / inconclusive |
| ds006850 | -0.04178 | [-0.07553, -0.00644] | [-0.06975, -0.01237] | inconclusive / inconclusive |

The negative intervals show worse CCC after adding the fitted LaBraM residual in
these two analyses. They do not establish practical equivalence and they do not
predict the untouched FACED outcome.

## Field-audit boundary

Only 13 of the prespecified 36 main/reserve records had a validated public full
text after two dated attempts. The 24-paper codable cohort therefore adds the
next eligible records with validated index-declared public PDFs in the frozen
within-year order. This makes the coder task feasible while creating access
bias. Both the intent-to-sample access attrition and the access-conditioned
cohort must be reported; the latter cannot be called an unbiased prevalence
estimate for all EEG affect papers.

## Dual-AI sensitivity audit

Two independent AI coders reviewed the same 24-paper accessible cohort without
seeing each other's judgments before both files were hash-locked. Both classified
24/24 papers as lacking an eligible learned no-EEG comparator. This is useful
triage evidence, not completion of the preregistered two-human-coder audit. The
evaluation-unit definitions also require post-lock harmonization: coder A made
57 dataset/protocol rows and coder B made 59. Until that mapping is adjudicated,
evaluation-level kappa is not reported.
