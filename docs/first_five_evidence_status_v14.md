# Evidence status for the first five upgrades (v14)

Checked 2026-09-22. A computation launch or prepared form is not counted as a
completed scientific result.

| Upgrade | Verified evidence | Remaining completion condition |
|---|---|---|
| Untouched FACED confirmation | FACED v1.1.3; 123 participants; 3,444 eligible trials; no exclusions; 100 crossed splits; 2,500 resource cells; observed and label-permutation EEGNet/LaBraM stages complete | Both locked sinusoidal EEGNet positive controls failed. The redesigned positive controls passed only as post-failure exploratory subsets, so the cohort is untouched but the complete pipeline cannot be called fully preregistered confirmation |
| Strong models and five seeds | ds005540 and ds006850 LaBraM final-four-block adaptation complete; FACED EEGNet and LaBraM observed stages, full-grid LaBraM label permutation (2,500 fits), and LaBraM signed-power subset (75 fits) completed with five seeds and no failed records | Computational criterion complete; interpretation retains the post-outcome-control limitation |
| Training uncertainty | Joint participant/stimulus, executed-seed, and supplied-rotation bootstrap complete for all reported FACED stages; 2,000/2,000 valid iterations for LaBraM label permutation and 1,999/2,000 for each signed-power subset | Computational criterion complete; arbitrary architecture and tuning-policy uncertainty remain outside the estimand |
| Practical equivalence | Fixed CCC margins 0.05 and 0.025 applied without retroactive changes; FACED EEGNet observed is inconclusive; FACED LaBraM observed and label permutation are equivalent at both margins | Statistical criterion complete; conclusions must remain model-specific |
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

## Interim FACED EEGNet observed result

The five-seed observed EEGNet stage completed on 2026-09-16. The uniform-grid
mean matched EEG CCC increment is -0.04139, with a joint 95% interval of
[-0.08747, 0.00017]. The prespecified 90% equivalence interval is
[-0.07962, -0.00010], so equivalence is not established at either the primary
±0.05 margin or the strict ±0.025 sensitivity margin. All 2,000 requested
bootstrap iterations were valid and included 123 participants, 28 stimuli,
five training seeds, and five supplied fold rotations. The cell-table SHA-256
is `c8d2ce8d413dce2b5062a3f22180bf2bf16f44859952b8dc25c6200fbd8faaf0`.

This is an interim model-stage result. The synthetic-signal control and all
LaBraM stages remain required before the FACED confirmation is interpreted as
a whole.

The EEGNet label-permutation control subsequently completed with a uniform-grid
mean increment of -0.03995, a joint 95% interval of [-0.08564, 0.00021], and a
90% interval of [-0.07667, 0.00004]. Its prespecified diagnostic did not trigger
because the increment did not exceed +0.10. The close observed and permuted
point estimates are descriptive until the positive synthetic-signal control and
LaBraM stages establish whether the pipeline can recover injected information.
The label-permutation cell-table SHA-256 is
`a4dd4784fb106110e417f81de9f5b388ff4098d7a93e0034d0ca14760d89326a`.

The original EEGNet synthetic-signal control failed its required diagnostic:
the increment was -0.14755, with a 95% interval of [-0.23976, 0.05729], rather
than the required value above +0.10. Inspection showed that it encoded raw
outcomes although the neural model learns residual outcomes and is added back
to the population prior. The failed result is retained. A dated post-outcome
amendment adds a separately named residual-aligned control; see
`docs/confirmatory_positive_control_amendment_v14.md`. Until that amended
control succeeds, the FACED neural result is not treated as confirmed.

The residual-aligned EEGNet control completed on 2026-09-19 and also failed:
its mean increment was -0.05623, with a joint 95% interval of
[-0.11926, 0.02672], rather than the required value above +0.10. All 2,000
bootstrap iterations were valid. This rules out target alignment as a
sufficient explanation for the first failure. The guarded sequence stopped
before the LaBraM controls, and both EEGNet control failures remain part of the
evidence record.

A deterministic oracle-residual diagnostic produced a +0.61761 mean increment
(200-iteration diagnostic 95% interval [0.58107, 0.67126]), showing that the
+0.10 threshold is attainable under the unchanged calibration and scoring
pipeline. This localizes the unresolved failure to learned recovery or signal
construction rather than to an impossible downstream metric threshold.

A subsequent exploratory one-cell signed-power pilot removed the carrier
sign/phase ambiguity and succeeded. Frozen cost-controlled subsets then passed
for both architectures. EEGNet produced +0.65124 (95% [0.59682, 0.78544]) from
150 candidate fits; LaBraM produced +0.63032 (95% [0.57698, 0.75945]) from 75
fits. These results show that both pipelines recover the redesigned signal, but
remain post-failure exploratory diagnostics rather than replacements for the
failed locked controls.

## Interim FACED LaBraM observed result

The five-seed LaBraM observed stage completed on 2026-09-19. The uniform-grid
mean matched EEG CCC increment is -0.00532, with a joint 95% interval of
[-0.01124, 0.00085]. The prespecified 90% equivalence interval is
[-0.01025, -0.00011], establishing equivalence at both the primary ±0.05 margin
and the strict ±0.025 sensitivity margin. All 2,000 requested bootstrap
iterations were valid and included 123 participants, 28 stimuli, five training
seeds, and five supplied fold rotations. The prediction SHA-256 is
`1dd155dbca9b6bd3e1d8c34ed514db78ec6b8dbb92da3ba8f298caa7985dd0231`; the
cell-table SHA-256 is
`5cbce648f8f7269ee9a8165d3e76ac5c0ff3c81353c923115acdb5604df212cd`.

The full-grid LaBraM label-permutation control also completed. Its mean matched
increment was -0.00584, with a joint 95% interval of [-0.00765, -0.00413] and
a 90% interval of [-0.00736, -0.00442]. It did not trigger the spurious-gain
warning and was equivalent at both margins. All 2,000 bootstrap iterations were
valid. Its prediction SHA-256 is
`84028e0dd5cfc76ec4025f378f4b2a11622a5e22d7a7528b372544f2b56c8499` and its
cell-table SHA-256 is
`da25e99b827721450d3d6a934618cf7d83974c05944c3eecd921b30ff3fbd8372`.

The observed-model result is therefore supported by a well-behaved negative
control and a successful redesigned positive diagnostic across two
architectures. The remaining limitation is temporal: the successful positive
diagnostic was designed after both locked controls failed, so the evidence is
stronger than an unvalidated null result but weaker than a fully preregistered
confirmation.

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
