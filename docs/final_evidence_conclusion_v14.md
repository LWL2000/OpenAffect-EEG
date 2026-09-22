# Final evidence conclusion for upgrades 1--5 (v14)

Checked 2026-09-22 after the guarded server workflow reached
`signed_power_recovery_all`.

## Results that can support the paper

- FACED is a previously untouched cohort for this project: 123 participants,
  3,444 eligible trials, 28 stimuli, 100 crossed splits, and no post-outcome
  exclusions.
- With matched label resources, five-seed EEGNet did not show a reliable
  positive increment: -0.04139, 95% [-0.08747, 0.00017]. Its interval is too
  wide to establish equivalence at either fixed margin.
- Five-seed LaBraM produced -0.00532, 95% [-0.01124, 0.00085], with the 90%
  interval [-0.01025, -0.00011]. This establishes practical equivalence at
  both the fixed +/-0.05 and +/-0.025 margins for this model and estimand.
- The full-grid LaBraM label-permutation control produced -0.00584, 95%
  [-0.00765, -0.00413], did not trigger its +0.10 spurious-gain warning, and
  was equivalent at both margins.
- Redesigned signed-power controls recovered large positive increments for
  EEGNet (+0.65124, 95% [0.59682, 0.78544]) and LaBraM (+0.63032, 95%
  [0.57698, 0.75945]) on the frozen cost-controlled subset. This demonstrates
  that both fitted pipelines can recover a crop-robust injected signal.

## Interpretation boundary

The defensible claim is that EEG benefit estimates depend materially on label
resource matching, and that in FACED the strong LaBraM model's matched
increment is practically negligible under the fixed uniform-grid estimand.
The evidence does not justify the universal claim that EEG is useless. EEGNet
remains statistically inconclusive, the two same-data LaBraM analyses show
negative rather than equivalent increments, and datasets, targets, acquisition
quality, deployment settings, and model families can change the answer.

Both locked sinusoidal positive controls failed. The signed-power controls were
designed after those failures and therefore cannot retroactively restore fully
preregistered confirmation status. They support a specific sign/phase failure
mechanism and validate the redesigned pipeline only as transparent exploratory
diagnostics. The manuscript must report both failures, the amendment timing,
the stopped 52-fit full-grid attempt, and the successful frozen subsets.

## Completion status

The computational parts of upgrades 1--4 are finished and hash-verified. The
pre-registered field-prevalence upgrade is not complete: two independent AI
codings agree that all 24 accessible papers lack an eligible learned no-EEG
comparator, but two real human coders have not independently locked and
adjudicated their files. The accessible cohort is also selection-biased toward
papers with public full text. These AI results may be reported as sensitivity
or triage evidence, not as the planned human prevalence estimate.

Accordingly, all server experiments are complete, but the first five evidence
upgrades as a package are not complete until the two-human audit is finished or
the manuscript explicitly removes the human-prevalence claim and presents the
AI audit as exploratory.
