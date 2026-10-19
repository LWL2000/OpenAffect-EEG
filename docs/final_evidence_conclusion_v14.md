# Final evidence conclusion for the v14 evidence upgrades

Checked 2026-10-19 after the guarded server workflow reached
`signed_power_recovery_all`, the model-assisted field audit was locked, and the
controlled public artifact was rerun in a clean environment.

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

The computational parts of upgrades 1--4 are finished and hash-verified. Two
new independently locked model-assisted coding passes and a separate post-lock
adjudication are also complete. Both passes agree that all 24 accessible papers
lack an eligible learned no-EEG comparator. Paper-level raw agreement is 100%
for comparator presence and overall parity, but protocol unitization differs
(60 versus 57 rows), and one target-granularity choice remains deliberately
unresolved. The accessible cohort is selection-biased toward papers with public
full text. These results may be reported as an exploratory sensitivity audit,
not as the preregistered two-human prevalence estimate.

Accordingly, all server experiments are complete. The manuscript now follows
the second permissible path: it removes the human-prevalence claim and labels
the model-assisted audit as exploratory. The original two-human completion
criterion remains unmet and must not be claimed.

## Artifact and cross-domain evidence

A separately instructed model operator cloned the public branch, created a
fresh Python 3.12 environment, and ran the reviewer quickstart. The first run
correctly exposed 29 stale records in the old 280-file release manifest. After
the repository's controlled release builder regenerated the current 365-file
boundary, all six acceptance checks and 16 targeted tests passed, the manifest
had zero mismatches, and a second build was byte-identical. The acceptance
packet is deliberately labelled `author_assisted`: this is a useful independent
execution path, but it is not an outside-person or independent-human
reproduction and it does not reproduce full EEG training.

The manuscript now states the cross-domain contribution as an identification
argument. A component-specific gain is identified by comparing systems that
hold the remaining resource contract fixed; changing both the component and
resources confounds the component effect with the resource change and their
interaction. Primary-source examples from multimodal learning, active learning,
personalized federated learning, domain adaptation, and vision--language
pretraining show that the same comparison issue matters beyond EEG. These
examples support transfer of the evaluation template, not a prevalence claim
about those fields.
