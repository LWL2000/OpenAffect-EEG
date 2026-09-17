# FACED residual-positive-control amendment (v14)

Status: dated post-outcome diagnostic amendment, 2026-09-17

The originally locked synthetic EEG positive control added sinusoids whose
amplitudes encoded the raw valence and arousal outcomes. The neural estimator,
however, was trained to predict residual outcomes after subtracting the
stimulus-population prior, and its output was added back to that prior during
evaluation. Encoding the raw outcome can therefore make a successful learner
double-count the prior. The original unit test checked signal construction but
did not check alignment with the residual training objective.

This mismatch was discovered because the completed EEGNet positive control
triggered its prespecified warning. Its uniform-grid mean matched CCC increment
was -0.14755, with a 95% interval of [-0.23976, 0.05729], below the required
+0.10 increment. The failed result and its hashes are retained; it must not be
relabelled as a successful positive control.

The amended diagnostic is named `synthetic_residual_signal`. For each frozen
split and stimulus-resource dose, it encodes exactly the residual targets used
for training and validation. Other trials encode the outcome minus the
fit-only population prior used during scoring. This amendment does not change
the FACED data, assignments, observed predictions, label-permutation control,
resource grid, model selection, equivalence margins, or inferential code.

Because the defect was identified after FACED outcomes and the original control
result were visible, the amended control is a transparent post-outcome pipeline
diagnostic. It cannot retroactively convert the observed EEGNet result into a
confirmed scientific finding. Both the failed original control and the amended
control result will be reported. Confirmatory interpretation remains withheld
until the amended control recovers the injected residual information and the
LaBraM stages are complete.
