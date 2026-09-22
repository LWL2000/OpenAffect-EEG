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

## Completed amended-control result (2026-09-19)

The residual-aligned sinusoidal control also failed its diagnostic. Its
uniform-grid mean matched CCC increment was -0.05623, with a joint 95% interval
of [-0.11926, 0.02672] and a 90% interval of [-0.10587, 0.02459], below the
required +0.10 increment. All 2,000 requested joint bootstrap iterations were
valid. The prediction SHA-256 is
`70e514d7fca135516fc97aa6195a6c84ed9adaf6cf4065d89e40aa9fc9dfd36c`; the
cell-table SHA-256 is
`00c865764841dba5b08d6b0714ad1539e97483477c69679a67ef03e7ad1000e5`.

This second failure shows that raw-target versus residual-target alignment was
not a sufficient explanation for the original failure. One plausible unresolved
mechanism is that signed sinusoidal amplitude is confounded with carrier phase
under random cropping and the model's temporal invariances. That mechanism is a
diagnostic hypothesis, not an established cause. The guarded recovery driver
therefore stopped before the LaBraM controls. No FACED neural result is treated
as confirmed, and any further redesigned control must be labelled exploratory
and retain both failed controls.

## Threshold-attainability diagnostic

A deterministic oracle check then replaced the learned residual with the exact
outcome-minus-prior residual while preserving the frozen assignments,
calibration, resource grid, and scoring code. Its uniform-grid mean matched CCC
increment was +0.61761; a 200-iteration diagnostic bootstrap gave a 95% interval
of [0.58107, 0.67126]. Thus the +0.10 threshold is attainable under the scoring
pipeline. The failed neural controls cannot be explained by an impossible
threshold or by the downstream CCC calculation. This oracle repeats one
deterministic prediction under five seed labels, so it is a pipeline diagnostic,
not evidence about training uncertainty or EEG information.

## Exploratory signed-power pilot

After the two full sinusoidal controls failed, a separately labelled one-cell
pilot encoded positive and negative residual magnitudes in different carrier
channels and replaced the first four input channels. This removes the
sign-versus-phase ambiguity under random cropping. On split 0, stimulus dose 8,
and training seed 2026091401, validation-selected EEGNet reached residual MAE
0.03407. Its personalized combined macro CCC was 0.99611 at participant dose 0
and 0.99614 at participant dose 8, versus prior-only CCCs of 0.34070 and
0.52684. This supports the sign/phase mechanism and shows that the training
pipeline can recover a crop-robust injected signal in one cell. It remains an
exploratory post-failure diagnostic; it is neither a full-grid result nor a
replacement for either failed control.

## Cost-controlled diagnostic subset (2026-09-20)

The initially launched signed-power full-grid recovery was stopped after 52 of
5,000 EEGNet candidate fits. Successful signal recovery required roughly 96
epochs per fit, compared with about seven epochs before early stopping in the
failed controls. The observed throughput was about 20 fits per hour, implying
roughly ten days for EEGNet alone. The 52 partial fits are retained as an
incomplete computational record and are excluded from inference.

Before restarting or inspecting any subset-level outcome, a cost-controlled
exploratory diagnostic was fixed: split indices 0, 20, 40, 60, and 80; stimulus
doses 0, 2, and 8; all five participant doses; and all five training seeds.
This gives 150 EEGNet candidate fits across two learning rates and 75 LaBraM
fits. The subset samples every fold rotation and the lower, middle, and upper
stimulus-resource regimes. It tests whether the successful one-cell pilot
generalizes across rotations, doses, and seeds, but it is not a replacement for
the locked full 5x5 confirmation grid and cannot support a confirmatory claim.

The EEGNet subset completed all 150 candidate fits and selected 75 fits. The
uniform mean matched CCC increment was +0.65124, with a joint 95% interval of
[0.59682, 0.78544]; 1,999 of 2,000 bootstrap iterations were valid. It therefore
passed the pre-run +0.10 diagnostic gate and supports the proposed sign/phase
failure mechanism across the sampled rotations, doses, and seeds. The prediction
SHA-256 is `c2a4fb48e6dea4cc0b3db21bd1f6ed3765f40e246b9f59ac828d8bcb0e7d57cc`;
the cell-table SHA-256 is
`972dfdbc97014b9492bbf042bec013dbea93c52fb0afc63a0bdcfb7a21d8d606`.
This remains an exploratory pipeline diagnostic and does not repair the failed
locked positive controls or change the status of the observed scientific result.

## Completed LaBraM controls (2026-09-22)

The full-grid LaBraM label-permutation control completed all 2,500 fits across
100 splits, five stimulus doses, and five training seeds. Its uniform mean
matched CCC increment was -0.00584, with a joint 95% interval of
[-0.00765, -0.00413] and a 90% interval of [-0.00736, -0.00442]. The
prespecified spurious-gain warning did not trigger, and the interval lies within
both equivalence margins. All 2,000 bootstrap iterations were valid. The
prediction SHA-256 is
`84028e0dd5cfc76ec4025f378f4b2a11622a5e22d7a7528b372544f2b56c8499`;
the cell-table SHA-256 is
`da25e99b827721450d3d6a934618cf7d83974c05944c3eecd921b30ff3fbd8372`.

The cost-controlled LaBraM signed-power subset completed all 75 fits. Its
uniform mean matched CCC increment was +0.63032, with a joint 95% interval of
[0.57698, 0.75945]; 1,999 of 2,000 bootstrap iterations were valid. It passed
the fixed +0.10 diagnostic gate. The prediction SHA-256 is
`f29a170d927f8a9bd5801950a1fba23bc685d91af34ca9c164dece67f5947d3a`;
the cell-table SHA-256 is
`7ad89211d566d50104da2feeed2790ceff24a1c9eb94fc09702a5da88c4e5de2f`.

Together with the EEGNet signed-power result, this shows that both neural
pipelines can recover the redesigned crop-robust signal across the sampled
rotations, doses, and seeds. It strengthens the sign/phase explanation for the
two locked-control failures, but remains post-failure exploratory evidence. It
does not retroactively turn either failed sinusoidal control into a pass or make
the FACED observed result fully preregistered confirmation.
