# Next E&D submission workplan (v14)

This file tracks the seven evidence upgrades requested after the v13 review.
Passing a software check is not counted as scientific completion.

| Upgrade | Concrete completion criterion | Current status |
|---|---|---|
| Untouched confirmation dataset | Outcome-blind protocol locked; pinned FACED v1.1.3 data ingested; all prespecified analyses reported | Dataset ingestion, assignments, observed five-seed EEGNet and LaBraM stages, and full-grid label-permutation controls are complete. Both locked sinusoidal EEGNet positive controls failed and are retained. Post-failure signed-power subsets passed for EEGNet (+0.65124, 95% [0.59682, 0.78544]) and LaBraM (+0.63032, 95% [0.57698, 0.75945]), but are exploratory and do not restore fully preregistered confirmation status |
| Strong adapted model and seeds | LaBraM final-four-block adaptation and EEGNet run for five seeds on the confirmation design; failures retained | Computationally complete. Same-data LaBraM completed on ds005540 and ds006850; FACED EEGNet and LaBraM observed stages used five seeds. Full-grid LaBraM label permutation completed 2,500 fits; its signed-power subset completed 75 fits with five seeds and no failed records |
| Training uncertainty | Five fold rotations are fully refit; crossed identity and training-seed uncertainty reported for the primary uniform mean | Computationally complete for all reported FACED observed and control results. LaBraM label permutation used 2,000/2,000 valid joint replicates; signed-power subsets used 1,999/2,000 for each architecture. The boundary still excludes arbitrary architecture and tuning-policy uncertainty |
| Practical equivalence | FACED-only margins fixed before label access; TOST-compatible interval decision reported without retroactive claims | Complete for the fixed estimands. FACED EEGNet observed remains inconclusive. FACED LaBraM observed estimate is -0.00532, 95% [-0.01124, 0.00085], with equivalence at ±0.05 and ±0.025. LaBraM label permutation is also equivalent: -0.00584, 95% [-0.00765, -0.00413]. The post-failure positive-control results are diagnostics rather than equivalence evidence about real EEG |
| Field prevalence | 20--30-paper sampling frame and coding manual frozen; two independent human coders complete masked coding; agreement and adjudication reported | Original 24 main/12 reserve flow retained; 24 validated public full texts (four/year), 24 evidence packets, two isolated coder archives, hash-lock validator, agreement analysis, and adjudication template complete. Two real independent coder locks remain required; the public-access cohort is explicitly access-conditioned |
| Independent artifact reproduction | A person uninvolved in implementation runs the public quickstart and signs a factual environment/time/failure report | Instructions, structured form, and packet validator implemented; actual outside-person run pending |
| Manuscript revision | Contribution type, abstract, main figure, results, limitations, and artifact version all match completed evidence | Deferred until evidence freezes |

## Integrity rules

- Existing datasets cannot be relabelled as untouched confirmation.
- Agents, repeated clean installs by the authors, and automated tests cannot be
  described as independent human reproduction.
- Two model runs or two automated coders cannot be described as two independent
  human literature coders.
- Probability estimates are planning judgements and never manuscript evidence.
- A conflicting FACED result changes the conclusion; it is not a reason to
  switch datasets or suppress the run.
