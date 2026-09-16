# Next E&D submission workplan (v14)

This file tracks the seven evidence upgrades requested after the v13 review.
Passing a software check is not counted as scientific completion.

| Upgrade | Concrete completion criterion | Current status |
|---|---|---|
| Untouched confirmation dataset | Outcome-blind protocol locked; pinned FACED v1.1.3 data ingested; all prespecified analyses reported | Access-only amendment locked; 618 source files (31.41 GiB) checksummed outside repository; 123 participants and 3,444 trials passed QC/ingestion with no exclusions; 100 splits and 2,500 resource cells verified; five-seed EEGNet observed and label-permutation stages complete; synthetic-signal is active and LaBraM remains |
| Strong adapted model and seeds | LaBraM final-four-block adaptation and EEGNet run for five seeds on the confirmation design; failures retained | Same-data LaBraM completed on ds005540 and ds006850 (125 fits and five seeds each); FACED EEGNet observed and label-permutation completed with five seeds; synthetic-signal is active, followed resumably by all LaBraM stages |
| Training uncertainty | Five fold rotations are fully refit; crossed identity and training-seed uncertainty reported for the primary uniform mean | Same-data five-seed LaBraM joint analyses complete; FACED EEGNet observed and label-permutation analyses each completed with 2,000/2,000 valid joint bootstrap replicates; remaining FACED stages pending |
| Practical equivalence | FACED-only margins fixed before label access; TOST-compatible interval decision reported without retroactive claims | Margins locked at 0.05 primary and 0.025 sensitivity; alpha 0.05 TOST uses a 90% decision interval. FACED EEGNet observed estimate is -0.04139, 95% interval [-0.08747, 0.00017], and 90% interval [-0.07962, -0.00010], inconclusive at both margins; controls and LaBraM remain |
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
