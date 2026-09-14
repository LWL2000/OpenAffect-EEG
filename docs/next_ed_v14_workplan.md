# Next E&D submission workplan (v14)

This file tracks the seven evidence upgrades requested after the v13 review.
Passing a software check is not counted as scientific completion.

| Upgrade | Concrete completion criterion | Current status |
|---|---|---|
| Untouched confirmation dataset | Outcome-blind protocol locked; authorised AMIGOS data ingested; all prespecified analyses reported | Protocol locked; EULA/data access pending |
| Strong adapted model and seeds | LaBraM final-four-block adaptation and EEGNet run for five seeds on the confirmation design; failures retained | Specification locked; implementation pending |
| Training uncertainty | Five fold rotations are fully refit; crossed identity and training-seed uncertainty reported for the primary uniform mean | Specification locked; implementation pending |
| Practical equivalence | AMIGOS-only margins fixed before label access; TOST-compatible interval decision reported without retroactive claims | Margins locked at 0.05 primary and 0.025 sensitivity |
| Field prevalence | 20--30-paper sampling frame and coding manual frozen; two independent human coders complete masked coding; agreement and adjudication reported | Sampling/coding protocol pending |
| Independent artifact reproduction | A person uninvolved in implementation runs the public quickstart and signs a factual environment/time/failure report | Requires external person; template pending |
| Manuscript revision | Contribution type, abstract, main figure, results, limitations, and artifact version all match completed evidence | Deferred until evidence freezes |

## Integrity rules

- Existing datasets cannot be relabelled as untouched confirmation.
- Agents, repeated clean installs by the authors, and automated tests cannot be
  described as independent human reproduction.
- Two model runs or two automated coders cannot be described as two independent
  human literature coders.
- Probability estimates are planning judgements and never manuscript evidence.
- A conflicting AMIGOS result changes the conclusion; it is not a reason to
  switch datasets or suppress the run.

