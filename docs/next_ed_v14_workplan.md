# Next E&D submission workplan (v14)

This file tracks the seven evidence upgrades requested after the v13 review.
Passing a software check is not counted as scientific completion.

| Upgrade | Concrete completion criterion | Current status |
|---|---|---|
| Untouched confirmation dataset | Outcome-blind protocol locked; pinned FACED v1.1.3 data ingested; all prespecified analyses reported | Access-only amendment locked; 618 source files (31.41 GiB) checksummed outside repository; 123/123 participants and 3,444 windows passed outcome-blind structural/signal QC; frozen ingestion next |
| Strong adapted model and seeds | LaBraM final-four-block adaptation and EEGNet run for five seeds on the confirmation design; failures retained | Resumable LaBraM, validation-selected EEGNet, and Ridge runners tested; five-seed same-data LaBraM grid running; FACED runs pending ingestion |
| Training uncertainty | Five fold rotations are fully refit; crossed identity and training-seed uncertainty reported for the primary uniform mean | Joint identity/seed/rotation analysis tested; same-data outputs pending; FACED rotations pending ingestion |
| Practical equivalence | FACED-only margins fixed before label access; TOST-compatible interval decision reported without retroactive claims | Margins locked at 0.05 primary and 0.025 sensitivity; alpha 0.05 TOST uses a 90% decision interval |
| Field prevalence | 20--30-paper sampling frame and coding manual frozen; two independent human coders complete masked coding; agreement and adjudication reported | Protocol locked; 24 main and 12 reserve papers selected; six main full texts and neutral evidence packets acquired; remaining full texts and two human coding locks pending |
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
