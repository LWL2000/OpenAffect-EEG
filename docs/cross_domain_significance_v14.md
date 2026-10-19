# Cross-domain significance of resource-matched attribution (v14)

Checked 2026-10-19. This note states the broad claim that the evidence supports
and the stronger claims that it does not support.

## Identification claim

Let a system score be written as `S(M, R; T)`, where `M` is the component whose
added value is being claimed, `R` is the remaining resource contract, and `T`
is fixed evaluation support. The paired increment

`S(M=1, R; T) - S(M=0, R; T)`

attributes a score difference to the addition of `M` only for that declared
contract. If the two systems instead use `R1` and `R0`, the observed contrast

`S(1, R1; T) - S(0, R0; T)`

combines the component change with the resource change and their interaction.
No algebraic subtraction can identify the component contribution without an
additional assumption or an explicit factorial design. This is an estimand and
comparison-design statement, not a claim that the component is causally useless.

OpenAffect-EEG instantiates this principle for EEG by matching target-user and
repeated-stimulus labels, holding test support and population-training size
fixed, and reporting the remaining paired EEG increment. FACED demonstrates
that the distinction is consequential in one real setting: the adapted LaBraM
increment is practically equivalent to zero under the fixed resource-grid
estimand, while EEGNet remains inconclusive.

## Transferable evaluation template

| Setting | Component claim | Resources that must be declared or controlled | What an unmatched contrast can support |
|---|---|---|---|
| Multimodal prediction | Added value of a modality or cross-modal interaction | Training examples and labels, target-user feedback, preprocessing, model selection, test support | Performance of the complete system package |
| Personalized or federated learning | Added value of the personalization rule | Local labelled examples, client participation, global pretraining, tuning and validation access | Benefit of algorithm plus local-data access |
| Active and semi-supervised learning | Added value of a query or learning strategy | Annotation budget, unlabelled pool, annotator policy, pretraining, stopping and tuning budget | Utility under the combined acquisition-and-learning policy |
| Domain adaptation | Added value of the adaptation mechanism | Target-domain labels, in-domain pretraining data, source data, compute and tuning budget | Utility of the full adaptation package |
| Foundation-model comparison | Added value of architecture or adaptation | Pretraining corpora, supervision, model-selection budget, retrieval tools and task labels | End-to-end system performance under unequal upstream resources |

The equality requirement depends on the scientific question. If the claim is
that an active-learning policy acquires better examples, selected labels are
part of the treatment and the relevant invariant is annotation cost, not label
identity. If the claim is that an encoder adds information after calibration,
calibration labels belong in the shared contract. Resource matching therefore
does not require every benchmark to equalize every resource; it requires the
comparator to hold fixed the resources outside the component named in the
attribution claim.

## Evidence connecting the template to other fields

- Hessel and Lee (2020) showed that higher multimodal performance need not
  identify cross-modal interaction and recommended diagnostics beyond unimodal
  baselines. This is the same attribution problem at a different layer: a system
  gain and a component-specific gain are distinct estimands.
- Bugliarello et al. (2021) found in controlled vision-language experiments
  that training data and hyperparameters explained most reported differences
  between several architectures. This provides empirical precedent for treating
  data and selection resources as part of the comparison contract.
- Hacohen et al. (2022) showed that the preferred active-learning strategy
  depends on the labelled-data budget. Bai et al. (2021) likewise evaluated
  domain adaptation as allocation between annotation and pretraining under a
  constrained budget. These works support reporting resource-response curves
  rather than a single score detached from the acquisition budget.
- Marfoq et al. (2022) explicitly held the global model fixed when varying local
  datastore size to isolate that resource's effect in personalized federated
  learning. This is a direct example of the same controlled-comparison logic.

## Scope boundary

The FACED experiment does not establish that resource mismatch is common in
all multimodal, personalized, active-learning, domain-adaptation, or foundation-
model papers. It establishes that resource matching can reverse or eliminate an
apparent component attribution in the audited EEG setting. The cross-domain
contribution is the reusable estimand, contract, and audit procedure. Any claim
about prevalence outside the 24-paper EEG access cohort requires a separate,
representative audit.

## Sources

- Hessel, J. and Lee, L. (2020). *Does my multimodal model learn cross-modal
  interactions? It's harder to tell than you might think!* EMNLP.
  https://doi.org/10.18653/v1/2020.emnlp-main.62
- Bugliarello, E. et al. (2021). *Multimodal Pretraining Unmasked*. TACL.
  https://doi.org/10.1162/tacl_a_00408
- Hacohen, G. et al. (2022). *Active Learning on a Budget*. ICML.
  https://proceedings.mlr.press/v162/hacohen22a.html
- Bai, F. et al. (2021). *Pre-train or Annotate? Domain Adaptation with a
  Constrained Budget*. EMNLP. https://doi.org/10.18653/v1/2021.emnlp-main.409
- Marfoq, O. et al. (2022). *Personalized Federated Learning through Local
  Memorization*. ICML. https://proceedings.mlr.press/v162/marfoq22a.html
