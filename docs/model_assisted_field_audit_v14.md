# Model-assisted field audit conclusion (v14)

Checked 2026-10-19 against two independently locked coding passes and a
separate post-lock adjudication.

## Result

- Both independent model-assisted coders classified 24/24 papers in the
  access-conditioned cohort as having no eligible learned no-EEG comparator.
- Raw agreement was 100% for `no_eeg_comparator` and `overall_parity`.
  Cohen's kappa is undefined for these fields because both coders used one
  constant category.
- The descriptive Clopper--Pearson 95% interval for 24/24 within this cohort is
  [0.858, 1.000]. It must not be presented as a representative field-prevalence
  interval.
- Paper-level claim type agreed for 23/24 papers (95.8%); confidence agreed for
  20/24 (83.3%, kappa 0.484). All five paper-level disagreements were recorded
  and adjudicated after both source files were locked.
- Coder A created 60 paper--dataset--protocol units and coder B created 57.
  Four papers differed in unit count. Post-lock comparison found factual
  omissions or material protocol splits for P011, P014, and P032. For X2503,
  the coding manual does not uniquely determine whether two DEAP targets should
  be one or two units, so both granularities remain visible and no evaluation-
  level kappa is reported.

## Defensible manuscript statement

> In a protocol-locked sensitivity audit of 24 EEG-affect papers with validated
> public full text, two independent model-assisted coding passes found no
> eligible learned no-EEG comparator in any sampled paper (24/24; raw agreement
> 100%). Because the cohort was conditioned on public access, the coders were
> automated rather than human, and protocol unitization differed for four
> papers, this result is exploratory and is not a representative prevalence
> estimate for the field.

## Prohibited stronger statement

Do not describe this as a two-human-coder audit, a representative systematic
review, proof that all EEG-affect papers use invalid baselines, or evidence that
EEG itself lacks information. Absence of a learned no-EEG comparator is a
reporting and estimand limitation; it is not automatically a resource mismatch.

## Released evidence

The compact release in `paper/generated/literature_resource_audit_v14/`
contains both paper- and protocol-level coding tables, declarations, lock
files, pre-adjudication agreement, and the post-lock crosswalk. Source PDFs and
extracted full texts are excluded from the release.
