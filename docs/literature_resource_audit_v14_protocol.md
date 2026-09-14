# Outcome-blind protocol: label-resource audit of EEG affect prediction papers

Locked 2026-09-14, before paper-level label-resource outcomes are coded. This
audit asks how often published EEG affect evaluations permit a resource-matched
claim about EEG's incremental value. It does not estimate whether EEG itself is
useful in every deployment.

## Sampling frame

The retrieval window is 2021-01-01 through 2026-09-14. We query OpenAlex for
works whose titles match either `"EEG emotion recognition"` or
`"EEG-based emotion recognition"`, retain the raw responses, and deduplicate by
DOI and OpenAlex identifier. Local title screening requires EEG or
electroencephalography, an affect or emotion term, and a prediction,
classification, recognition, or decoding term.

A work is eligible when it is an English original empirical article or
proceedings paper, evaluates an EEG-input affect predictor, and reports at least
one evaluation on DEAP, DREAMER, AMIGOS, or MAHNOB-HCI. These datasets were
chosen before coding because each has repeated audiovisual stimuli and
participant-level affect ratings. Reviews, editorials, non-human studies, and
dataset descriptions without an empirical prediction evaluation are excluded.
Screening decisions and reasons are retained.

The main sample contains 24 papers: four from each publication year 2021--2026.
Within a year, eligible records are ordered by SHA-256 of the fixed seed,
publication year, DOI or OpenAlex ID, and normalized title. The next two records
per year form a reserve. If full text remains unavailable after two documented
attempts on different days, the next same-year reserve replaces it; both the
inaccessible record and replacement remain in the flow table. This rule creates
possible access bias, which must be reported.

Before paper-level resource coding, the retrieval produced 380 records passing
the mechanical title-and-dataset check. To avoid unnecessary screening without
changing the sample, title/abstract screening proceeds in the frozen hash order
within each year and stops after six eligible records are found (four main and
two reserve). Later records are retained as `not_screened_after_quota`. Because
no later record precedes the sixth eligible record in the frozen order, this is
identical to selecting the first six eligible records from the full randomized
ordering. This clarification was committed after retrieval counts were known but
before resource-access outcomes were coded.

## Coding and estimands

Two people independently code the same version of every paper using
`literature_resource_coding_manual_v14.md`. Each coder locks a CSV and SHA-256
before seeing the other coder's answers. Automated extraction can prepare text
and locate candidate passages, but an automated agent is not a second human
coder. Disagreements are compared only after both locks and are then adjudicated
with a recorded reason.

The primary descriptive quantities are:

1. among all sampled papers, the proportions with a resource-matched no-EEG
   comparator, a resource-unmatched comparator, no such comparator, or
   insufficient reporting;
2. among papers that include a no-EEG comparator, the proportion whose EEG and
   no-EEG systems receive the same participant-specific, stimulus-specific, and
   population label resources;
3. counts by dataset, split unit, target-label source, and claim type.

The second estimand is conditional on having a no-EEG comparator. It must not be
presented as the prevalence among all EEG affect papers. A missing no-EEG
baseline is not automatically a mismatch and is reported separately.

We report exact binomial intervals as descriptive uncertainty and Cohen's kappa
plus raw agreement for nominal fields. We do not run a significance test against
an arbitrary prevalence threshold. Paper-dataset rows are secondary because a
single paper can contribute correlated evaluations.

## Bias controls and deviations

- Search-index coverage, title-query wording, English-only screening, incomplete
  2026 coverage, and full-text access can all bias the sample.
- High reported prediction accuracy is not evidence of resource mismatch and is
  not used for screening or sampling.
- Coders record `unclear` when the text does not establish access; they do not
  infer a split or label source from conventional practice.
- Any protocol change is appended with a timestamp, rationale, and whether it
  was made before or after coder outcomes were visible. The original files stay
  in version control.
