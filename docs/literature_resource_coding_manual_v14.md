# Coding manual: EEG affect label-resource parity

Use the paper's full methods, supplement, and linked code when available. Record
page, section, table, figure, or code location for every non-`unclear` judgment.
Do not use reported accuracy to resolve an access judgment.

## Paper-level fields

`paper_id`, `coder_id`, `document_version`, `datasets_evaluated`,
`prediction_target`, `claim_type`, `no_eeg_comparator`, `overall_parity`,
`confidence`, `evidence_locator`, and `notes` are required.

`claim_type` is one of:

- `eeg_prediction_only`: evaluates an EEG predictor without claiming an
  increment over a non-EEG system;
- `eeg_incremental`: text attributes improvement or unique information to EEG;
- `multimodal_incremental`: compares a fusion model with a non-EEG modality;
- `method_comparison`: compares EEG architectures or features only;
- `unclear`.

`no_eeg_comparator` is `present`, `absent`, or `unclear`. A majority-class,
random, or chance line is not a learned no-EEG comparator unless it uses labels
or metadata available at the target deployment.

`overall_parity` is:

- `matched`: identical label-resource access and identical test support;
- `eeg_favored`: EEG path receives an extra participant-specific,
  stimulus-specific, population-label, or test-support advantage;
- `baseline_favored`: the no-EEG path receives such an extra advantage;
- `mixed`: advantages differ across resource dimensions;
- `no_comparator`: no eligible no-EEG comparator;
- `unclear`: reporting is insufficient to classify at least one material
  dimension.

## Paper-dataset evaluation fields

Create one row per dataset and materially different evaluation protocol. Record:

- target label source: `recorded_participant_self_report`,
  `external_stimulus_norm`, `intended_stimulus_category`, `hybrid`, or `unclear`;
- prediction unit and test support;
- split isolation for participants and stimuli: `held_out`, `overlap`, or
  `unclear`;
- window-to-trial grouping and whether windows from one trial cross partitions;
- label resources available to the EEG path and the no-EEG path at training and
  test time;
- participant calibration count, stimulus-label count, population-label count,
  and whether each count is exact, bounded, zero, or unclear;
- whether model selection uses the final test labels;
- comparator prediction columns and matched metric rows.

Resource access means information used to fit, choose, calibrate, or construct a
prediction. Merely reporting a dataset's labels in a descriptive table does not
count as model access. If a paper trains window samples with a repeated trial
label, every such label exposure is attributed to the parent trial.

## Independent lock and adjudication

Coders must not discuss individual judgments before both files are locked. Each
CSV is validated for allowed values, hashed, and timestamped. After comparison,
every disagreement receives `final_value`, `adjudicator`, `evidence_locator`, and
`reason`. Agreement is calculated on the pre-adjudication files. An author or
agent may adjudicate, but the paper must identify who served as the two human
independent coders and disclose any author involvement.

