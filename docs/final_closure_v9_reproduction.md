# Bounded V9 Evidence Closure: Reproduction

This revision is a bounded support/training sensitivity and execution-contract audit. It does not replace the earlier MusicEEG, Urban, EMOD, or interval-diagnostic results. It does not certify latent emotion disentanglement, full official-model reproduction, or literature-wide error prevalence.

## Inspect The Contract Tool Without EEG

Use a fresh Python 3.11/3.12 environment and a fresh output directory:

```bash
python -m pip install '.[audit,dev]'
python scripts/run_workflow_contract_toy.py workflow_toy
python -m openaffect_eeg.workflow_audit \
  --trials workflow_toy/trials.tsv \
  --execution workflow_toy/matched_execution.json \
  --contract workflow_toy/matched_contract.json \
  --output workflow_toy/rechecked.json
```

The four toy statuses are `allow`, `block`, `unverifiable`, `block`. A returned `allow` only covers supplied observable records. `authenticity_status` remains `unverifiable` even in the clean example. CLI exit codes are 0 (observable allow), 2 (block), and 3 (missing evidence).

The v9 execution contract accepts only four Boolean fields: `subject_holdout`, `stimulus_holdout`, `validation_trial_holdout`, and `require_matched_predictors`. Unsupported keys are rejected, not silently certified. This is a focused adapter, not a replacement for all legacy deployment-contract fields. Complete source UIDs and actual execution traces must be supplied; a manually invented trace is not independently authenticated by the tool.

For matched predictors, population labels must equal the declared train-plus-validation UID set, both predictors must use exactly the declared calibration UIDs, and evaluation supports must match. Additional normative label resources must be declared for both branches. The preprocessing and model-selection execution lists are separately checked; they cannot include the held-out identities when those holdouts are requested. Trial chronology, aliases, acquisition provenance, and upstream foundation-model pretraining remain outside this certificate.

## Pinned External Source Exercise

Retrieve official source bytes at these commits, using `git archive` to avoid checkout line-ending conversion:

- EEGain: `d6892f5586181f345969651a9baedabd828bb1c5`, https://github.com/EmotionLab/EEGain.
- LibEER: `dddff9776dbdae21195fe320dff0a5ba61628a18`, https://github.com/XJTU-EEG/LibEER.

```bash
python scripts/run_external_workflow_audit_v9.py \
  "$ELIGIBLE_TRIALS" "$EEGAIN_SOURCE" "$LIBEER_SOURCE" "$AUDIT_OUTPUT"
```

This executes unchanged official split functions and one exact LibEER control-flow branch with supplied imports. It uses eligible EmoEEG-MC trial identities and three generated index markers per trial to trace window membership. It does not train an upstream emotion classifier or reproduce its published accuracy. Required source-file SHA-256 values are enforced before execution.

The 20 native/scoped cases and 70 injected cases are reported separately. Injected resource mismatches are not upstream defects. Five markers/folds/configurations are not independent studies. An elementary comparator checks the same basic schema, trial intersections, and requested identity isolation; the extended checker adds explicitly named resource/provenance rules. Its broader rule coverage on developer-designed cases is a functional validation, not an unbiased population-level detection-rate estimate. No independent human user study has been completed.

Public outputs contain aggregate case diagnostics and source hashes. `private_traces/` contains executed identities and must remain outside the allowlisted release. An initial fixture inadvertently aliased its preprocessing list with its training list; this was corrected and regression-tested before the verified audit was exported. Earlier diagnostic outputs were preserved privately rather than passed off as the final result.

## Frozen GPU Sensitivity

The configuration is `configs/final_robustness_v9.json`. This is a support-coverage sensitivity on already-used data, not a new blinded confirmation cohort.

- Eligible EmoEEG-MC and Urban trial supports are intersected with the existing feature and raw-tensor archives before assigning roles.
- Five deterministic participant and stimulus blocks cover every eligible identity once as a test identity. Only diagonal crossed blocks are evaluated. This does not cover every possible trial pair.
- Within a block, all models and 25 dose cells share identical test trials and a fixed population-training row count. Actual exposure counts, calibration identities, and unavailable claims are recorded.
- Two predeclared ridge alpha grids and two standard EEGNet learning rates are retained, with no choice based on test results. Frozen LaBraM is a feature audit, not full-model fine-tuning.
- EEGNet selects epochs on validation MAE and refits on train plus validation. Both direct and prior-residual branches are retained. Personalization uses the same calibration labels as the non-EEG prior. It is an output-offset calibration experiment, not a claim to optimize every possible adaptation strategy.
- Emo uses real valence/arousal targets. Urban uses one real valence target; no second target is synthesized.

Install the existing EEG/foundation dependencies and Braindecode 1.2.0 in an isolated GPU environment. Select the working CUDA library paths for that environment without altering unrelated environments. Set BLAS/PyTorch CPU threads to at most two on a shared server. Do not install dependencies in the shared base environment.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/prepare_urban_tensors_v9.py "$DATA_ROOT"
for dataset in ds005540 ds006850; do
  for phase in prepare ridge eegnet aggregate; do
    python scripts/run_final_robustness_v9.py configs/final_robustness_v9.json \
      "$DATA_ROOT" "$RESPONSE_ROOT" --dataset "$dataset" --phase "$phase"
  done
done
python scripts/verify_final_robustness_v9.py \
  configs/final_robustness_v9.json "$RESPONSE_ROOT" "$QA_OUTPUT"
for method in percentile normal_t; do
  python scripts/analyze_exposure_added_value.py "$RESPONSE_ROOT" "$STATS_ROOT/$method" \
    --bootstrap 2000 --seed 20260909 --interval-method "$method"
done
```

The data-root layout is declared in the JSON configuration. Raw data, trial identifiers, individual predictions, and checkpoint arrays stay on the authorized compute host. The verifier checks complete model/cell coverage, assignment and prediction hashes, exact target support, resource parity, unchanged population size, zero-dose identities, validation checkpoint selection, and the genuine number of targets. It does not certify inaccessible pretraining provenance.

The primary contrast is the equally weighted mean of `CCC(EC)-CCC(PC)` over the 25 cells and five blocks, not pooled-trial CCC and not a deployment-prevalence-weighted utility. Crossed bootstrap uses shared participant and stimulus multiplicities for all models, cells, and blocks. It conditions on the fitted predictions. Percentile and heuristic t-normal intervals are both retained; neither is a proof of correct finite-sample coverage. Finite valid-draw counts must be reported. Do not interpret a confidence interval containing zero as equivalence.

## Independent Acceptance Record

A person not involved in implementation should record the downloaded version/hash, OS/Python, installation commands, toy statuses, failures and elapsed time. Authorized data owners can additionally run one private data adapter and verify that no restricted files enter the exported bundle. Do not label agent-driven clean installs as independent-user acceptance.

Versioned public code, a locally anonymized archive, anonymous review hosting, and a persistent DOI are different deliverables. Report their actual states separately. No citation, public address, or human acceptance record should be invented to fill a checklist.
