# V14 evidence and reproduction

Current integration: 2026-10-20. The next NeurIPS E&D format has not yet been
published; the `paper/neurips2026` directory and template are a working reference,
not a claim that this is a valid late NeurIPS 2026 submission.

## What the reviewer can reproduce without data or training

The release contains ten frozen neural analysis sets: FACED EEGNet and adapted
LaBraM observed results, both full-grid permutations, both failed EEGNet
sinusoidal controls, both successful exploratory signed-power subsets, and
five-seed adapted LaBraM sensitivities for EmoEEG-MC and Urban. Every set has a
summary JSON and cell CSV copied byte for byte from the completed analysis.
`paper/generated/evidence_v14/archive_manifest.json` records their hashes.
The Ridge archive separately contains scalar metrics and record hashes for all
2,500 completed cells, with descriptive means only and no new joint inference.

```bash
python -m pip install ".[audit]"
python scripts/build_v14_paper_assets.py --verify \
  --output v14-table-check
python scripts/verify_review_artifact.py \
  --project-root . --output acceptance-run
```

Use absent or empty output directories for the reviewer acceptance command.
The v14 table check validates all source hashes, grid keys, primary means and
equivalence decisions, then reproduces the frozen tables. It imports no trainer,
loads no participant predictions, samples no bootstrap, and downloads nothing.
Its primary table, complete control table, all-results CSV and resource-corner
CSV agree with the manuscript. Cell tables contain the already computed
pointwise and simultaneous intervals and secondary MAE differences.

For a source edit, compile the paper without rebuilding earlier analyses:

```bash
python scripts/build_neurips_submission.py --reuse-existing-evidence
```

This requires the paper/LaTeX environment. To regenerate the already selected
heatmap, run `python scripts/plot_v14_faced_heatmap.py` in a Matplotlib/Pillow
environment. Both commands consume existing results. The older development
figures and appendices keep their own fixed-fit inferential boundary.

## Full scientific reproduction is a separate, explicit workflow

The commands below document an expensive source-data workflow; they are not
invoked by acceptance or by manuscript integration. Install the documented EEG,
GPU and foundation-model dependencies and obtain the official LaBraM repository
and checkpoint under their own terms. Set `OPENAFFECT_DATA_ROOT`,
`LABRAM_REPOSITORY`, and `LABRAM_CHECKPOINT` to your own paths. The dataset is
NEMAR `nm000112`, version `v1.1.3`; no floating `latest` version is substituted.

```bash
DATA="$OPENAFFECT_DATA_ROOT"
RAW="$DATA/raw/nemar/nm000112/v1.1.3"
INPUT="$DATA/derived/confirmatory_faced_v14_inputs"
ASSIGN="$DATA/derived/confirmatory_faced_v14_assignments"
CFG=configs/confirmatory_faced_v14_experiment.yaml
python scripts/fetch_faced_v14.py "$RAW" --include-events --include-bdf
python scripts/qc_faced_remote_v14.py "$DATA/derived/faced-remote-qc.json"
python scripts/qc_faced_signal_v14.py "$RAW" \
  "$DATA/derived/faced-remote-qc.json" "$DATA/derived/faced-signal-qc.json"
python scripts/ingest_faced_v14.py "$RAW" \
  "$DATA/derived/faced-signal-qc.json" "$INPUT"
python scripts/prepare_confirmatory_assignments_v14.py "$CFG" "$DATA" \
  "$ASSIGN" --dataset faced_confirmation_v14
python scripts/run_bandpower_ridge_v14.py "$CFG" "$DATA" "$ASSIGN" \
  "$DATA/derived/faced-ridge" --dataset faced_confirmation_v14
python scripts/run_eegnet_v14.py "$CFG" "$DATA" "$ASSIGN" \
  "$DATA/derived/faced-eegnet-observed" \
  --dataset faced_confirmation_v14 --control observed
python scripts/run_labram_peft_v14.py "$CFG" "$DATA" "$ASSIGN" \
  "$LABRAM_REPOSITORY" "$LABRAM_CHECKPOINT" \
  "$DATA/derived/faced-labram-observed" \
  --dataset faced_confirmation_v14 --control observed --trainable-final-blocks 4
python scripts/analyze_training_uncertainty_v14.py \
  "$DATA/derived/faced-labram-observed/identity_exposure_predictions.tsv.gz" \
  "$DATA/derived/faced-labram-observed-analysis" \
  --bootstrap 2000 --seed 2026091400 --control observed \
  --equivalence-scope confirmatory
```

The last invocation's `confirmatory` flag records the fixed-margin analysis
scope. It does not overrule the complete pipeline's failed-control limitation.
Apply the same analysis command to EEGNet's local predictions. All five seed
and five rotation choices are frozen in the configuration; participant-dose
offsets reuse a fitted residual rather than creating redundant retraining.
Source identifiers and upstream checkpoint histories are not certified by
resource hashes alone.

## Controls, with no outcome-dependent substitution

Use separate output directories for every control. For EEGNet run the full
design with `--control label_permutation`, `--control synthetic_signal` and
`--control synthetic_residual_signal`. The latter two are retained failures.
For LaBraM the completed full-grid negative control uses
`--control label_permutation`. Analyze synthetic controls with
`--equivalence-scope exploratory`; no synthetic control is confirmatory real-EEG
equivalence evidence.

For each architecture, the completed signed-power diagnostic uses:

```text
--control synthetic_signed_power_residual
--split-index 0 --split-index 20 --split-index 40
--split-index 60 --split-index 80
--stimulus-dose 0 --stimulus-dose 2 --stimulus-dose 8
```

It retains all five participant doses and training seeds, yielding 150 EEGNet
candidates / 75 selected fits and 75 LaBraM fits. Its realized support is 85
participants and 23 videos, not the full 123-by-28 cohort. The stopped 52-fit
full-grid attempt is not included in inference. The sequence, failed controls,
oracle check, pilot and amendment timing are documented in
`confirmatory_positive_control_amendment_v14.md`. Existing source outputs must
not be overwritten or silently relabelled after changing code.

## Interpretation and independence

The archived statistic is the uniform mean of CCC increments over the resource
grid and targets. Equivalence concerns that average for that fitted model and
design. No target-, cell-, participant-, acquisition-cost-, or all-model
equivalence claim follows. The margins are pre-outcome benchmark resolutions,
not clinical minimum important differences. The older Gaussian simulation
coverage does not validate the newer joint training bootstrap.

The two literature coders and artifact operator are automated model roles.
Their executions are isolated, but they do not establish independent human
judgment or eliminate shared model bias. A passing quickstart establishes
software integrity and table reconstruction, not full experimental reproduction.
The original clean-run packet is classified as `author_assisted`.

## Release boundary

Build with `configs/submission_release_v14.yaml` (also mirrored by
`configs/public_release.yaml`). Release only tracked allowlisted files. Raw
EEG, stimulus media, weights, arrays, trial-level predictions and the private
`derived/` tree are excluded. The archive builder writes a current internal
manifest; its verified copy is synchronized to the repository root before
publication. The anonymous service must expose the same release when submitted.
