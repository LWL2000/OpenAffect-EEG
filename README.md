# OpenAffect-EEG

OpenAffect-EEG evaluates EEG affect prediction under explicitly controlled
participant-calibration and stimulus-exposure resources, comparing EEG and
non-EEG predictors with matched resources. It does not establish causal or
latent-emotion disentanglement. The anonymous review artifact contains no raw
EEG, stimulus media, participant-level predictions, feature arrays, or weights.

## Reviewer acceptance

Reviewers and independent testers should begin with
`docs/REVIEWER_QUICKSTART.md`. In a Python 3.11 or 3.12 environment, the
following command verifies the release manifest, dependency consistency,
synthetic audit path, and frozen V11 evidence hashes:

```bash
python -m pip install ".[audit]"
python scripts/verify_review_artifact.py \
  --project-root . \
  --output acceptance-run
```

This acceptance path does not download restricted data or retrain EEG models.

## Server layout

- Code and experiment metadata: `${OPENAFFECT_PROJECT_ROOT}`
- Raw and derived data: `${OPENAFFECT_DATA_ROOT}`
- Project data link: `${OPENAFFECT_PROJECT_ROOT}/data`

Raw recordings, derivatives, caches, and model weights must never be committed.

## Public workflow modes

The benchmark has three explicit entry points. `metadata` validates the public
contract, Croissant metadata, and evaluation cards without reading EEG;
`smoke` runs deterministic synthetic split/baseline/probe/registry checks; and
`full` validates the frozen real-data result registry and rebuilds paper tables.

```bash
python scripts/run_benchmark.py metadata \
  --output-root /tmp/openaffect-metadata
python scripts/run_benchmark.py smoke \
  --output-root /tmp/openaffect-smoke
python scripts/run_benchmark.py full \
  --data-root ${OPENAFFECT_DATA_ROOT} \
  --output-root ${OPENAFFECT_DATA_ROOT}/derived/reproduction/neurips_ed_v1
```

See `docs/reproducibility.md`, `cards/datasets`, and `cards/models` for the
public artifact boundary, environment roles, and model-specific limitations.

## Initial dataset tiers

The core tier contains EmoEEG-MC (`ds005540`), DENS (`ds003751`), and MusicEEG
(`ds002721`). Their latest pinned snapshots require about 55.4 GiB before Git
or DataLad overhead. The first external dataset is the 148-subject emotion
regulation study (`ds006866`, snapshot `1.0.0`, 116.2 GiB). Its versioned
metadata, event tables, and trial manifest are present; the raw EEG transfer is
resumable and tracked independently from Git.

## Setup

```bash
conda env create -f environment.yml
conda activate openaffect-eeg
python -m pip install -e ".[smoke]"
python scripts/run_benchmark.py metadata --output-root /tmp/openaffect-metadata
python scripts/run_benchmark.py smoke --output-root /tmp/openaffect-smoke
```

Install the heavier development, EEG I/O, foundation-model, and paper extras
only for the corresponding workflow:

```bash
python -m pip install -e ".[dev,eeg,paper]"
python scripts/check_environment.py
python scripts/build_evaluation_cards.py \
  configs/benchmark.yaml cards/evaluations
python scripts/audit_datasets.py --tiers core
python scripts/prepare_openneuro.py --datasets ds002721
# Add --get-data only after the metadata checkout and capacity check succeed.
python scripts/prepare_openneuro.py --datasets ds002721 --get-data
# On high-latency links, use resumable public-S3 transfers with annex verification.
python scripts/download_openneuro_annex.py \
  ${OPENAFFECT_DATA_ROOT}/raw/openneuro/ds002721 --jobs 4
python scripts/build_music_eeg_trials.py \
  ${OPENAFFECT_DATA_ROOT}/raw/openneuro/ds002721 \
  ${OPENAFFECT_DATA_ROOT}/raw/stimuli/eerola_soundtracks_osf_p6vkg/mean_ratings_set1.csv \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests
python scripts/build_dens_trials.py \
  ${OPENAFFECT_DATA_ROOT}/raw/openneuro/ds003751 \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests
python scripts/build_strict_splits.py \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests/ds002721_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests/ds003751_trials.tsv.gz \
  --output-dir ${OPENAFFECT_DATA_ROOT}/derived/splits/core_partial_v0
python scripts/extract_bandpower_features.py \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests/ds003751_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/raw/openneuro/ds003751 \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds003751_bandpower.npz \
  --skip-unreadable
python scripts/run_bandpower_baselines.py \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_partial_v0/core_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_partial_v0 \
  ${OPENAFFECT_DATA_ROOT}/derived/results/bandpower_baselines_partial_v0.json \
  --features ${OPENAFFECT_DATA_ROOT}/derived/features/ds002721_bandpower.npz \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds003751_bandpower.npz
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/run_identity_probes.py \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_partial_v0/core_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_partial_v0 \
  ${OPENAFFECT_DATA_ROOT}/derived/results/identity_probes_partial_v0.json \
  --features ${OPENAFFECT_DATA_ROOT}/derived/features/ds002721_bandpower.npz \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds003751_bandpower.npz
pytest
```

`configs/benchmark.yaml` is the machine-readable evaluation contract. It records
whether each public dataset supports a protocol, is outside that protocol's
scope, or is blocked by source metadata. The generated cards make unavailable
cells explicit; in particular, `ds006866` is never treated as stimulus-held-out
because its public event tables do not expose trial-level image identity.

The reproducible core-v1 workflow additionally builds EmoEEG-MC, the unified
three-dataset splits, and the transparent disentanglement controls:

```bash
python scripts/build_emo_eeg_trials.py \
  ${OPENAFFECT_DATA_ROOT}/raw/openneuro/ds005540 \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests
python scripts/extract_emo_de_features.py \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests/ds005540_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/raw/openneuro/ds005540 \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds005540_de.npz
python scripts/build_strict_splits.py \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests/ds002721_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests/ds003751_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests/ds005540_trials.tsv.gz \
  --output-dir ${OPENAFFECT_DATA_ROOT}/derived/splits/core_v1
python scripts/build_context_transfer_splits.py \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests/ds005540_trials.tsv.gz \
  --output-dir ${OPENAFFECT_DATA_ROOT}/derived/splits/emo_context_v1
python scripts/run_disentanglement_baseline.py \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_v1/core_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/emo_context_v1 \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds005540_de.npz \
  ${OPENAFFECT_DATA_ROOT}/derived/results/disentanglement_context_transfer_v1.json
```

The optional `foundation` dependency group pins the PyTorch-side requirements.
After placing the official CBraMod repository and checkpoint under the data
root, frozen or fixed-random trial representations can be extracted and audited
with the same strict assignments:

```bash
python scripts/extract_cbramod_features.py \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests/ds005540_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/raw/openneuro/ds005540 \
  ${OPENAFFECT_DATA_ROOT}/models/CBraMod \
  ${OPENAFFECT_DATA_ROOT}/models/CBraMod/pretrained_weights/pretrained_weights.pth \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds005540_cbramod.npz \
  --device cuda --batch-size 1
python scripts/run_disentanglement_baseline.py \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_v1/core_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_v1 \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds005540_cbramod.npz \
  ${OPENAFFECT_DATA_ROOT}/derived/results/disentanglement_cbramod_regularized_core_v1.json \
  --feature-view foundation \
  --predictions-output \
  ${OPENAFFECT_DATA_ROOT}/derived/results/disentanglement_cbramod_regularized_core_v1_predictions.tsv.gz
python scripts/analyze_paired_predictions.py \
  ${OPENAFFECT_DATA_ROOT}/derived/results/paired_representation_statistics_core_v1.json \
  --prediction de=${OPENAFFECT_DATA_ROOT}/derived/results/disentanglement_de_regularized_core_v1_predictions.tsv.gz \
  --prediction cbramod=${OPENAFFECT_DATA_ROOT}/derived/results/disentanglement_cbramod_regularized_core_v1_predictions.tsv.gz \
  --prediction random=${OPENAFFECT_DATA_ROOT}/derived/results/disentanglement_cbramod_random_core_v1_predictions.tsv.gz
python scripts/analyze_paired_predictions.py \
  ${OPENAFFECT_DATA_ROOT}/derived/results/paired_conditional_cbramod_core_v1.json \
  --prediction Base=${OPENAFFECT_DATA_ROOT}/derived/results/disentanglement_cbramod_projection_core_v1_predictions.tsv.gz::disentangled_prior_plus_eeg_residual \
  --prediction Conditional=${OPENAFFECT_DATA_ROOT}/derived/results/disentanglement_cbramod_projection_core_v1_predictions.tsv.gz::conditional_stimulus_invariant_eeg_residual
python scripts/analyze_residual_invariance.py \
  ${OPENAFFECT_DATA_ROOT}/derived/results/disentanglement_cbramod_projection_core_v1_predictions.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/results/residual_invariance_cbramod_projection_core_v1.json
python scripts/run_adversarial_disentanglement.py \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_v1/core_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_v1 \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds005540_cbramod.npz \
  ${OPENAFFECT_DATA_ROOT}/derived/results/adversarial_cbramod_core_v1.json \
  --feature-view foundation \
  --predictions-output \
  ${OPENAFFECT_DATA_ROOT}/derived/results/adversarial_cbramod_core_v1_predictions.tsv.gz
python scripts/analyze_paired_predictions.py \
  ${OPENAFFECT_DATA_ROOT}/derived/results/paired_adversarial_cbramod_core_v1.json \
  --prediction Zero=${OPENAFFECT_DATA_ROOT}/derived/results/adversarial_cbramod_core_v1_predictions.tsv.gz::neural_zero_adversary_prior_plus_eeg_residual \
  --prediction Selected=${OPENAFFECT_DATA_ROOT}/derived/results/adversarial_cbramod_core_v1_predictions.tsv.gz::neural_selected_adversary_prior_plus_eeg_residual \
  --bootstrap 2000 --seed 20260818
python scripts/extract_emo_raw_trials.py \
  ${OPENAFFECT_DATA_ROOT}/derived/manifests/ds005540_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/raw/openneuro/ds005540 \
  ${OPENAFFECT_DATA_ROOT}/derived/tensors/ds005540_raw_100hz.npy \
  ${OPENAFFECT_DATA_ROOT}/derived/tensors/ds005540_raw_100hz_trial_uids.npy \
  --target-frequency 100
python scripts/run_eegnet_disentanglement.py \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_v1/core_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/core_v1 \
  ${OPENAFFECT_DATA_ROOT}/derived/tensors/ds005540_raw_100hz.npy \
  ${OPENAFFECT_DATA_ROOT}/derived/tensors/ds005540_raw_100hz_trial_uids.npy \
  ${OPENAFFECT_DATA_ROOT}/derived/results/eegnet_joint_v1.json \
  --predictions-output \
  ${OPENAFFECT_DATA_ROOT}/derived/results/eegnet_joint_v1_predictions.tsv.gz \
  --protocol subject_stimulus_holdout --device cuda
```

The large-sample emotion-regulation validation uses OpenNeuro's versioned
GraphQL/S3 interface because a GitHub metadata clone is not required for direct
snapshot verification:

```bash
python scripts/download_openneuro_snapshot.py ds006866 1.0.0 \
  --jobs 12 --annexed no \
  --include README.md --include participants.tsv --include participants.json \
  --include dataset_description.json --include 'task-*_events.json' \
  --include 'sub-*/eeg/*_events.tsv' \
  --include 'sub-*/eeg/*_channels.tsv' --include 'sub-*/eeg/*_eeg.json'
python scripts/download_openneuro_snapshot.py ds006866 1.0.0 \
  --jobs 6 --annexed yes --include 'sub-*/eeg/*_eeg.set'
python scripts/build_emotion_regulation_trials.py \
  ${OPENAFFECT_DATA_ROOT}/raw/openneuro/ds006866 \
  ${OPENAFFECT_DATA_ROOT}/derived/trial_tables
python scripts/build_emotion_regulation_splits.py \
  ${OPENAFFECT_DATA_ROOT}/derived/trial_tables/ds006866_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/ds006866_subject_v1
python scripts/extract_bandpower_features.py \
  ${OPENAFFECT_DATA_ROOT}/derived/trial_tables/ds006866_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/raw/openneuro/ds006866 \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds006866_bandpower.npz \
  --skip-unreadable
python scripts/run_emotion_regulation_baselines.py \
  ${OPENAFFECT_DATA_ROOT}/derived/trial_tables/ds006866_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/ds006866_subject_v1 \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds006866_bandpower.npz \
  ${OPENAFFECT_DATA_ROOT}/derived/results/ds006866_bandpower_channel_v1.json \
  --feature-view channel \
  --predictions-output \
  ${OPENAFFECT_DATA_ROOT}/derived/results/ds006866_bandpower_channel_v1_predictions.tsv.gz
python scripts/run_emotion_regulation_condition_probe.py \
  ${OPENAFFECT_DATA_ROOT}/derived/trial_tables/ds006866_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/splits/ds006866_subject_v1 \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds006866_bandpower.npz \
  ${OPENAFFECT_DATA_ROOT}/derived/results/ds006866_condition_probe_channel_v1.json \
  --feature-view channel
python scripts/analyze_external_conditions.py \
  ${OPENAFFECT_DATA_ROOT}/derived/results/ds006866_bandpower_channel_v1_predictions.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/results/ds006866_condition_effects_channel_v1.json \
  --candidate condition_plus_eeg_residual_ridge
```

The final `sub-051` event contains only 2.88 seconds of readable EEG inside the
requested five-second stimulus window and has no self-report targets. The
audited extraction therefore records and skips that single unsupervised trial;
it is not silently padded or truncated.

Long extraction runs may use disjoint subject-level manifest shards. Each shard
is produced by the same extractor, then merged in the canonical full-manifest
order with exact trial-coverage and preprocessing-metadata checks:

```bash
python scripts/merge_bandpower_features.py \
  ${OPENAFFECT_DATA_ROOT}/derived/trial_tables/ds006866_trials.tsv.gz \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds006866_bandpower.npz \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds006866_bandpower_part1.npz \
  ${OPENAFFECT_DATA_ROOT}/derived/features/ds006866_bandpower_part2.npz
```

Dataset preparation uses DataLad and checks out the exact snapshot in
`configs/datasets.yaml`. Metadata-only preparation is the default; downloading
annexed recordings requires the explicit `--get-data` flag.

`download_openneuro_annex.py` is an alternative transfer path for an already
initialized dataset. It accepts only OpenNeuro's public S3 URLs, resumes partial
downloads, checks the size encoded in each annex key, and lets `git annex
reinject` verify the content hash before marking an object present.

`download_openneuro_snapshot.py` works without a Git checkout. It pins the
OpenNeuro tag in a JSON file manifest, accepts only versioned objects in the
official OpenNeuro S3 bucket, resumes partial files, and checks every final
object against the byte size returned by the GraphQL snapshot.

The MusicEEG trial builder recovers one stimulus token and eight self-reported
ratings per trial, joins the OSF Set 1 normative ratings, and writes both the
stimulus prior and the participant-specific residual. See `docs/data_audit.md`
for source discrepancies and the exact parsing evidence.

The DENS trial builder corrects sample-index event timing, joins behavior by
stimulus identity instead of row position, and retains trials with unavailable
ratings under an explicit label flag. Its public stimulus mapping and channel
metadata discrepancies are recorded in the same audit document.

Strict splits are deterministic, stratified by dataset, and written before any
EEG window is created. Five fixed seeds cover subject, stimulus, and joint
subject-stimulus holdout. Input manifest hashes and identity-overlap audits are
stored beside every assignment file; see `docs/split_protocol.md`.

The bandpower extractor uses an audited channel set, common-average reference,
and five Welch log-power bands without fitting any dataset statistics. DENS
requires `--skip-unreadable` because most public `.fdt` objects are shorter
than their `.set` headers declare; every skipped trial is listed in the output
metadata rather than silently padded. See `docs/data_audit.md` for counts.

The first regression benchmark compares a training-set global mean, a
stimulus-identity group mean, and an EEG global-bandpower Ridge model. All
scaling is fitted inside the assigned training partition, alpha is selected on
validation data, and test metrics are reported overall and per dataset. Initial
partial-core results and their limitations are in `docs/baseline_results.md`.

Identity probes use subject-held-out assignments, linear Ridge classifiers, and
permuted-label controls. The global representation is probed for dataset ID;
dataset-specific channel representations are probed for stimulus ID. Results
are summarized in `docs/identity_probe_results.md`.

The current three-dataset results, explicit stimulus-prior plus EEG-residual
baseline, multimodal negative control, and bidirectional video/imagery transfer
are summarized in `docs/core_v1_results.md`. A Chinese project brief suitable
for a supervisor update is in `docs/group_leader_brief_zh.md`.
The two linear conditional-debiasing negative ablations and their paired
identity-versus-affect audit are documented in `docs/linear_debiasing_results.md`.
The nonlinear gradient-reversal experiment, its identical zero-adversary
control, and the paired comparison with linear residual models are documented
in `docs/nonlinear_adversarial_results.md`.
The raw-waveform EEGNet architecture audit and its strict joint-holdout negative
result are documented in `docs/eegnet_results.md`.
The `ds006866` event reconstruction, source anomalies, and absence of public
per-trial image identity are documented in `docs/data_audit.md`.
The four-dataset frequency-band and channel-region audit, including 2,000-draw
paired clustered intervals and participant-ID probes under unseen stimuli, is
documented in `docs/feature_ablation_results.md`.

Final model comparisons retain trial-aligned predictions. The paired analysis
resamples fixed evaluation seeds and complete held-out subject or stimulus
clusters, independently resamples both crossed factors for joint holdout, and
also reports exact seed-level sign-flip tests rather than treating correlated
trials as independent observations.

## Evaluation guardrails

1. Windows from the same trial never cross train, validation, or test folds.
2. Subject-held-out and stimulus-held-out results are reported separately.
3. The strict test holds out both subjects and stimuli.
4. Preprocessing statistics are fitted only on the training partition.
5. Stimulus-only, dataset-ID, stimulus-ID, and random-encoder controls are
   mandatory, not optional ablations.

See `docs/research_protocol.md` for the research question and planned evidence.
