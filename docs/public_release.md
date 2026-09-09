# Anonymous public release

## Build

```bash
python scripts/build_public_release.py \
  . configs/public_release.yaml OpenAffect-EEG-anonymous.tar.gz
```

When the development checkout contains selected manuscript changes that must
not be staged or committed yet, first materialize an independent allowlisted
snapshot. The snapshot is a fresh Git index containing only policy-selected
files; it does not modify the source checkout or add Git history to the archive.

```bash
python scripts/freeze_public_release.py \
  . configs/public_release.yaml /tmp/openaffect-anonymous-snapshot
python scripts/build_public_release.py \
  /tmp/openaffect-anonymous-snapshot configs/public_release.yaml \
  OpenAffect-EEG-anonymous.tar.gz
```

Use repeated `--forbid` options during a private pre-release audit to reject any
literal lab-specific host or account identifiers in addition to the built-in
local-path and email checks.

For the V11 submission closure, use the narrower policy that also includes the
hash-verified evaluator simulations and source-linked claim-repair record:

```bash
python scripts/freeze_public_release.py \
  . configs/submission_release_v11.yaml /tmp/openaffect-v11-snapshot
python scripts/build_public_release.py \
  /tmp/openaffect-v11-snapshot \
  /tmp/openaffect-v11-snapshot/configs/submission_release_v11.yaml \
  OpenAffect-EEG-anonymous-submission-v11.tar.gz
```

The freeze step captures only the policy allowlist into a separate Git index;
it does not stage or commit the development checkout. A developer-operated
clean installation verifies packaging and executable behavior but is not an
independent-human usability study.

## Acceptance levels

The lightweight `smoke` extra validates the public metadata and synthetic audit
path; it is not the dependency set for every repository test:

```bash
python -m pip install -e ".[smoke]"
python scripts/run_benchmark.py metadata --output-root /tmp/openaffect-metadata
python scripts/run_benchmark.py smoke --output-root /tmp/openaffect-smoke
openaffect toy --output /tmp/openaffect-toy
python -m pytest tests/test_audit_cli.py tests/test_audit_evidence.py \
  tests/test_deployment_selection.py
```

The complete test suite additionally imports EEG I/O, foundation-model, and
figure modules. Run it only after installing the corresponding extras (or the
locked `environment.yml`). A lightweight-environment import failure is neither
a scientific failure nor a passing full-suite result.

## Included boundary

The allowlist contains tracked source code, tests, benchmark configuration,
evaluation and model cards, Croissant metadata, reproducibility instructions,
and the manuscript claim configuration. It excludes Git history, internal
planning documents, editable office files, generated arrays, raw recordings,
stimulus media, model checkpoints, and all untracked files.

The builder rejects symlinks and members with EEG, array, checkpoint, media,
archive, or office-document suffixes. It also scans text for local home/data
paths, Windows user paths, email addresses, and caller-supplied forbidden
tokens.

## Integrity

The gzip and tar metadata are normalized, so identical tracked inputs produce
an identical archive hash. `PUBLIC_RELEASE_MANIFEST.json` records every included
path, mode, size, and SHA-256. A sidecar manifest additionally records the final
archive hash and size.

The archive does not contain raw EEG, DE files, precomputed feature arrays,
restricted stimuli, or model weights. Users obtain those assets from their
original providers and supply an explicit data root for `full` mode.

The package also contains a source-safe aggregate record for one separately
authorized SEED-V protocol-audit case study. Its boundary and deterministic
import procedure are documented in `docs/seedv_case_study.md`; it is not a
redistribution of SEED-V or a fifth public OpenAffect-EEG source.

The independent MusicEEG protocol audit is likewise aggregate-only. Its
source-safe four-protocol band-power, frozen-LaBraM, identity-probe, and
self-report summaries are in
`paper/generated/music_eeg_external_audit_v2.json`; its result boundary and
full rerun recipe are documented in `docs/cross_dataset_foundation_audit.md`.
Raw source data and extracted feature archives remain outside the package.

The second full raw-EEG external case is distributed only as source-safe
aggregates in `paper/generated/ds006866_external_audit_v2.json` and
`paper/generated/ds006866_channel_audit_v2.json`. The package includes its
five-protocol split and evaluation code, pretrained-versus-random LaBraM
comparison, participant/condition probes, channel-contract report, and frozen
result hashes. It excludes the source EEGLAB files, image stimuli, band-power
and LaBraM feature arrays, and trial-level predictions.

The independent `ds006850` confirmation is also aggregate-only. Its audited
62-participant image-appraisal response surface, resource-matched EEG contrasts,
crossed-bootstrap intervals, source exclusions, and provenance hashes are in
`paper/generated/closure_v7/urban_confirmation_v7_public_aggregate.json`.
The portable full-rerun entry point is `scripts/run_urban_confirmation_v7.sh`;
the exact input and claim boundaries are documented in
`docs/independent_confirmation_v7_reproduction.md`. Raw BrainVision files,
derived features, trial-level predictions, stimulus media, and the LaBraM
checkpoint are excluded.

The identity-exposure response-surface release contains only the five
hash-locked aggregate CSV tables under
`paper/generated/identity_exposure_v1/analysis/` plus the claim-extension
table. Runtime logs and the internal statistics JSON are excluded because they
retain machine-local input paths; neither is required to regenerate the 44
response-surface claims. The release includes the response-surface runner,
analysis, claim-card compiler, configuration, and tests, but no trial-level EEG
features or source labels beyond the published aggregates.
