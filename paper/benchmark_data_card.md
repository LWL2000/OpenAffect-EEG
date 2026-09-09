# OpenAffect-EEG Benchmark Data Card

## Summary

OpenAffect-EEG is a metadata and evaluation benchmark for auditing EEG affect
models under changes in participant, stimulus, context, and dataset identity.
Version `neurips_ed_v1` contains harmonized trial metadata, deterministic split
builders, evaluation cards, model cards, aggregate result tables, claim-locked
statistics, and executable code.

The public artifact does not contain raw EEG, source DE or feature arrays,
foundation-model checkpoints, or movie, music, and image stimuli.

## Motivation

EEG affect scores can change when training and test sets share participants or
stimuli. OpenAffect-EEG binds every result to an explicit identity-isolation
contract and records whether each evaluation cell is supported, executed,
blocked, or not applicable. The benchmark is designed for auditing deployment
claims, not for creating another random-split leaderboard.

## Composition

The collection links version-pinned snapshots of MusicEEG (`ds002721`), DENS
(`ds003751`), EmoEEG-MC (`ds005540`), and an external socio-affective emotion-
regulation dataset (`ds006866`). The first three form the stimulus-elicited core;
the fourth is a condition-level external stress test because public events do
not expose trial-level image identity.

Trial rows identify the dataset, participant, trial, context, target
availability, and stimulus when the source supports that mapping. Split rows
record the protocol, fixed seed, trial identifier, and train, validation, test,
or excluded role. Prediction and result artifacts are keyed and SHA-256 locked.

## Sources and licensing

Users obtain every upstream dataset and optional model from its original
provider. Snapshot identifiers and citations are recorded in the dataset cards.
The OpenAffect-EEG software is MIT licensed, but that license does not relicense
upstream recordings, stimuli, derivatives, repositories, or checkpoints. See
`THIRD_PARTY_LICENSES.md` for the redistribution boundary. Where upstream DENS
metadata expose conflicting license labels, the public package redistributes no
DENS signal or stimulus and flags the conflict instead of choosing a permissive
interpretation.

## Processing

Processing verifies source snapshots, parses events and behavior, creates
dataset-namespaced trial and identity keys, harmonizes valence and arousal,
records target and signal availability, constructs splits before any windowing,
and hashes outputs. Unreadable signals are not imputed, missing stimulus identity
is not replaced by row order, and inferred mappings retain their provenance
state.

## Intended uses

- Compare model conclusions across trial-random, subject, stimulus, joint,
  context-transfer, and dataset-holdout contracts.
- Compare pretrained EEG encoders with matched random initialization under the
  same trial support and prediction head.
- Audit participant, stimulus, dataset, and condition decodability.
- Test whether a mitigation changes both affect prediction and identity-related
  sensitivity.
- Add a model to the same split, prediction, and reporting contract.

## Out-of-scope uses

- Treating self-reports as error-free ground truth for private internal states.
- Claiming new-participant or new-stimulus generalization from trial-random
  evaluation.
- Inventing stimulus holdout for `ds006866` without image identifiers.
- Identifying real people from EEG or using benchmark scores for clinical,
  employment, education, surveillance, or other high-stakes decisions.
- Ranking scores that use different targets, protocols, or available trial sets
  without disclosing those differences.

## Limitations and risks

MusicEEG stimulus mapping includes evidence-backed inference. DENS has behavior
gaps and unreadable public signal windows. EmoEEG-MC contains behavior fields
whose broadcast stimulus codes cannot all be trusted, so only verified mappings
enter supervised analyses. `ds006866` lacks image identity. Devices, montages,
sampling rates, stimulus modalities, and rating scales remain heterogeneous;
target scaling does not establish psychometric equivalence. Demographic
coverage is inherited from the source studies.

EEG is sensitive physiological data. Identity probes are risk audits and must
not be repurposed for real-world identification. Users remain responsible for
the source studies' consent, ethics, and licensing conditions.

## Maintenance

Releases bind code, source snapshots, fixed seeds, input hashes, artifact
manifests, and the numerical claim ledger. Changes that affect trial mappings,
protocols, or frozen values create a new release. See `docs/versioning.md` and
`docs/correction_policy.md` for maintenance, challenge, and correction rules.
