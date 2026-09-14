# Confirmatory-dataset amendment (v14)

Status: locked before trial-level FACED or AMIGOS outcome access
Amendment date: 2026-09-14 (Asia/Shanghai)
Parent repository commit: `1832b6bfb198de11236aa30c514a420dd4644421`

## Decision and reason

The untouched confirmation dataset is changed from the AMIGOS individual
short-video arm to FACED, pinned to the NEMAR release `nm000112` version
`v1.1.3` (dataset DOI `10.82901/nemar.nm000112`). The AMIGOS plan remains in
the repository as a superseded audit record.

This amendment is based on access feasibility, not an observed label pattern or
model result. At the amendment lock, no AMIGOS recording, trial label, or model
outcome had been received or inspected. AMIGOS required a separately approved
EULA and its public delivery endpoint was unavailable during the attempted
access. FACED provides a versioned, direct, CC-BY-4.0 distribution that an
independent reviewer can retrieve without credentials.

Public descriptions and metadata known before the lock report 123 participants,
28 repeated video stimuli, 32-channel EEG, and participant-specific Valence and
Arousal ratings. The repository had cited the FACED data paper and a paper about
stimulus-identity classification, but this project had not downloaded, fitted,
or analysed FACED trial-level data. The confirmation is therefore outcome-blind
but not literature-naive.

The selection rule is fixed as follows: among otherwise eligible datasets, use
the one with direct lawful access, the largest reported participant count, and
a stable version and licence. Expected compatibility with the paper's claim,
published benchmark accuracy, and the sign of any EEG effect are not selection
criteria. A contradictory FACED result will be retained and will change the
paper's conclusion.

## Known pre-outcome caveat

The NEMAR catalogue reports that some recording views could not be generated
because of structural BDF errors. This is a signal-access fact, not an outcome.
Eligibility is therefore determined by the label-blind structural rules in the
locked FACED protocol. The analysis remains confirmatory only if at least 25
participants retain all 28 common stimuli; no exclusion may depend on Valence,
Arousal, an EEG/no-EEG score, or an effect direction.

## Files frozen by this amendment

- `docs/confirmatory_faced_v14_protocol.md`
- `configs/confirmatory_faced_v14.yaml`
- the original AMIGOS protocol and config, marked superseded but not deleted

Only source-schema metadata, licence information, public aggregate descriptions,
and file names/shapes were examined before this amendment. Trial-level event
rows, self-rating values, EEG samples, and model outcomes remained unopened.
