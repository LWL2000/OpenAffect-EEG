# `neurips_ed_v1` reproduction report

Run date: 2026-08-21
Candidate source state: allowlisted anonymous snapshot; source worktree is not staged or committed by release packaging

## Server environments

Statistical and artifact-validation environment (`openaffect-eeg`):

- Linux 6.8.0-124-generic, glibc 2.35, x86_64
- Python 3.11.15
- NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1
- scikit-learn 1.9.0, MNE 1.12.1, PyYAML 6.0.3

Foundation feature environment (`lerobot`):

- Python 3.12.13
- PyTorch 2.11.0+cu130, CUDA runtime 13.0
- `timm` 1.0.28
- NVIDIA RTX 6000 Ada Generation; CUDA availability verified at run time

The original `neurips_ed_v1` run kept PyTorch outside the statistical
environment. For the prespecified ds006866 extension on 2026-08-31, the user-
selected `openaffect-eeg` environment was augmented with PyTorch 2.11.0+cu130,
torchvision 0.26.0+cu130, torchaudio 2.11.0+cu130, and `timm` 1.0.28. CUDA 13.0
availability and the official LaBraM checkpoint were verified before feature
extraction. The canonical extension records CUDA preprocessing and
`torchaudio` band-limited sinc resampling in per-feature metadata; linear
evaluation and table compilation remain in scikit-learn.

## Recompute recipe validation

`configs/recompute_neurips_ed_v1.yaml` defines a 63-step, 103-output typed DAG
covering manifests, splits, transparent and foundation features, evaluations,
identity and feature audits, external validation, paired statistics, benchmark
scope, feature-missingness analysis, semantic-prior ablations, and final
registry compilation. The server plan reported `ready=true` with no missing
requirement.

A formal `--resume` execution marked 62 steps `skipped_existing` because every
declared scientific output already existed; the final registry compilation was
rebuilt in 9.10 seconds. This validates dependency resolution, source
requirements, output declarations, and restart behavior. It is not represented
as a cold raw-EEG recomputation.

Plan report SHA-256:
`d7e49078e3dd0e76a0822a6f6a789a9ca87823af9cc8aef14a6e57bf61f46c74`.

Run report SHA-256:
`c95a71342eeb2d8dd1c6e7bd067df11666cc2d0fb2d0bd28f64f6a706a492ef0`.

## Independent MusicEEG foundation audit

`configs/cross_dataset_music_labram_audit_v1.yaml` is a five-step, separately
resumable DAG for an independent MusicEEG LaBraM audit: manifest construction,
complete-trial splits, pretrained feature extraction, fixed-random feature
extraction, and paired evaluation. It uses 1,240 labelled 12-second audio
trials from 31 participants and five fixed seeds. The server `--resume` run
reported `ready=true` and skipped all five existing outputs only after checking
their declared requirements.

The source-safe aggregate record is
`paper/generated/music_labram_cross_dataset_audit_v1.json`. It reports a
pretrained-versus-random macro-CCC difference of 0.0703 under unseen-participant
evaluation and 0.0337 under joint unseen-participant/unseen-music evaluation.
The same record also reports high pretrained subject-probe decodability under
unseen music (balanced accuracy 0.9136); this is interpreted as a paired
representation audit, not as a new model claim.

## Manuscript assets

The 56-input frozen registry generated 17 deterministic manuscript assets and
a 104-entry claim ledger. The build verifies each input hash before resolving
any manuscript value. The generated manuscript asset manifest has SHA-256:

`4418782224ac34dd01fc603d0ae04617bfeaa716efb4f00bc2b0864632c2a81f`.

Two additional aggregate-only case-study artifacts are deliberately outside the
claim ledger because they are reported as source-linked supplementary evidence:
the authorized SEED-V negative control and the independent MusicEEG paired
LaBraM audit. Each has a dedicated importer or recomputation configuration,
source boundary, and regression test; neither includes raw data, feature arrays,
media, trial predictions, or participant identifiers.

The Chinese main text, Chinese supplement, data card, English main text, and
English supplement pass citation-key, source-verification, claim-resolution,
numeric-consistency, and anonymity audits. Figure files in the worktree are
evidence previews; final figures are approved separately for each scientific
question.

The anonymous English manuscript uses the official NeurIPS 2026 E&D style and
generated claim-value macros. A native `pdflatex`/BibTeX build completes without
undefined citations, undefined references, overfull boxes, or sensitive source
strings. The content-page marker is on page 9, exactly at the nine-page limit;
references, supplement, and the official checklist follow outside that count.
A figure-height stress test reserves 1.75--1.95 inches for four main-paper
figures and 2.05 inches for the debiasing audit as Supplementary Fig. S1; this
layout remains at 9/9 pages. Final-figure integration must still repeat the
page-limit check using the selected artwork's actual bounding boxes.

## Anonymous release status

The release builder operates from an independent allowlisted snapshot with a
fresh temporary Git index. This permits strict tracked-member checks without
staging, committing, or changing the development worktree. The snapshot is not
included in the archive. The archive builder records a member manifest and an
external sidecar checksum; the checksum is intentionally not written back into
the archive, which keeps deterministic bytes well-defined.

The release boundary is unchanged: no raw EEG, source DE/feature arrays,
foundation checkpoints, restricted media, local paths, hostnames, emails, or
Git history may enter the anonymous package. Users obtain upstream resources
from their original providers and pass an explicit data root to full mode.

## Current verification

- The final clean-package server suite reports 271 tests passed; Ruff passes
  across `src`, `tests`, and `scripts` in the canonical `openaffect-eeg`
  environment.
- Recompute readiness: 63/63 steps ready; zero missing requirements.
- Independent MusicEEG recompute DAG: 5/5 steps ready; a `--resume` run skipped
  only declared existing outputs.
- Registered evidence: 56/56 inputs hash verified during manuscript build.
- Frozen numeric claims: 104/104 resolved.
- Current manuscript audits: Chinese main text, supplement, data card, English
  main text, and English supplement pass.
- Official anonymous draft: compiles successfully at 9/9 content pages.

The final clean-package run is a release validation, not a claim that an
independent third party has reproduced the raw-data computation.

## 2026-08-31 evidence-extension addendum

The independent MusicEEG audit was rerun in the `openaffect-eeg` environment
using only flattened band-power inputs and five frozen complete-trial split
assignments. No raw EEG, DE array, music label, text descriptor, normative
rating, or trial-local rating was used as a prediction feature. The source-safe
aggregate is `paper/generated/music_eeg_external_audit_v2.json`; its private
counterpart is deliberately excluded from the public release.

For valence/arousal macro-CCC, the transparent band-power Ridge reference was
0.1031 under random trial holdout, 0.0175 under unseen participants, 0.1130
under unseen music, and 0.0227 under joint unseen-participant/unseen-music
holdout. The paired random-minus-joint difference was 0.0804 across the five
frozen seeds. These seed summaries are descriptive protocol contrasts, not
independent replication samples or causal variance attribution.

The same external audit includes representation probes and a self-report
structure diagnostic. Under unseen music, a linear band-power participant probe
reached balanced accuracy 0.9995 versus 0.0431 after label permutation
(31-class uniform chance 0.0323). Its closed-set music-identity counterpart
was 0.0000 observed versus 0.0033 permuted, with 0.7875--0.9125 seen-class test
coverage across seeds; this negative probe must not be read as evidence that
music identity is absent from every EEG representation. Leave-one-participant-
out consensus ratings within the same music had CCC 0.1437 for valence and
0.1634 for arousal. They describe rating structure and do not provide a noise
ceiling or identify the private experience of an individual participant.

The cached MusicEEG LaBraM audit was also expanded to all four frozen protocols.
Pretrained-minus-random macro-CCC differences were 0.0732 (random trial),
0.0703 (unseen participant), 0.0703 (unseen music), and 0.0337 (joint). The
pretrained feature participant probe under unseen music was 0.9136 versus
0.0313 after permutation (uniform chance 0.0323). Thus the added external
evidence supports a conditional audit finding: a pretraining gain under a
particular holdout does not, by itself, establish identity-invariant affect
representation.

### Full ds006866 raw-EEG external case

The second external case uses 35,504 complete-rating trials from 148
participants. A formal header audit confirms one ordered 64-channel EEG
contract for all 148 recordings; 146 match the base source pattern and two add
only the preregistered excluded EMG channel. The frozen pipeline evaluates
band-power, official pretrained LaBraM, and the same architecture under a fixed
random initialization across subject holdout, both directional context-only
transfers, and both directional joint participant-context transfers. Every cell
uses seeds 11, 23, 47, 71, and 101 and 500 participant-cluster bootstrap
replicates.

When participants repeat across contexts, the pretrained hybrid reaches macro
CCC 0.5507 for nonsocial-to-social and 0.5486 in the reverse direction, while
participant-probe balanced accuracy is 0.9997 and 0.9999. Under joint
participant-context holdout, pretrained EEG-only CCC falls to 0.0323 and
0.0289, and pretrained hybrid CCC is 0.3718 and 0.3761. The pretrained hybrid
does not beat fixed-random LaBraM consistently across seeds, and its
participant-cluster intervals do not establish improvement over the semantic
prior. This supports a known-person personalization versus participant-
independent transfer distinction. It does not establish a causal identity
shortcut, and condition families are not treated as image identities.

The final aggregate and channel-contract SHA-256 values are
`1bb2b98349cf053eff8ce4a0ea91c5d674f7e374991c44e6f14938cc03049617`
and
`1ddc32a003a510ea45df343ffd559981a56a5212223d825f6f6341fb00ca0ee7`.
Both are registered in the manuscript asset compiler; raw EEG and derived
feature archives remain outside the release.

Validation performed after this extension: the complete local test suite,
targeted server tests for MusicEEG, Chinese-PDF, environment, Croissant, and
workflow paths, and Ruff on all new or changed Python files passed. The local
English build remained at 9/9 content pages and the paired Chinese reading PDF
was regenerated. The in-repository Croissant structure and RAI-field validation
passes, while the strict publishability gate intentionally fails until a real
anonymous code/archive URL replaces the `.invalid` placeholder. This remaining
hosting action is required before formal submission.

### Final server snapshot validation

The final allowlisted snapshot was validated in the `openaffect-eeg`
environment against the declared data root. Its complete unit-test suite passed,
and `ruff check src tests scripts` passed under the repository configuration.
The only cross-platform lint adjustment is a documented per-file exception for
the executable-bit rule on `scripts/*.py`: these entry points are intentionally
invoked as `python scripts/...` in the release recipe, so a Windows-generated
archive must not require a POSIX executable bit.

This server run also caught and fixed a NumPy 2.x compatibility defect in the
band-power integration helper: the fallback reference to the removed `np.trapz`
was eagerly evaluated even when `np.trapezoid` was available. The helper now
selects the available integrator without evaluating the unavailable fallback;
the dedicated spectral-feature regression test passes in the server environment.

The final deterministic release candidate contains 244 allowlisted public
members plus its internal manifest. It was built twice on the server after the
source scan and Git-indexed snapshot checks; the two archive checksums matched.
No raw EEG, derived feature array, restricted stimulus, local path, hostname,
email, or Git history is included. A clean extraction passed 271 tests, Ruff,
all three metadata steps, all five smoke steps, and all four full-workflow
steps against the declared data root. The archive checksum and external
sidecar checksum are emitted by the release command and must be recorded beside
the eventual anonymous hosting record rather than copied into this source
document, which would make the archive checksum self-referential. The remaining
submission action is to upload this candidate to a reviewer-accessible anonymous
host and replace the placeholder URL in Croissant metadata.
