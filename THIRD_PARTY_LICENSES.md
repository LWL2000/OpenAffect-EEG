# Third-party resources

OpenAffect-EEG code is released under the MIT License. That license does not
change the terms of upstream datasets, stimuli, model implementations, or
checkpoints.

| Resource | Use in this project | Upstream terms | Redistribution boundary |
|---|---|---|---|
| MusicEEG, OpenNeuro `ds002721` | User-supplied EEG and metadata | Consult the version-pinned OpenNeuro snapshot | Raw EEG and soundtrack media are not included |
| DENS, OpenNeuro `ds003751` | User-supplied EEG and metadata | Consult the version-pinned OpenNeuro snapshot; any conflicting metadata must be resolved against the upstream snapshot before release | Raw EEG and video stimuli are not included |
| EmoEEG-MC, OpenNeuro `ds005540` | User-supplied EEG, DE/reorder derivatives, and metadata | Consult the version-pinned OpenNeuro snapshot and derivative provider | EEG, DE arrays, reorder arrays, and movies are not included |
| Emotion regulation, OpenNeuro `ds006866` | User-supplied EEG and metadata | Consult the version-pinned OpenNeuro snapshot | EEG and image stimuli are not included |
| CBraMod source code | Optional feature extraction | MIT, according to the official repository | Source and checkpoint are not vendored |
| LaBraM source code | Optional feature extraction | MIT, according to the official repository | Source and checkpoint are not vendored |
| CBraMod and LaBraM checkpoints | Optional frozen feature extraction | Follow the checkpoint provider's terms; a code license is not assumed to license weights or training data | Checkpoints are not included |
| NeurIPS 2026 style files | Manuscript compilation | Distributed by the conference for submission preparation | Included only in the submission source bundle |

The public artifact contains benchmark code, configuration, cards, harmonized
schema definitions, split-building logic, and hash-locked evaluation metadata.
Users obtain every upstream recording, stimulus, derivative, repository, and
checkpoint from its original provider. Source URLs, snapshot identifiers,
commits, and hashes are recorded in the corresponding cards and artifact
registry.
