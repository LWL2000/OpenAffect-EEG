# LaBraM

## Role

LaBraM is used as a frozen EEG foundation encoder in the development benchmark.
V14 also evaluates final-four-block adaptation, described below. The adapter
uses the official base patch-200 architecture and checkpoint from
https://github.com/935963004/LaBraM at commit
`c431221e6cfd23dbfa9950e0180682fb322b0548`.

## Input and pooling

- EmoEEG-MC reorder arrays: 64 channels, 200 Hz, 30 seconds, microvolts
- One-second 200-sample patches
- Three 8-patch windows plus one 6-patch window, with patch-count-weighted token
  pooling
- Official encoder normalization; pretraining-only mask and decoder heads are
  excluded

The official checkpoint SHA-256 is
`7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`.
Pretrained and fixed-random features cover the same 1,260 unique trials. The
pretrained representation strongly encodes participant identity and does not
outperform random initialization under joint or directional context transfer.

## V14 adapted-model evaluation

The final four transformer blocks, final normalization and a linear residual
regression head are trained with five fixed seeds. FACED uses the harmonized
30-channel montage, 100 Hz stored signals and ten-second crops converted to
the official patch interface. Validation MAE selects duration before refitting
on the permitted training and validation union. Backbone and head learning
rates are 1e-5 and 1e-3; AdamW weight decay is 0.01, with batch size eight,
40 maximum epochs and patience eight. The configuration, protocol and CLI map
are in `docs/v14_reproduction.md`.

The FACED observed grid includes five full-refit rotations. Same-data
EmoEEG-MC/Urban sensitivities have a single fixed rotation and are not
independent confirmation. FACED average equivalence is model/design-specific;
failed locked neural controls remain a limitation despite subsequent strong
signed-power recovery. Upstream pretraining exposure is not certified by the
checkpoint digest. Weights and fitted checkpoints are not redistributed.
