# LaBraM

## Role

LaBraM is the second frozen EEG foundation encoder in the benchmark. The adapter
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
