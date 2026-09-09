# EEGNet

## Role

EEGNet is the end-to-end specialist control trained on fixed trial crops. It
tests whether learning directly from waveforms changes the benchmark conclusion
relative to frozen spectral and foundation representations.

## Audit boundary

- Only the joint subject-stimulus protocol is used for the frozen v1 result.
- Crop policy and trial IDs are recorded in the result sidecar.
- Zero-adversary and selected-adversary variants share data and optimization
  settings.

The current result is a controlled specialist baseline, not a claim that EEGNet
is exhaustively tuned for every source dataset.
