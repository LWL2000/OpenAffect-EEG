# DE / bandpower Ridge

## Role

The transparent baseline uses channel-by-band differential entropy or Welch
log power and a validation-selected Ridge estimator. It is the primary sanity
check for whether a foundation representation adds information beyond standard
spectral features.

## Audit boundary

- Fitting statistics are learned from train data only.
- Alpha selection uses validation MAE.
- Trial assignments are frozen before feature windows are created.
- Global, channel, band-leave-out, and region-leave-out views are reported.

The representation is interpretable but strongly encodes subject and dataset
identity. Its accuracy must not be read as direct evidence of subjective affect.
