# CBraMod

## Role

CBraMod is evaluated as a frozen EEG foundation encoder. The official
pretrained checkpoint and a fixed-seed same-architecture random initialization
are extracted over the same EmoEEG-MC trials and scored with identical heads.

## Audit boundary

- Channel order, sampling, crop policy, checkpoint hash, and source commit are
  recorded with each feature archive.
- Pretrained-versus-random comparisons use trial-aligned predictions and 2,000
  clustered bootstrap samples.
- No downstream encoder fine-tuning is performed.

Pretraining improves selected seen-subject settings but does not establish
strict joint or cross-context self-report prediction.
