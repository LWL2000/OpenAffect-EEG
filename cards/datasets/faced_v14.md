# FACED v14 evaluation data card

Source: NEMAR nm000112 v1.1.3, DOI 10.82901/nemar.nm000112, CC BY 4.0.
Original study: Chen et al. (2023), DOI 10.1038/s41597-023-02650-w.
The BIDS conversion and original collection are credited separately.

Evaluation uses 123 complete participants, 28 common videos and 3,444 trials.
Targets are individual valence and arousal ratings transformed to [-1, 1].
Input is the final 30 seconds before video offset, a common 30-channel scalp
montage, common-average reference, 0.5–45 Hz filtering and 100 Hz resampling.
The source has two channel schemas; a pre-outcome header-only amendment
specifies their intersection. No outcome-dependent rejection or imputation is
used. All eligible observed trials are retained.

Five-by-four crossed identity folds and five full-refit rotations define 100
splits. Participant and stimulus label doses are independently 0, 1, 2, 4 or 8.
Nonzero doses expose that identity under a declared budget; they cannot be
called zero-calibration or joint unseen-identity deployments.

This was a previously unused cohort under a locked observed-data design.
Failed locked neural controls and later exploratory signal redesign prevent a
fully preregistered complete-pipeline confirmation claim. The signed-power
subset has a different support (85 participants and 23 videos) and is not a
replacement for the observed full grid.

The public release distributes only protocol/configuration files, aggregate
statistics, source digests and code. EEG, individual features, trial predictions
and source media are not redistributed. Consent and original use terms apply.
Self-ratings are measurements of reported affect, not clinical state or direct
access to a private emotion. Cross-population, device, session and pretraining
exposure generalization is not established.
