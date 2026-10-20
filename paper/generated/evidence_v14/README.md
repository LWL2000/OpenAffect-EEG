# Frozen v14 aggregate evidence

This public bundle imports completed analyses without training, simulation,
resampling or downloads. The original summaries and cell CSVs in `archive/`
are byte-preserved and hash-linked by `archive_manifest.json`.

`ridge/` additionally preserves scalar metrics and source-record hashes from
all 2,500 already executed deterministic Ridge cells. The builder verifies the
complete keys and descriptive means. No joint Ridge interval was archived,
so these rows receive no new significance or equivalence inference.

`all_results.csv` includes every completed FACED neural stage and both same-data
LaBraM adaptation analyses. `primary.tex`, `controls.tex`, `values.tex`, and
`resource_corners.csv` are deterministic views, rebuilt with
`scripts/build_v14_paper_assets.py`. The plot source is
`scripts/plot_v14_faced_heatmap.py`. Every resource cell remains visible in its
source CSV; only prespecified corners are singled out for discussion.

The frozen original `equivalence_scope=confirmatory` fields describe the
analysis invocation. They are not a claim that failed locked controls have
been repaired. FACED's complete pipeline cannot be called fully preregistered
confirmation. Signed-power subsets are exploratory; same-data development
cohorts do not acquire prospective equivalence margins.

Five-seed, five-rotation FACED records have repeated trial occurrences, not
additional independent participants. The signed-power subset has 85
participants, 23 stimuli and 15 resource cells; it is not the full crossed
cohort. Older five-block Gaussian validation does not establish coverage of
the newer joint bootstrap. Raw predictions, EEG and model weights are excluded.

See `docs/v14_reproduction.md` for the aggregate-only check and the separate,
expensive source-data reproduction map. The manuscript integration does not
repeat any experiment. All source file digests come from machine records,
superseding typographical hashes in earlier prose status notes.
