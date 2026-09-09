# Reviewer revision V12

This revision reuses the primary EEG predictions and 200 recorded EEGNet fits. It adds no EEG training, data download, or independent cohort. Prior results and negative cases remain available.

## What changed

- All 300 matched CCC dose cells are shown in one common-scale heatmap; all 48 deployment corners and unrounded paired intervals are exported. This is post-analysis descriptive presentation.
- Resource matching is a fitted-predictor contrast. At fixed stimulus dose, the unmatched resource in EC versus P is participant calibration. Label access does not guarantee equally effective adaptation.
- The exact primary five-block topology is tested with four fixed Gaussian scenarios, 300 replicates per task/scenario and 2,000 shared crossed-bootstrap draws. All 2,400 runs completed, with no failures; 187,200 output rows are dependent cell/method summaries.
- Percentile mean coverage is 0.900/0.927 (Emo/Urban trial signal) and 0.903/0.970 (participant-constant signal). The worst Emo cell covers at 0.193 [Monte Carlo 0.153, 0.242]. Even heuristic t-normal covers that cell at only 0.887. These intervals are conditional diagnostics, not calibrated 95% guarantees.
- Current evidence precedes the historical appendix. English and Chinese PDFs report the same new findings.

## Reproduce without downloading EEG

Use Python 3.11 or 3.12 in an isolated environment:

```bash
python -m pip install ".[audit,paper,dev]"
python -m pytest tests/test_block_validation.py -q
python scripts/verify_block_validation_v12.py
python scripts/build_reviewer_revision_assets.py
```

The first command installs dependencies, not EEG datasets. The plot command reads frozen aggregates. To rerun the new synthetic study only, choose an output directory that does not exist:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/run_block_validation.py configs/block_validation_v12.json validation-rerun
```

The study ran on CPU in the existing research environment in about 439 seconds. Runtime is hardware dependent. Compare configuration/source hashes and aggregate estimates; no training uncertainty is simulated. See `results/block_validation_v12/manifest.json`, `completion.json`, `failures.json`, `summary.csv`, and the full `replicates.csv`.

To rebuild both manuscripts, install a TeX distribution with pdfLaTeX, BibTeX, XeLaTeX, and the Chinese font/packages used by the reading edition, then run:

```bash
python scripts/build_paper_bundle.py --require-final-figures
```

The original primary EEG pipeline is documented in `docs/final_closure_v9_reproduction.md`; source-data access and model dependencies are separate from artifact acceptance. Do not rerun it merely to view the revised results.

## Remaining limits

The new study exposes interval miscalibration; it does not repair general inferential validity. A calibrated replacement requires separate development/validation or a changed support and estimand. Synthetic positive controls do not prove optimal real-task EEG adaptation. The existing external code audits exercise execution paths, not published classifier performance or literature-wide defect prevalence. Author-reported use by a colleague is distinct from a newly inspected independent acceptance log. No new human study, complete retraining uncertainty, or persistent DOI is claimed.
