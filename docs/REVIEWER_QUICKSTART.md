# Reviewer quickstart

This artifact requires Python 3.11 or 3.12 and does not download EEG data,
stimulus media, feature arrays, or model weights during acceptance testing.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ".[audit]"
python scripts/verify_review_artifact.py \
  --project-root . \
  --output acceptance-run
```

On Windows PowerShell, activate with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[audit]"
.\.venv\Scripts\python.exe scripts\verify_review_artifact.py `
  --project-root . `
  --output acceptance-run
```

A successful run writes `acceptance-run/independent_acceptance_report.json` and
returns exit code zero. It verifies every file against the release manifest,
runs `pip check`, executes the synthetic audit, rebuilds the source-linked
evidence summaries with their frozen hashes, and verifies the V13 five-block
block-$t$ derivation and simulation artifacts. The final release manifest
records every included file.

A passing report establishes executable behavior on the tested machine. It is
independent-user evidence only if the operator did not participate in artifact
development. The test does not download restricted data, retrain EEG models,
or establish independent replication of the empirical paper results.
