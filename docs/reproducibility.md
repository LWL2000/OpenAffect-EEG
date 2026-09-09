# Reproducibility

OpenAffect-EEG separates public metadata checks, synthetic code checks, and
full-data result validation. None of the modes downloads data implicitly.

## Install

```bash
conda env create -f environment.yml
conda activate openaffect-eeg
python -m pip install -e ".[smoke]"
```

The direct Conda dependencies are pinned to the linux-64 versions exercised by
the anonymous-package gate. Use `.[dev,eeg,paper]` for development, EEG I/O,
and manuscript assets; install foundation dependencies only in the GPU
environment used for feature extraction.

Foundation feature extraction may run in a separate GPU environment. The frozen
LaBraM v1 features were produced in `lerobot` with Python 3.12, PyTorch 2.11,
CUDA 13, and `timm` 1.0.28. Statistical evaluation runs in `openaffect-eeg`.

## Metadata mode

```bash
python scripts/run_benchmark.py metadata \
  --output-root /tmp/openaffect-metadata
```

This validates `configs/benchmark.yaml` and `croissant.json`, then regenerates
the evaluation cards. It does not resolve or inspect a data root.

## Synthetic smoke mode

```bash
python scripts/run_benchmark.py smoke \
  --output-root /tmp/openaffect-smoke
```

This runs synthetic split, baseline, probe, and registry tests and compiles a
small hashed synthetic result registry. It requires the `smoke` dependencies but
does not require an OpenNeuro checkout.

## Anonymous release

```bash
python scripts/build_public_release.py \
  . configs/public_release.yaml OpenAffect-EEG-anonymous.tar.gz
```

The release builder includes only tracked allowlisted files, rejects raw EEG,
feature arrays, checkpoints, media, symlinks, local paths, and email addresses,
and writes a deterministic archive plus an external hash manifest. Every file
inside the archive is covered by `PUBLIC_RELEASE_MANIFEST.json`.

The review-ready V11 closure uses `configs/submission_release_v11.yaml` and
`scripts/freeze_public_release.py`. It packages the current allowlisted
manuscript state in a separate Git-indexed snapshot, including the known-truth
evaluator diagnostics and the machine-readable external claim repair, without
modifying the development checkout. External anonymous hosting remains a
submission-time responsibility.

## Full frozen-result mode

```bash
python scripts/run_benchmark.py full \
  --data-root "${OPENAFFECT_DATA_ROOT}" \
  --output-root "${OPENAFFECT_DATA_ROOT}/derived/reproduction/neurips_ed_v1"
```

Full mode requires an explicit existing data root. It verifies all paths,
protocols, five-seed coverage, finite values, duplicate prediction keys, and
SHA-256 values declared in `configs/paper_artifacts.yaml`, then regenerates the
paper tables and provenance manifest. It validates frozen results; raw feature
extraction and model-training commands remain explicit in `README.md` and are
never started by metadata or smoke mode.

## Raw recomputation plan

The frozen-result check above is intentionally fast. To inspect or execute the
full raw-data-to-paper dependency graph, use the separately named recomputation
entry point:

```bash
python scripts/recompute_benchmark.py plan \
  --data-root "${OPENAFFECT_DATA_ROOT}" \
  --output-root "${OPENAFFECT_DATA_ROOT}/derived/recomputed/neurips_ed_v1" \
  --statistics-python "${CONDA_PREFIX}/bin/python" \
  --foundation-python /path/to/gpu-environment/bin/python \
  --foundation-device cuda \
  --cbramod-repository /path/to/CBraMod \
  --labram-repository /path/to/LaBraM \
  --report /tmp/openaffect-recompute-plan.json
```

`plan` never executes a model. It resolves the typed DAG, lists every command,
reports missing source recordings/checkpoints/repositories, and confirms the
declared outputs. The versioned plan contains stages for manifests, splits,
transparent features, foundation features, evaluations, identity/feature
audits, paired statistics, and final registry compilation.

After the readiness report has `"ready": true`, execute all stages with:

```bash
python scripts/recompute_benchmark.py run \
  --data-root "${OPENAFFECT_DATA_ROOT}" \
  --output-root "${OPENAFFECT_DATA_ROOT}/derived/recomputed/neurips_ed_v1" \
  --statistics-python "${CONDA_PREFIX}/bin/python" \
  --foundation-python /path/to/gpu-environment/bin/python \
  --foundation-device cuda \
  --cbramod-repository /path/to/CBraMod \
  --labram-repository /path/to/LaBraM \
  --resume \
  --report /tmp/openaffect-recompute-run.json
```

Use repeated `--stage` options to select explicit stages. `--resume` skips a
step only when every output declared by that step already exists; missing
partial outputs cause the step to run again. Commands are executed as argument
tuples without a shell. The plan does not download restricted stimuli or model
checkpoints implicitly.

## Data boundary

The repository may publish harmonized trial metadata, split assignments,
evaluation cards, hashes, derived aggregate tables, and source code. Users must
obtain raw EEG, precomputed source derivatives, and movie/music/image stimuli
from the original providers. Do not upload raw EEG, DE arrays, or restricted
stimuli to the public artifact.
