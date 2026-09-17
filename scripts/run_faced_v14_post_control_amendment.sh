#!/usr/bin/env bash
set -euo pipefail

ROOT=${OPENAFFECT_ROOT:-/home/lwl/OpenAffect-EEG-v14}
DATA=${OPENAFFECT_DATA_ROOT:-/opt/lwl_data/OpenAffect-EEG}
ASSIGN=$DATA/derived/confirmatory_faced_v14_assignments
DERIVED=$DATA/derived
LOG=$DERIVED/faced_v14_post_control_amendment.log
STATUS=$DERIVED/faced_v14_all_status.tsv
PY=${OPENAFFECT_PYTHON:-/home/lwl/miniforge3/envs/openaffect-eeg/bin/python}
LABRAM_REPO=${LABRAM_REPOSITORY:-/home/lwl/third_party/LaBraM}
CHECKPOINT=${LABRAM_CHECKPOINT:-$DATA/models/labram/labram-base.pth}

NVIDIA_LIBS=$(find /home/lwl/miniforge3/envs/lerobot/lib/python3.12/site-packages/nvidia -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:/usr/local/cuda-13.2/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$ROOT/src:$ROOT/scripts
cd "$ROOT"

exec 9>/home/lwl/run_faced_v14_post_control_amendment.lock
flock -n 9 || { echo "post-amendment driver already active"; exit 3; }

run_step() {
  local name=$1; shift
  printf '%s\tstart\t%s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$STATUS" "$LOG"
  "$@" >>"$LOG" 2>&1
  printf '%s\tcomplete\t%s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$STATUS" "$LOG"
}

if [[ -n "${WAIT_PID:-}" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

LABRAM_OBSERVED=$DERIVED/faced_labram_observed_v14
test -f "$LABRAM_OBSERVED/completion.json"
if [[ ! -f "$DERIVED/faced_labram_observed_analysis_v14/training_uncertainty_summary.json" ]]; then
  run_step labram_observed_analysis "$PY" scripts/analyze_training_uncertainty_v14.py \
    "$LABRAM_OBSERVED/identity_exposure_predictions.tsv.gz" \
    "$DERIVED/faced_labram_observed_analysis_v14" \
    --bootstrap 2000 --seed 2026091400 --control observed --equivalence-scope confirmatory
fi

EEGNET_POSITIVE=$DERIVED/faced_eegnet_synthetic_residual_signal_v14
run_step eegnet_synthetic_residual_signal "$PY" scripts/run_eegnet_v14.py \
  configs/confirmatory_faced_v14_experiment.yaml "$DATA" "$ASSIGN" "$EEGNET_POSITIVE" \
  --dataset faced_confirmation_v14 --control synthetic_residual_signal
run_step eegnet_synthetic_residual_signal_analysis "$PY" scripts/analyze_training_uncertainty_v14.py \
  "$EEGNET_POSITIVE/identity_exposure_predictions.tsv.gz" \
  "$DERIVED/faced_eegnet_synthetic_residual_signal_analysis_v14" \
  --bootstrap 2000 --seed 2026091400 --control synthetic_residual_signal --equivalence-scope exploratory
"$PY" - "$DERIVED/faced_eegnet_synthetic_residual_signal_analysis_v14/training_uncertainty_summary.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if json.loads(path.read_text())["control_diagnostic"]["warning"]:
    raise SystemExit("Corrected EEGNet positive control failed; stopping before further controls")
PY

for control in label_permutation synthetic_residual_signal; do
  out="$DERIVED/faced_labram_${control}_v14"
  run_step "labram_${control}" "$PY" scripts/run_labram_peft_v14.py \
    configs/confirmatory_faced_v14_experiment.yaml "$DATA" "$ASSIGN" \
    "$LABRAM_REPO" "$CHECKPOINT" "$out" \
    --dataset faced_confirmation_v14 --control "$control"
  run_step "labram_${control}_analysis" "$PY" scripts/analyze_training_uncertainty_v14.py \
    "$out/identity_exposure_predictions.tsv.gz" \
    "$DERIVED/faced_labram_${control}_analysis_v14" \
    --bootstrap 2000 --seed 2026091400 --control "$control" --equivalence-scope exploratory
done

"$PY" - "$DERIVED/faced_labram_synthetic_residual_signal_analysis_v14/training_uncertainty_summary.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if json.loads(path.read_text())["control_diagnostic"]["warning"]:
    raise SystemExit("Corrected LaBraM positive control failed")
PY

printf '%s\tcomplete\tpost_control_amendment_all\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS" "$LOG"
