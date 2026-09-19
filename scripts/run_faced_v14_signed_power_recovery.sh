#!/usr/bin/env bash
set -euo pipefail

ROOT=${OPENAFFECT_ROOT:-/home/lwl/OpenAffect-EEG-v14}
DATA=${OPENAFFECT_DATA_ROOT:-/opt/lwl_data/OpenAffect-EEG}
ASSIGN=$DATA/derived/confirmatory_faced_v14_assignments
DERIVED=$DATA/derived
LOG=$DERIVED/faced_v14_signed_power_recovery.log
STATUS=$DERIVED/faced_v14_all_status.tsv
PY=${OPENAFFECT_PYTHON:-/home/lwl/miniforge3/envs/openaffect-eeg/bin/python}
LABRAM_REPO=${LABRAM_REPOSITORY:-/home/lwl/third_party/LaBraM}
CHECKPOINT=${LABRAM_CHECKPOINT:-$DATA/models/labram/labram-base.pth}
LABRAM_RUNNER=${LABRAM_RUNNER:-scripts/run_labram_peft_v14.py}

NVIDIA_LIBS=$(find /home/lwl/miniforge3/envs/lerobot/lib/python3.12/site-packages/nvidia -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH="$NVIDIA_LIBS:/usr/local/cuda-13.2/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$ROOT/src:$ROOT/scripts
cd "$ROOT"

exec 9>/home/lwl/run_faced_v14_signed_power_recovery.lock
flock -n 9 || { echo "signed-power recovery driver already active"; exit 3; }

run_step() {
  local name=$1; shift
  printf '%s\tstart\t%s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$STATUS" "$LOG"
  "$@" >>"$LOG" 2>&1
  printf '%s\tcomplete\t%s\n' "$(date --iso-8601=seconds)" "$name" | tee -a "$STATUS" "$LOG"
}

PILOT=$DERIVED/faced_eegnet_signed_power_pilot_v14/pilot_report.json
"$PY" - "$PILOT" <<'PY'
import json, sys
from pathlib import Path
x = json.loads(Path(sys.argv[1]).read_text())
if x["selected_validation_mae"] >= 0.10:
    raise SystemExit("Signed-power pilot validation MAE is not below 0.10")
for dose in ("0", "8"):
    if x["evaluations"][dose]["combined"]["macro"]["ccc"] < 0.90:
        raise SystemExit(f"Signed-power pilot CCC failed at participant dose {dose}")
PY

EEGNET_POSITIVE=$DERIVED/faced_eegnet_synthetic_signed_power_residual_v14
run_step eegnet_synthetic_signed_power_residual "$PY" scripts/run_eegnet_v14.py \
  configs/confirmatory_faced_v14_experiment.yaml "$DATA" "$ASSIGN" "$EEGNET_POSITIVE" \
  --dataset faced_confirmation_v14 --control synthetic_signed_power_residual
run_step eegnet_synthetic_signed_power_residual_analysis "$PY" scripts/analyze_training_uncertainty_v14.py \
  "$EEGNET_POSITIVE/identity_exposure_predictions.tsv.gz" \
  "$DERIVED/faced_eegnet_synthetic_signed_power_residual_analysis_v14" \
  --bootstrap 2000 --seed 2026091400 --control synthetic_signed_power_residual \
  --equivalence-scope exploratory
"$PY" - "$DERIVED/faced_eegnet_synthetic_signed_power_residual_analysis_v14/training_uncertainty_summary.json" <<'PY'
import json, sys
from pathlib import Path
if json.loads(Path(sys.argv[1]).read_text())["control_diagnostic"]["warning"]:
    raise SystemExit("Full-grid signed-power EEGNet diagnostic failed")
PY

for control in label_permutation synthetic_signed_power_residual; do
  out="$DERIVED/faced_labram_${control}_v14"
  run_step "labram_${control}" "$PY" "$LABRAM_RUNNER" \
    configs/confirmatory_faced_v14_experiment.yaml "$DATA" "$ASSIGN" \
    "$LABRAM_REPO" "$CHECKPOINT" "$out" \
    --dataset faced_confirmation_v14 --control "$control"
  run_step "labram_${control}_analysis" "$PY" scripts/analyze_training_uncertainty_v14.py \
    "$out/identity_exposure_predictions.tsv.gz" \
    "$DERIVED/faced_labram_${control}_analysis_v14" \
    --bootstrap 2000 --seed 2026091400 --control "$control" \
    --equivalence-scope exploratory
done

"$PY" - "$DERIVED/faced_labram_synthetic_signed_power_residual_analysis_v14/training_uncertainty_summary.json" <<'PY'
import json, sys
from pathlib import Path
if json.loads(Path(sys.argv[1]).read_text())["control_diagnostic"]["warning"]:
    raise SystemExit("Full-grid signed-power LaBraM diagnostic failed")
PY

printf '%s\tcomplete\tsigned_power_recovery_all\n' "$(date --iso-8601=seconds)" | tee -a "$STATUS" "$LOG"
