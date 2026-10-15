#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_rossler_cphi_shifted_split.sh [GPU_ID]
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-$ROOT_DIR/training_data/single_new_regimes_const/trajectories}"
BIFURCATION_DIR="${BIFURCATION_DIR:-$ROOT_DIR/diffrentbutconst_prams}"
regime=rossler_a_005_to_035
DATA_DIR="$TRAINING_DATA_ROOT/$regime"
if [[ ! -f "$DATA_DIR/trajectories.npy" && -d "$DATA_DIR/values_8" ]]; then
  DATA_DIR="$DATA_DIR/values_8"
fi
RUN_ROOT="$SCRIPT_DIR/runs/$regime"
OUTPUT_DIR="$SCRIPT_DIR/bifurcation_diagrams/alrnn_cphi/$regime/shifted_split"
LOG_DIR="$SCRIPT_DIR/runs/new_regimes_cphi_logs"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${1:-${GPU_ID:-0}}"

latents_dim=300
epochs=1000
pwl_units=30
batch_size=80
batches_per_epoch=100
cpu_threads=4

export OMP_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads"
export PYTHONUNBUFFERED=1

RUN_ALL="$RUN_ROOT/all_m${latents_dim}_p${pwl_units}_cphi"
RUN_NEGATIVE="$RUN_ROOT/negative_without_m0p142857_m${latents_dim}_p${pwl_units}_cphi"
RUN_POSITIVE="$RUN_ROOT/positive_plus_m0p142857_m${latents_dim}_p${pwl_units}_cphi"

for required_file in \
  "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "$DATA_DIR/trajectories.npy" \
  "$DATA_DIR/trajectory_phi.npy" \
  "$DATA_DIR/trajectory_phi_constants.npy" \
  "$DATA_DIR/metadata.json" \
  "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Missing required file: %s\n' "$required_file" >&2
    exit 1
  fi
done

"$python_bin" - "$gpu_id" <<'PY'
import sys
import torch

gpu_id = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected Python environment.")
if not 0 <= gpu_id < torch.cuda.device_count():
    raise SystemExit(f"Invalid GPU ID {gpu_id}; visible GPUs: {torch.cuda.device_count()}")
print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
PY

phi_groups="$("$python_bin" - "$DATA_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path
import numpy as np

data_dir = Path(sys.argv[1])
with (data_dir / "metadata.json").open(encoding="utf-8-sig") as stream:
    metadata = json.load(stream)
values = metadata.get("args", {}).get("phi_values")
if values is None:
    values = metadata.get("requested_phi_values")
if values is None:
    raise SystemExit("metadata.json contains no phi_values or requested_phi_values.")
values = sorted(set(float(value) for value in values))
if len(values) != 8 or not all(math.isfinite(value) for value in values):
    raise SystemExit(f"Expected eight finite phi values, found: {values}")
available = np.unique(np.load(data_dir / "trajectory_phi_constants.npy"))
if len(available) != len(values) or not np.allclose(available, values, atol=1e-5):
    raise SystemExit("Metadata phi values do not match trajectory constants.")
moved = [v for v in values if math.isclose(v, -0.142857, rel_tol=0, abs_tol=1e-5)]
if len(moved) != 1:
    raise SystemExit(f"Expected exactly one phi near -0.142857, found: {moved}")
# Shift this one value between groups without modifying the phi fed to the model.
negative = [v for v in values if v < 0 and v != moved[0]]
positive = [v for v in values if v > 0 or v == moved[0]]
if len(negative) != 3 or len(positive) != 5:
    raise SystemExit(f"Expected a 3/5 split; found {negative} and {positive}.")
for group in (values, negative, positive):
    print(" ".join(str(value) for value in group))
PY
)"
mapfile -t phi_lines <<< "$phi_groups"
read -r -a all_phi <<< "${phi_lines[0]}"
read -r -a negative_phi <<< "${phi_lines[1]}"
read -r -a positive_phi <<< "${phi_lines[2]}"

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --multi-trajectory
  --use-phi
  --gpu-id "$gpu_id"
  --cpu-threads "$cpu_threads"
  --latent-dim "$latents_dim"
  --pwl-units "$pwl_units"
  --batch-size "$batch_size"
  --epochs "$epochs"
  --batches-per-epoch "$batches_per_epoch"
)

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${regime}_shifted_split_$(date +%Y%m%d_%H%M%S)_$$.log"
printf 'Sequential Rossler trainings on GPU %s. Log: %s\n' "$gpu_id" "$LOG_FILE"
for subset in all negative positive; do
  case "$subset" in
    all) selected_phi=("${all_phi[@]}"); run_dir="$RUN_ALL" ;;
    negative) selected_phi=("${negative_phi[@]}"); run_dir="$RUN_NEGATIVE" ;;
    positive) selected_phi=("${positive_phi[@]}"); run_dir="$RUN_POSITIVE" ;;
  esac
  printf '\n[%s] Training %s: phi = %s\n' "$regime" "$subset" "${selected_phi[*]}" | tee -a "$LOG_FILE"
  "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
    "${COMMON_ARGS[@]}" \
    --train-phi-values "${selected_phi[@]}" \
    --output-dir "$run_dir" 2>&1 | tee -a "$LOG_FILE"
done

printf '\nComparing Rossler bifurcation diagrams for a in [0.05, 0.35]\n' | tee -a "$LOG_FILE"
cd "$BIFURCATION_DIR"
"$python_bin" compare_model_bifurcation_diagrams.py \
  --model-type alrnn \
  --regime "$regime" \
  --model-runs "$RUN_ALL" "$RUN_NEGATIVE" "$RUN_POSITIVE" \
  --model-labels \
    "Cphi: all 8 values" \
    "Cphi: negative except phi=-0.142857" \
    "Cphi: positive plus phi=-0.142857" \
  --parameter-min 0.05 \
  --parameter-max 0.35 \
  --components z \
  --output "$OUTPUT_DIR/${regime}_cphi_bifurcation_compare.png" \
  --cache "$OUTPUT_DIR/${regime}_cphi_bifurcation_cache.npz" \
  --force-recompute \
  --device cuda \
  --gpu-id "$gpu_id" \
  --torch-threads "$cpu_threads" \
  --num-parameters 320 2>&1 | tee -a "$LOG_FILE"

printf '\nFinished. Bifurcation diagram: %s\n' \
  "$OUTPUT_DIR/${regime}_cphi_bifurcation_compare.png" | tee -a "$LOG_FILE"
