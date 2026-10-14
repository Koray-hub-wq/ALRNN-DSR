#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_new_regimes_cphi_parallel.sh [DADRAS_GPU_ID] [TSUCS1_GPU_ID]
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-$ROOT_DIR/training_data/single_new_regimes_const}"
BIFURCATION_DIR="${BIFURCATION_DIR:-$ROOT_DIR/diffrentbutconst_prams}"
OUTPUT_DIR="$SCRIPT_DIR/bifurcation_diagrams/alrnn_cphi"
LOG_DIR="$SCRIPT_DIR/runs/new_regimes_cphi_logs"
python_bin="${PYTHON_BIN:-python}"
dadras_gpu="${1:-${DADRAS_GPU_ID:-0}}"
tsucs1_gpu="${2:-${TSUCS1_GPU_ID:-1}}"

latents_dim=200
epochs=800
pwl_units=20
batch_size=80
batches_per_epoch=100
cpu_threads=4  # Per system; both parallel jobs use this setting.

export OMP_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads"
export PYTHONUNBUFFERED=1

REGIMES=(dadras_e_49_to_64 tsucs1_e_20_to_245)
GPU_IDS=("$dadras_gpu" "$tsucs1_gpu")

for required_file in \
  "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Missing required file: %s\n' "$required_file" >&2
    exit 1
  fi
done
for regime in "${REGIMES[@]}"; do
  for filename in trajectories.npy trajectory_phi.npy trajectory_phi_constants.npy metadata.json; do
    if [[ ! -f "$TRAINING_DATA_ROOT/$regime/values_8/$filename" ]]; then
      printf 'Missing required file: %s\n' "$TRAINING_DATA_ROOT/$regime/values_8/$filename" >&2
      exit 1
    fi
  done
done

# Training otherwise falls back to CPU when CUDA is unavailable.
"$python_bin" - "$dadras_gpu" "$tsucs1_gpu" <<'PY'
import sys
import torch

gpu_ids = [int(value) for value in sys.argv[1:]]
if len(set(gpu_ids)) != 2:
    raise SystemExit("Choose two different GPU IDs for the parallel jobs.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected Python environment.")
for gpu_id in gpu_ids:
    if not 0 <= gpu_id < torch.cuda.device_count():
        raise SystemExit(f"Invalid GPU ID {gpu_id}; visible GPUs: {torch.cuda.device_count()}")
    print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
PY

run_regime() {
  local regime="$1"
  local gpu_id="$2"
  local data_dir="$TRAINING_DATA_ROOT/$regime/values_8"
  local run_root="$SCRIPT_DIR/runs/$regime"
  local output_dir="$OUTPUT_DIR/$regime"
  local phi_groups

  # Use metadata rather than rounded, hard-coded parameter lists.
  phi_groups="$("$python_bin" - "$data_dir" <<'PY'
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
    raise SystemExit(f"Metadata phi values {values} do not match trajectory constants {available}.")
for group in (values, [v for v in values if v < 0], [v for v in values if v > 0]):
    if not group:
        raise SystemExit("Both negative and positive training parameters are required.")
    print(" ".join(str(value) for value in group))
PY
)"
  local -a phi_lines all_phi negative_phi positive_phi
  mapfile -t phi_lines <<< "$phi_groups"
  read -r -a all_phi <<< "${phi_lines[0]}"
  read -r -a negative_phi <<< "${phi_lines[1]}"
  read -r -a positive_phi <<< "${phi_lines[2]}"

  local -a common_args=(
    --data-dir "$data_dir"
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
  local -a selected_phi
  local subset run_dir
  for subset in all negative positive; do
    case "$subset" in
      all) selected_phi=("${all_phi[@]}") ;;
      negative) selected_phi=("${negative_phi[@]}") ;;
      positive) selected_phi=("${positive_phi[@]}") ;;
    esac
    run_dir="$run_root/${subset}_m${latents_dim}_p${pwl_units}_cphi"
    printf '\n[%s, GPU %s] Training %s: phi = %s\n' "$regime" "$gpu_id" "$subset" "${selected_phi[*]}"
    "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
      "${common_args[@]}" \
      --train-phi-values "${selected_phi[@]}" \
      --output-dir "$run_dir"
  done

  printf '\n[%s, GPU %s] Creating bifurcation comparison\n' "$regime" "$gpu_id"
  cd "$BIFURCATION_DIR"
  "$python_bin" compare_model_bifurcation_diagrams.py \
    --model-type alrnn \
    --regime "$regime" \
    --model-runs \
      "$run_root/all_m${latents_dim}_p${pwl_units}_cphi" \
      "$run_root/negative_m${latents_dim}_p${pwl_units}_cphi" \
      "$run_root/positive_m${latents_dim}_p${pwl_units}_cphi" \
    --model-labels "Cphi: all 8 values" "Cphi: negative values" "Cphi: positive values" \
    --components z \
    --output "$output_dir/${regime}_cphi_bifurcation_compare.png" \
    --cache "$output_dir/${regime}_cphi_bifurcation_cache.npz" \
    --force-recompute \
    --device cuda \
    --gpu-id "$gpu_id" \
    --torch-threads "$cpu_threads" \
    --num-parameters 320
  printf '\n[%s] Finished. Diagram: %s\n' "$regime" "$output_dir/${regime}_cphi_bifurcation_compare.png"
}

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)_$$"
pids=()
logs=()
for i in "${!REGIMES[@]}"; do
  log_file="$LOG_DIR/${REGIMES[$i]}_${timestamp}.log"
  logs+=("$log_file")
  run_regime "${REGIMES[$i]}" "${GPU_IDS[$i]}" > "$log_file" 2>&1 &
  pids+=("$!")
  printf 'Started %s on GPU %s (PID %s). Log: %s\n' "${REGIMES[$i]}" "${GPU_IDS[$i]}" "${pids[$i]}" "$log_file"
done

# Always wait for both jobs, even if one fails.
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    printf 'Completed: %s\n' "${REGIMES[$i]}"
  else
    job_status=$?
    printf 'Failed: %s (exit %s). See %s\n' "${REGIMES[$i]}" "$job_status" "${logs[$i]}" >&2
    status=1
  fi
done
exit "$status"
