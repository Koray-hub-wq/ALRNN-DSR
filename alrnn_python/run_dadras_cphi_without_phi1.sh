#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_dadras_cphi_without_phi1.sh [GPU_ID]
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DATA_ROOT="${TRAINING_DATA_ROOT:-$ROOT_DIR/training_data/single_new_regimes_const/trajectories}"
BIFURCATION_DIR="${BIFURCATION_DIR:-$ROOT_DIR/diffrentbutconst_prams}"
regime=dadras_e_49_to_64
DATA_DIR="$TRAINING_DATA_ROOT/$regime"
if [[ ! -f "$DATA_DIR/trajectories.npy" && -d "$DATA_DIR/values_8" ]]; then
  DATA_DIR="$DATA_DIR/values_8"
fi
python_bin="${PYTHON_BIN:-python}"
gpu_id="${1:-${GPU_ID:-0}}"

# Training settings.
latents_dim=200
epochs=750
pwl_units=20
batch_size=80
batches_per_epoch=100
cpu_threads=4

export OMP_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads"
export PYTHONUNBUFFERED=1

RUN_ALL="$SCRIPT_DIR/runs/$regime/all_without_phi1_m${latents_dim}_p${pwl_units}_cphi"
RUN_POSITIVE="$SCRIPT_DIR/runs/$regime/positive_without_phi1_m${latents_dim}_p${pwl_units}_cphi"
OUTPUT_DIR="$SCRIPT_DIR/bifurcation_diagrams/alrnn_cphi/$regime/without_phi1_m${latents_dim}_p${pwl_units}"

for required_file in \
  "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
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

# Exclude phi=+1 from training only; keep the original dataset intact.
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
if not all(math.isfinite(value) for value in values):
    raise SystemExit("Metadata phi values must be finite.")
available = np.unique(np.load(data_dir / "trajectory_phi_constants.npy"))
if len(available) != len(values) or not np.allclose(available, values, atol=1e-5):
    raise SystemExit("Metadata phi values do not match trajectory constants.")
selected = [v for v in values if not math.isclose(v, 1.0, rel_tol=0, abs_tol=1e-5)]
positive = [v for v in selected if v > 0]
if len(selected) != 7 or len(positive) != 3:
    raise SystemExit(f"Expected seven retained values, three positive; found {selected}.")
for group in (selected, positive):
    print(" ".join(str(value) for value in group))
PY
)"
mapfile -t phi_lines <<< "$phi_groups"
read -r -a all_phi <<< "${phi_lines[0]}"
read -r -a positive_phi <<< "${phi_lines[1]}"

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

printf '\nTraining Dadras with Cphi on all values except phi=+1: %s\n' "${all_phi[*]}"
"$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "${COMMON_ARGS[@]}" \
  --train-phi-values "${all_phi[@]}" \
  --output-dir "$RUN_ALL"

# Optional positive-only training (disabled).
# printf '\nTraining Dadras with Cphi on positive values except phi=+1: %s\n' "${positive_phi[*]}"
# "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
#   "${COMMON_ARGS[@]}" \
#   --train-phi-values "${positive_phi[@]}" \
#   --output-dir "$RUN_POSITIVE"

printf '\nComparing Dadras AL-RNN with ground truth for e in [4.9, 6.2]\n'
cd "$BIFURCATION_DIR"
"$python_bin" "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  --backend-script "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
  --model-type alrnn \
  --regime "$regime" \
  --model-runs "$RUN_ALL" \
  --model-labels "Cphi: all except phi=+1" \
  --parameter-min 4.9 \
  --parameter-max 6.2 \
  --components z \
  --output "$OUTPUT_DIR/dadras_without_phi1_bifurcation_compare.png" \
  --cache "$OUTPUT_DIR/dadras_without_phi1_bifurcation_cache.npz" \
  --force-recompute \
  --device cuda \
  --gpu-id "$gpu_id" \
  --torch-threads "$cpu_threads" \
  --num-parameters 320

printf '\nFinished. Bifurcation diagram: %s\n' \
  "$OUTPUT_DIR/dadras_without_phi1_bifurcation_compare.png"
