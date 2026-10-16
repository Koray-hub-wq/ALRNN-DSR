#!/usr/bin/env bash
set -euo pipefail

# From alrnn_python: bash parameter_drift/run_halvorsen_parameter_drift.sh [GPU_ID] [RUN_DIR ...]
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALRNN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ALRNN_DIR"

gpu_id="${1:-${GPU_ID:-0}}"
if (( $# > 0 )); then
  shift
fi
if (( $# > 0 )); then
  model_runs=("$@")
else
  model_runs=("$ALRNN_DIR/runs/halvorsen_14to209_noisy_m200_p20_cphi")
fi

python_bin="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

"$python_bin" "$SCRIPT_DIR/evaluate_parameter_drift.py" \
  --model-runs "${model_runs[@]}" \
  --regime halvorsen \
  --steps 12000 \
  --burn-in-steps 4000 \
  --drift linear \
  --ground-truth "$ALRNN_DIR/runs/parameter_drift/ground_truth/halvorsen_linear_12000.npz" \
  --output-dir "$ALRNN_DIR/runs/parameter_drift/halvorsen_linear" \
  --metric dh:8000:12000 \
  --metric dstsp:0:3000 \
  --metric mse:0:1000 \
  --device cuda \
  --gpu-id "$gpu_id" \
  --cpu-threads 4
