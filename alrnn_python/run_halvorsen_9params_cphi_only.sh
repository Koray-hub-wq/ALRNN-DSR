#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_halvorsen_9params_cphi_only.sh [GPU_ID]
# DRY_RUN=1 prints commands without training or creating output directories.
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${DATA_DIR:-$ROOT_DIR/training_data/single_halvorsen_constparams/14to209_noisy_9params_trajectories}"
BIFURCATION_DIR="${BIFURCATION_DIR:-$ROOT_DIR/diffrentbutconst_prams}"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${1:-${GPU_ID:-0}}"
dry_run="${DRY_RUN:-0}"
if (( $# > 1 )) || [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  printf 'Usage: bash %s [nonnegative GPU_ID]\n' "${BASH_SOURCE[0]}" >&2
  exit 2
fi

latents_dim=200
pwl_units=100
epochs=1000
batch_size=80
batches_per_epoch=100
cpu_threads=4
run_id="$(date +%Y%m%d_%H%M%S)_$$_m${latents_dim}_p${pwl_units}"
RUN_ROOT="$SCRIPT_DIR/runs/term_ablations/halvorsen/cphi/$run_id"
OUTPUT_DIR="$SCRIPT_DIR/bifurcation_diagrams/term_ablations/halvorsen/cphi/$run_id"
RUN_NEGATIVE="$RUN_ROOT/negative"
RUN_MIXED="$RUN_ROOT/mixed"
RUN_FIVE="$RUN_ROOT/five_params"

for required_file in \
  "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  "$DATA_DIR/trajectories.npy" \
  "$DATA_DIR/trajectory_phi.npy" \
  "$DATA_DIR/trajectory_phi_constants.npy" \
  "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Missing required file: %s\n' "$required_file" >&2
    exit 1
  fi
done

export OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads" NUMEXPR_NUM_THREADS="$cpu_threads"
export PYTHONUNBUFFERED=1

run_command() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "$dry_run" != 1 ]]; then
    "$@"
  fi
}

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --multi-trajectory --use-phi
  --gpu-id "$gpu_id" --cpu-threads "$cpu_threads"
  --latent-dim "$latents_dim" --pwl-units "$pwl_units"
  --epochs "$epochs" --batch-size "$batch_size" --batches-per-epoch "$batches_per_epoch"
)

printf 'Three Cphi-only models on GPU %s. Models: %s\n' "$gpu_id" "$RUN_ROOT"
if [[ "$dry_run" == 1 ]]; then
  printf 'DRY RUN: no training, plotting, or output directory creation.\n'
fi
run_command "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -1 -0.75 -0.5 \
  --output-dir "$RUN_NEGATIVE"

run_command "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -0.25 0.25 0.75 \
  --output-dir "$RUN_MIXED"

run_command "$python_bin" "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -1 -0.5 0 0.5 1 \
  --output-dir "$RUN_FIVE"

printf '\nComparing three Cphi models with ground truth over phi=-1..1\n'
cd "$BIFURCATION_DIR"
run_command "$python_bin" "$SCRIPT_DIR/compare_model_bifurcation_diagrams.py" \
  --backend-script "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
  --model-type alrnn --regime halvorsen \
  --model-runs "$RUN_NEGATIVE" "$RUN_MIXED" "$RUN_FIVE" \
  --model-labels \
    "Cphi: phi {-1, -0.75, -0.5}" \
    "Cphi: phi {-0.25, 0.25, 0.75}" \
    "Cphi: phi {-1, -0.5, 0, 0.5, 1}" \
  --phi-min -1 --phi-max 1 --num-parameters 320 --components z \
  --output "$OUTPUT_DIR/halvorsen_cphi_bifurcation_compare.png" \
  --cache "$OUTPUT_DIR/halvorsen_cphi_bifurcation_cache.npz" \
  --force-recompute --device cuda --gpu-id "$gpu_id" --torch-threads "$cpu_threads"

printf '\nFinished. Models: %s\nDiagrams/cache: %s\n' "$RUN_ROOT" "$OUTPUT_DIR"
