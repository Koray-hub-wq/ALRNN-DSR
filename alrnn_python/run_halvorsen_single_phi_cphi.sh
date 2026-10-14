#!/usr/bin/env bash
set -euo pipefail

# Optional first argument: GPU ID.
# Example: bash run_halvorsen_single_phi_cphi.sh 0
ROOT_DIR="${ROOT_DIR:-$HOME}"
BTIP_REPO="$ROOT_DIR/DynaMix-python-b-tipping"
source "$BTIP_REPO/venv/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/export/home/klenkeit/training_data/single_halvorsen_constparams/14to209_noisy_9params_trajectories"
BIFURCATION_DIR="/export/home/klenkeit/diffrentbutconst_prams"
OUTPUT_DIR="$SCRIPT_DIR/bifurcation_diagrams/alrnn"

gpu_id="${1:-${GPU_ID:-0}}"
python_bin="${PYTHON_BIN:-python}"
latents_dim=200
epochs=700
pwl_units=20
batch_size=80
batches_per_epoch=100

RUN_DIR="$SCRIPT_DIR/runs/halvorsen_9params_train_m0p75_m200_p20_cphi"

for required_file in \
  "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "$DATA_DIR/trajectories.npy" \
  "$DATA_DIR/trajectory_phi.npy" \
  "$DATA_DIR/trajectory_phi_constants.npy" \
  "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Missing required file: %s\n' "$required_file" >&2
    exit 1
  fi
done

cd "$SCRIPT_DIR"
printf '\nTraining AL-RNN with Cphi on phi = -0.75 (GPU %s)\n' "$gpu_id"
"$python_bin" train_halvorsen_plain.py \
  --data-dir "$DATA_DIR" \
  --multi-trajectory \
  --use-phi \
  --train-phi-values -0.75 \
  --gpu-id "$gpu_id" \
  --cpu-threads 4 \
  --latent-dim "$latents_dim" \
  --pwl-units "$pwl_units" \
  --batch-size "$batch_size" \
  --epochs "$epochs" \
  --batches-per-epoch "$batches_per_epoch" \
  --output-dir "$RUN_DIR"

printf '\nCreating the bifurcation diagram for the single-phi Cphi model\n'
cd "$BIFURCATION_DIR"
"$python_bin" compare_model_bifurcation_diagrams.py \
  --model-type alrnn \
  --regime halvorsen \
  --model-runs "$RUN_DIR" \
  --model-labels "Cphi: trained on phi {-0.75}" \
  --components z \
  --output "$OUTPUT_DIR/halvorsen_train_m0p75_cphi_bifurcation_compare.png" \
  --cache "$OUTPUT_DIR/halvorsen_train_m0p75_cphi_bifurcation_cache.npz" \
  --force-recompute \
  --device cuda \
  --gpu-id "$gpu_id" \
  --torch-threads 4 \
  --num-parameters 320

printf '\nFinished. Bifurcation diagram: %s\n' \
  "$OUTPUT_DIR/halvorsen_train_m0p75_cphi_bifurcation_compare.png"
