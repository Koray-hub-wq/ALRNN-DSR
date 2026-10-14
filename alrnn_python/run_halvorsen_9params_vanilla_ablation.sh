#!/usr/bin/env bash
set -euo pipefail

# Activate the Python environment before starting. Optional first argument: GPU ID.
# Example: bash run_halvorsen_9params_vanilla_ablation.sh 0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/export/home/klenkeit/training_data/single_halvorsen_constparams/14to209_noisy_9params_trajectories"
BIFURCATION_DIR="/export/home/klenkeit/diffrentbutconst_prams"
EXISTING_RUN="/export/home/klenkeit/ALRNN-DSR/alrnn_python/runs/halvorsen_9params_all_m100_p10_vanilla"
OUTPUT_DIR="$SCRIPT_DIR/bifurcation_diagrams/alrnn_vanilla"

gpu_id="${1:-${GPU_ID:-0}}"
python_bin="${PYTHON_BIN:-python}"
latents_dim=200
epochs=1000
pwl_units=20
batch_size=80
batches_per_epoch=100

RUN_NEGATIVE="$SCRIPT_DIR/runs/halvorsen_9params_train_m1_m0p75_m0p5_m200_p20_vanilla"
RUN_MIXED="$SCRIPT_DIR/runs/halvorsen_9params_train_m0p25_p0p25_p0p75_m200_p20_vanilla"
RUN_SINGLE="$SCRIPT_DIR/runs/halvorsen_9params_train_m0p75_m200_p20_vanilla"

# Check required inputs before starting the long training runs.
for required_file in \
  "$SCRIPT_DIR/train_halvorsen_plain.py" \
  "$DATA_DIR/trajectories.npy" \
  "$DATA_DIR/trajectory_phi_constants.npy" \
  "$BIFURCATION_DIR/compare_model_bifurcation_diagrams.py" \
  "$EXISTING_RUN/model.pt" \
  "$EXISTING_RUN/metrics.json"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'Missing required file: %s\n' "$required_file" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --multi-trajectory
  --gpu-id "$gpu_id"
  --cpu-threads 4
  --latent-dim "$latents_dim"
  --pwl-units "$pwl_units"
  --batch-size "$batch_size"
  --epochs "$epochs"
  --batches-per-epoch "$batches_per_epoch"
)

cd "$SCRIPT_DIR"

# No --use-phi: phi values only select/balance training data, not model inputs.
printf '\nTraining vanilla AL-RNN on phi = -1, -0.75, -0.5 (GPU %s)\n' "$gpu_id"
"$python_bin" train_halvorsen_plain.py \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -1 -0.75 -0.5 \
  --output-dir "$RUN_NEGATIVE"

printf '\nTraining vanilla AL-RNN on phi = -0.25, 0.25, 0.75\n'
"$python_bin" train_halvorsen_plain.py \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -0.25 0.25 0.75 \
  --output-dir "$RUN_MIXED"

printf '\nTraining vanilla AL-RNN on phi = -0.75\n'
"$python_bin" train_halvorsen_plain.py \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -0.75 \
  --output-dir "$RUN_SINGLE"

printf '\nComparing bifurcation diagrams for all four vanilla models\n'
cd "$BIFURCATION_DIR"
"$python_bin" compare_model_bifurcation_diagrams.py \
  --model-type alrnn \
  --regime halvorsen \
  --model-runs "$RUN_NEGATIVE" "$RUN_MIXED" "$RUN_SINGLE" "$EXISTING_RUN" \
  --model-labels \
    "Vanilla: phi {-1, -0.75, -0.5}" \
    "Vanilla: phi {-0.25, 0.25, 0.75}" \
    "Vanilla: phi {-0.75}" \
    "Vanilla: all 9 phi values" \
  --components z \
  --output "$OUTPUT_DIR/halvorsen_vanilla_ablation_bifurcation_compare.png" \
  --cache "$OUTPUT_DIR/halvorsen_vanilla_ablation_bifurcation_cache.npz" \
  --force-recompute \
  --device cuda \
  --gpu-id "$gpu_id" \
  --torch-threads 4 \
  --num-parameters 320

printf '\nFinished. Bifurcation diagram: %s\n' \
  "$OUTPUT_DIR/halvorsen_vanilla_ablation_bifurcation_compare.png"
