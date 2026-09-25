#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="/export/home/klenkeit/training_data/single_halvorsen_constparams/14to209_noisy_9params_trajectories"

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --multi-trajectory
  --use-phi
  --gpu-id 0
  --cpu-threads 4
  --latent-dim 100
  --pwl-units 10
  --batch-size 24
  --epochs 2000
  --batches-per-epoch 50
)

python train_halvorsen_plain.py \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -1 -0.75 -0.5 \
  --output-dir runs/halvorsen_9params_train_m1_m0p75_m0p5_m100_p10_cphi

python train_halvorsen_plain.py \
  "${COMMON_ARGS[@]}" \
  --train-phi-values -0.25 0.25 0.75 \
  --output-dir runs/halvorsen_9params_train_m0p25_p0p25_p0p75_m100_p10_cphi

VANILLA_ARGS=(
  --data-dir "$DATA_DIR"
  --multi-trajectory
  --gpu-id 0
  --cpu-threads 4
  --latent-dim 100
  --pwl-units 10
  --batch-size 24
  --epochs 2000
  --batches-per-epoch 50
)

python train_halvorsen_plain.py \
  "${VANILLA_ARGS[@]}" \
  --output-dir runs/halvorsen_9params_all_m100_p10_vanilla
