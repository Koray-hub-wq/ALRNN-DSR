#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="/export/home/klenkeit/training_data/single_halvorsen_constparams/14to209_noisy_9params_trajectories"
gpu_id=0
latents_dim=200
epochs=1000
pwl_units=20
batch_size=80
batches_per_epoch=100


COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --multi-trajectory
  --use-phi
  --gpu-id "$gpu_id"
  --cpu-threads 4
  --latent-dim "$latents_dim"
  --pwl-units "$pwl_units"
  --batch-size "$batch_size"
  --epochs "$epochs"
  --batches-per-epoch "$batches_per_epoch"
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
  --gpu-id "$gpu_id"
  --cpu-threads 4
  --latent-dim "$latents_dim"
  --pwl-units "$pwl_units"
  --batch-size "$batch_size"
  --epochs "$epochs"
  --batches-per-epoch "$batches_per_epoch"
)

python train_halvorsen_plain.py \
  "${VANILLA_ARGS[@]}" \
  --output-dir runs/halvorsen_9params_all_m100_p10_vanilla
