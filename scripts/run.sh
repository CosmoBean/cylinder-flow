#!/usr/bin/env bash

# Stop immediately if any run fails.
set -e

# Always run from the repository root.
cd "$(dirname "$0")/.."

python train.py \
  --model gnn \
  --epochs 10 \
  --hidden-dim 128 \
  --num-layers 4 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/proper_gnn_baseline

python train.py \
  --model transolver \
  --epochs 10 \
  --hidden-dim 128 \
  --num-layers 4 \
  --num-heads 4 \
  --num-slices 32 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/proper_transolver_baseline

python train.py \
  --model flare \
  --epochs 10 \
  --hidden-dim 128 \
  --num-layers 4 \
  --num-heads 4 \
  --num-slices 32 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/proper_flare_baseline

python train.py \
  --model gnot \
  --epochs 10 \
  --hidden-dim 128 \
  --num-layers 4 \
  --num-heads 4 \
  --learning-rate 3e-4 \
  --min-learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/final_gnot_tuned

python train.py \
  --model lno \
  --epochs 6 \
  --hidden-dim 192 \
  --num-layers 4 \
  --num-heads 6 \
  --num-slices 64 \
  --learning-rate 3e-4 \
  --min-learning-rate 1e-5 \
  --weight-decay 5e-5 \
  --grad-clip 0.5 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/final_lno_tuned

python train.py \
  --model fno \
  --epochs 5 \
  --hidden-dim 64 \
  --num-layers 4 \
  --fno-modes 8 \
  --fno-grid-size 32 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --max-train-samples 2000 \
  --max-valid-samples 200 \
  --save-dir out/fno_baseline_mid
