#!/usr/bin/env bash

# Stop immediately if any run fails.
set -e

# Always run from the repository root.
cd "$(dirname "$0")/.."

flowpde/bin/python train.py \
  --model gnn \
  --epochs 10 \
  --target-params 134000 \
  --hidden-dim-min 16 \
  --hidden-dim-max 256 \
  --hidden-dim-step 8 \
  --num-layers 4 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/fair_gnn

flowpde/bin/python train.py \
  --model transolver \
  --epochs 10 \
  --target-params 134000 \
  --hidden-dim-min 16 \
  --hidden-dim-max 256 \
  --hidden-dim-step 8 \
  --num-layers 4 \
  --num-heads 4 \
  --num-slices 32 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/fair_transolver

flowpde/bin/python train.py \
  --model flare \
  --epochs 10 \
  --target-params 134000 \
  --hidden-dim-min 16 \
  --hidden-dim-max 256 \
  --hidden-dim-step 8 \
  --num-layers 3 \
  --num-heads 4 \
  --num-slices 32 \
  --learning-rate 5e-4 \
  --min-learning-rate 1e-5 \
  --weight-decay 1e-5 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/fair_flare

flowpde/bin/python train.py \
  --model gnot \
  --epochs 10 \
  --target-params 134000 \
  --hidden-dim-min 16 \
  --hidden-dim-max 256 \
  --hidden-dim-step 8 \
  --num-layers 4 \
  --num-heads 4 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/fair_gnot

flowpde/bin/python train.py \
  --model lno \
  --epochs 10 \
  --target-params 134000 \
  --hidden-dim-min 16 \
  --hidden-dim-max 256 \
  --hidden-dim-step 8 \
  --num-layers 4 \
  --num-heads 4 \
  --num-slices 32 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/fair_lno

flowpde/bin/python train.py \
  --model fno \
  --epochs 10 \
  --target-params 134000 \
  --hidden-dim-min 16 \
  --hidden-dim-max 256 \
  --hidden-dim-step 8 \
  --num-layers 4 \
  --fno-modes 8 \
  --fno-grid-size 32 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/fair_fno
