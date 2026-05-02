# cylinder-flow

Simple baselines for the cylinder-flow dataset on an irregular mesh.

## Python

Use Python `3.10+`.

This repo was tested with:

- Python `3.11`
- `uv` for environment management
- PyTorch `2.11.0+cu130`

## Install

Create the environment with `uv`:

```bash
uv venv flowpde
uv sync --python flowpde/bin/python
```

If you prefer `pip` instead:

```bash
python3 -m venv flowpde
flowpde/bin/python -m pip install --upgrade pip
flowpde/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
```

For macOS, use the CPU-only requirements file instead:

```bash
python3 -m venv flowpde
flowpde/bin/python -m pip install --upgrade pip
flowpde/bin/python -m pip install -r requirements-mac.txt
```

## Download The Dataset

The training code uses the downloaded Hugging Face dataset referenced in `problem.md`.

Dataset page:

- https://huggingface.co/datasets/ayz2/ldm_pdes/tree/main

Direct download used here:

```bash
mkdir -p data
curl -L --fail -o data/cylinder_flow_captioned.zip \
  'https://huggingface.co/datasets/ayz2/ldm_pdes/resolve/main/cylinder_flow_captioned.zip?download=true'
```

The loader will automatically extract the required files from the zip into `data/` when needed.

## Reproduce Reported Results Without Retraining

The reported baseline artifacts are already saved under:

- `out/checkpoints/flare`
- `out/checkpoints/transolver`
- `out/checkpoints/gnn`
- `out/checkpoints/fno`
- `out/checkpoints/gnot`
- `out/checkpoints/lno`

Each checkpoint folder contains:

- `*_best.pt`: saved model checkpoint
- `*_results.json`: best saved train/validation metrics
- `history.json`: per-epoch training history used for plots

The main reported summary files are:

- `out/results.md`
- `out/tables.tex`

To inspect the saved metrics directly:

```bash
cat out/checkpoints/flare/flare_results.json
cat out/checkpoints/transolver/transolver_results.json
cat out/checkpoints/gnn/gnn_results.json
cat out/checkpoints/fno/fno_results.json
cat out/checkpoints/gnot/gnot_results.json
cat out/checkpoints/lno/lno_results.json
```

To regenerate the training-dynamics analysis from the saved histories only:

```bash
flowpde/bin/python scripts/analyze_training_dynamics.py \
  --root-dir out/checkpoints \
  --output-dir out/reproduced_training_dynamics \
  --models flare transolver gnn fno gnot lno
```

To inspect the saved plots already included in the repo:

- `out/plots/validation_curves.png`
- `out/plots/validation_nrmse.png`
- `out/plots/validation_rmse.png`
- `out/plots/training_loss_curves.png`
- `out/plots/validation_loss_curves.png`
- `out/plots/peak_gpu_memory_by_epoch.png`
- `out/plots/epoch_time_by_epoch.png`

You do not need to rerun training to reproduce the tables and plots above.

## Train From Scratch

To rerun the full sweep in one command:

```bash
bash scripts/run.sh
```

The script now bootstraps the repo end to end:

- reuses `flowpde/bin/python` if it already exists
- otherwise creates `.venv/` and installs dependencies
- downloads `data/cylinder_flow_captioned.zip` automatically if needed
- runs the six training jobs from the README

Useful overrides:

```bash
# Run just one model.
MODELS=gnn bash scripts/run.sh

# Smoke test on a small subset.
MODELS=gnn EPOCHS=1 MAX_TRAIN_SAMPLES=8 MAX_VALID_SAMPLES=4 bash scripts/run.sh

# Use an existing Python instead of creating .venv.
PYTHON_BIN=$(which python3) MODELS=gnn bash scripts/run.sh
```

For evaluation, use `NRMSE` as the primary metric. `RMSE` can still be logged as a secondary reference, but model comparisons in this repo should be made by validation `NRMSE`.

Latest runs use parameter matching with `--target-params 134000`. The resolved hidden sizes from the latest saved runs were:

- `gnn`: `72`
- `transolver`: `64`
- `flare`: `48`
- `gnot`: `24`
- `lno`: `56`
- `fno`: `16`

GNN run:

```bash
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
  --save-dir out/checkpoints/gnn
```

Transolver run:

```bash
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
  --save-dir out/checkpoints/transolver
```

FLARE run:

```bash
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
  --save-dir out/checkpoints/flare
```

GNOT run:

```bash
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
  --save-dir out/checkpoints/gnot
```

LNO run:

```bash
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
  --learning-rate 5e-4 \
  --min-learning-rate 5e-5 \
  --weight-decay 1e-5 \
  --grad-clip 0.5 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/checkpoints/lno
```

FNO run:

```bash
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
  --save-dir out/checkpoints/fno
```

## Repo Layout

- `datasets/`: dataset loading and preprocessing
- `metrics.py`: RMSE, NRMSE, and MSE helpers
- `models/`: GNN, Transolver, FLARE, GNOT, LNO, and FNO baselines
- `train.py`: training loop
- `main.py`: CLI entrypoint
