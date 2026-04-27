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
source flowpde/bin/activate
uv sync
```

If you prefer `pip` instead:

```bash
python3 -m venv flowpde
source flowpde/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
```

For macOS, use the CPU-only requirements file instead:

```bash
python3 -m venv flowpde
source flowpde/bin/activate
pip install --upgrade pip
pip install -r requirements-mac.txt
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

## Train

GNN baseline:

```bash
source flowpde/bin/activate
python train.py \
  --model gnn \
  --epochs 10 \
  --hidden-dim 128 \
  --num-layers 4 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/proper_gnn_baseline
```

Transolver baseline:

```bash
source flowpde/bin/activate
python train.py \
  --model transolver \
  --epochs 10 \
  --hidden-dim 128 \
  --num-layers 4 \
  --num-heads 4 \
  --num-slices 32 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/proper_transolver_baseline
```

FLARE baseline:

```bash
source flowpde/bin/activate
python train.py \
  --model flare \
  --epochs 10 \
  --hidden-dim 128 \
  --num-layers 4 \
  --num-heads 4 \
  --num-slices 32 \
  --device cuda \
  --window-stride 10 \
  --save-dir out/proper_flare_baseline
```

## Repo Layout

- `datasets/`: dataset loading and preprocessing
- `metrics/`: RMSE, NRMSE, and MSE helpers
- `models/`: simple MLP, GNN, and Transolver baselines
- `train.py`: training loop
- `main.py`: thin CLI entrypoint
