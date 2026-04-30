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

To reproduce runs without copying commands manually:

```bash
bash scripts/run.sh
```

For evaluation, use `NRMSE` as the primary metric. `RMSE` can still be logged as a secondary reference, but model comparisons in this repo should be made by validation `NRMSE`.

Latest fair-size runs use parameter matching with `--target-params 134000`. The resolved hidden sizes from the latest saved runs were:

- `gnn`: `72`
- `transolver`: `64`
- `flare`: `48`
- `gnot`: `24`
- `lno`: `56`
- `fno`: `16`

GNN fair run:

```bash
source flowpde/bin/activate
python train.py \
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
```

Transolver fair run:

```bash
source flowpde/bin/activate
python train.py \
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
```

FLARE fair run:

```bash
source flowpde/bin/activate
python train.py \
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
```

GNOT fair run:

```bash
source flowpde/bin/activate
python train.py \
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
```

LNO fair run:

```bash
source flowpde/bin/activate
python train.py \
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
```

FNO fair run:

```bash
source flowpde/bin/activate
python train.py \
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
```

## Repo Layout

- `datasets/`: dataset loading and preprocessing
- `metrics.py`: RMSE, NRMSE, and MSE helpers
- `models/`: GNN, Transolver, FLARE, GNOT, LNO, and FNO baselines
- `train.py`: training loop
- `main.py`: CLI entrypoint
