# Stable Results

These values were re-checked against the saved `*_results.json` and `history.json` files for each stable run.

Stable run sources:
- `flare`: `out/launch_oc3_flare`
- `transolver`: `out/launch_oc3_transolver`
- `gnn`: `out/launch_oc3_gnn`
- `fno`: `out/launch_oc1_fno`
- `gnot`: `out/launch_oc3_gnot`
- `lno`: `out/lno_stability_c`

Common setup:
- train/valid samples: `10000 / 1000`
- input/output steps: `1 / 1`
- epochs: `10`
- window stride: `10`
- device: `cuda`

### 1. Model Details

| Model | Parameters | Hidden Dim | Best Epoch |
| --- | ---: | ---: | ---: |
| flare | 133,203 | 48 | 10 |
| transolver | 136,467 | 64 | 9 |
| gnn | 132,771 | 72 | 10 |
| fno | 132,772 | 16 | 10 |
| gnot | 140,867 | 24 | 10 |
| lno | 144,739 | 56 | 9 |

### 2. Training Behaviour

| Model | Train Loss | Valid Loss |
| --- | ---: | ---: |
| flare | 0.004285 | 0.004993 |
| transolver | 0.007152 | 0.008109 |
| gnn | 0.008825 | 0.009819 |
| fno | 0.062094 | 0.068865 |
| gnot | 0.255387 | 0.355921 |
| lno | 0.266147 | 0.365839 |

### 3. Results & Performance

| Model | Train NRMSE | Valid NRMSE | Train RMSE | Valid RMSE |
| --- | ---: | ---: | ---: | ---: |
| flare | 0.016107 | 0.016653 | 0.008540 | 0.009382 |
| transolver | 0.021331 | 0.021461 | 0.011401 | 0.012319 |
| gnn | 0.022241 | 0.022757 | 0.012824 | 0.013806 |
| fno | 0.030535 | 0.031537 | 0.015444 | 0.017029 |
| gnot | 0.055458 | 0.059688 | 0.033082 | 0.038479 |
| lno | 0.055567 | 0.060094 | 0.033543 | 0.038960 |

Sorted by validation NRMSE:
- flare
- transolver
- gnn
- fno
- gnot
- lno

## Training Curves

Epoch-by-epoch validation curves for the stable runs:

![Stable baseline training curves](plots/stable_training_curves.png)

Validation NRMSE:

![Stable validation NRMSE](plots/stable_validation_nrmse.png)

Validation RMSE:

![Stable validation RMSE](plots/stable_validation_rmse.png)
