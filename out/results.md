# Model Results

| Model | Parameters | Valid RMSE | Epochs | Train/Valid Samples | Source |
| --- | ---: | ---: | ---: | --- | --- |
| gnn | 415,235 | 0.029777 | 10 | 10000 / 1000 | `out/proper_gnn_baseline/gnn_results.json` |
| transolver | 565,379 | 0.043973 | 10 | 10000 / 1000 | `out/proper_transolver_baseline/transolver_results.json` |
| flare | 978,819 | 0.018551 | 10 | 10000 / 1000 | `out/proper_flare_baseline/flare_results.json` |
| gnot | 1,545,871 | 0.044718 | 10 | 10000 / 1000 | `out/oc1_tuned/gnot_results.json` |
| lno | 1,775,171 | 0.051878 | 6 | 10000 / 1000 | `out/oc1_tuned/lno_results.json` |
| fno | 2,131,651 | 0.029030 | 5 | 2000 / 200 | `out/fno_baseline_mid/fno_results.json` |

Notes:
- Parameter counts are computed from the current model definitions with `input_dim=15` and `output_steps=1`.
- RMSE values come from the saved `*_results.json` files listed in the table.
