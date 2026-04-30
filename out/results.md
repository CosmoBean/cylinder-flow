# Model Results

| Model | Parameters | Valid RMSE | Epochs | Train/Valid Samples |
| --- | ---: | ---: | ---: | --- |
| gnn | 132,771 | 0.013991 | 10 | 10000 / 1000 |
| transolver | 136,467 | 0.012215 | 10 | 10000 / 1000 |
| flare | 133,203 | 0.009052 | 10 | 10000 / 1000 |
| gnot | 140,867 | 0.038479 | 10 | 10000 / 1000 |
| lno | 144,739 | 0.047910 | 10 | 10000 / 1000 |
| fno | 132,772 | 0.017029 | 10 | 10000 / 1000 |

## Fair Comparison

These runs are the latest saved fair-size comparisons, with models kept near the same parameter budget and trained on the same dataset size.

Common setup:
- train/valid samples: `10000 / 1000`
- input/output steps: `1 / 1`
- epochs: `10`
- window stride: `10`
- device: `cuda`

| Model | Host | Parameters | Hidden Dim | Valid RMSE | Valid NRMSE | Train RMSE | Train NRMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flare | oc4 | 133,203 | 48 | 0.009052 | 0.016041 | 0.008315 | 0.015636 |
| transolver | oc3 | 136,467 | 64 | 0.012215 | 0.021594 | 0.010772 | 0.020153 |
| gnn | oc3 | 132,771 | 72 | 0.013991 | 0.023105 | 0.012934 | 0.022485 |
| fno | oc4 | 132,772 | 16 | 0.017029 | 0.031537 | 0.015444 | 0.030535 |
| gnot | oc4 | 140,867 | 24 | 0.038479 | 0.059688 | 0.033082 | 0.055458 |
| lno | oc4 | 144,739 | 56 | 0.047910 | 0.078807 | 0.042278 | 0.076141 |

Sorted by validation RMSE:
- flare
- transolver
- gnn
- fno
- gnot
- lno
/sbandred/cylinder-flow/out/oc4_fair/lno/lno_results.json)
