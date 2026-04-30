# Model Results

| Model | Parameters | Valid NRMSE | Epochs | Train/Valid Samples |
| --- | ---: | ---: | ---: | --- |
| gnn | 132,771 | 0.023105 | 10 | 10000 / 1000 |
| transolver | 136,467 | 0.021594 | 10 | 10000 / 1000 |
| flare | 133,203 | 0.016041 | 10 | 10000 / 1000 |
| gnot | 140,867 | 0.059688 | 10 | 10000 / 1000 |
| lno | 144,739 | 0.078807 | 10 | 10000 / 1000 |
| fno | 132,772 | 0.031537 | 10 | 10000 / 1000 |

## Fair Comparison

These runs are the latest saved fair-size comparisons, with models kept near the same parameter budget and trained on the same dataset size. NRMSE is the primary evaluation metric.

Common setup:
- train/valid samples: `10000 / 1000`
- input/output steps: `1 / 1`
- epochs: `10`
- window stride: `10`
- device: `cuda`

| Model | Parameters | Hidden Dim | Valid NRMSE | Valid RMSE | Train NRMSE | Train RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flare | 133,203 | 48 | 0.016041 | 0.009052 | 0.015636 | 0.008315 |
| transolver | 136,467 | 64 | 0.021594 | 0.012215 | 0.020153 | 0.010772 |
| gnn | 132,771 | 72 | 0.023105 | 0.013991 | 0.022485 | 0.012934 |
| fno | 132,772 | 16 | 0.031537 | 0.017029 | 0.030535 | 0.015444 |
| gnot | 140,867 | 24 | 0.059688 | 0.038479 | 0.055458 | 0.033082 |
| lno | 144,739 | 56 | 0.078807 | 0.047910 | 0.076141 | 0.042278 |

Sorted by validation NRMSE:
- flare
- transolver
- gnn
- fno
- gnot
- lno
