# Base (YOLOX-S) at batch 8

Identical to `results/Base-fixedwd-imagenet/` except the batch size: 8 instead of 4, which
under the batch-linear LR rule means peak LR 1.25e-3 (instead of 6.25e-4) and 224 steps per
epoch (instead of 449), i.e. 22,400 total updates instead of 44,900.

Run to test whether the paper's Base (83.59 AP50) is explained by the authors using batch 8
on their two RTX3090s. See `BASELINE-FINDINGS.md` for the full investigation.

## Results

| | Epoch | AP50 | Pr | Re | F1 |
|---|---|---|---|---|---|
| Best | 48 | **86.77** | 94.72 | 92.18 | 93.43 |
| Final | 100 | 86.39 | 96.34 | 90.39 | 93.27 |
| Batch 4, same recipe | 51 | 88.83 | 95.88 | 93.06 | 94.44 |
| Paper, Base | - | 83.59 | 94.27 | 89.34 | 91.74 |

Batch 8 converges about 2 points below batch 4 - consistent with half the optimizer steps -
but still ~3 points above the paper's Base. Stable across training: no collapse, no late decay.

Per-video at the best epoch (48), AP50 / recall:

| Video | AP50 | recall |
|---|---|---|
| data12 | 100.00 | 100.0 |
| data11 | 93.26 | 94.9 |
| data18 | 92.09 | 95.0 |
| data20 | 88.41 | 89.0 |
| data6 | 87.86 | 91.0 |
| data15 | 64.08 | 94.9 |
| data21 | 27.12 | 61.2 |

Checkpoints are not committed; they are on the server under `~/mocid_checks/runs/base-batch8/`.
