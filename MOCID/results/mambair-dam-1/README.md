# R0 stage 2 with the MambaIR DAM (`mambair-dam-1`)

Stage 2 (DAM) of R0 with the DAM rebuilt around a MambaIR residual state-space block
around its TIDS scan (commit `51c1ea3`, `components/dam.py`). That code is on the
`dam-mambair` branch only; `main` keeps the previous DAM, which scored higher (below). The
starting checkpoint and
settings are R0's, as in [R0-DAM-fixed](../R0-DAM-fixed/README.md), so that run is the
comparison.

## Setup

- **Model:** MOCID with the MambaIR DAM, 12.91 M parameters (stage 1 9.49 M, DAM 3.42 M; the
  previous DAM was 3.03 M, the paper's is 3.60 M). Stage 2 freezes the backbone and trains the
  DAM, FPN and head (`set_stage(2)`).
- **Start:** the team's R0 stage-1 checkpoint,
  `/srv/proj-mamba/mocid-baseline/checkpoints/r0_fista_best.pth`.
- **Settings:** `--profile r0`, as R0-DAM-fixed: SGD with Nesterov momentum 0.937, weight decay
  5e-4 on every parameter, LR 1e-3 → 1e-4 (6-epoch warmup, then cosine), 100 epochs over the
  whole training set, AMP, gradient clipping at 10, random flip, ÷255 input, EMA 0.999.
  Evaluated every 2 epochs on the repo loader's 4,767 validation frames (VOC AP50, F1 at the best
  threshold), with BatchNorm statistics recomputed from 300 training batches before each
  evaluation (`c103a2c`).
- **Command:**
  `python main.py --profile r0 train --tag mambair-dam-1 --stage1-from /srv/proj-mamba/mocid-baseline/checkpoints/r0_fista_best.pth`
- **Hardware and time:** 1× L40S (pl-lawr7615), 6 Oct 2026 13:06 → 21:09.

Files: `eval_log.csv` (one row per evaluation). The console log was not kept.

## Results

VOC AP50 on 4,767 frames, as logged:

| Stage 2 | Best | Final (ep 100) | Mean, all 50 evaluations | Mean, epochs 80–100 |
|---|---|---|---|---|
| **MambaIR DAM (this run)** | **89.53** (ep 32; F1 90.53) | **88.53** (F1 90.68) | **88.87** | **88.72** |
| DAM fixed ([R0-DAM-fixed](../R0-DAM-fixed/README.md)) | 90.12 (ep 52; F1 90.21) | 89.20 (F1 89.66) | 89.49 | 89.37 |
| *Starting point: R0 stage 1* | *88.65 (F1 90.05)* | | | |
| *Paper: +FISTA → MOCID* | *92.42 → 95.93* | | | |

Every tenth evaluation:

| Epoch | 2 | 12 | 22 | 32 | 42 | 52 | 62 | 72 | 82 | 92 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MambaIR DAM | 88.85 | 88.54 | 89.21 | **89.53** | 88.77 | 88.58 | 88.86 | 89.06 | 88.72 | 88.68 | 88.53 |
| DAM fixed | 89.12 | 89.67 | 89.67 | 89.41 | 89.53 | **90.12** | 89.42 | 89.51 | 89.39 | 89.32 | 89.20 |

The MambaIR DAM is about 0.6 AP50 below the fixed DAM on every summary (best, final and both
means) and ends level with its stage-1 starting point (88.53 vs 88.65), so it adds nothing
over stage 1 here. Its F1 at the best threshold is higher than the fixed DAM's late in training
(90.68 vs 89.66 at epoch 100). Training loss falls steadily (0.93 → 0.76) while AP50 stays flat
from epoch 2.

Not directly comparable: R0-DAM-fixed was scored without the BatchNorm recompute, but its
README re-scores it with the recompute and the numbers move by at most 0.1.
