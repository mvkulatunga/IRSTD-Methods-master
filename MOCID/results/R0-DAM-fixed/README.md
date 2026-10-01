# R0 stage 2 with the DAM handover fixed (`r0-dam-refres`)

Stage 2 (DAM) of R0, rerun with the DAM handover fix: `DAMBlock` takes its residual from the
reference frame `F_R` instead of the target `F_T` (commit `21f5717`,
[CODE-REVIEW.md](../../CODE-REVIEW.md) finding 1). Everything else is identical to R0's own
stage 2: same starting checkpoint, same settings, same code.

## Setup

- **Model:** the repo's MOCID (12.52 M parameters). Stage 2 freezes the backbone and trains the
  DAM, FPN and head (`set_stage(2)`, as R0 did).
- **Start:** R0's stage-1 checkpoint, `runs/server-r0/fista_best.pth` (epoch 100, VOC AP50 88.65,
  F1 90.05).
- **Settings:** `--profile r0`. SGD with Nesterov momentum 0.937, weight decay 5e-4 on every
  parameter, LR 1e-3 → 1e-4 (6-epoch warmup, then cosine), 100 epochs over the whole training set
  (2,235 steps at batch 4), AMP, gradient clipping at 10, random flip, ÷255 input, EMA 0.999.
  Evaluated every 2 epochs on the repo loader's 4,767 validation frames (VOC AP50, F1 at the best
  threshold).
- **Command:**
  `python main.py --profile r0 train --tag r0-dam-refres --stage1-from runs/server-r0/fista_best.pth`
- **Hardware and time:** 1× L40S (pl-lawr7615), 30 Sep 2026 12:40 → 23:13. The GPU was shared
  with another group's job for part of the run.
- Trained before BatchNorm statistics were recomputed at evaluation (`6e59c6d`); re-scored with
  that afterwards, below.

Files: `eval_log.csv` (one row per evaluation) and `train_log.txt` (the console log without
progress bars).

## Results

VOC AP50 on 4,767 frames, as logged:

| Stage 2 | Best | Final (ep 100) | Mean, all 50 evaluations | Mean, epochs 80–100 |
|---|---|---|---|---|
| **DAM fixed (this run)** | **90.12** (ep 52; F1 90.21) | **89.20** (F1 89.66) | **89.49** | **89.37** |
| R0's original DAM | 89.86 (ep 10; F1 90.83) | 88.53 (F1 89.96) | 88.87 | 88.42 |
| *Starting point: R0 stage 1* | *88.65 (F1 90.05)* | | | |
| *Paper: +FISTA → MOCID* | *92.42 → 95.93* | | | |

Every tenth evaluation:

| Epoch | 2 | 12 | 22 | 32 | 42 | 52 | 62 | 72 | 82 | 92 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DAM fixed | 89.12 | 89.67 | 89.67 | 89.41 | 89.53 | **90.12** | 89.42 | 89.51 | 89.39 | 89.32 | 89.20 |
| R0's DAM | 88.91 | 89.57 | 88.99 | 89.04 | 89.49 | 89.19 | 89.18 | 88.46 | 88.27 | 88.41 | 88.53 |

### Re-scored with BatchNorm statistics recomputed

Recomputed from 300 training batches for the layers stage 2 trains (FPN and head), as
`train.py` now does before every evaluation:

| Checkpoint | Logged | Recomputed |
|---|---|---|
| DAM fixed, best (ep 52) | 90.12 | 90.11 (F1 90.19) |
| DAM fixed, final (ep 100) | 89.20 | 89.18 (F1 89.68) |
| R0's DAM, best (ep 10) | 89.86 | 89.94 (F1 90.93) |
| R0's DAM, final (ep 100) | 88.53 | 88.50 (F1 89.92) |

Differences of at most 0.1, so the logged stage-2 numbers are fair. (Stage 2 trains at a lower LR
with a fast EMA, so its averaged statistics don't fall behind the weights the way R0's stage 1
did.)

## What this shows

1. **The fix makes stage 2 consistently better than R0's,** by +0.6 AP50 on average over the run,
   +0.95 over the last 20 epochs and +0.67 at the final epoch. R0's stage 2 began by repairing the
   handover damage (switching its DAM on cost 64 AP50 before training); this run starts from the
   stage-1 model intact.
2. **The DAM still adds little.** Over its starting point of 88.65 it gains +1.5 at best and
   +0.55 at the final epoch, against +3.51 in the paper. The handover was not the main limitation.
   The next suspects, in the order planned: the scan parameters not being computed per frame
   (each token should get its own frame's B, C and Δ), the missing Δ bias and initialisation, and
   the DAM's size (3.03 M against the paper's 3.60 M).
3. **Both runs drift down in the second half** while their training loss keeps falling (here
   0.93 → 0.75), which points to overfitting. Stage 2 trains the FPN and head alongside the DAM,
   whereas the paper trains only the DAM.
4. **F1 at the final epoch is slightly lower** for this run (89.66 against 89.96), although its
   AP50 is higher.
