# R0 stage 2 with CFMW's fusion Mamba in the DAM (`r0-cfm`)

Stage 2 (DAM) of R0 with CFMW's cross-modality fusion Mamba added to every DAM block beside TIDS,
the target and reference frames as its two streams. Design, code mapping and checks:
[CFMW-DAM.md](../../CFMW-DAM.md).

## Setup

- **Model:** MOCID with the CFM branch, 13.78 M parameters (DAM 4.29 M). Stage 2 freezes the
  backbone and trains the DAM, FPN and head, as R0 does.
- **Start:** R0's stage-1 checkpoint, `/srv/proj-mamba/mocid-baseline/checkpoints/r0_fista_best.pth`,
  the same as every other DAM experiment.
- **Settings:** `--profile r0-cfm` = R0 (SGD with Nesterov, LR 1e-3 → 1e-4, weight decay 5e-4
  on every parameter except CFM's scan parameters, 100 epochs, AMP, BatchNorm recomputed before
  each evaluation) plus `EVAL_DAM_OFF`. Every evaluation is scored twice, with the DAM on and off.
- **Command:** `python main.py --profile r0-cfm train --tag r0-cfm --stage1-from <checkpoint above>`
- **Hardware and time:** 1× L40S (pl-lawr7615), 16 Oct 23:55 → 17 Oct 08:04, GPU not shared.

Files: `eval_log.csv` (one row per evaluation; the last two columns are the DAM-off scores) and
`train_log.txt` (the console log without progress bars).

## Results

VOC AP50 on the repo loader's 4,767 validation frames:

| | Best (epoch) | Final (epoch 100) | Mean of 50 evaluations |
|---|---|---|---|
| **DAM with CFM** | **90.27** (70) | **89.44** (F1 90.37) | 89.64 |
| DAM without CFM (`results/R0-DAM-fixed`) | 90.12 (52) | 89.20 (F1 89.66) | 89.49 |
| DAM with CFM, scored with the DAM off | 91.99 (32) | 89.56 (F1 89.25) | 91.09 |

- CFM's effect (+0.15) is within the spread of the other DAM variants.
- Switching the DAM off scores higher at 48 of 50 evaluations, as it does for the DAM without CFM.
  So CFM did not make the DAM contribute to AP50.
