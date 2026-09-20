# Base (YOLOX-S), weight decay excluded from BatchNorm/biases + ImageNet normalisation

From-scratch Base run using SSTNet's released DAUB recipe. This is the run that first held
its accuracy across all 100 epochs, and the one that identified the two settings behind the
original Base collapse (75.93 AP50).

## Setup

- **Weight decay on conv/linear weights only**; BatchNorm scales and biases excluded
  (YOLOX's and SSTNet's grouping; this repo's `build_optimizer` still decays everything)
- **Input: ÷255 then ImageNet mean/std**, bicubic letterbox resize
- LR 6.25e-4 -> 6.25e-6 (batch-scaled), 3-epoch quadratic warmup, cosine, last 5 epochs flat
- Each epoch = a random 1/5 of the training set (SSTNet's `utils_fit.py`), 100 epochs, batch 4
- No augmentation (SSTNet's augmentation call is commented out), fp32, no gradient clipping
- EMA 0.9999, trained from scratch (no pretrained weights)
- Eval every epoch over all 4,795 val frames; AP50 = COCO 101-point at IoU 0.5, with
  SSTNet's Pr/Re/F1 convention (`vid_map_coco.py`); the VOC columns are this repo's metric

1x L40S (pl-lawr7615), 1.2 h for 100 epochs. Trained with `~/mocid_checks/base_train_sstnet.py`,
which is not in this repo.

## Results

| | Epoch | AP50 | Pr | Re | F1 |
|---|---|---|---|---|---|
| **Best (`best_ap50.pth`)** | **51** | **88.83** | 95.88 | 93.06 | 94.44 |
| Final | 100 | 87.99 | 96.94 | 91.39 | 94.08 |
| Paper, Base | - | 83.59 | 94.27 | 89.34 | 91.74 |

It climbs to ~88.6 by epoch 40 and stays there: no collapse, and no late decay. Validation
loss bottoms out around epoch 34 and then rises while AP50 keeps improving, so AP50 (not
val loss) is the right checkpoint criterion.

Per-video at the best checkpoint (`per_video_best.txt`), recall / AP50:

| Video | recall | AP50 |
|---|---|---|
| data12 | 100.0 | 100.00 |
| data18 | 97.6 | 95.22 |
| data11 | 95.0 | 94.26 |
| data20 | 93.5 | 92.30 |
| data6 | 91.0 | 88.14 |
| data15 | 90.7 | 62.40 |
| data21 | 69.6 | 49.97 |
| **all** | **93.06** | **88.83** |

27 frames in total have no detection, against 463 in data15 alone for the original Base.

## Why these two settings

Single-variable ablations from the original Base configuration (LR 0.01, full-size epochs,
flip, AMP, clip 10), 14 epochs each:

| Configuration | AP50 @ ep14 |
|---|---|
| All original settings (control) | **28.94** (collapse; peak 77.74 @ ep4) |
| - weight decay on BatchNorm/biases | 73.35 (no collapse) |
| - that, and ÷255 -> ImageNet normalisation | **85.80** (peak 88.06 @ ep8) |

Two settings that were *not* the cause, tested the same way:

- **Learning rate**: at LR 0.01 with the decay scope fixed, Base reached 89.93 by epoch 10.
- **Epoch size**: with full-size epochs, Base reached 88.91 by epoch 2.

Parameter-level evidence for the decay finding, comparing epoch-14 checkpoints: BatchNorm
gamma fell 1.000 -> 0.231 and the objectness bias drifted -4.595 -> +0.610 in the collapsed
run, while the run with those parameters excluded from decay kept gamma at 1.003. Momentum
0.937 amplifies weight decay by ~1/(1-m) ~ 16x, which is why a nominal 5e-4 does this.

Checkpoints are not committed (34-103 MB); they are on the server under
`~/mocid_checks/runs/base-sstnet/`.

## Files

- `eval_log.csv`: per-epoch LR, losses, COCO AP50 / Pr / Re / F1, VOC AP50 / best-F1, epoch time
- `per_video_best.txt`: per-video recall, AP50, frames with no detection, at the best checkpoint
- `train_log.txt`: console log, one line per epoch
