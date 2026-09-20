# Base (YOLOX-S) with the recipe-fix settings + COCO-pretrained init

Base rerun after `afb81a6` ("apply the recipe fix"), this time starting from the official
YOLOX-S COCO checkpoint instead of random init.

## Setup

Settings from `afb81a6` that this run used:

- **LR 6.25e-4** (`LR_INIT`), min 6.25e-6, same warmup + cosine shape, batch 4, 512 px, 100 epochs
- **Best checkpoint tracked from epoch 1** (`TRACK_BEST_AFTER = 0`)
- **External pretrained weights** (as `load_external_pretrained` does): Megvii's official
  `yolox_s.pth`. 456 of 462 tensors loaded; the 6 skipped are `head.cls_preds.*`, the
  80-class predictors, which start fresh at the 1e-2 prior
- **Per-video evaluation** every epoch (`per_video.csv`), as `utils/perseq.py` provides
- Full-size epochs (2,245 steps), random flip, EMA 0.9999

Deliberate deviations from this repo's current code:

| | This run | Repo at `afb81a6` |
|---|---|---|
| Input | raw 0-255, letterbox pad 114 (what the COCO checkpoint expects) | ÷255 + ImageNet mean/std |
| Weight decay | conv/linear weights only; BatchNorm scales and biases excluded | all parameters (`build_optimizer`) |
| Precision | fp32, no gradient clipping | AMP + clip 10 |
| Eval | every epoch, all 4,795 frames, COCO 101-point AP50 at IoU 0.5 + SSTNet's Pr/Re/F1 | every 2 epochs, 4,767 clips, VOC all-points |

Trained with `~/mocid_checks/base_train_sstnet.py` on pl-lawr7615 (1x L40S), not with
`main.py`; that script is not in this repo. 3.0 h for 100 epochs.

## Results

| | Epoch | AP50 | Pr | Re | F1 |
|---|---|---|---|---|---|
| **Best (`best_ap50.pth`)** | **5** | **89.82** | 95.96 | 94.52 | 95.23 |
| Final | 100 | 72.10 | 98.67 | 73.79 | 84.43 |
| Paper, Base | - | 83.59 | 94.27 | 89.34 | 91.74 |

The pretrained start works: epoch 1 is already at 80.38 and epoch 2 passes the paper's
83.59, peaking at 89.82 by epoch 5.

After that the run decays to 72.10, with precision rising to 98.7 and recall falling to
73.8 while training loss keeps dropping (2.18 -> 0.38) and validation loss rises
(3.50 -> 6.58). `per_video.csv` shows where it goes (recall):

| Video | ep 5 | ep 100 |
|---|---|---|
| data12 | 100.0 | 100.0 |
| data18 | 99.8 | 87.4 |
| data11 | 96.4 | 87.4 |
| data6 | 93.5 | 75.2 |
| data20 | 91.8 | 86.8 |
| **data15** | **92.0** | **17.4** (443 frames with no detection) |
| **data21** | **76.8** | **34.4** |

For context, an earlier from-scratch run using ImageNet normalisation (not in this repo)
held 88.83 at epoch 51 and 87.99 at epoch 100, with data15 at 90.7% recall - i.e. it did
not show this decay. Input normalisation is the main difference between the two runs, so
it is the first thing to test next; this run also differs in using pretrained init.

Checkpoints are not committed (34-103 MB); they are on the server under
`~/mocid_checks/runs/base-pretrained/`.

## Files

- `eval_log.csv`: per-epoch LR, losses, COCO AP50 / Pr / Re / F1, VOC AP50 / best-F1, epoch time
- `per_video.csv`: per-epoch, per-video AP50, recall and count of frames with no detection
- `train_log.txt`: console log, one line per epoch
