# Base (YOLOX-S) sanity run on DAUB

PLAN.md Phase 1, step 2: train the paper's "Base" detector to check this repo's evaluation against Table 2 (AP50 83.59).

## Model

- YOLOX-S backbone and PAFPN (official YOLOX code, commit `6ddff482`), run on the **target frame only**, with this repo's `YOLOXHead` (width 0.5) and `YOLOLoss`.
- 8.94 M parameters with 1 class. This matches Table 2's *Base* row, which has the same AP50 and F1 as Table 1's *YOLOx* row.

## Training

Same machinery as R0 stage 1 (`train.py: run_stage`, `utils/utils.py`):

- SGD, LR 0.01, 6-epoch warmup then cosine to 1e-4, EMA 0.9999, batch 4, 512 px, random flip, AMP.
- 100 epochs, from scratch, on 1x L40S (pl-lawr7615).
- The training script is not in this repo (`~/mocid_checks/base_train.py` on the server).

The run was interrupted after epoch 18 and resumed from the epoch-15 checkpoint. `eval_log.csv` therefore has two rows each for epochs 16 and 18; the rows dated 2026-09-17 are the resumed run. Resuming restarts the EMA warm-up.

## Results (validation: 7 videos)

| Metric | This run, epoch 100 (best) | Paper, Base |
|---|---|---|
| Repo metric: VOC all-points AP50 / best F1 over thresholds (4,767 clips) | 75.93 / 82.38 | — |
| Paper metric: COCO AP50 / Pr / Re / F1 (all 4,795 frames) | **76.95 / 98.83 / 78.37 / 87.42** | **83.59 / 94.27 / 89.34 / 91.74** |

The paper metric is computed as in SSTNet's `vid_map_coco.py`: `pycocotools` COCOeval at IoU 0.5, with Re = max recall and Pr = mean precision up to that recall. It was checked against `pycocotools` directly, and the images were resized with PIL bicubic as in SSTNet's data loader.

- AP50 peaked at **82.35 at epoch 8**. It then settled at 73-76 from about epoch 20, while training loss kept falling (2.9 to 0.88). Best-model tracking only starts after epoch 40, so the early peak was not kept.
- The gap to the paper is recall, concentrated in two validation videos. `perseq_recall.txt` (epoch 98) shows recall of **data15 29.6%** and **data21 20.0%**, against 83-100% on the other five. After 1 epoch, data15 recall was 94.7%.

## Differences from SSTNet's released DAUB code

The paper follows SSTNet (UESTC-nnLab/SSTNet) for the data split and the Pr/Re/F1 formula. Its `train_DAUB.py` differs from this run in:

- **Effective LR**: 0.01 x batch/64 = **6.25e-4**, decaying to 6.25e-6 (3-epoch warmup, cosine, last 5 epochs flat).
- **Epoch length**: each epoch uses a random **1/5** of the training set (`utils_fit.py`).
- **Augmentation**: none; the augmentation call is commented out.
- **Input**: ImageNet mean/std normalisation. Training is fp32 with no gradient clipping.
- **Initial weights**: `model_data/pre_trained.pth`, not random.
- **Checkpoint choice**: evaluated and saved every epoch; the best checkpoint is picked on validation.

A from-scratch re-run matching these settings is in progress.

## Files

- `eval_log.csv`: AP50 / F1 (repo metric) every 2 epochs.
- `train_log.txt`: training log with progress-bar lines removed.
- `perseq_recall.txt`: recall per validation video at the best checkpoint (epoch 98 at the time).
