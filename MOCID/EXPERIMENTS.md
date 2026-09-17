# MOCID Experiment Ledger

One row per training run. Keep it append-only. `git rev-parse --short HEAD` for the
base commit. Copy `runs/<tag>/eval_log.csv` numbers into the AP50 / F1 columns.

## Targets (DAUB, from paper Table 2 / Table 1)

| Config | AP50 | F1 | Params |
|---|---|---|---|
| Base (YOLOX-ish) | 83.59 | 91.74 | 8.94 M |
| .+FISTA (stage 1) | 92.42 | 96.40 | 9.45 M |
| .+FISTA+DAM (MOCID) | 95.93 | 98.22 | 13.05 M |
| MOCID on IRDST | 94.74 | 97.88 | — |

## Runs

| id | date | commit | dataset | GPU | change vs previous | stage-1 AP50 / F1 | stage-2 AP50 / F1 | params (noDAM / DAM) | cost | notes |
|----|------|--------|---------|-----|--------------------|-------------------|-------------------|----------------------|------|-------|
| _param-check_ | 2026-09-15 | b82300f | — | — | none — `python main.py params` | — | — | 9.49 / 12.52 M | ~$0 | must be ≈ 9.45 / 13.05 M. noDAM matches (+0.04 M); DAM is 0.53 M short |
| R0 | 2026-09-15 | b82300f | DAUB | 1× L40S (pl-lawr7615) | baseline, code as-is | 88.65 / 90.05 | not run | 9.49 / 12.52 M | ~$0 (own server, stage 1 ≈ 7.7 h) | Phase 1 baseline, stage 1 only. Best = final epoch 100 (EMA weights). AP50 peaked at 88.29 @ ep4, collapsed to 0.68 @ ep20 at peak LR, recovered to ~82 by ep54, reached 88.65 only as LR decayed. Ran with `TORCH_COMPILE_DISABLE=1` (`torch.compile` fails on step 1 with Inductor `ValueRangeError`). VMamba CUDA kernel not built, so `csms6s` uses its pure-PyTorch scan: stage 2 ≈ 9.5 days/epoch, not run. Eval log + training log in `results/R0/` |

## Deviation-alignment sub-runs (Phase 3)

Track which PLAN.md §7 rows each run touches.

| id | §7 rows changed | hypothesis | result | verdict |
|----|-----------------|------------|--------|---------|
| R1 | 1, 2, 3, 8, 9, 10 | paper-recipe bundle closes most of the gap | | |
