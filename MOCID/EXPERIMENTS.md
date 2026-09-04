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
| _param-check_ | | | — | — | none — `python main.py params` | — | — | ? / ? | ~$0 | must be ≈ 9.45 / 13.05 M |
| R0 | | | DAUB | | baseline, code as-is | | | | | Phase 1 baseline |

## Deviation-alignment sub-runs (Phase 3)

Track which PLAN.md §7 rows each run touches.

| id | §7 rows changed | hypothesis | result | verdict |
|----|-----------------|------------|--------|---------|
| R1 | 1, 2, 3, 8, 9, 10 | paper-recipe bundle closes most of the gap | | |
