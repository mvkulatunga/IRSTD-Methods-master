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
| R0 | 2026-09-15 | b82300f | DAUB | 1× L40S (pl-lawr7615) | baseline, code as-is | 88.65 / 90.05 | not run | 9.49 / 12.52 M | ~$0 (own server, stage 1 ≈ 7.7 h) | Phase 1 baseline, stage 1 only. Best = final epoch 100 (EMA weights). AP50 peaked at 88.29 @ ep4, collapsed to 0.68 @ ep20 at peak LR, recovered to ~82 by ep54, reached 88.65 only as LR decayed. Ran with `TORCH_COMPILE_DISABLE=1` (`torch.compile` fails on step 1 with Inductor `ValueRangeError`). VMamba CUDA kernel not built, so `csms6s` uses its pure-PyTorch scan: stage 2 ≈ 9.5 days/epoch, not run. Eval log + training log in `results/R0/`. **Note: stage 1 never touches DAM/`selective_scan` — the ep20 collapse is a pure FISTA/optimizer/loss issue, not the kernel problem.** |
| R0-colab-a1 | 2026-09-07 → 09-14 | b82300f | DAUB (Colab) | A100-40GB, CUDA kernel built | same config as `R0`, independent run | ep2: 86.69/82.37 · ep4: 85.90/86.06 · ep6: 89.41/86.41 · ep8: **92.92**/88.78 · ep10: 90.80/89.59 · ep12: 87.26/85.37 · ep14: 88.05/84.56 (~8 min/ep, `TORCHDYNAMO_DISABLE=1`) | — | — | ~$15 | Repeatedly interrupted by Colab session drops (resumed 3× via Drive-cached checkpoints); never reached ep20+, so can't confirm whether the same collapse would hit here. ep8 already touched the .+FISTA target (92.42) before bouncing — consistent with `R0`'s early peak (88.29 @ ep4) followed by instability. Two independent runs, two different GPUs, same code, same collapse-prone middle-training regime → strengthens the case that this is a **recipe** issue (LR schedule/loss weighting, PLAN §7 rows 1–3) rather than a hardware fluke. |
| Base | 2026-09-17 | cc31326 | DAUB | 1× L40S (pl-lawr7615) | official YOLOX-S code + this repo's `YOLOXHead`/`YOLOLoss`, no FISTA/DAM — PLAN §2 Tier-1 check | VOC: 75.93/82.38 · paper-style (pycocotools COCOeval, IoU 0.5, SSTNet's Pr/Re convention): AP50 76.95, Pr 98.83, Re 78.37, F1 87.42 | n/a | 8.94 M (matches) | ~$0 (own server) | **`Base` ≠ 83.59 → the shared recipe has a real problem independent of FISTA/DAM; the "assume baseline is clean" working assumption doesn't hold.** VOC vs. paper-style COCO differ by only ~1 pt (75.93 vs 76.95) — **confirms eval methodology is not the explanation for any gap seen so far.** Gap is recall: 78.4% vs 89.3%, concentrated on 2/7 val videos (data15 29.6%, data21 20.0%; data15 was **94.7%** after epoch 1 — lost during training, same collapse-then-partial-recover shape as `R0`'s FISTA run, just milder). Prime suspect, per `results/Base-YOLOX-S/README.md`: recipe deviations from SSTNet (which the paper follows) — **LR 0.01 unscaled for batch 4 vs. SSTNet's batch-linear-scaled 6.25e-4 (16× higher here)**, training from random init vs. SSTNet's `pre_trained.pth`, SSTNet using 1/5 of the data per epoch, no augmentation, fp32/no-AMP/no-clip vs. our AMP+clip. A from-scratch rerun matching SSTNet's recipe is in progress. |
| Base-pretrained | 2026-09-20 | afb81a6 | DAUB | 1× L40S (pl-lawr7615) | recipe-fixed LR (6.25e-4) + `TRACK_BEST_AFTER=0` + COCO-pretrained YOLOX-S init (456/462 tensors) — **but raw 0-255/pad-114 input, not this repo's ImageNet normalisation**; fp32, no grad clip; eval every epoch, all 4,795 frames | best: **89.82/95.23 @ ep5** (already past paper's 83.59/91.74 by ep2) → decays to 72.10/84.43 @ ep100 | n/a | 8.94 M | ~$0 (own server, 3.0 h) | **The LR+init fix is confirmed correct for getting to a good state fast — but the mid/late-training decay persists even with proper LR and a strong pretrained start**, concentrated on the same two videos again: data15 92.0%→17.4% (443 zero-det frames by ep100), data21 76.8%→34.4%. Precision keeps climbing (95.96→98.67) while recall falls (94.52→73.79) as training loss keeps falling and val loss rises — the classic overfitting/generalisation-gap shape, not a raw instability, since 5/7 videos stay healthy throughout. **Confound, not yet isolated:** README also references an off-repo from-scratch run using this repo's ImageNet normalisation that did *not* decay (held ~88 AP50 through ep51 and ep100, data15 at 90.7%) — so normalisation and init both differ between the stable and decaying runs; can't yet attribute the decay to either alone. **Do not revert `utils/data.py`'s ImageNet normalisation** on the strength of this run — if anything the evidence currently favours keeping it. Next isolating test: pretrained init **with** ImageNet normalisation (not raw-pixel), holding everything else fixed. See `results/Base-pretrained/README.md`. |

## Recipe fix, part 2 (2026-09-20) — weight decay

Landed in response to `Base-pretrained`'s README, which also ran with weight decay
excluding BatchNorm scale/bias and all bias terms (matches official YOLOX's own
optimizer setup) — uncontested regardless of the normalisation confound above:

- `utils/utils.py`: `build_optimizer` now splits parameters into two SGD groups —
  every 1-D parameter (BatchNorm weight/bias, conv/linear biases) gets
  `weight_decay=0`; everything else keeps `cfg.WEIGHT_DECAY`. Previously decay
  applied uniformly to every parameter.

**Explicitly not changed:** `utils/data.py`'s ImageNet normalisation stays as-is
(see the confound noted in `Base-pretrained` above) — do not "correct" it to
raw-pixel without a controlled test showing that's actually the fix, since the
one data point we have suggests the opposite.

## Recipe fix (2026-09-20) — applied ahead of the next Base / R0 rerun

Landed in the repo, in response to the `Base` row above:

- `config.py`: `LR_INIT` 0.01 → **6.25e-4** (SSTNet's batch-linear-scaled rate for
  batch 4), `MIN_LR` scaled by the same 1/16 factor, `TRACK_BEST_AFTER` 40 → **0**
  (both Base and R0 lost a genuine good early-epoch result to this gate with no
  checkpoint saved to recover it).
- `utils/data.py`: clips are now ImageNet mean/std normalised (previously plain
  `/255.0`) — matches what SSTNet's recipe does and what a COCO-pretrained
  backbone expects its input to look like. `MOCIDDataset` also now returns the
  source image path per item.
- `utils/utils.py`: new `load_external_pretrained(model, ckpt_path)` — shape-
  filtered loading of an outside checkpoint (e.g. official YOLOX-S COCO
  weights). For the stock-architecture `Base` run this transfers the whole
  backbone/neck; for this repo's own FISTA backbone it currently transfers
  nothing (key names differ from stock CSPDarknet — a name-mapping table is a
  follow-up, not yet done).
- `utils/perseq.py` (new) + `main.py eval --perseq`: per-video recall/
  confidence breakdown, at the same best-F1 operating point the aggregate F1
  already uses, so we can check directly whether a fix heals the data15/data21
  pattern instead of only watching the aggregate score.
- `utils/eval.py`: `compute_ap50_f1`/`evaluate` unchanged (still return
  `(ap, f1)`); added `compute_ap50_f1_full`/`evaluate_full`, which also expose
  the best-F1 threshold and Pr/Re at that point — needed by `perseq.py` and
  useful for the fuller AP50+F1+Pr+Re comparison discussed for `.+FISTA`.

**Not done here — needs the same treatment manually on the server:**
`~/mocid_checks/base_train.py` is not in this repo, so its LR/`TRACK_BEST_AFTER`
constants need updating by hand to match `config.py` above, and it needs an
explicit call to `load_external_pretrained(model, "yolox_s.pth")` after building
the model. The normalisation fix applies automatically once that script's
checkout of `utils/data.py` is updated, since it already reuses `MOCIDDataset`.

## Deviation-alignment sub-runs (Phase 3)

Track which PLAN.md §7 rows each run touches.

| id | §7 rows changed | hypothesis | result | verdict |
|----|-----------------|------------|--------|---------|
| R1 | 1, 2, 3, 8, 9, 10 | paper-recipe bundle closes most of the gap | | |
