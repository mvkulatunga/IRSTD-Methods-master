# MOCID Reproduction & Verification — Plan

**Task.** A PhD-student prototype of MOCID (Zhang et al., AAAI-25) does not reach the
published numbers. Verify the implementation module-by-module, get it to the paper's
performance, and produce (1) working code and (2) a written verification report.

**Paper.** *MOCID: Motion Context and Displacement Information Learning for Moving
Infrared Small Target Detection*, AAAI-25 (pp. 10022–10030). Official code:
`https://github.com/TanzanOY/MOCID`.

---

## 1. Success criteria

Primary target — Table 1 (paper):

| Config | Dataset | AP50 | Pr | Re | F1 |
|---|---|---|---|---|---|
| MOCID | DAUB | **95.93** | 99.12 | 97.34 | **98.22** |
| MOCID | IRDST | **94.74** | 98.92 | 96.86 | **97.88** |

Intermediate targets — Table 2 ablation (DAUB), our diagnostic ladder:

| Row | AP50 | F1 | Params | Inference |
|---|---|---|---|---|
| Base (spatial-only CSPDarknet ≈ YOLOX) | 83.59 | 91.74 | 8.94 M | 0.003 s |
| .+FISTA (stage 1, `use_dam=False`) | 92.42 | 96.40 | **9.45 M** | 0.022 s |
| .+DAM (no FISTA) | 90.22 | 95.14 | 12.54 M | 0.032 s |
| .+FISTA+DAM = **MOCID** | 95.93 | 98.22 | **13.05 M** | 0.041 s |

"Reached" = within ~0.3 AP50 of target, averaged over ≥2 seeds, on DAUB and IRDST.

Paper recipe (Experimental Details section):
SGD, weight decay 5e-4, momentum 0.937, **initial LR 0.01, reduction coefficient 0.1**
(step decay), input 512×512, **T = 5**, augmentation = random clip flip,
`L = L_reg + L_cls` (IoU loss + BCE), **STB trained 100 epochs → frozen → DAM trained
100 epochs**. Train on 2×RTX3090, test on 1×RTX3090.

**Do DAUB first** — the ablation ladder is on DAUB, it is ~6× smaller than IRDST, and it
gives two intermediate checkpoints to bisect against. IRDST is a final-phase confirmation.

---

## 2. Deliverables & repo layout

| Deliverable | Location | Status |
|---|---|---|
| Working code hitting the targets | `MOCID/` (this tree) | in progress |
| Written verification report | `MOCID/REPORT.md` | to scaffold |
| Experiment ledger | `MOCID/EXPERIMENTS.md` | to scaffold |
| Colab driver notebook | `MOCID/notebooks/run_colab.ipynb` | to write |
| Trained checkpoints + eval logs | Google Drive `MOCID_runs/` | — |

The report is written **incrementally**: each module gets its verdict the moment its
behaviour is confirmed by a run, not in a batch at the end.

---

## 3. Environment (Google Colab)

1. **Runtime.** A100 40 GB is the closest match to the paper's 2×3090. L4 is the
   cost-efficient option for iteration; T4 only for smoke tests (2–3× slower).
2. **Persistent storage.** Mount Google Drive; all of `runs/` (checkpoints,
   `eval_log.csv`) lives there so a session timeout never loses progress. The code
   already auto-resumes from `fista_last.pth` / `dam_last.pth`
   ([train.py](train.py), [utils/utils.py](utils/utils.py) `load_checkpoint`).
3. **VMamba selective-scan kernel.** [components/dam.py](components/dam.py) imports
   `selective_scan_fn` from an external VMamba checkout (`VMAMBA_PATH`,
   [config.py](config.py#L40)). On Colab: clone VMamba, build the CUDA kernel with
   `nvcc` once per image, set `VMAMBA_PATH`. **Risk item** — if the kernel build is
   fragile, fall back to VMamba's pure-PyTorch `selective_scan_fn` reference path and
   note the (speed-only) difference in the report.
4. **Data staging.** Copy DAUB (~13.8k frames) and IRDST (~40.7k frames) from Drive to
   the local Colab disk (or a single tar) before training — per-frame reads off Drive
   are too slow for a dataloader.
5. **`torch.compile`.** [train.py:224](train.py#L224) wraps the model. First-step
   compile latency on Colab can be minutes; if it is flaky, disable for iteration and
   re-enable for the final timed runs.
6. **Determinism.** Fixed seed, log `torch`, CUDA, VMamba commit, GPU type into every
   `EXPERIMENTS.md` row.

---

## 4. Cost model & run budget

Budget: **$750** of Colab compute units (~7,500 units).

Rough estimate, DAUB, batch 4, ~2,200 steps/epoch:

| GPU | ~min/epoch | Full 2-stage (200 ep) | ~units | ~$ |
|---|---|---|---|---|
| A100 | ~10 | ~33 h | ~500 | ~$50 |
| L4 | ~18 | ~60 h | ~290 | ~$29 |
| T4 | ~35 | (smoke only) | — | — |

Implications:
- **~10–15 full DAUB runs** available, or ~2× that if stage 1 is cached and only
  stage 2 is re-run.
- IRDST is ~2.3× the cost per run — budget for **2–3 IRDST runs total**, at the end.
- **Cache the stage-1 backbone** the moment it matches 92.42 AP50 / 9.45 M params, then
  iterate stage 2 only. This roughly doubles the effective run count.
- Phase 3 therefore **bundles** low-risk paper-alignment changes into one run and only
  bisects if that run regresses (see §7).

Indicative ledger (adjust as results come in):

| Run | What | Scope | ~$ |
|---|---|---|---|
| R0 | Baseline, code as-is | full 2-stage | 50 |
| R1 | Paper-aligned recipe bundle | full 2-stage | 50 |
| R2–R7 | Bisection / architecture levers | stage-2-only mostly | 25–50 ea |
| R8–R9 | Seed variance on best config | full | 100 |
| R10–R12 | IRDST confirmation | full | 120–360 |
| — | Reserve | — | ~150 |

---

## 5. Phase 1 — Run it as-is (no code changes)

1. **Param count (free, do first).** `python main.py params`. Must print ≈ 9.45 M
   (no DAM) and ≈ 13.05 M (MOCID). If not, architecture width/depth is wrong —
   stop and fix before spending any GPU time. The prototype already carries these
   exact targets as comments ([ablations_and_old_notebooks/mocid.py](ablations_and_old_notebooks/mocid.py)).
2. **Eval-harness sanity.** Evaluate an untrained / spatial-only path and confirm the
   pipeline produces sane boxes; after a short YOLOX-style baseline train, it should
   land near the paper's **Base = 83.59 AP50**. This validates
   [utils/eval.py](utils/eval.py) (AP method, NMS, decode) independently of MOCID.
3. **Smoke test.** One `train` step + one `eval` pass on ~50 clips. No NaNs, finite
   loss, non-empty detections.
4. **Full baseline run (R0).** Full 2-stage on DAUB, current [config.py](config.py).
   Record:
   - stage-1 eval vs 92.42 / 96.40 and param 9.45 M
   - final eval vs 95.93 / 98.22
   - loss-component trace (`YOLOLoss.last_parts`), NaN-skip count, epoch time,
     per-clip inference time (target 0.041 s)
   - the `runs/<tag>/eval_log.csv`

**Output:** your-vs-paper table at both stages. This defines the gap.

---

## 6. Phase 2 — Localize the gap

Read the ladder:

| Observation | Conclusion |
|---|---|
| stage-1 `.+FISTA` ≈ 92.4 | backbone/FISTA/head/loss/eval OK → gap is in **DAM / stage 2** |
| stage-1 well below ~92 | problem is **upstream of DAM** — fix first, DAM can't rescue a weak backbone |
| stage-1 OK, stage-2 doesn't lift to ~95.9 (or drops) | **DAM architecture or stage-2 recipe** |

Then bucket every suspect and test **cheapest-first**:

1. **Eval protocol** (free, no retrain): AP integration method (VOC-all-points in
   [eval.py:96](utils/eval.py#L96) vs COCO-style), NMS 0.65, conf 1e-3, and
   "best-F1-over-sweep" ([eval.py:140](utils/eval.py#L140)) vs F1 at a fixed operating
   point. The paper's Pr/Re/F1 are self-consistent at one threshold
   (2·99.12·97.34/(99.12+97.34) = 98.22) — to match the Pr/Re/F1 columns you must
   report at a threshold, not the sweep max.
2. **Data / splits** (free): video & frame counts vs paper (DAUB 10/7 videos →
   8,983/4,795 frames; IRDST 42/43 → 20,398/20,258), clip-window logic in
   [utils/data.py](utils/data.py), flip aug.
3. **Training recipe** (1 run each): §7 rows 1–3, 9, 10.
4. **Architecture** (1 run each): §7 rows 4–8, 11, 12.

---

## 7. Phase 3 — Align to the paper (deviations already found in the code)

Each row needs a verdict in the report: **revert to paper** or **keep + justify**.
Execution: put the low-risk recipe reverts (1, 2, 3, 8, 9, 10) into **one bundled run
R1**. If R1 ≥ paper, done. If R1 regresses vs R0, bisect the bundle. Architecture rows
(4–7, 11, 12) are tested one at a time, stage-2-only where possible.

| # | Location | Paper | Code | Planned action |
|---|---|---|---|---|
| 1 | [utils/utils.py:122](utils/utils.py#L122) | LR 0.01, ×0.1 step decay | warmup(6) + cosine → 1e-4 | revert to MultiStep γ=0.1 |
| 2 | [config.py:13](config.py#L13) | LR 0.01 throughout | stage-2 LR = 1e-3 | test 0.01; keep 1e-3 only if it truly diverges |
| 3 | [utils/losses.py:202](utils/losses.py#L202) | `L = L_reg + L_cls` | `5·L_iou + L_obj + L_cls` | try `reg_weight = 1`; keep obj term (YOLOX-inherent), document |
| 4 | [components/components.py:216](components/components.py#L216) | n conv + n FISTA blocks | `proj_in/out` halve channels (÷2) | check vs 9.45 M budget — may be load-bearing |
| 5 | [components/components.py:195](components/components.py#L195) | `f_out = W_t * f`, no residual | `f + f_out` | keep, confirm it isn't masking an init bug |
| 6 | [components/dam.py:52](components/dam.py#L52) `SDS` | `B,C,Δ = 3DCDC(x)` | bottleneck 3DCDC + 1×1 expand | test paper-literal (`cdc_hidden = d_inner+2N`) |
| 7 | [components/dam.py](components/dam.py) `DAMBlock` init | standard init | `mid`=0, `out`=identity | keep, but retry paper-faithful init once LR is fixed |
| 8 | [components/components.py:335](components/components.py#L335) `TemporalPooling` | fuse T−1 DAM outputs | max over DAM outputs **+ F_T** | exclude F_T; test |
| 9 | [utils/utils.py:132](utils/utils.py#L132) `set_stage(2)` | freeze STB, train DAM | trains DAM + pool + fpn + head | try freezing fpn/head too |
| 10 | [utils/utils.py:9](utils/utils.py#L9) `ModelEMA` | not mentioned | EMA for eval/ckpt | ablate; keep only if it helps |
| 11 | [model.py:27](model.py#L27) | FPN fuses target + F_f | in-ch `c*2` then `width=0.5` halves back | verify this is a no-op, not a silent bug |
| 12 | [utils/eval.py:96](utils/eval.py#L96) | "AP50" | VOC post-2010 all-points | fix method so a YOLOX baseline reproduces Base = 83.59 |

**Rule: one lever per run, every run logged in `EXPERIMENTS.md`.**

---

## 8. Phase 4 — Close the remainder

- Tune eval knobs (NMS, conf) on val; report Pr/Re/F1 at a defined operating point.
- 2–3 seeds on the best config — decide whether a residual <0.5 AP gap is noise.
- If still short: still-image pretraining of STB (the paper pretrains video methods on
  still images before adding temporal modules — check whether MOCID's STB stage needs
  this too), augmentation, epoch budget.
- Repeat the loop on **IRDST** (build its split file to the paper's 42/43-video split).

---

## 9. Phase 5 — Verification report (`REPORT.md`)

One row per module: **paper spec (eq/section ref) → implementation → verdict
(correct / deviation + rationale / bug + fix) → evidence (which ablation number
confirms it)**.

Modules to cover:
- Dataset & clip construction ([utils/data.py](utils/data.py))
- SpatialFISTA — paper Eq (1)
- TemporalFISTA — paper Eq (2)–(4)
- MotionGuidedSpatialConv (dynamic kernel) — paper Eq (5)–(6)
- FISTABlock / FISTALayer / SpatioTemporalBackbone — paper "Spatio-temporal Backbone"
- CDC3D — 3D central-difference conv (Yu et al. 2021)
- SDS — paper "SDS"
- TIS / TIDS (interpolation + bidirectional selective scan) — paper Eq (9)–(13)
- DAMBlock — paper Fig. 4a
- DisplacementNet / TemporalPooling — paper "Overview" (T−1 shared DAMs + pooling)
- FPN — Lin et al. 2017 vs the custom variant here
- YOLOXHead — Ge et al. 2021
- YOLOLoss + SimOTA — paper loss `L = L_reg + L_cls`
- Evaluation metric — AP50 definition ([utils/eval.py](utils/eval.py))
- Two-stage training schedule ([train.py](train.py))

Close with the final results table (DAUB + IRDST) vs paper, and a summary of every
retained deviation with its justification.

---

## 10. Experiment ledger format (`EXPERIMENTS.md`)

One row per run:

| id | date | base commit | GPU | one-line change | stage-1 AP50 | stage-2 AP50 | F1 | params | notes |
|---|---|---|---|---|---|---|---|---|---|

---

## 11. Risk register

| Risk | Mitigation |
|---|---|
| VMamba CUDA kernel won't build on Colab | pure-PyTorch `selective_scan_fn` fallback; document speed-only impact |
| Colab session timeout mid-run | all `runs/` on Drive; code auto-resumes per epoch |
| `torch.compile` instability | disable for iteration, re-enable for final timed runs |
| Param count already off in Phase 1 | architecture bug — fix before any training |
| Gap is pure eval-protocol | Phase 2 checks this first, at zero GPU cost |
| Budget overrun from too many full runs | cache stage-1; bundle recipe changes; IRDST last |
| Prototype's stability hacks (rows 5, 7) mask a real bug | when reverting the hack, watch for the failure mode it was added to prevent |

---

## 12. Immediate next steps

1. Scaffold `REPORT.md` and `EXPERIMENTS.md`.
2. Write `notebooks/run_colab.ipynb` (Drive mount, VMamba build, data staging,
   train/eval wrappers).
3. Fix `VMAMBA_PATH` and dataset paths in [config.py](config.py) for the Colab layout.
4. Phase 1 step 1: `python main.py params` → confirm 9.45 M / 13.05 M.
5. Phase 1 step 4: launch R0 (baseline DAUB 2-stage).
