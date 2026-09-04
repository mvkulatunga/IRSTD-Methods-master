# MOCID Implementation — Verification Report

Status: **DRAFT / living document.** Preliminary verdicts below come from a static
read of the code against the paper (Zhang et al., *MOCID*, AAAI-25). Empirical
verdicts (⏳) are filled in as runs land — see [EXPERIMENTS.md](EXPERIMENTS.md).

## Verdict legend

| | meaning |
|---|---|
| ✅ FAITHFUL | matches the paper (or a source it cites) closely enough |
| ⚠️ DEVIATION (documented) | differs from paper; code carries a rationale; keep unless it hides a bug |
| 🔧 DEVIATION (to test) | differs from paper; revert-and-measure in Phase 3 |
| ❓ OPEN QUESTION | possible correctness issue, needs scrutiny |
| ⏳ PENDING | verdict needs an empirical run |

## Summary table

| # | Module | Paper ref | Verdict (prelim) | PLAN §7 |
|---|---|---|---|---|
| 1 | Dataset & clip construction | Exp. Details | ✅ / ❓ box clamping | — |
| 2 | SpatialFISTA | Eq (1) | ✅ (GFNet-style rfft) | — |
| 3 | TemporalFISTA | Eq (2)–(4) | ✅ (note L2 interpretation) | — |
| 4 | MotionGuidedSpatialConv | Eq (5)–(6) | ✅ | — |
| 5 | FISTABlock residual | Eq (5) | ⚠️ documented | 5 |
| 6 | ConvBlock / FISTALayer | "STB" / Fig 2 | 🔧 channel bottleneck | 4 |
| 7 | SpatioTemporalBackbone | "STB" | ✅ | — |
| 8 | CDC3D | Yu et al. 2021 | ✅ | — |
| 9 | SDS | "SDS" | 🔧 3DCDC bottleneck | 6 |
| 10 | TIS / TIDS | Eq (9)–(13) | ✅ / ❓ param–sequence alignment | — |
| 11 | DAMBlock | Fig 4a | ⚠️ zero/identity init | 7 |
| 12 | DisplacementNet / TemporalPooling | "Overview" | 🔧 F_T in the pool | 8 |
| 13 | FPN | Lin et al. 2017 | ✅ custom variant / ❓ `c*2`×`width=0.5` | 11 |
| 14 | YOLOXHead | Ge et al. 2021 | ✅ | — |
| 15 | YOLOLoss + SimOTA | `L = L_reg + L_cls` | 🔧 `reg_weight=5`, `min=3` | 3 |
| 16 | Evaluation metric | "AP50" | 🔧 AP method / F1 operating point | 12 |
| 17 | Two-stage training schedule | Exp. Details | ✅ structure / 🔧 recipe | 1,2,9,10 |

---

## 1. Dataset & clip construction — [utils/data.py](utils/data.py)

**Paper.** Clip `{x_i}_{i=1}^T`, `x_T` the target frame, `{x_i}_{i=1}^{T-1}` references;
output on `x_T`. Frames resized to 512×512, T=5, augmentation = random clip flip.
DAUB split follows SSTNet (10 train / 7 val videos → 8,983 / 4,795 frames);
IRDST 42 / 43 videos → 20,398 / 20,258 frames.

**Implementation.** `MOCIDDataset` groups annotation lines by sequence, slides a
T-wide window, drops windows spanning a frame-number gap; last frame carries labels.
Direct `cv2.resize` to 512×512 (no letterbox). Shared horizontal flip across the clip,
p=0.5. `collate_train` → cxcywh+cls; `collate_eval` → xyxy GT.

**Verdict.** ✅ faithful to the paper's description. Direct resize (aspect distortion)
is consistent with "resized to 512×512". Confirm the DAUB split file line counts
match 8,983 / 4,795 (`dataset_helpers/coco_*_DAUB.txt` are 8,981 / 4,794 — check the
±2).

**❓ Open.** Scaled boxes are not clamped to `[0, img_size]` and zero-area boxes
(possible after rounding a 1–2 px IR target) are not filtered — these feed SimOTA.
Add a clamp + degenerate-box drop and check it doesn't move AP50.

---

## 2. SpatialFISTA — [components/components.py](components/components.py) (`SpatialFISTA`), Eq (1)

**Paper.** `F_s = DFT2D(f)`; `F_s ← F_s ⊙ K`, `K ∈ C^{C×H×W}` learnable complex filter;
`f_s = IDFT2D(F_s)`. Cites GFNet (Rao et al. 2021).

**Implementation.** `torch.fft.rfft2(f, norm="ortho")` (half spectrum, `W//2+1`),
`K = complex(weight_real, weight_imag)` shape `(1,C,H,W//2+1)`, multiply, `irfft2`.
Runs in fp32 under `autocast(enabled=False)`; `@torch.compiler.disable`.

**Verdict.** ✅ FAITHFUL. The rfft/irfft half-spectrum form is exactly GFNet's and is
mathematically equivalent to a full DFT with a Hermitian filter for real input.
`norm="ortho"` is a harmless normalization choice the paper doesn't pin. Init
`randn*0.02` is reasonable.

---

## 3. TemporalFISTA — `TemporalFISTA`, Eq (2)–(4)

**Paper.** `F_t = DFT1D(f_s)` over T; `F_t ← F_t ⊙ K_t`, `K_t ∈ C^{T×C×1×1}`;
`f_hat = IDFT1D(F_t)`; `M = f ⊙ ‖f_hat‖_2` — "multiply the original features f by the
L2 normalization of f_hat"; "regions associated with target movement exhibit higher
amplitudes".

**Implementation.** `fft(f_s, dim=1, norm="ortho")`; `K_t` shape `(1,T,C,1,1)`;
`f_hat = ifft(...).real`; `f_hat_norm = vector_norm(f_hat, ord=2, dim=1, keepdim=True)`
(L2 over the T axis → one scalar per (C,H,W)); `return f_orig * f_hat_norm`.

**Verdict.** ✅ FAITHFUL to the paper's *text* and intent (multiply by the temporal
amplitude, not divide). Note the interpretation: "L2 normalization" in Eq (4) is
read as the L2 *norm/amplitude* over T, applied identically to all frames — which
matches the "amplitude … represents the temporal dynamics at each spatio-temporal
location" sentence. If the reproduction underperforms in stage 1, an alternative
reading (per-frame magnitude `|f_hat|`, no T-collapse) is worth one ablation.

---

## 4. MotionGuidedSpatialConv — `MotionGuidedSpatialConv`, Eq (5)–(6)

**Paper.** `f_out = W_t * f = (α_t · W_b) * f`; `α_t = FC(GAP_s(M))`,
`W_b ∈ R^{C×C×k²}` shared base kernel, `α_t ∈ R^{T×C×1×1}`, FC across the temporal
dimension.

**Implementation.** `gap = M.mean((-2,-1))` → `(B,T,C)`; `alpha_t = fc(gap^T)^T` with
`fc = Linear(T, T)` mixing across frames; reshaped to per-(B,T,C) scalars; multiply
the shared `Wb`; grouped `conv2d` with `groups=B*T` so every (sample, frame) applies
its own calibrated kernel in one call.

**Verdict.** ✅ FAITHFUL — GAP → temporal FC → per-frame kernel calibration of a
shared base weight is exactly Eq (5)–(6). No activation on `fc`; `Wb` uses
`kaiming_uniform_(a=√5)`. Note: if `α_t ≈ 0` at init the block output collapses —
directly relevant to item 5.

---

## 5. FISTABlock residual — `FISTABlock.forward`

**Paper.** Eq (5): `f_out = W_t * f`. No residual around the FISTA block.

**Implementation.** `return f + f_out` — comment: *"residual is not in the MOCID
paper; without it features collapse"*.

**Verdict.** ⚠️ DEVIATION (documented). Plausible: the dynamic conv (item 4) with a
near-zero `α_t` at init would otherwise zero the signal. **Keep**, but during Phase 3
confirm it is not masking a real init bug — try paper-faithful (no residual) once the
`α_t` / `fc` init is sane, and check stage-1 AP50 vs 92.42. PLAN §7 row 5.

---

## 6. ConvBlock / FISTALayer — `ConvBlock`, `FISTALayer`

**Paper.** FISTA layer = `n` convolution blocks + `n` FISTA blocks. Fig 2
bottom-right shows a conv block as `3×3 Conv → 1×1 Conv ∥ 1×1 Conv`.

**Implementation.** `ConvBlock` = depthwise 3×3 → (`1×1_a` + `1×1_b`) summed → matches
the figure (depthwise is an unstated efficiency choice). `FISTALayer` wraps
`n_blocks × (ConvBlock → FISTABlock)` in a **channel bottleneck**: `proj_in`
`C → C/2`, blocks run at `C/2`, `proj_out` `C/2 → C`. `n_blocks = [4, 4, 1]` for the
three stages.

**Verdict.** 🔧 DEVIATION (to test). The `C/2` bottleneck is not in the paper. It may
be **load-bearing for the 9.45 M budget** — check `python main.py params` first; if
removing it overshoots 9.45 M substantially, the bottleneck is effectively intended
and should be kept + documented. `n_blocks` per stage is unspecified by the paper;
record the values used. PLAN §7 row 4.

---

## 7. SpatioTemporalBackbone — `SpatioTemporalBackbone`

**Paper.** Keep the first two CSPDarknet layers for low-level features; replace the
last three spatial layers with three cascaded FISTA layers. Outputs at ×1/8, ×1/16,
×1/32 (Fig 2).

**Implementation.** `stem(3×3,s1) → spatial_layer1(s2 + CSPLayer) → spatial_layer2(s2 +
CSPLayer)` (→ ×1/4), then `downsample_k(s2) + fista_layer_k` for k = 1,2,3 producing
×1/8, ×1/16, ×1/32. `Ft` = target-frame maps, `Fr_list` = reference-frame maps.

**Verdict.** ✅ FAITHFUL. Resolution schedule and the "2 spatial + 3 FISTA" split match
Fig 2. `base_channels = 16` → `ch = [128, 256, 512]` (paper doesn't pin; validated by
the param count).

---

## 8. CDC3D — [components/dam.py](components/dam.py) (`CDC3D`)

**Paper.** 3D central-difference conv (Yu et al. 2021):
`y = conv3d(x) − θ·(Σ w)·x_center`, θ = 0.7.

**Implementation.** `out = conv(x)`; `w_sum = weight.sum((2,3,4))`;
`center = conv3d(x, w_sum[:,:,None,None,None])` (1×1×1); `return out − θ·center`.

**Verdict.** ✅ FAITHFUL to the standard CDC formulation; θ = 0.7 matches Yu et al.

---

## 9. SDS — `SDS`

**Paper.** `x = concat[F_T, F_R] ∈ R^{2×C×H×W}`; `B, C, Δ = 3DCDC(x)` — 3DCDC is the
selection function ϕ, producing the input-dependent SSM parameters directly.

**Implementation.** `x = stack([F_R, F_T], dim=2)`; `h = CDC3D(x).mean(dim=2)`
(collapse the 2-frame axis by mean); `p = proj(h)` — a **1×1 conv from a bottleneck
width** `hidden = max(2N, C//8)` up to `d_inner + 2N`; split → `dt, Bp, Cp`. Comment:
*"Bottlenecked (efficiency, NOT from the MOCID paper) … set `cdc_hidden = d_inner+2N`
to disable"*.

**Verdict.** 🔧 DEVIATION (to test). Two departures: (a) the CDC→1×1 **bottleneck**
vs. a single wide 3DCDC; (b) `mean` over the temporal pair to reach 2D (paper is
vague on the collapse). Phase 3: run `cdc_hidden = d_inner + 2*d_state` (paper-literal)
and compare stage-2 AP50. PLAN §7 row 6.

---

## 10. TIS / TIDS — `TIDS`, Eq (9)–(13)

**Paper.** `SP(1×2)` on `F_T, F_R` → `F̄_T, F̄_R ∈ R^{L/2×C}`;
`X_W = Interpolation(F̄_T, F̄_R) = [f¹_R, f¹_T, f²_R, f²_T, …] ∈ R^{L×C}`;
same along height with `SP(2×1)` → `X_H`; VMamba expanding scan TL→BR and BR→TL on
each, merge by add; merge `X_W`, `X_H` results by add.

**Implementation.** `_interp_width` = `stack([r_pool, t_pool], -1).reshape(...)` →
`[r,t,r,t,…]` along W (matches Eq 11). `_bidir_scan` packs `seq` and `seq.flip(-1)`
as K=2 groups into one `selective_scan_fn`, returns `fwd + bwd.flip(-1)`. Width branch
scans row-major (TL→BR); height branch transposes so the scan runs column-major, then
transposes back. `A_log` init `log(arange(1,N+1))` (VMamba / S4D-real). `dt` clamped
to `[−15,15]`; scan output `nan_to_num`'d. Runs fp32.

**Verdict.** ✅ FAITHFUL in structure — interleave order, bidirectional expanding scan,
two-axis merge all match. Numerical clamps are harmless guards not in the paper.

**❓ Open.** `dt, Bp, Cp` are computed by SDS on the **full-res** `H×W` grid and then
`reshape`/`transpose` to length `L`, while `X_W` / `X_H` are **pooled + interleaved**
sequences of the same length `L` but different spatial layout. Token `i` of the
selection parameters does not correspond to the same spatial cell as token `i` of
`X_W`. Eq (12)–(13) index `B_i, C_i` along the interpolated sequence, implying the
params should be interpolated too. Dimensionally it runs; semantically the alignment
is approximate. **Scrutinize** — this could be a genuine correctness gap in the DAM.
Candidate fix: derive `dt/Bp/Cp` from the same pooled+interleaved sequence (or pool
them the same way) so token indices line up.

---

## 11. DAMBlock — `DAMBlock`, Fig 4a

**Paper (Fig 4a).** Top: `LN → Linear → DWConv → SiLU → TIDS`. Gate: `LN → Linear →
SiLU`. Then `⊗` multiply → `Linear` → `+` residual (block input) → `Linear` → out.

**Implementation.** `norm = GroupNorm(1,C)` (= LayerNorm over C), shared for `t` and
`r`. `z = silu(in_z(t))`; `xt = silu(dw(in_x(t)))`, `xr = silu(dw(in_x(r)))` (target &
reference share `in_x`/`dw`); `d = tids(xt, xr)`; `y = mid(d*z)`;
`return out(F_T + y)`. `mid.weight` **zero-init**, `out.weight` **identity-init**
(per-channel eye). Comment: *"AI based weight re-init (model would not train with
random init) … makes the block an exact no-op at step 0"*.

**Verdict.** ⚠️ DEVIATION (documented). Structurally matches Fig 4a; the shared
`in_x`/`dw` for both streams and the gate taken from the target are reasonable reads
of an ambiguous figure. The zero/identity init is a stage-2 stability device (start
DAM as identity so the frozen FPN/head see the stage-1 distribution on step 0).
**Keep**, but Phase 3: once the LR schedule is paper-correct (item 17), retry standard
init and see whether the "won't train" failure still occurs. PLAN §7 row 7.

---

## 12. DisplacementNet / TemporalPooling — `DisplacementNet`, `TemporalPooling`

**Paper.** Displacement Modeling Network = `T−1` **weight-sharing** DAMs (target vs.
each reference); a Temporal Pooling function fuses **the results of the DAMs**.

**Implementation.** One `DAMBlock` per scale (3 total), reused across all `T−1`
references → weight sharing ✅. `disp = [block(F_T, feat[:,r]) for r in 0..T-2]`, then
**`disp.append(F_T)`** and `stack`. `TemporalPooling` = `amax(dim=1)` over all `T`
slices — i.e. over the `T−1` DAM outputs **plus the raw target feature**.

**Verdict.** 🔧 DEVIATION (to test). Including `F_T` in the max-pool changes the fused
result vs. "fuse the results of the DAMs" (`T−1` items). Comment says it keeps
`TemporalPooling` unchanged. Phase 3: pool over the `T−1` DAM outputs only; also try
mean vs. max (paper says only "Temporal Pooling"). PLAN §7 row 8.

---

## 13. FPN — `FPN`

**Paper.** FPN (Lin et al. 2017) fuses clip-level features (`F_T`) and frame-level
features (`F_f`); Fig 2 shows `⊕` then a standard top-down pyramid.

**Implementation.** `x_k ← x_k + ff[k]` (elementwise add of clip-target and pooled
motion), lateral `1×1 → 256`, top-down `nearest` upsample + add, `3×3` smooth
projecting **back to `[c3,c4,c5]`** so the head keeps CSP channel counts.

**Verdict.** ✅ custom but reasonable — a standard top-down FPN with an extra
project-back so downstream channel counts are unchanged; add-fusion matches Fig 2's
`⊕`.

**❓ Cosmetic.** [model.py](model.py) builds the head with
`in_channels=[c*2 for c in ch]` and `width=0.5`; inside `YOLOXHead` the stem uses
`int(in_channels*width) = c`, so the `×2` and `×0.5` cancel. Confirm it is an exact
no-op (it appears to be) and simplify — it reads like a leftover from a concat-fusion
design. PLAN §7 row 11.

---

## 14. YOLOXHead — `YOLOXHead`

**Paper.** Detection head from YOLOX (Ge et al. 2021).

**Implementation.** Decoupled cls / (reg+obj) branches per level, `1×1` stem →
`2×(3×3 conv)` trunks → `1×1` preds. `initialize_biases(1e-2)` → obj/cls bias ≈ −4.6.

**Verdict.** ✅ FAITHFUL to YOLOX.

---

## 15. YOLOLoss + SimOTA — [utils/losses.py](utils/losses.py)

**Paper.** "Binary Cross Entropy loss for classification and IoU loss for regression:
`L = L_reg + L_cls`." (No explicit obj term or weighting given.)

**Implementation.** Near-verbatim YOLOX port. `loss = 5.0·L_iou + L_obj + L_cls`
(`reg_weight = 5.0`); IoU loss `1 − iou²` (`loss_type="iou"`); SimOTA with
`center_radius = 2.5`, cost `= cls + 3·iou_loss + 1e5·(¬in_center)`,
`dynamic_ks = clamp(topk10(iou).sum, min=3, max=·)`.

**Verdict.** 🔧 DEVIATION from the paper *text*; faithful to YOLOX defaults. Two knobs
to test in Phase 3: (a) `reg_weight = 1` to match `L = L_reg + L_cls` literally (the
obj term is a YOLOX-head necessity — document it); (b) `dynamic_ks` floor is `3` here
vs YOLOX's `1` — for 1–3 px targets forcing ≥3 positives per GT may hurt precision;
try `min=1`. PLAN §7 row 3.

---

## 16. Evaluation metric — [utils/eval.py](utils/eval.py)

**Paper.** AP50 (TP if IoU ≥ 0.5), plus Pr, Re, F1. No integration method or
operating point stated.

**Implementation.** `voc_ap` = VOC **post-2010 all-points** interpolation. Score =
`sigmoid(obj)·max_c sigmoid(cls)`; `conf_thr = 1e-3`; NMS at `0.65`; greedy
score-ordered matching, one detection per GT. **F1 reported as the max over the PR
sweep** (`f1.max()`), not at a fixed threshold.

**Verdict.** 🔧 to pin. (a) The AP integration method must be fixed so a plain YOLOX
baseline reproduces the paper's **Base = 83.59** on DAUB — if VOC-all-points gives a
different baseline, try COCO-101-point or VOC-11-point. (b) Paper's Pr/Re/F1 are
self-consistent at one operating point (`2·99.12·97.34/(99.12+97.34) = 98.22`); to
match those columns, report Pr/Re/F1 at a fixed `conf` chosen on val, not the sweep
max. AP50 itself is unaffected by (b). PLAN §7 row 12.

---

## 17. Two-stage training schedule — [train.py](train.py), [utils/utils.py](utils/utils.py), [config.py](config.py)

**Paper.** SGD, wd 5e-4, momentum 0.937, **initial LR 0.01, reduction coefficient
0.1** (step decay), 512×512, T=5, random clip flip. STB trained 100 epochs → frozen →
DAM trained 100 epochs. `L = L_reg + L_cls`.

**Implementation.** Stage 1: train `backbone+pool+fpn+head`, `use_dam=False`, 100 ep.
Stage 2: `seed_from_fista_best` (load best stage-1, drop `disp.*`), freeze backbone
(`.eval()` for BN), train `disp+pool+fpn+head`, `use_dam=True`, 100 ep. SGD+Nesterov,
`WARMUP_EPOCHS=6` then **cosine** decay to `MIN_LR=1e-4`. Stage-2 LR **`LR_DAM=1e-3`**.
`ModelEMA` (stage 2: fast EMA, decay .999, τ 300) used for eval/checkpoints. AMP with
finite-loss step-skipping and grad-clip 10.

**Verdict.** ✅ two-stage structure (100 + 100, freeze STB) is faithful.
🔧 recipe deviations to test in Phase 3:
- **row 1** — cosine + 6-ep warmup → MultiStep γ=0.1 (paper).
- **row 2** — stage-2 LR `1e-3` → `0.01` (paper does not distinguish stages);
  comment says `1e-2` diverges a fresh Mamba branch — verify after the schedule fix.
- **row 9** — stage 2 trains `pool+fpn+head` too; paper says "train the DAM". Try
  freezing FPN/head in stage 2.
- **row 10** — `ModelEMA` is not in the paper; ablate (keep only if it strictly helps).
- Warmup itself is not in the paper.

**❓ Open.** The comparison paragraph mentions pretraining video methods on still
images before adding temporal modules — check whether MOCID's STB stage is meant to
start from a still-image-pretrained CSPDarknet rather than from scratch.

---

## Final results (to fill in)

| Config | Dataset | AP50 | Pr | Re | F1 | Params | vs paper |
|---|---|---|---|---|---|---|---|
| .+FISTA (stage 1) | DAUB | | | | | | Δ vs 92.42 / 96.40 |
| MOCID | DAUB | | | | | | Δ vs 95.93 / 98.22 |
| MOCID | IRDST | | | | | | Δ vs 94.74 / 97.88 |

## Retained deviations (to justify in the writeup)

_List each ⚠️/🔧 that we keep, with the measured cost of reverting it._
