# MOCID code review: model and DAM findings

A review of the model code (`model.py`, `components/components.py`, `components/dam.py`,
`utils/utils.py`) against the MOCID paper, done while reproducing the Base → +FISTA →
+FISTA+DAM ablation on DAUB. It complements [REPORT.md](REPORT.md), which audits the
code module by module. The difference here is that most findings come with a
measurement, and several of them settle verdicts REPORT.md left as "to test".

All numbers are DAUB validation, 4,795 frames, COCO 101-point AP50 at IoU 0.5 (the
paper's metric). The checkpoints come from the ladder run recorded in
`EXPERIMENTS.md`: Base 88.83, +FISTA 89.52 (stage 1, epoch 16), +FISTA+DAM 87.07.

## Summary

| # | Finding | Measured impact | REPORT.md | Suggested change |
|---|---|---|---|---|
| 1 | The DAM "exact no-op" init is not a no-op | **−22.1 AP50** the moment DAM is switched on | item 11 (⚠️ keep) | Take the residual from `F_R`, not `F_T`; verified to restore 89.52 exactly |
| 2 | The target frame is included in the temporal pool | Part of #1; contradicts Fig. 2 | item 12 (🔧) | Pool the T−1 DAM outputs only |
| 3 | DAM is 0.57 M under the paper's budget; the "paper-literal" SDS is ruled out | 3.03 M vs 3.60 M | item 9 (🔧) | Widen `cdc_hidden` to ~86; drop the paper-literal Phase 3 run |
| 4 | Stage 2 trains FPN + head, not just the DAM | Stage 2 decays from 87.07 to 79.93 | item 17 | Freeze FPN/head after fix #1, as the paper states |
| 5 | Base uses PAFPN, MOCID uses plain FPN | Ablation is not neck-controlled | — | Add a Base with MOCID's FPN |
| 6 | `nan_to_num` in the scan disables the finite-check assert | Divergence is silent | — | Count and log non-finite values instead |
| 7 | `GroupNorm(1, C)` is not a channel LayerNorm | Unmeasured | item 11 (says "= LN") | Use a per-pixel channel LN, or fix the comment |
| 8 | `A_log` and the identity-initialised `out` get weight decay | Unmeasured | — | Exclude by name, as Mamba does |
| 9 | Two pairs of duplicated files | Risk of silent drift | — | Delete the unused copies |

Smaller notes are at the end.

---

## 1. The DAM initialisation is not a no-op

**Where.** `components/dam.py`, `DAMBlock.__init__` and `DAMBlock.forward`.

**What the code claims.** `mid` is zero-initialised and `out` is identity-initialised,
with the comment that this makes the block "an EXACT no-op at step 0", so that stage 2
"begins at a stable, deterministic operating point" and the FPN/head "see the stage-1
distribution". REPORT.md item 11 accepts this reasoning and recommends keeping it.

**What actually happens.** At init the block computes `out(F_T + mid(...)) = F_T`, so
every reference slot is replaced by a copy of the **target** feature. The temporal max
pool then returns `amax(F_T, F_T, F_T, F_T, F_T) = F_T`. In stage 1 (DAM off) the same
pool returned `amax` over the five real frames. So switching DAM on does not preserve the
stage-1 features; it deletes every reference frame from the pooled path.

At the tensor level, on a random clip in train mode, the pooled features change by
83–87% relative (`||pool_DAM_off − pool_DAM_on|| / ||pool_DAM_off||` = 0.867 / 0.832 /
0.846 at strides 8 / 16 / 32), and the DAM-on pool equals `F_T` to 1.5e-4.

**Measured on the trained model.** The same stage-1 checkpoint (+FISTA, epoch 16), whose
DAM weights are still exactly at init (`mid == 0`, `out == I`, checked), evaluated two
ways:

| video | DAM off | DAM on (as shipped) | DAM on, residual from `F_R` |
|---|---|---|---|
| data6 | 87.22 | 68.32 | 87.22 |
| data11 | 92.46 | 84.63 | 92.46 |
| data12 | 100.00 | 98.84 | 100.00 |
| data15 | 61.13 | 20.52 | 61.15 |
| data18 | 98.87 | 46.43 | 98.87 |
| data20 | 90.40 | 83.09 | 90.40 |
| data21 | 63.99 | 15.33 | 63.98 |
| **All AP50** | **89.52** | **67.44** | **89.52** |
| All recall | 95.85 | 69.43 | 95.85 |
| Frames with no detection | 14 | 1,278 | 14 |

Turning DAM on costs 22.1 AP50 and 26 points of recall before a single stage-2 step.
data15 and data21, the two hard videos, lose most (−40.6 and −48.7), which is what you
would expect if the reference frames were carrying the motion signal.

**Why it matters for our results.** Stage 2 does not start from the stage-1 model; it
starts from a damaged one and has to relearn the lost signal through a freshly
initialised Mamba branch. In our run it partly recovered (86.90 after the first epoch,
best 87.07 at epoch 3) but never returned to 89.52. This is the most likely reason our
+FISTA+DAM (87.07) sits *below* +FISTA (89.52), the opposite of the paper's ordering
(92.42 → 95.93), though it has not been tested yet: overfitting in stage 2 (finding 4)
could also explain it. We cannot yet say whether DAM helps, because the comparison is
confounded by this handover. The test is a stage-2 rerun with only the change below.

**Suggested change.** Take the residual from the reference frame:

```python
# components/dam.py, DAMBlock.forward
def forward(self, F_T, F_R):
    t, r = self.norm(F_T), self.norm(F_R)
    z = F.silu(self.in_z(t))
    xt = F.silu(self.dw(self.in_x(t)))
    xr = F.silu(self.dw(self.in_x(r)))
    y = self.mid(self.tids(xt, xr) * z)
    return self.out(F_R + y)          # was: self.out(F_T + y)
```

At init this returns `F_R` for each reference, so the pool reproduces the stage-1 `amax`
exactly. We verified this: the patched model scores 89.52 AP50 with DAM on, identical
per video to DAM off (third column above; the 0.01–0.02 differences on data15/21 are
floating-point). It is also a natural reading of the architecture: each DAM output is
"reference frame *r*, refined by its displacement relative to the target". Once the
DAM learns, `mid(...)` becomes the displacement correction.

**How to verify after the change.** Evaluating a stage-1 checkpoint with `use_dam=True`
must reproduce its DAM-off AP50. That check deserves a place in the test suite, since it
guards the whole two-stage design.

---

## 2. The target frame is included in the temporal pool

**Where.** `components/dam.py` `DisplacementNet.forward` (`disp.append(F_T)`).

**Paper.** The text says "a Temporal Pooling function is employed to fuse the results of
DAMs", and Fig. 2 (`diagrams/mocid_architecture.png`) shows exactly that: only the T−1
DAM outputs have arrows into Temporal Pooling. The target feature `F_T` takes two other
paths: into every DAM, and directly into the FPN, where it is added (⊕) to the pooled
features `F_f` at each level.

**Code.** The FPN addition (`x3 = x3 + ff[0]` in `FPN.forward`) matches Fig. 2. The
deviation is that `F_T` is also appended as a T-th slice of the pool, so the pool runs
over 5 items instead of 4. REPORT.md item 12 already flags this.

**Why it matters.** A displacement feature only reaches the neck where it exceeds the
target's own activation, because the max also sees `F_T`. Combined with finding 1, at
init the max is saturated by the target everywhere. Fig. 4(a) does not settle which
frame the DAM residual comes from: the block is drawn with a single input, while TIDS
takes two, so the residual choice in finding 1 is an open reading of the paper rather
than a contradiction of it.

**Suggested change.** Apply finding 1 first. Then run the Fig. 2 variant: pool over the
T−1 DAM outputs only, keeping the FPN addition as the target path. Mean pooling versus
max is worth a run at the same time; the paper says only "Temporal Pooling".

**Open question.** If the pool only sees DAM outputs, the paper does not say what stage 1
pools, since there is no DAM yet. The code pools the raw backbone features of all five
frames in stage 1. Any change here has to keep the stage-1 → stage-2 handover lossless
(the check in finding 1).

---

## 3. The DAM is 0.57 M under budget, and the "paper-literal" SDS is ruled out

**Where.** `components/dam.py` `SDS.__init__`: `hidden = cdc_hidden or max(2 * d_state, C // 8)`.

**Paper, Table 2.** Base 8.94 M, +FISTA 9.45 M, +DAM 12.54 M, MOCID 13.05 M. So FISTA adds
0.51 M and **DAM adds 3.60 M**.

**Code.**

| | ours | paper | gap |
|---|---|---|---|
| Stage 1 (backbone + FISTA + FPN + head) | 9.49 M | 9.45 M | +0.4% |
| DAM | 3.03 M | 3.60 M | −16% |
| Full MOCID | 12.53 M | 13.05 M | −4% |

The FISTA side is faithful. The whole gap is in the DAM. The SDS is the part the paper
leaves least specified, and its width alone is enough to close the gap:

| DAM configuration | DAM params |
|---|---|
| `expand=1, d_state=32`, `cdc_hidden=64` (shipped) | 3.03 M |
| `expand=1, d_state=32`, `cdc_hidden≈86` | **3.59 M** (matches) |
| `expand=2, d_state=32` | 7.55 M |
| `cdc_hidden = d_inner + 2·d_state` ("paper-literal") | 12.7 M (model 22.2 M) |

Two conclusions:

- **`expand=1` is correct.** The Mamba default of 2 would double the paper's DAM budget.
- **The paper-literal SDS cannot be what the authors built.** The code comment calls the
  bottleneck "NOT from the MOCID paper" and proposes `cdc_hidden = d_inner + 2·d_state`
  as the paper-literal setting, and REPORT.md item 9 schedules that run for Phase 3. But
  that model would be 22.2 M against a reported 13.05 M, so the authors must have kept
  the 3DCDC smaller somehow. The paper does not show how. Fig. 4(b) has the 3DCDC output
  B, C and Δ directly (its separate Embedding produces A, which matches the code's learned
  `A_log`), so the code's CDC → 1×1 bottleneck is indeed not drawn in the paper, as the
  code comment says. A bottleneck, a grouped 3DCDC or narrower channels would all fit
  the budget.

**Suggested change.** Make `cdc_hidden` an explicit config value and set it to match the
3.60 M budget (86, or the nearest round width such as 96, which gives 3.84 M). This keeps
the code's bottleneck design and only fixes its size. Update the comment in `SDS` to say
the paper-literal width is ruled out by Table 2. Remove the paper-literal run from the
Phase 3 plan.

---

## 4. Stage 2 trains more than the paper says, and degrades

**Where.** `utils/utils.py` `set_stage`.

**Paper.** "The spatio-temporal backbone is initially trained for 100 epochs, after which
it is frozen while we proceed to train the DAM for an additional 100 epochs."

**Code.** `set_stage(2)` unfreezes `disp`, `pool`, `fpn` and `head`. Its own docstring
says "freeze backbone, train disp", so the docstring and the code disagree.

**Observed.** Our stage 2 (seeded from the 89.52 stage-1 checkpoint, LR 1e-3 → 1e-5)
peaked at epoch 3 (87.07) and then declined steadily to 79.93 at epoch 100, while
validation loss rose from 3.89 to 5.72 and training loss fell. That is overfitting,
and unfreezing FPN and head gives the model 4.2 M extra trainable parameters to do it
with.

**Suggested change.** After finding 1 is fixed, the reason for unfreezing FPN/head (the
distribution shift at the handover) is gone, so train `disp` only, as the paper states.
If `pool` stays in the list it makes no difference (it has no parameters). Keep
best-checkpoint tracking on for stage 2 either way.

---

## 5. Base and MOCID have different necks

**Paper.** The Base row in Tables 1 and 2 is "YOLOx (Ge et al. 2021)", i.e. YOLOX-S with
a **PAFPN** (top-down plus bottom-up). MOCID's neck is a plain top-down **FPN** (Lin et
al. 2017), and the code follows the paper here.

**Why it matters.** The Base → +FISTA step changes both the backbone and the neck, so it
does not isolate FISTA. The same is true of our ladder: our Base is real YOLOX-S
(`YOLOPAFPN`), our +FISTA uses the repo's `FPN`. Our +FISTA gain (+0.69 AP50) and the
paper's (+8.83) both carry this confound.

**Suggested change.** Train one extra Base: single-frame CSPDarknet with MOCID's `FPN`
and head, same recipe. That gives a neck-controlled Base → +FISTA step. It is a
single-frame model and trains in about the time of our existing Base runs.

---

## 6. The finite-check in the forward pass can never catch a divergent scan

**Where.** `components/dam.py` `_bidir_scan`:
`out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)`, and `model.py`
`forward`: `assert torch.isfinite(t).all()`.

**Problem.** The scan output is sanitised before the assert sees it, so a scan that
diverges produces zeros and ±1e4 values and training carries on silently. The assert
still runs every step and forces three GPU→CPU synchronisations per forward pass.

**Suggested change.** Replace `nan_to_num` with a counter: record how many non-finite
values the scan produced, log it per epoch, and raise if it exceeds a threshold. Move the
assert behind a debug flag, or run it every N steps.

---

## 7. `GroupNorm(1, C)` is not a channel LayerNorm

**Where.** `components/dam.py` `DAMBlock.__init__`: `self.norm = nn.GroupNorm(1, C)  # LayerNorm over channels`.
REPORT.md item 11 repeats the claim ("= LayerNorm over C").

**Problem.** With one group, GroupNorm normalises each sample over channels **and** all
spatial positions together. The LN in Fig. 4(a), as in Mamba and VMamba, normalises each
pixel over its channels. For a target of a few pixels the two behave differently: with
GroupNorm the statistics are almost entirely background, so the target has almost no
influence on its own normalisation; a per-pixel LN normalises the target's feature
vector on its own.

**Suggested change.** Either use a per-pixel channel LayerNorm (permute to channels-last
and apply `nn.LayerNorm(C)`, as VMamba does) and measure, or keep GroupNorm and correct
the comment and REPORT.md. The first is the paper-faithful option.

---

## 8. Weight decay on `A_log` and on the identity-initialised `out`

**Where.** `utils/utils.py` `build_optimizer` excludes parameters with `ndim <= 1` from
decay (commit `6c609d6`).

**Problem.** Two DAM parameters fall on the wrong side of that rule:

- **`A_log`** has shape `(d_inner, d_state)`, so it is 2-D and decayed. Mamba and VMamba
  exclude it by name (`_no_weight_decay`). Decay pulls `A_log` towards 0, i.e. `A`
  towards −1, which erodes the log-spaced state timescales it was initialised with.
  `D` is 1-D and correctly excluded.
- **`out.weight`** is initialised to the identity and is decayed like any conv weight.
  Decay shrinks it towards zero even with no gradient, so the residual path through
  the block gradually attenuates.

**Suggested change.** Add a name-based exclusion (`A_log`, and optionally `out.weight` in
`DAMBlock`) to `build_optimizer`. This affects stage 2 only.

---

## 9. Duplicated files

**Where.** `dam.py` duplicates `components/dam.py`; `mocid_module.py` duplicates
`model.py`.

**Check.** With comments and formatting stripped, `dam.py` and `components/dam.py` are
functionally identical. `train.py` and `main.py` import `model.py` and
`components/dam.py`, so `mocid_module.py` and the top-level `dam.py` are never run.
The top-level `dam.py` also lacks the `VMAMBA_PATH` setup, so it only imports if VMamba is already on `sys.path`.

**Suggested change.** Delete the top-level `dam.py`. Keep one MOCID class: if the
registered `mocid_module.py` is the one the benchmark harness needs, make `model.py`
import it rather than redefine it.

---

## Smaller notes

- **Forward and reverse scans share `A_log` and `D`** (`_bidir_scan` concatenates the same
  tensors for both directions). VMamba's cross-scan, which the paper cites, gives each
  direction its own. Separate parameters would cost about 30 K extra per model. Low
  priority.
- **The spatial FISTA filter is half-spectrum**, shape `(1, C, H, W/2+1)`. Using `rfft2`
  forces the effective filter to be conjugate-symmetric, whereas the paper specifies
  `K ∈ ℂ^{C×H×W}`. This is the GFNet convention and probably intended. The filter is
  also tied to the 512×512 input size. These raw tensors hold 1.72 M parameters (14% of
  the model), which is why the optimizer's parameter grouping mattered.
- **`YOLOXHead(..., width=0.5, in_channels=[c * 2 for c in ch])`** (`model.py`). The
  doubling and halving cancel; the head input equals `ch` and the hidden width is fixed
  at 128. Passing `in_channels=ch` and an explicit hidden width would say the same thing
  directly.
- **The stem runs at stride 1** (`SpatioTemporalBackbone.stem`), so the first conv
  processes the full 512×512 frame. YOLOX downsamples in its stem. This is a speed
  cost, not a correctness issue.
- **The FISTA-block residual** (`f + f_out`) is not in the paper's Eq. 5. It is
  documented in the code and in REPORT.md item 5; no new finding.

## Suggested order of work

1. Apply finding 1 (one-line change) and add the DAM-on/DAM-off equality check.
2. Rerun stage 2 from the existing +FISTA checkpoint with `disp` alone trainable
   (finding 4), with `A_log` excluded from decay (finding 8) and the scan's non-finite
   counter in place (finding 6).
3. Widen `cdc_hidden` to the paper's budget (finding 3) and run the T−1 pooling variant
   (finding 2).
4. Train the FPN-neck Base (finding 5) so the ladder is neck-controlled.
5. Clean up findings 7 and 9 and the smaller notes.

## Reproducing the measurements

The evaluation scripts live on the server in `~/mocid_checks/` (not yet in this repo).
The finding-1 table comes from:

```bash
cd ~/mocid_checks
export PYTHONPATH=$HOME/.local/ss-kernel:$HOME/.local/mocid-deps:$HOME/VMamba VMAMBA_PATH=$HOME/VMamba
export MOCID_VAL_PATH=$HOME/mocid_min/daub_val_server.txt TORCH_COMPILE_DISABLE=1
python eval_seq_fista.py runs/fista-sstnet/best_ap50.pth imagenet False   # 89.52
python eval_seq_fista.py runs/fista-sstnet/best_ap50.pth imagenet True    # 67.44
```

The residual-from-`F_R` column was produced by the same evaluation with
`DAMBlock.forward` patched as shown in finding 1. Parameter counts were computed with
`sum(p.numel() for p in module.parameters())` on `model.MOCID(num_frames=5, img_size=512)`.
