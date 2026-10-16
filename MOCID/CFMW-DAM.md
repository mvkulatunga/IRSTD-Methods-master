# CFMW's fusion Mamba inside the DAM (branch `cfmw-test`)

CFMW (Li et al., *IEEE TCSVT* 2025; [arXiv 2404.16302](https://arxiv.org/abs/2404.16302);
[github.com/lhy-zju/CFMW](https://github.com/lhy-zju/CFMW)) fuses two streams of the same
scene, RGB and infrared, with a Cross-modality Fusion Mamba (CFM). In MOCID every DAM also
relates two streams of the same scene: the target frame and one reference frame. This branch
adds CFM to every DAM, with the two frames as its streams, beside the DAM's own scan (TIDS).

## Where it goes

```
5-frame clip  x1..x4 (references), x5 (target)
      │
      ▼
┌──────────────────────────────────────────────┐
│ Spatio-temporal backbone: CSPDarknet + FISTA │   clip-level motion context (frozen in stage 2)
└──────────────────────────────────────────────┘
      │  F_T and F_R1..F_R4 at 1/8, 1/16, 1/32
      ▼
┌──────────────────── Displacement Modeling Network ─────────────────────────────┐
│  4 weight-sharing DAMs per scale, one per (F_T, F_Rk) pair                     │
│  ┌────────────────────────────── DAM ──────────────────────────────────────┐   │
│  │  LN → Linear → DWConv → SiLU  ──►  xt, xr                               │   │
│  │                                     │                                   │   │
│  │                 ┌───────────────────┴──────────────────┐                │   │
│  │                 ▼                                      ▼                │   │
│  │  ┌──────────────────────────┐   ┌────────────────────────────────────┐  │   │
│  │  │ TIDS (paper)             │   │ ★ CFM (CFMW), new                  │  │   │
│  │  │ SDS: 3D-CDC → B, C, Δ    │   │ pool both frames to 8×8 tokens     │  │   │
│  │  │ TIS: interleave ref/tgt  │   │ channel swap (half the channels)   │  │   │
│  │  │ tokens, scan W and H     │   │ [target ; reference] + position    │  │   │
│  │  │ both ways                │   │ bidirectional Mamba → DWConv → LN  │  │   │
│  │  │ = LOCAL displacement     │   │ upsample = GLOBAL cross-frame ctx  │  │   │
│  │  └────────────┬─────────────┘   └─────────────────┬──────────────────┘  │   │
│  │               └──────────────── + ────────────────┘   (CFMW's Add2)     │   │
│  │                                 │                                       │   │
│  │  LN → Linear → SiLU (gate z) ──►×──► Linear (mid, zero init)            │   │
│  │                                 │                                       │   │
│  │                     F_Rk ──────►+──► Linear (out) ──► slot k            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┘
      │  4 DAM slots + F_T
      ▼
Temporal pooling (max) ──► FPN (+ F_T, clip level) ──► YOLOX head
```

## Why there

- **The DAM is MOCID's only two-stream module.** CFM fuses two streams of the same scene, which
  is exactly what each DAM does with (F_T, F_R). The backbone (FISTA) already works on the whole
  clip, and the FPN and head work within one frame. Nothing else in MOCID compares two frames.
- **Beside TIDS, not instead of it.** TIDS is the paper's displacement model: it puts each target
  token next to the matching reference token (1×2 / 2×1 pooled) and scans, so it sees *local*,
  frame-to-frame displacement. CFM puts the whole reference frame and the whole target frame in
  one sequence, so every token sees the other frame's entire layout: *global* context. That's
  what distinguishes a target's own motion from scene-wide change. In data15, the video MOCID
  finds hardest, the camera pans among target-like clutter, so local differences fire everywhere.
- **Inside the gate.** CFM's output joins TIDS's output before the gate and `mid`. So it uses
  the DAM's zero-init handover (switching the DAM on still leaves stage 1's output unchanged), and
  the DAM's output format, the slot in the temporal pool, doesn't change.

## CFMW's code, mapped (models/common.py `GPT`, vim/mamba_module.py)

| CFMW | Here (`components/cfm.py`) |
|---|---|
| avg-pool each stream to 8×8 tokens | same (`CFM_GRID = 8`) |
| `TokenSwapMamba`: swap the first half of the channels (its Mamba encoders are commented out) | same (`channel_swap`) |
| concatenate both streams' tokens + learned position embedding (zero init) | same, target tokens first |
| `CrossMamba` × n_layer: LayerNorm → Mamba → 3×3 depthwise conv + residual | same, 1 layer; the Mamba scans **both directions**, so each frame also sees the other; the conv runs on each frame's own 8×8 grid (CFMW views the joint sequence as one 8×16 map, which mixes the two frames' rows) |
| LayerNorm, split per stream, bilinear upsample, `Add2` to each stream | same; both streams' outputs are summed and added to TIDS's output |

CFMW's Mamba is a patched `mamba_ssm` with an `extra_emb` input, but CFMW passes the same tokens
there, so it behaves as a standard Mamba. Here the scan is VMamba's CUDA kernel, as in TIDS.

**Departures, from our own measurements (2026-10-08).** In r0-damv2, a Mamba with Mamba's Δ
initialisation got almost no gradient under R0's SGD and weight decay on every parameter, and its
state term fell to ~0% of its output. So here:
- B and C are RMS-normalised (Mamba-3's BC normalisation). That raised those gradients about 2000×.
- A_log, D, the Δ bias, the B/C scales and the position embedding are kept out of weight decay
  (Mamba's own convention), through `_no_weight_decay`, which `build_optimizer` now honours.
  Nothing outside CFM sets it.

## Checks before training (all pass)

- The two-direction scan matches a step-by-step reference implementation exactly.
- Channel swap and token/grid reshapes are correct; each frame keeps its own rows.
- **Handover:** switching a fresh DAM on changes the head outputs by 0 in full fp32. Under TF32
  it's 6.4e-4, identical with and without CFM; that's the DAM's identity `out` conv rounding.
- CFM's memory is active at initialisation: zeroing B and C changes its output by 6–32%
  (r0-damv2's DAM: ~1%).
- 40 real training steps: losses and weights finite, and all 54 CFM tensors move.
- The fixed-DAM checkpoint (`r0_dam_fixed_best.pth`) still loads under `--profile r0`.
- DisplacementNet now runs the 4 references in one batched call. In full fp32 that equals the old
  per-reference loop to 6e-7 (exactly 0 without CFM), and it cuts a training step from 96 to
  76 ms. With CFM a step takes 78 ms.

Size: CFM adds 1.26 M parameters. DAM 3.03 → 4.29 M, MOCID 12.53 → 13.78 M (paper: 13.05 M).

## Run and compare

```bash
cd ~/IRSTD-Methods-master/MOCID
python main.py --profile r0-cfm train --tag r0-cfm \
    --stage1-from /srv/proj-mamba/mocid-baseline/checkpoints/r0_fista_best.pth
column -s, -t runs/r0-cfm/eval_log.csv   # DAM on vs DAM off at every evaluation
python main.py --profile r0-cfm eval --ckpt runs/r0-cfm/dam_best.pth --perseq
```

`--profile r0-cfm` is R0's settings plus `DAM_CFM` and `EVAL_DAM_OFF`. The latter scores every
stage-2 evaluation a second time with the DAM switched off (two extra columns in
`eval_log.csv`), so the log shows what the DAM adds at each epoch.

Compare against the same stage 2 without CFM: R0-DAM-fixed, 90.12 best / 89.20 final. Two
earlier findings matter for reading the result:
- in R0-DAM-fixed, switching the DAM off scored higher (91.59 vs 90.12);
- per-video results on data15 and data21 vary a lot between identical runs.

So judge CFM by its DAM-on vs DAM-off gap and by data15 recall, and confirm any gain with a
second seed.
