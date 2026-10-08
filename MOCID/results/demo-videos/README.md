# Demo videos: detection on DAUB validation videos

Made with `tools/animate.py` on the lab server (1× L40S), 8 Oct 2026, then re-encoded to
H.264 so they play in QuickTime and PowerPoint. Each frame is the last frame of a 5-frame
clip, as in evaluation, so the first 4 frames of each video are skipped. Blue: ground truth
(drawn 2 px outside the box). Green: the model's detections with score ≥ 0.3. The inset
enlarges the area around the target 4×. data6 and data15 are validation videos (never
trained on); data15 is the one every model finds hardest.

| File | Model | Checkpoint | Trained with |
|---|---|---|---|
| `data6_plain.mp4` | none: the video with ground truth only | — | — |
| `data6_stage1.mp4`, `data15_stage1.mp4` | MOCID stage 1 (+FISTA, DAM off) | `checkpoints/r0_fista_best.pth` (R0) | `--profile r0` |
| `data6_mocid_best.mp4`, `data15_mocid_best.mp4` | MOCID, previous DAM (R0-DAM-fixed) | `checkpoints/r0_dam_fixed_best.pth` | `--profile r0` |
| `data6_mambair.mp4`, `data15_mambair.mp4` | MOCID, MambaIR DAM (branch `dam-mambair`) | `runs/a1879572/mambair-dam-1/dam_best.pth` | `--profile r0` |

## Results

| Model | AP50 best / final (all 7 val videos) | data6: target found | data15: target found | Model time per frame |
|---|---|---|---|---|
| Stage 1 (+FISTA) | 88.65 / 88.65 | 95.4% (377/395) | 45.6% (341/747) | 14.0 ms (71 fps) |
| MOCID, previous DAM | 90.12 / 89.20 | 97.0% (383/395) | 53.5% (400/747) | 24.4 ms (41 fps) |
| MOCID, MambaIR DAM | 89.53 / 88.53 | 96.2% (380/395) | 53.1% (397/747) | 28.2 ms (35 fps) |
| *Paper: +FISTA → MOCID* | *92.42 → 95.93* | | | |

- **AP50** comes from each run's training evaluations (`EXPERIMENTS.md`). The videos use the
  best checkpoints, so "best" is the matching number.
- **Target found** is the share of frames where every ground-truth target has a detection
  with score ≥ 0.3 at IoU ≥ 0.5. It is a per-video hit rate at one threshold, not AP50.
- **Model time** is moving the clip to the GPU, the forward pass, decoding and NMS, after a
  warm-up, timed with the GPU synchronised; reading images from disk is not included. The
  numbers above are from earlier renders and agree across repeats: 13.9–14.0 ms (stage 1,
  2 runs), 24.0–24.4 ms (previous DAM, 4 runs), 28.1–29.7 ms (MambaIR DAM, 4 runs).

**The speeds printed inside the videos are not these.** The GPU was shared with another
job while these copies were rendered, so the per-frame time shown on screen is 2–5× slower
(e.g. about 70 ms for the previous DAM instead of 24 ms). Detections and hit rates are
unaffected; use the table above for speed.

All three models were trained with the R0 settings, which apply weight decay to BatchNorm
and bias parameters (the bug fixed in `build_optimizer`; see `EXPERIMENTS.md`). That is the
most likely reason data15 stays near 50%: a Base model trained with the fix reached about
91% recall on data15.
