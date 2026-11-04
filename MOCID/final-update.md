# Final update: CFMW and MambaTree inference timing

Measured on 4 November 2026 on the lab server (1× NVIDIA L40S), after access to it was restored.
These results fill the two blank rows of Table 16 and the outstanding hit rates in Table 22.
The last section lists the report text that changes as a result.

## Results

### Inference time (for Table 16)

| Model | Inference (ms) | Frames per second | Full-model parameters (M) |
|---|---|---|---|
| CFMW-inspired DAM | 32.3 | 31 | 13.78 |
| MambaTree-inspired DAM, N = 1 | 44.2 | 23 | 12.09 |

Each time is the mean of three timing runs: data6 twice and data15 once. FPS is 1000 divided by
that mean.

What the numbers show:

- The CFM branch adds about 8 ms per clip to the corrected original DAM (24.4 ms in Table 16).
- The tree scan costs about 20 ms more than a raster scan at the same state size. A raster-scan
  control at N = 1 (`bidir-d1-leah-1`) took 24.3 ms (41 fps), the same as the original DAM at
  N = 32. MambaTree's extra cost therefore comes from the tree scan, not from the state size.
  This agrees with the training result of about 50% more time per step.
- MambaTree is the only model below 25 frames per second.

### Hit rates (for Section 7.7 and Table 22)

| Model | data6 | data15 |
|---|---|---|
| CFMW-inspired DAM | 95.7% (378/395) | 50.6% (378/747) |
| MambaTree-inspired DAM, N = 1 | 96.7% (382/395) | 55.2% (412/747) |

The definition is the same as in Section 7.7: in a frame, every ground-truth target has a
detection with score ≥ 0.3 at IoU ≥ 0.5. On data15, MambaTree's 55.2% is the highest of any model
(stage 1 45.6%, corrected original DAM 53.5%, MambaIR 53.1%).

## How it was measured

- **Same scope as the existing rows of Table 16.** `tools/animate.py` times each clip from
  transfer to the GPU through the forward pass, box decoding and NMS. It runs 3 warm-up calls
  first and synchronises the GPU around each call. Reading images from disk is not included.
  Batch size 1, five 512 × 512 frames per clip, GPU otherwise idle.
- **One session.** The three existing rows (stage 1, corrected original DAM, MambaIR) were
  re-timed alongside the new ones and agreed with Table 16 within 3%. Their values therefore stay
  as they are, and the new rows can sit beside them.
- **Each model on its own code.** CFMW ran from branch `cfmw-test` (commit `5cd8495`).
  MambaTree ran from Leah's checkout on the server, which is the code that trained
  `tree-d1-leah-1`, using the CUDA tree scan built in `third_party/TreeScan`.
- **One change, made for a fair comparison.** `cfmw-test` sends the four reference frames
  through the DAM in one batched call. `main`, `dam-mambair` and `mamba-tree` call the DAM once
  per reference frame. For this timing, CFMW used the one-per-reference loop like the other rows.
  Both ways give the same detections: 378/395 on data6.

### Reproducing

Run from `MOCID/`. Checkpoints are under `/srv/proj-mamba/runs/`.

| Model | Code | Checkpoint | Command |
|---|---|---|---|
| CFMW | `cfmw-test`, with the loop below | `a1832453/r0-cfm/dam_best.pth` | `python tools/animate.py --video data6 --profile r0-cfm --ckpt <ckpt> --out x.mp4` |
| MambaTree, N = 1 | `mamba-tree` | `a1851923/tree-d1-leah-1/dam_best.pth` | `MOCID_D_STATE=1 python tools/animate.py --video data6 --profile r0-tree --ckpt <ckpt> --out x.mp4` |
| Raster control, N = 1 | `mamba-tree` | `a1851923/bidir-d1-leah-1/dam_best.pth` | `MOCID_D_STATE=1 python tools/animate.py --video data6 --profile r0 --ckpt <ckpt> --out x.mp4` |

- **CFMW:** in `DisplacementNet.forward` (`components/dam.py`), replace the three batched lines
  (`refs = ...`, `disp = self.blocks[k](...)`, `disp = disp.view(...)`) with:

  ```python
  disp = torch.stack([self.blocks[k](F_T, feat[:, r]) for r in range(T - 1)], dim=1)
  ```

- **MambaTree:** `mamba-tree` has no `tools/animate.py`. Copy it from `main`.

The script prints the mean time per frame, the FPS and the hit rate at the end of each run.

### Numbers not to use

- **CFMW at about 19 ms / 52 fps.** This is the batched call. Batching alone speeds up every DAM:
  the corrected original DAM drops to about 17 ms with identical detections. Putting it next to
  the per-reference rows would make CFMW look faster than the DAM it extends. If it is
  mentioned, it belongs in a footnote, as an implementation speed-up that applies to all variants.
- **`tools/check_fps.py`** (in Leah's checkout). It times the forward pass only, on random input,
  with cuDNN autotuning, averaged over a loop without a sync per call. That scope differs from
  Table 16.

### Caveats

- MambaTree was timed from Leah's server checkout: her last commit `ce4cd98` plus her uncommitted
  tree wiring. The report cites `mamba-tree` commit `6f622ee`, which is not on the server, so
  Leah should confirm the two match.
- Per-video AP50 for CFMW and MambaTree is still not measured. Only hit rates were added.

## Report changes

Page numbers are those printed on the report pages.

### Executive Summary (p. iii)

- **Current:** "The displacement module adds about 10 milliseconds to the 14.0 milliseconds per
  clip of the detector without it, and the MambaIR version adds a further 4 milliseconds. The
  tree scan needed about 50% more time per training step."
- **Change:** add the two new costs, for example: "The CFMW branch adds about 8 milliseconds,
  and the tree scan about 20 milliseconds more than the row-by-row scan at the same memory size,
  which takes that model below 25 frames per second."

### Section 1.4, Objectives (p. 3), last paragraph

- **Current:** "and inference timing was measured for all variants except CFMW and MambaTree,
  because the project's GPU server became unavailable late in the year."
- **Change:** "and inference timing was measured for all variants."

### Section 5.12 (p. 37)

- **Current:** "Four configurations were compared in Tables 9 and ??".
- **Change:** fix the broken reference. Either cite Table 9 only, or cite the table that was
  meant.

### Section 6.4 (p. 39)

- **Current:** "the GPU server was lost late in the year"
- **Change (optional):** say it was unavailable from 21 October until access was restored. Use
  the actual restore date if known; it was working by 4 November.

### Section 7.7 (p. 45)

- **Current:** "The CFMW and MambaTree variants could not be rendered, for the reason given in
  Section 7.9."
- **Change:** replace with the hit rates. Suggested text: "On data15 the rate was 50.6% for the
  CFMW variant and 55.2% for the MambaTree variant, and on data6 it was 95.7% and 96.7%."

### Section 7.8 (pp. 45–46)

- **Table 16:** fill in CFMW 32.3 ms / 31 fps and MambaTree (N = 1) 44.2 ms / 23 fps.
- **Table 16 caption:** delete "Training cost is relative to the corrected original DAM, from
  the steps per second in Table 9." The table has no training-cost column and Table 9 has no
  steps per second. Delete "Dashes mark values that were not measured." once no dashes remain.
  Consider adding: "All rows call the DAM once per reference frame."
- **"The DAM adds about 10 ms per clip and the MambaIR additions a further 4 ms."** Add: "the CFM
  branch adds about 8 ms, and the tree scan about 20 ms more than a raster scan at the same state
  size (24.3 ms at N = 1)."
- **"All measured models run faster than the 25 frames per second of typical video on this
  GPU"** is no longer true. Change to: "All models except the MambaTree-inspired DAM (23 fps) run
  faster than ...".
- **"The CFMW and MambaTree variants have training measurements in place of inference times."**
  Rewrite the opening, for example: "Training cost was also recorded." Keep the 78 ms vs 76 ms
  sentence: both of those steps used the batched call, so they are comparable with each other
  but not with Table 16. Keep the sentence about 50% longer per step. Optionally add the batching
  footnote from "Numbers not to use" above.

### Section 7.9 (p. 46)

- **Current:** "The project GPU server became unavailable on 21 October 2026 following a system
  update, and restoring access was beyond the team's control. Inference timing, per-video
  results and demonstration hit rates for the CFMW and MambaTree variants were therefore not
  completed."
- **Change:** "The project GPU server was unavailable from 21 October 2026 following a system
  update. Once access was restored, inference timing and demonstration hit rates for the CFMW
  and MambaTree variants were measured (Sections 7.7 and 7.8). Per-video AP50 for these two
  variants remains outstanding (Appendix I)." Keep the sentence saying AP50 and F1 are
  unaffected.

### Section 9, Conclusion (p. 49)

- **Current:** "with benchmark integration and the remaining timing measurements passed on as
  recommendations"
- **Change:** "with benchmark integration passed on as a recommendation"

### Appendix G.1, Table 18 (p. 73)

- **Change:** add a milestone row: "4 November: server access restored; CFMW and MambaTree
  inference timing and hit rates measured."

### Appendix G.4, Table 20 (p. 74)

- **Change:** add `MOCID/final-update.md` (this file) to the "Section 7.8, timing and
  demonstrations" row.

### Appendix I (p. 78)

- **Table 21, row 2 ("Remaining inference measurements"):** timing and hit rates are done.
  Narrow the row to per-video AP50 for CFMW and MambaTree, or remove it.
- **Section I.2:** "measurements were completed before the GPU server became unavailable on
  21 October 2026" needs rewording, since the timing has now been completed.
- **Table 22:** for CFMW and MambaTree, change "Inference time" and "Hit rates" from Outstanding
  to Done.
