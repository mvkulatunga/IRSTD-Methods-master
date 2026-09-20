# What training settings produce the paper's Base (83.59 AP50)?

The paper reports Base = AP50 **83.59**, Pr 94.27, Re 89.34, F1 91.74 (Table 2's Base row is
the same result as Table 1's "YOLOx" row). It does not state Base's batch size, epoch count,
LR scaling or initialisation. Reconstructing the recipe and training it out gives **86.4-88.8**,
so something has to explain the difference. This is what we tested.

## The reconstructed recipe, and why we believe it

- MOCID's own repository (`github.com/TanzanOY/MOCID`) contains only a README, no code.
- The paper says it follows SSTNet (Chen et al. 2024) for the DAUB split, and its stated
  hyperparameters are SSTNet's `yolox-pytorch`-derived defaults verbatim - including
  **momentum 0.937**, which is that codebase's idiosyncrasy (stock YOLOX uses 0.9).
- The paper's Pr/Re/F1 reproduce exactly under SSTNet's `vid_map_coco.py` convention
  (Re = maximum recall at conf 0.001, Pr = mean precision up to that recall):
  2 x 94.27 x 89.34 / (94.27 + 89.34) = **91.74**, the paper's F1 to the decimal.
- Architecture confirmed: YOLOX-S at 1 class is **8.94 M** parameters, matching Table 2's Base;
  the same head at 80 classes gives 8.97 M, matching published YOLOX-S.

So: YOLOX-S, 512 px, SGD + Nesterov, momentum 0.937, weight decay 5e-4 on conv/linear weights
only, LR 0.01 scaled by batch/64, warmup + cosine, EMA, ImageNet-normalised input, evaluated at
conf 0.001 / NMS 0.65 with COCO AP50 at IoU 0.5 (our implementation is verified identical to
`pycocotools`, see `Base-fixedwd-imagenet/README.md`).

## Hypotheses tested

| # | Hypothesis | Test | Outcome | Verdict |
|---|---|---|---|---|
| 1 | They evaluated at a stricter confidence threshold | rescored the converged Base at conf 0.05 / 0.1 / 0.25 / 0.5 / 0.7 | AP50 falls 85.5 -> 78.8, but precision **rises** to 97-98 | **Ruled out.** The paper's Pr (94.27) is *lower* than ours at every threshold, so they were not truncating |
| 2 | They reported the checkpoint SSTNet's code saves by lowest validation loss | evaluated `best_valloss.pth` | epoch 34, AP50 **88.51** | **Ruled out** |
| 3 | They used batch 8 (two RTX3090s), halving the updates | full 100-epoch run at batch 8 | best **86.77**, final 86.39 | **Partial** - explains about 2 of the ~5 points |

## The decisive observation

Both of our runs pass through the paper's four Base metrics *early in training*:

| Run | Closest epoch | Updates | AP50 | Pr | Re | F1 |
|---|---|---|---|---|---|---|
| batch 4 (`Base-fixedwd-imagenet`) | 29 of 100 | 13,021 | 83.90 | 94.59 | 89.84 | 92.16 |
| batch 8 (`Base-batch8`) | 19 of 100 | 4,256 | 82.57 | 93.23 | 89.18 | 91.16 |
| **Paper, Base** | - | - | **83.59** | **94.27** | **89.34** | **91.74** |

Four independent metrics matching within ~0.5 points is not coincidence: the recipe is right,
and the paper's Base sits at an early point on its training curve.

## Converged results with the same recipe

| Configuration | Best AP50 | Final AP50 |
|---|---|---|
| Batch 4 | 88.83 (ep51) | 87.99 |
| Batch 8 | 86.77 (ep48) | 86.39 |
| Paper, Base | 83.59 | - |

## Conclusion

**The paper's Base is reproducible as a point on the training curve, not as a converged model.**
Trained to convergence with the same architecture, data, split, metric and recipe, the same
baseline reaches 86.4-88.8 AP50. We cannot determine *why* theirs stopped where it did - fewer
epochs, a checkpoint that was not the best, or simply an untuned comparison baseline - but it
is not explained by the evaluation protocol, the checkpoint-selection rule, or batch size alone,
each of which was tested and rejected above.

**Implication for the ablation.** The paper's headline FISTA gain of +8.8 AP50 (83.59 -> 92.42)
is measured against a baseline 3-5 points below what that baseline reaches when trained out.
Measured with the same pipeline on both sides, FISTA's aggregate advantage largely disappears;
its measurable benefit shows up per-video, as recall on the cluttered sequence (data15: 97.2%
with FISTA against 90.7% for Base). Per-video numbers are worth reporting alongside the
aggregate - the paper reports only aggregates, and five of the seven validation videos sit at
88-100 AP50 regardless of model.

## Caveats

- SSTNet's released code is the reference, not MOCID's (unreleased). The recipe is inferred from
  matching hyperparameters, metric convention and data split - strong evidence, not proof.
- Our runs train **from scratch**. The reference code initialises from `model_data/pre_trained.pth`,
  which is a Baidu download we could not obtain. A COCO-pretrained run is in
  `results/Base-pretrained/` (peak 89.82), but it used YOLOX-native 0-255 input rather than
  ImageNet normalisation, so it is not a controlled comparison.
- All numbers here: COCO AP50 at IoU 0.5 with SSTNet's Pr/Re/F1 convention, conf 0.001, NMS 0.65,
  over all 4,795 validation frames.
- Training scripts live on the server (`~/mocid_checks/`), not in this repository.

## Runs referenced

- `results/Base-fixedwd-imagenet/` - batch 4, the main converged Base
- `results/Base-batch8/` - batch 8 (hypothesis 3)
- `results/Base-pretrained/` - COCO-pretrained init
- `results/Base-YOLOX-S/` - the original run under the repo's pre-fix recipe (76.95)
