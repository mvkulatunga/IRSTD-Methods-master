# Findings: the repo's MOCID vs the ablation models

How the repo's MOCID (`model.py` / `mocid_module.py`) differs from the two ablation models
added alongside it, `MOCIDBase` ([base.py](base.py)) and `MOCIDBaseFISTA`
([base_fista.py](base_fista.py)), and how each lines up with the paper's Table 2.

The ablation models exist so that the paper's rows can be trained as separate models:

| Paper row (Table 2) | Model |
|---|---|
| Base | `MOCIDBase` |
| +FISTA | `MOCIDBaseFISTA` |
| +FISTA+DAM (MOCID) | repo MOCID, but see §4 |

## 1. Architecture side by side

| Part | Repo MOCID | `MOCIDBase` | `MOCIDBaseFISTA` | Paper |
|---|---|---|---|---|
| Front end ("first two layers") | own stem: 3→16 conv at stride 1, then 16→32 and 32→64 stages with CSP (1 and 2 bottlenecks) | YOLOX-S Focus stem + dark2 | YOLOX-S Focus stem + dark2 (identical to `MOCIDBase`) | "retain the first two layers of CSPDarknet" |
| Stages at 1/8, 1/16, 1/32 | stride-2 conv + FISTA layer | YOLOX-S dark3/4/5: stride-2 conv + CSP stack, SPP in dark5 | stride-2 conv + FISTA layer; CSP stacks and SPP removed | "replace the last three spatial layers with three FISTA layers" |
| FISTA layers | `FISTALayer`, 4/4/1 blocks | none | same `FISTALayer`, 4/4/1 blocks | block counts not given |
| Neck | plain top-down FPN | YOLOX PAFPN | plain top-down FPN (the repo's `FPN` class) | Base: YOLOX (PAFPN); MOCID: "Feature Pyramid Network (Lin et al. 2017)" |
| What the neck receives | stage 1: `F_T` + max over all 5 frames; stage 2: `F_T` + max over DAM outputs and `F_T` | target frame | target frame `F_T` only | Fig. 2: `F_T` ⊕ pooled DAM outputs |
| DAM | yes | no | no | yes (MOCID, +DAM rows) |
| Head | YOLOX head, 128 channels | same | same | "Detection Head from (Ge et al. 2021)" |

## 2. Parameter counts

| | Backbone | Neck | Head | DAM | Total | Paper |
|---|---|---|---|---|---|---|
| `MOCIDBase` | 4.213 | 2.835 | 1.890 | — | **8.938** | Base 8.94 |
| `MOCIDBaseFISTA` | 5.298 | 2.297 | 1.890 | — | **9.485** | +FISTA 9.45 |
| repo MOCID, DAM excluded | 5.303 | 2.297 | 1.890 | — | **9.490** | +FISTA 9.45 |
| repo MOCID, full | 5.303 | 2.297 | 1.890 | 3.034 | **12.525** | MOCID 13.05 |

Both ablation models are within 0.4% of the paper. The remaining 0.035 M on +FISTA sits in
widths the paper does not give (the FPN's internal width, the FISTA block counts), so it has
been left at the repo's values.

## 3. The differences, and why each choice was made

### 3.1 The neck changes between Base and +FISTA, as in the paper

The paper's Base is stock YOLOX-S, which uses PAFPN (top-down and bottom-up). Its MOCID uses a
plain FPN. The parameter counts confirm this: 9.45 M for +FISTA is only reachable with the
plain FPN, since keeping PAFPN adds 0.54 M. `MOCIDBaseFISTA` therefore uses the plain FPN, and
the Base → +FISTA step changes FISTA **and** the neck. That is the paper's own comparison, but
it means the +FISTA gain cannot be attributed to FISTA alone.

### 3.2 SPP is removed

YOLOX-S's dark5 has an SPP block (0.657 M) between its downsampling conv and its CSP stack.
The paper replaces "the last three spatial layers" without saying whether SPP counts, but the
count decides it: keeping SPP puts the model at 10.14 M instead of 9.45 M. The repo's
backbone has no SPP either.

### 3.3 The front end differs between the repo's MOCID and the ablation models

The repo's MOCID builds its own front end (a stride-1 stem that runs at full 512×512, then two
stride-2 stages). The ablation models use YOLOX-S's Focus stem and dark2, because the paper
says it retains the first two layers of CSPDarknet and its Base is YOLOX-S. Both front ends end
at 64 channels and 1/4 resolution with almost the same size (1.60 M vs 1.59 M, excluding FISTA),
so the parameter counts cannot distinguish them. They are different layers, though.

### 3.4 What reaches the neck without the DAM

This is the difference most likely to change results.

- **Repo MOCID, stage 1 (DAM off):** the neck receives `F_T + max(F_1, …, F_5)`, the target
  frame's features plus a max over the raw features of all five frames. Reference-frame
  information therefore reaches the neck directly, not only through FISTA.
- **`MOCIDBaseFISTA`:** the neck receives `F_T` only. Temporal information reaches it only
  through the FISTA layers, which mix the frames inside the backbone.

The paper's +FISTA row has no displacement network, and in Fig. 2 the pooled features `F_f`
come from the DAM outputs, so without the DAM there is nothing to pool. `MOCIDBaseFISTA` follows
that reading. The repo's stage-1 pool over raw frames is not described in the paper.

## 4. The full MOCID is not built on `MOCIDBaseFISTA`

Because of §3.3 and §3.4, going from `MOCIDBaseFISTA` to the repo's full MOCID changes more than
the DAM: it also swaps the front end and adds the raw-frame pool. The paper's ablation adds the
DAM alone. A consistent third row would be `MOCIDBaseFISTA` plus the DAM and temporal pooling.
With the DAM as currently written that comes to about 12.52 M; the paper's 13.05 M needs the DAM
widened by the 0.57 M identified in [CODE-REVIEW.md](CODE-REVIEW.md) finding 3.

## 5. A caution about the parameter-count match

The paper's "+Convs" row (13.51 M) replaces the FISTA blocks with "an equivalent number of
convolution blocks". Doing that to the repo's layout gives 6.75 M with its `ConvBlock`, or 7.77 M
with a full 3×3 convolution, nowhere near 13.51 M. The repo's layout reproduces the +FISTA count
but not the +Convs count, so the authors' internal configuration (block counts, widths, what a
"convolution block" is) must differ from it somewhere. Matching the totals shows the models are
the right size; it does not show they have the same internal structure.

## 6. Checks

- `MOCIDBase` loads the checkpoint of the Base-fixedwd-imagenet run with `strict=True` and
  reproduces its numbers exactly on all 4,795 validation frames: 88.83 AP50, 95.88 Pr, 93.06 Re.
- `MOCIDBaseFISTA`'s Focus stem and dark2 are identical to `MOCIDBase`'s (all 42 tensors, same
  names and shapes).
- Forward and backward on a real batch of 4 DAUB clips: correct output shapes, finite loss,
  every parameter receives a gradient, 10 GB peak GPU memory.
- Both classes register in the harness's `detectors.registry`.
- The YOLOX-S backbone and PAFPN in [components/yolox/](components/yolox/) are Megvii's files,
  unmodified.
