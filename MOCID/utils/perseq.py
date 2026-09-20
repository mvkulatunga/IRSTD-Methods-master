"""Per-video-sequence recall breakdown.

The aggregate AP50/F1 can look mediocre-but-fine while hiding a small number
of validation videos failing almost completely -- that's exactly what the
Base ablation's data15/data21 collapse looked like (EXPERIMENTS.md). Run this
after any eval, not just the aggregate score, to catch that pattern early.

Usage: `python main.py eval --ckpt <path> --perseq`
"""

import os
from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

from .eval import compute_ap50_f1_full, decode_outputs, iou_matrix, predict_image


def collate_eval_perseq(batch):
    """batch -> ((B,T,3,H,W) clips, list of (n_gt,4) xyxy GT tensors, list of
    target-frame source paths)."""
    clips = torch.stack([b[0] for b in batch])
    gts = [b[1]["boxes"].clone() for b in batch]
    paths = [b[1]["path"] for b in batch]
    return clips, gts, paths


@torch.no_grad()
def perseq_breakdown(
    model, loader, device, use_dam, strides, num_classes, conf_thr=1e-3, nms_thr=0.65
):
    """Run eval and break the result down per source video (folder name).

    Recall / zero-detection counts are taken at the same best-F1 confidence
    threshold the aggregate F1 already reports elsewhere in this codebase --
    not an independently chosen cutoff -- so this is the per-sequence view of
    that exact operating point, not a different measurement.

    -> (summary dict from compute_ap50_f1_full,
        {seq_id: {frames, gt_frames, hits, recall_pct, zero_det,
                   median_top_score}}, sorted by seq_id).
    """
    model.eval()
    all_dets, gt_by_img, seq_ids, per_frame = [], [], [], []
    img_idx = 0

    for clips, gts, paths in tqdm(
        loader, desc=f"eval(perseq, use_dam={use_dam})", mininterval=10
    ):
        clips = clips.to(device)
        decoded = decode_outputs(model(clips, use_dam=use_dam), strides)
        for b in range(decoded.shape[0]):
            box, scores = predict_image(decoded[b], num_classes, conf_thr, nms_thr)
            box_np = box.cpu().numpy()
            sc_np = scores.cpu().numpy()
            for s, bx in zip(sc_np, box_np):
                all_dets.append((float(s), img_idx, bx))
            gt_np = gts[b].cpu().numpy()
            gt_by_img.append(gt_np)
            seq_ids.append(os.path.basename(os.path.dirname(paths[b])))
            per_frame.append((box_np, sc_np, gt_np))
            img_idx += 1

    summary = compute_ap50_f1_full(all_dets, gt_by_img)
    thr = summary["thr"]

    stats = defaultdict(
        lambda: {"frames": 0, "gt_frames": 0, "hits": 0, "zero_det": 0, "scores": []}
    )
    for seq_id, (box_np, sc_np, gt_np) in zip(seq_ids, per_frame):
        s = stats[seq_id]
        s["frames"] += 1
        s["scores"].append(float(sc_np.max()) if len(sc_np) else 0.0)

        keep = sc_np >= thr
        if keep.sum() == 0:
            s["zero_det"] += 1

        if len(gt_np) > 0:
            s["gt_frames"] += 1
            if keep.sum() > 0:
                ious = iou_matrix(box_np[keep], gt_np)
                if ious.size and ious.max() >= 0.5:
                    s["hits"] += 1

    out = {}
    for seq_id, s in stats.items():
        out[seq_id] = {
            "frames": s["frames"],
            "gt_frames": s["gt_frames"],
            "hits": s["hits"],
            "recall_pct": 100.0 * s["hits"] / max(s["gt_frames"], 1),
            "zero_det": s["zero_det"],
            "median_top_score": float(np.median(s["scores"])) if s["scores"] else 0.0,
        }
    return summary, dict(sorted(out.items()))


def print_perseq_table(summary, per_seq):
    """Pretty-print the breakdown to stdout."""
    print(
        f"AP50 {summary['ap']*100:.2f}  F1 {summary['f1']*100:.2f}  "
        f"(threshold {summary['thr']:.3f}, Pr {summary['prec']*100:.2f}, "
        f"Re {summary['rec']*100:.2f})"
    )
    print(f"{'seq':<10}{'frames':>8}{'recall%':>10}{'0-det':>8}{'median top score':>18}")
    tot_frames = tot_gt = tot_hits = 0
    for seq_id, s in per_seq.items():
        print(
            f"{seq_id:<10}{s['frames']:>8}{s['recall_pct']:>10.2f}"
            f"{s['zero_det']:>8}{s['median_top_score']:>18.3f}"
        )
        tot_frames += s["frames"]
        tot_gt += s["gt_frames"]
        tot_hits += s["hits"]
    print(
        f"{'ALL':<10}{tot_frames:>8}{100.0*tot_hits/max(tot_gt,1):>10.2f}"
    )
