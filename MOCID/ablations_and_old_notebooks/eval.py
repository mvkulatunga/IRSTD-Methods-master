import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def collate_eval(batch):
    """Collate (clip, target) pairs into a batch.

    Returns:
        clips: (B, T, C, H, W) stacked clips.
        gts: list of B ground-truth box tensors, each (n_gt, 4) xyxy.
    """
    clips = torch.stack([b[0] for b in batch])  # (B, T, C, H, W)
    gts = [b[1]["boxes"].clone() for b in batch]  # each (n_gt, 4) xyxy in 512-space
    return clips, gts


@torch.no_grad()
def decode_outputs(outputs, strides):
    """Decode raw per-scale grid predictions into absolute pixel boxes.

    Each scale is a (B, C, H, W) grid where every cell holds a raw box
    (offset xy, log wh), objectness, and class logits. This flattens
    each grid, adds each cell's position to the predicted xy offset,
    scales by stride to get pixel coords, and exp-scales wh. Objectness
    and class logits are passed through unchanged (sigmoid applied later
    in `predict_image`). wh is clamped (max=20.0) before exp to avoid
    inf/NaN blow-ups.

    Args:
        outputs: list of (B, C, H, W) tensors, one per scale.
            C = 4 (box) + 1 (obj) + num_classes.
        strides: list of ints, downsample stride per scale (e.g. [8, 16, 32]).

    Returns:
        (B, N, C) tensor, all scales concatenated. N = total cells.
        Columns are (cx, cy, w, h, obj_logit, cls_logits...) — xy is the
        box CENTER, not a corner.
    """
    decoded = []
    for out, stride in zip(outputs, strides):
        B, C, H, W = out.shape
        out = out.flatten(2).permute(0, 2, 1)  # B, C, HxW -> B, HxW, C
        yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid = torch.stack((xv, yv), 2).reshape(1, -1, 2).to(out.device).type_as(out)

        xy = (out[..., :2] + grid) * stride

        # [PATCH APPLIED HERE]: Clamp to max 20.0 to prevent inf explosions
        wh = torch.exp(torch.clamp(out[..., 2:4], max=20.0)) * stride

        rest = out[..., 4:]
        decoded.append(torch.cat([xy, wh, rest], dim=-1))
    return torch.cat(decoded, dim=1)


def cxcywh_to_xyxy(b):
    """Convert boxes (N, 4) from (cx, cy, w, h) to (x1, y1, x2, y2)."""
    x1 = b[:, 0] - b[:, 2] / 2
    y1 = b[:, 1] - b[:, 3] / 2
    x2 = b[:, 0] + b[:, 2] / 2
    y2 = b[:, 1] + b[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def nms_xyxy(boxes, scores, iou_thr):
    """Greedy NMS on xyxy boxes.

    Args:
        boxes: (N, 4) xyxy.
        scores: (N,) confidence per box.
        iou_thr: suppress a box if it overlaps a higher-scoring one above this.

    Returns:
        Long tensor of kept indices, highest score first.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest])
        yy2 = torch.min(y2[i], y2[rest])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


@torch.no_grad()
def predict_image(decoded_i, num_classes, conf_thr, nms_thr):
    """Turn one image's decoded predictions into final detections.

    Sigmoids obj/class logits, scores = obj * best-class-conf, filters by
    conf_thr, converts to xyxy, and runs NMS.

    Args:
        decoded_i: (N, 4 + 1 + num_classes) for one image (cxcywh + logits).
        num_classes: number of classes.
        conf_thr: min score to keep a detection.
        nms_thr: IoU threshold for NMS.

    Returns:
        (boxes, scores): (M, 4) xyxy boxes and (M,) scores, post-NMS.
    """
    box = cxcywh_to_xyxy(decoded_i[:, :4])
    obj = decoded_i[:, 4:5].sigmoid()
    cls = decoded_i[:, 5:].sigmoid()
    cls_conf, _ = cls.max(dim=1, keepdim=True)
    scores = (obj * cls_conf).squeeze(1)
    m = scores >= conf_thr
    box, scores = box[m], scores[m]
    if box.numel() == 0:
        return box, scores
    keep = nms_xyxy(box, scores, nms_thr)
    return box[keep], scores[keep]


def iou_matrix(a, b):
    """Pairwise IoU between two sets of xyxy boxes.

    Args:
        a: (Na, 4) xyxy.
        b: (Nb, 4) xyxy.

    Returns:
        (Na, Nb) IoU array (all-zero if either input is empty).
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def voc_ap(rec, prec):
    """Average Precision via the PASCAL VOC (post-2010) all-points method.

    Args:
        rec: recall values (non-decreasing).
        prec: precision values, same length as rec.

    Returns:
        float AP (area under the interpolated PR curve).
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap50_f1(all_dets, gt_by_img, iou_thr=0.5):
    """Compute AP@0.5 and best F1 over the PR sweep.

    Sorts detections by score, greedily matches each to an unmatched GT
    box in the same image (TP if IoU >= iou_thr, else FP), builds the PR
    curve, and returns AP plus the max F1 along it.

    Args:
        all_dets: list of (score, img_index, box) across all images.
        gt_by_img: per-image list of (n_gt, 4) xyxy GT arrays.
        iou_thr: match threshold (default 0.5 => AP50).

    Returns:
        (ap, best_f1). (0.0, 0.0) if no GT or no detections.
    """
    npos = sum(len(g) for g in gt_by_img)
    if npos == 0 or len(all_dets) == 0:
        return 0.0, 0.0

    all_dets.sort(key=lambda d: d[0], reverse=True)
    nd = len(all_dets)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    matched = {i: np.zeros(len(g), dtype=bool) for i, g in enumerate(gt_by_img)}

    for k, (score, img, box) in enumerate(all_dets):
        gts = gt_by_img[img]
        if len(gts) == 0:
            fp[k] = 1
            continue
        ious = iou_matrix(box[None, :], gts)[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[img][j]:
            tp[k] = 1
            matched[img][j] = True
        else:
            fp[k] = 1

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = tp_c / npos
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    ap = voc_ap(rec, prec)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return ap, float(f1.max())


@torch.no_grad()
def evaluate(
    model, loader, device, use_dam, strides, num_classes, conf_thr=1e-3, nms_thr=0.65
):
    """Run the model over the loader and compute AP50 / best-F1.

    For each batch: forward pass, decode, per-image post-process, then
    pool detections and GT for scoring.

    Args:
        model: detection model, called as model(clips, use_dam=use_dam).
        loader: yields (clips, gts) as from collate_eval.
        device: device for input clips.
        use_dam: enable the DAM branch in the forward pass.
        strides: output strides (for decode_outputs).
        num_classes: number of classes.
        conf_thr: keep-detection threshold (default 1e-3, low to populate
            the PR sweep).
        nms_thr: NMS IoU threshold (default 0.65).

    Returns:
        (ap50, f1) from compute_ap50_f1.
    """
    model.eval()
    all_dets, gt_by_img = [], []
    img_idx = 0
    for clips, gts in tqdm(loader, desc=f"eval(use_dam={use_dam})", mininterval=10):
        clips = clips.to(device)
        outs = model(clips, use_dam=use_dam)
        decoded = decode_outputs(outs, strides)
        for b in range(decoded.shape[0]):
            box, scores = predict_image(decoded[b], num_classes, conf_thr, nms_thr)
            box = box.cpu().numpy()
            scores = scores.cpu().numpy()
            for s, bx in zip(scores, box):
                all_dets.append((float(s), img_idx, bx))
            gt_by_img.append(gts[b].cpu().numpy())
            img_idx += 1
    return compute_ap50_f1(all_dets, gt_by_img)


def count_params_m(model, include_dam):
    """Count params in millions.

    Args:
        model: the model.
        include_dam: if False, exclude params named "disp.*" (the DAM branch).

    Returns:
        float, param count in millions.
    """
    if include_dam:
        n = sum(p.numel() for p in model.parameters())
    else:
        n = sum(
            p.numel()
            for name, p in model.named_parameters()
            if not name.startswith("disp.")
        )
    return n / 1e6


def _strip_compile(sd):
    """Drop the `_orig_mod.` prefix that torch.compile adds to state-dict keys."""
    return {
        k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v
        for k, v in sd.items()
    }


def main():
    """CLI entry point: load a checkpoint, then either print param counts
    (`--params-only`) or run eval and print AP50 / best-F1.
    """
    from mocid import MOCID, MOCIDDataset, Config  # lazy import breaks the cycle

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_last.pth")
    ap.add_argument("--dam", dest="dam", action="store_true")
    ap.add_argument("--no-dam", dest="dam", action="store_false")
    ap.add_argument("--params-only", action="store_true")
    ap.add_argument("--conf", type=float, default=1e-3)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.set_defaults(dam=None)
    args = ap.parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strides = [8, 16, 32]
    num_classes = 1

    model = MOCID(
        num_classes=num_classes, num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(_strip_compile(ckpt["model"]))
    stage = ckpt.get("stage", 2)
    use_dam = args.dam if args.dam is not None else (stage == 2)
    print(f"loaded {args.ckpt}  (stage={stage})  ->  use_dam={use_dam}")

    print(f"Params  .+FISTA (no DAM)      : {count_params_m(model, False):.2f} M")
    print(f"Params  .+FISTA+DAM (MOCID)   : {count_params_m(model, True):.2f} M")
    if args.params_only:
        return

    val_ds = MOCIDDataset(cfg.val_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=False)
    assert len(val_ds) > 0, f"Val set empty — check {cfg.val_path} (cwd={os.getcwd()})"
    loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_eval,
    )

    ap50, f1 = evaluate(
        model,
        loader,
        device,
        use_dam,
        strides,
        num_classes,
        conf_thr=args.conf,
        nms_thr=args.nms,
    )
    row = ".+FISTA+DAM (MOCID)" if use_dam else ".+FISTA"
    print(f"\n=== {row} ===")
    print(f"AP50 : {ap50 * 100:.2f}")
    print(f"F1   : {f1 * 100:.2f}   (best over PR sweep)")


if __name__ == "__main__":
    main()
