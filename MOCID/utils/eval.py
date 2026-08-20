import numpy as np
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def decode_outputs(outputs, strides):
    """list of (B,C,H,W) raw grids -> (B,N,C) with (cx,cy,w,h,obj,cls...) in pixels."""
    decoded = []
    for out, stride in zip(outputs, strides):
        B, C, H, W = out.shape
        out = out.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # add each cell's position to the predicted offset, then scale to pixels
        yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid = torch.stack((xv, yv), 2).reshape(1, -1, 2).to(out.device).type_as(out)
        xy = (out[..., :2] + grid) * stride

        # clamp before exp so a bad logit can't produce inf/NaN
        wh = torch.exp(torch.clamp(out[..., 2:4], max=20.0)) * stride

        rest = out[..., 4:]  # logits, sigmoided later in predict_image
        decoded.append(torch.cat([xy, wh, rest], dim=-1))
    return torch.cat(decoded, dim=1)


def cxcywh_to_xyxy(b):
    """(N,4) cxcywh -> (N,4) xyxy."""
    x1 = b[:, 0] - b[:, 2] / 2
    y1 = b[:, 1] - b[:, 3] / 2
    x2 = b[:, 0] + b[:, 2] / 2
    y2 = b[:, 1] + b[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def nms_xyxy(boxes, scores, iou_thr):
    """(N,4) xyxy + (N,) scores -> kept indices, highest score first."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]  # highest remaining score always survives
        keep.append(i.item())
        if order.numel() == 1:
            break

        # drop everything overlapping the kept box above the threshold
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest])
        yy2 = torch.min(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


@torch.no_grad()
def predict_image(decoded_i, num_classes, conf_thr, nms_thr):
    """(N, 5+num_classes) decoded preds -> ((M,4) xyxy boxes, (M,) scores) post-NMS."""
    box = cxcywh_to_xyxy(decoded_i[:, :4])

    # score = objectness * best class confidence
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
    """(Na,4) and (Nb,4) xyxy numpy arrays -> (Na,Nb) IoU."""
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
    """recall / precision curves -> float AP (VOC post-2010 all-points)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # make precision monotonically decreasing, then sum the recall-weighted steps
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap50_f1(all_dets, gt_by_img, iou_thr=0.5):
    """all_dets [(score, img_idx, box)] + per-image GT -> (AP, best F1)."""
    npos = sum(len(g) for g in gt_by_img)
    if npos == 0 or len(all_dets) == 0:
        return 0.0, 0.0

    all_dets.sort(key=lambda d: d[0], reverse=True)
    nd = len(all_dets)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    matched = {i: np.zeros(len(g), dtype=bool) for i, g in enumerate(gt_by_img)}

    # greedy matching in score order: each GT can only be claimed once
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
    """Run the model over loader and score it. -> (ap50, best_f1)."""
    model.eval()
    all_dets, gt_by_img = [], []
    img_idx = 0

    for clips, gts in tqdm(loader, desc=f"eval(use_dam={use_dam})", mininterval=10):
        clips = clips.to(device)
        decoded = decode_outputs(model(clips, use_dam=use_dam), strides)

        # post-process per image, pooling detections and GT into flat lists
        for b in range(decoded.shape[0]):
            box, scores = predict_image(decoded[b], num_classes, conf_thr, nms_thr)
            box = box.cpu().numpy()
            scores = scores.cpu().numpy()
            for s, bx in zip(scores, box):
                all_dets.append((float(s), img_idx, bx))
            gt_by_img.append(gts[b].cpu().numpy())
            img_idx += 1

    return compute_ap50_f1(all_dets, gt_by_img)
