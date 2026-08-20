import os
import argparse
import numpy as np
import torch
import torchvision  # <-- Added for fast NMS
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def collate_eval(batch):
    clips = torch.stack([b[0] for b in batch])  # (B, T, C, H, W)
    gts = [b[1]["boxes"].clone() for b in batch]  # each (n_gt, 4) xyxy in 512-space
    return clips, gts


def cxcywh_to_xyxy(b):
    x1 = b[:, 0] - b[:, 2] / 2
    y1 = b[:, 1] - b[:, 3] / 2
    x2 = b[:, 0] + b[:, 2] / 2
    y2 = b[:, 1] + b[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


@torch.no_grad()
def predict_image(decoded_i, conf_thr, nms_thr):
    box = cxcywh_to_xyxy(decoded_i[:, :4])
    scores = decoded_i[:, 4]

    m = scores >= conf_thr
    box, scores = box[m], scores[m]
    if box.numel() == 0:
        return box, scores

    # --- FIX: Use optimized C++ NMS instead of python loop ---
    keep = torchvision.ops.nms(box, scores, nms_thr)
    return box[keep], scores[keep]


def iou_matrix(a, b):
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
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap50_f1(all_dets, gt_by_img, iou_thr=0.5):
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
def evaluate(model, loader, device, conf_thr=1e-3, nms_thr=0.65):
    model.eval()
    all_dets, gt_by_img = [], []
    img_idx = 0

    for clips, gts in tqdm(loader, desc="evaluating", mininterval=10):
        clips = clips.to(device)

        # --- FIX: Safely infer using Mixed Precision ---
        with torch.amp.autocast("cuda"):
            outs = model(clips)

        if isinstance(outs, tuple):
            preds = outs[0]
        else:
            preds = outs

        preds = preds.transpose(1, 2)  # (B, num_anchors, 5)

        for b in range(preds.shape[0]):
            box, scores = predict_image(preds[b], conf_thr, nms_thr)
            box = box.cpu().numpy()
            scores = scores.cpu().numpy()

            for s, bx in zip(scores, box):
                all_dets.append((float(s), img_idx, bx))
            gt_by_img.append(gts[b].cpu().numpy())
            img_idx += 1

    return compute_ap50_f1(all_dets, gt_by_img)


def count_params_m(model):
    n = sum(p.numel() for p in model.parameters())
    return n / 1e6


def _strip_compile(sd):
    return {
        k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v
        for k, v in sd.items()
    }


def main():
    from mocid_r_elan import MOCID, MOCIDDataset, Config

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/batch_8_r_elan/ckpt_last.pth")
    ap.add_argument("--params-only", action="store_true")
    ap.add_argument("--conf", type=float, default=1e-3)
    ap.add_argument("--nms", type=float, default=0.65)
    args = ap.parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MOCID(num_classes=1, num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]).to(device)

    # --- FIX: Ensure strides are safely on the GPU for Detect anchor generation ---
    model.stride = model.stride.to(device)
    model.head.stride = model.head.stride.to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(_strip_compile(ckpt["model"]))
    stage = ckpt.get("stage", 1)
    print(f"loaded {args.ckpt} (stage={stage})")

    print(f"Total Params : {count_params_m(model):.2f} M")
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

    ap50, f1 = evaluate(model, loader, device, conf_thr=args.conf, nms_thr=args.nms)

    print(f"\n=== Evaluation Results ===")
    print(f"AP50 : {ap50 * 100:.2f}")
    print(f"F1   : {f1 * 100:.2f}   (best over PR sweep)")


if __name__ == "__main__":
    main()
