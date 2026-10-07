"""Animate a DAUB video, optionally with a model detecting targets frame by frame.

    # the video with its ground-truth boxes, no model
    python tools/animate.py --video data6 --out data6.mp4

    # with detections (use the --profile and --model the checkpoint was trained with)
    python tools/animate.py --video data6 --out data6_det.mp4 --profile r0 \
        --ckpt /srv/proj-mamba/mocid-baseline/checkpoints/r0_fista_best.pth

Each shown frame is the last frame of a T-frame clip, as in evaluation, so the first T-1
frames of the video are skipped and the boxes are exactly what evaluation scores (before
the confidence threshold). Ground truth is blue, detections green with their score, and an
enlarged inset follows the target. With a checkpoint, each frame shows the model's time
(clip to GPU, forward, decode, NMS; not disk reads), and the run ends with the mean time
per frame, so "real time" can be checked against --fps. Writes .mp4 (OpenCV) or .gif.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # the MOCID folder

import cv2
import numpy as np
import torch

from config import Config, get_device, setup_torch
from main import MODELS, build_model, load_weights
from utils.data import MOCIDDataset
from utils.eval import decode_outputs, iou_matrix, predict_image

GT_BGR, DET_BGR, TEXT_BGR = (255, 160, 0), (0, 220, 0), (255, 255, 255)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True, help="video folder name, e.g. data6")
    ap.add_argument("--out", required=True, help="output file, .mp4 or .gif")
    ap.add_argument("--split", choices=("val", "train"), default="val",
                    help="split file to read the video from (default: val, never trained on)")
    ap.add_argument("--profile", default=None, help="settings profile the checkpoint was trained with")
    ap.add_argument("--model", choices=sorted(MODELS), default="MOCID")
    ap.add_argument("--ckpt", default=None, help="checkpoint; without it, no detection")
    ap.add_argument("--dam", dest="dam", action="store_true")
    ap.add_argument("--no-dam", dest="dam", action="store_false")
    ap.set_defaults(dam=None)  # default: on for a stage-2 MOCID checkpoint, as main.py eval
    ap.add_argument("--conf", type=float, default=0.3, help="score needed to draw a detection")
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--fps", type=float, default=10, help="playback speed of the output")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    ap.add_argument("--no-gt", action="store_true", help="don't draw ground truth")
    ap.add_argument("--no-zoom", action="store_true", help="no enlarged inset")
    return ap.parse_args()


def video_clips(ds, video):
    """-> indices of ds's clips whose target frame is in video, in frame order."""
    seq = lambda i: os.path.basename(os.path.dirname(ds.clips[i][-1]["path"]))
    idx = [i for i in range(len(ds)) if seq(i) == video]
    if not idx:
        found = sorted({seq(i) for i in range(len(ds))}, key=lambda s: (len(s), s))
        raise SystemExit(f"no clips for {video!r}; this split has: {', '.join(found)}")
    return sorted(idx, key=lambda i: ds.clips[i][-1]["frame_num"])


@torch.no_grad()
def detect(model, clip, device, use_dam, cfg, conf, nms):
    """(T,3,H,W) clip -> ((M,4) xyxy, (M,) scores, model time in ms)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(clip[None].to(device), use_dam=use_dam)
    box, score = predict_image(decode_outputs(out, cfg.STRIDES)[0], cfg.NUM_CLASSES, conf, nms)
    box, score = box.cpu().numpy(), score.cpu().numpy()  # waits for the GPU
    return box, score, (time.perf_counter() - t0) * 1000


def draw_box(img, b, color, label=None):
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def add_inset(img, centre, crop=48, scale=4):
    """Paste an enlarged crop around centre into the top-right corner. -> None."""
    H, W = img.shape[:2]
    cx = int(np.clip(centre[0], crop // 2, W - crop // 2))
    cy = int(np.clip(centre[1], crop // 2, H - crop // 2))
    patch = img[cy - crop // 2 : cy + crop // 2, cx - crop // 2 : cx + crop // 2]
    big = cv2.resize(patch, (crop * scale, crop * scale), interpolation=cv2.INTER_NEAREST)
    s, pad = crop * scale, 8
    img[pad : pad + s, W - pad - s : W - pad] = big
    cv2.rectangle(img, (W - pad - s - 1, pad - 1), (W - pad, pad + s), TEXT_BGR, 1)
    cv2.rectangle(img, (cx - crop // 2, cy - crop // 2), (cx + crop // 2, cy + crop // 2), TEXT_BGR, 1)


def put_lines(img, lines):
    for k, line in enumerate(lines):
        y = img.shape[0] - 10 - 18 * (len(lines) - 1 - k)
        # black outline from shifted thin copies: a thicker stroke would also widen the
        # letter spacing, so the outline would drift away from the white text
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.putText(img, line, (8 + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_BGR, 1, cv2.LINE_AA)


class Writer:
    """Frames (BGR uint8) -> .mp4 via OpenCV, or .gif via Pillow."""

    def __init__(self, path, fps, size):
        self.path, self.fps, self.gif = path, fps, path.lower().endswith(".gif")
        if self.gif:
            self.frames = []
            return
        for codec in ("avc1", "mp4v"):  # H.264 plays everywhere; not every OpenCV build has it
            self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
            if self.vw.isOpened():
                self.codec = codec
                return
        raise SystemExit(f"OpenCV could not open a video writer for {path}")

    def add(self, frame):
        if self.gif:
            from PIL import Image
            self.frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            self.vw.write(frame)

    def close(self):
        if self.gif:
            self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:],
                                duration=int(1000 / self.fps), loop=0)
        else:
            self.vw.release()


def main():
    a = parse_args()
    setup_torch()
    cfg = Config(a.profile)
    device = get_device()

    ds = MOCIDDataset(cfg.val_path if a.split == "val" else cfg.train_path, T=cfg.T,
                      img_size=cfg.IMG_SIZE, is_train=False, norm=cfg.NORMALISE)
    idx = video_clips(ds, a.video)[: a.max_frames]

    model, use_dam, name = None, False, "no model"
    if a.ckpt:
        model = build_model(cfg, device, a.model)
        stage = load_weights(model, a.ckpt)
        model.eval()
        use_dam = (a.dam if a.dam is not None else stage == 2) and a.model == "MOCID"
        # run folder + file: every run's best checkpoint is called dam_best.pth / best.pth
        run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
        name = f"{a.model}{' +DAM' if use_dam else ''} ({run}/{os.path.basename(a.ckpt)})"
        for _ in range(3):  # warm-up: first calls include one-off CUDA setup
            detect(model, ds[idx[0]][0], device, use_dam, cfg, a.conf, a.nms)
    print(f"{a.video}: {len(idx)} frames, {name}, profile {cfg.PROFILE}")

    writer = Writer(a.out, a.fps, cfg.IMG_SIZE)
    times, found, with_gt = [], 0, 0
    for n, i in enumerate(idx):
        clip, target = ds[i]
        img = cv2.resize(cv2.imread(target["path"], cv2.IMREAD_COLOR), cfg.IMG_SIZE)
        gt = target["boxes"].numpy()
        lines = [f"{a.video}  frame {ds.clips[i][-1]['frame_num']}  ({n + 1}/{len(idx)})", name]

        if not a.no_gt:
            for b in gt:
                draw_box(img, b + np.array([-2, -2, 2, 2]), GT_BGR)  # 2 px out: stays visible under a matching detection
        box, score = np.zeros((0, 4)), np.zeros(0)
        if model is not None:
            box, score, ms = detect(model, clip, device, use_dam, cfg, a.conf, a.nms)
            times.append(ms)
            for b, s in zip(box, score):
                draw_box(img, b, DET_BGR, f"{s:.2f}")
            if len(gt):
                with_gt += 1
                found += int((iou_matrix(gt, box) >= 0.5).any(axis=1).all())
            lines.append(f"model {ms:.1f} ms/frame ({1000 / ms:.0f} fps)")
            lines.append(f"green: detection (score >= {a.conf:g})" + ("" if a.no_gt else "   blue: ground truth"))

        if not a.no_zoom:
            centre = gt[0] if len(gt) else (box[0] if len(box) else None)
            if centre is not None:
                add_inset(img, ((centre[0] + centre[2]) / 2, (centre[1] + centre[3]) / 2))
        put_lines(img, lines)
        writer.add(img)

    writer.close()
    codec = "" if writer.gif else f", codec {writer.codec}"
    print(f"wrote {a.out}: {len(idx)} frames at {a.fps:g} fps ({len(idx) / a.fps:.1f} s){codec}")
    if times:
        t = np.array(times)
        print(f"model time per frame: mean {t.mean():.1f} ms, median {np.median(t):.1f} ms "
              f"-> {1000 / t.mean():.0f} fps on {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}")
        if with_gt:
            print(f"target found (IoU >= 0.5, score >= {a.conf}) in {found} of {with_gt} frames "
                  f"({100 * found / with_gt:.1f}%)")


if __name__ == "__main__":
    main()
