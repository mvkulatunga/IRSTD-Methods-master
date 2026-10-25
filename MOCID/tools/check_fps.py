"""Inference speed (FPS) of MOCID's DAM variants on one GPU.

    python tools/check_fps.py              # default variants, batch 1
    python tools/check_fps.py --batch 4    # throughput at the eval batch size

One "frame" is one detection on the target frame of a 5-frame clip (512x512), so FPS here is
clips per second. Random input, eval mode, fp32 (as in utils/eval.py), no gradients. Times the
model's forward pass only: data loading, box decoding and NMS are excluded, as is usual when
papers report FPS. Weights don't affect speed, so no checkpoint is needed.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # the MOCID folder

import torch

from model import MOCID

# (label, scan, d_state): the configurations trained so far
VARIANTS = [
    ("R0-DAM-fixed (raster, N=32)", "bidir", 32),
    ("raster, N=1", "bidir", 1),
    ("tree, N=1", "tree", 1),
    ("tree, N=8", "tree", 8),
    ("tree, N=32", "tree", 32),
]


def measure(model, clip, n_warm, n_iter):
    """-> (clips per second, ms per forward) for model on clip."""
    with torch.no_grad():
        for _ in range(n_warm):  # cuDNN autotuning, caches, clocks
            model(clip)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n_iter):
            model(clip)
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / n_iter
    return clip.shape[0] / dt, dt * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    a = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")

    torch.backends.cudnn.benchmark = True
    clip = torch.randn(a.batch, a.frames, 3, a.size, a.size, device="cuda")
    print(f"{torch.cuda.get_device_name()}, batch {a.batch}, {a.frames} frames of {a.size}x{a.size}\n")
    print(f"{'variant':30s} {'FPS':>8s} {'ms/clip':>9s} {'params (M)':>11s}")

    for label, scan, n in VARIANTS:
        model = MOCID(num_frames=a.frames, img_size=a.size, d_state=n, scan=scan).cuda().eval()
        fps, ms = measure(model, clip, a.warm, a.iters)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{label:30s} {fps:8.1f} {ms / a.batch:9.2f} {params:11.3f}")

        # stage 1 alone (no DAM), once, as the reference point
        if label.startswith("R0"):
            model.forward = lambda clip, _f=model.forward: _f(clip, use_dam=False)
            fps, ms = measure(model, clip, a.warm, a.iters)
            print(f"{'stage 1 only (no DAM)':30s} {fps:8.1f} {ms / a.batch:9.2f} {'':>11s}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
