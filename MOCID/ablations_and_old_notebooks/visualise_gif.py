#!/usr/bin/env python3
"""Sliding-window GIF over 10 consecutive target frames of one (seeded) val sequence.

Each GIF frame shows 7 panels:
  - target frame with predicted (green) + GT (blue) boxes
  - 3 FISTA layer-output saliency surfaces (||f_hat||_2)
  - 3 DAM displacement-energy saliency surfaces

Standalone: imports model/dataset from mocid.py, decode helpers from eval.py.

Usage:
    python saliency_gif.py --ckpt runs/DAUB/dam.pth --seed 0 --frames 10
    python saliency_gif.py --ckpt runs/DAUB/dam.pth --seed 7 --surf-stride 12 --fps 3
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d)
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from mocid import MOCID, MOCIDDataset, Config
from eval import collate_eval, decode_outputs, predict_image, _strip_compile

try:
    import imageio.v2 as imageio
except Exception:
    import imageio


# ---------------------------------------------------------------- model + hooks
def load_model(ckpt_path, cfg, device):
    model = MOCID(num_classes=1, num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]).to(device)
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    model.load_state_dict(_strip_compile(sd), strict=False)
    model.eval()
    print(f"loaded {ckpt_path} (stage={ck.get('stage','?')})")
    return model


def fista_hooks(model):
    store = []

    def hook(mod, inp, out):
        f_s = inp[0].float()
        Ft = torch.fft.fft(f_s, dim=1, norm="ortho")
        Kt = torch.complex(mod.weight_real.float(), mod.weight_imag.float())
        f_hat = torch.fft.ifft(Ft * Kt, dim=1, norm="ortho").real
        store.append(
            torch.linalg.vector_norm(f_hat, ord=2, dim=1).detach()
        )  # (B,C,H,W)

    hs = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if type(m).__name__ == "TemporalFISTA"
    ]
    return hs, store


def dam_hooks(model):
    store = []

    def hook(mod, inp, out):
        for scale in out:  # (B,T,C,H,W) per scale
            disp = scale[:, :-1]  # refs only
            e = torch.linalg.vector_norm(disp, ord=2, dim=2)  # (B,T-1,H,W)
            store.append(e.mean(1).detach())  # (B,H,W)

    hs = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if type(m).__name__ == "DisplacementNet"
    ]
    return hs, store


def to_map(t, size, gamma=0.5):
    """(B,C,H,W) or (B,H,W) -> contrast-stretched (H,W) numpy in [0,1]."""
    s = t.mean(1, keepdim=True) if t.dim() == 4 else t.unsqueeze(1)
    s = F.interpolate(s, size=size, mode="bilinear", align_corners=False)[0, 0]
    s = s.cpu().numpy()
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    s = np.clip((s - lo) / (hi - lo + 1e-8), 0, 1) ** gamma
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)  # fill 0..1 so surface has height
    return s


# ------------------------------------------------- pick consecutive clip window
def consecutive_window(ds, start, n):
    """Return up to n clip indices that are contiguous frames of one sequence."""

    def seq_of(i):
        p = ds.clips[i][-1]["path"]
        return os.path.basename(os.path.dirname(p))

    def fnum(i):
        return ds.clips[i][-1]["frame_num"]

    idxs = [start]
    for j in range(start + 1, len(ds)):
        if len(idxs) >= n:
            break
        if seq_of(j) == seq_of(idxs[-1]) and fnum(j) == fnum(idxs[-1]) + 1:
            idxs.append(j)
        else:
            break
    return idxs


def pick_window(ds, n, rng, tries=200):
    for _ in range(tries):
        start = rng.randrange(len(ds))
        w = consecutive_window(ds, start, n)
        if len(w) == n:
            return w
    # fallback: longest run from a random start
    return consecutive_window(ds, rng.randrange(len(ds)), n)


# ------------------------------------------------------------- render one frame
def render_frame(det_img, fista_maps, dam_maps, title, st, xx, yy):
    ncol = 4
    fig = plt.figure(figsize=(5 * ncol, 10))

    ax0 = fig.add_subplot(2, ncol, 1)
    ax0.imshow(det_img)
    ax0.set_title(title)
    ax0.axis("off")

    def surf(pos, m, name):
        ax = fig.add_subplot(2, ncol, pos, projection="3d")
        z = m[::st, ::st]
        ax.plot_surface(
            xx,
            yy,
            z,
            cmap="jet",
            linewidth=0,
            antialiased=True,
            rcount=z.shape[0],
            ccount=z.shape[1],
        )
        ax.set_zlim(0, 1)  # fixed -> comparable across frames
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.view_init(elev=45, azim=-60)
        ax.invert_yaxis()

    for k, m in enumerate(fista_maps[:3]):
        surf(2 + k, m, f"FISTA scale {k}")
    for k, m in enumerate(dam_maps[:3]):
        surf(ncol + 2 + k, m, f"DAM scale {k}")

    sm = cm.ScalarMappable(cmap="jet", norm=mcolors.Normalize(0, 1))
    fig.colorbar(sm, ax=fig.axes[1:], shrink=0.5, pad=0.02, label="saliency (norm)")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def fourier_hooks(model):
    """Hook SpatialFISTA (2D spatial spectrum) and TemporalFISTA (temporal spectrum).
    Returns (handles, spatial_store, temporal_store).
      spatial_store: list of (H, W//2+1) log-magnitude arrays, mean over channels.
      temporal_store: list of (T,) amplitude arrays, mean over channels+space.
    """
    sp, tp = [], []

    def sp_hook(mod, inp, out):
        f = inp[0].float()  # (B,C,H,W)
        F_s = torch.fft.rfft2(f, dim=(-2, -1), norm="ortho")
        mag = F_s.abs().mean(1)[0]  # (H, W//2+1)
        sp.append(torch.log1p(mag).detach().cpu().numpy())

    def tp_hook(mod, inp, out):
        f_s = inp[0].float()  # (B,T,C,H,W)
        F_t = torch.fft.fft(f_s, dim=1, norm="ortho")
        amp = F_t.abs().mean(dim=(2, 3, 4))[0]  # (T,) amplitude per temporal bin
        tp.append(amp.detach().cpu().numpy())

    hs = []
    for m in model.modules():
        n = type(m).__name__
        if n == "SpatialFISTA":
            hs.append(m.register_forward_hook(sp_hook))
        elif n == "TemporalFISTA":
            hs.append(m.register_forward_hook(tp_hook))
    return hs, sp, tp


def _reduce_to(lst, k):
    """Pick k evenly-spaced entries (last of each group) to match FPN scales."""
    if k <= 0 or len(lst) == 0:
        return lst
    if len(lst) >= k:
        step = len(lst) // k
        return [lst[(i + 1) * step - 1] for i in range(k)]
    return lst


def render_fourier(det_img, spatial, temporal, title):
    """det | per-scale [spatial spectrum 2D | temporal amplitude 1D]."""
    n = len(spatial)
    fig = plt.figure(figsize=(11, 2.6 * (n + 1)))
    gs = fig.add_gridspec(n + 1, 2)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(det_img)
    ax0.set_title(title, fontsize=11)
    ax0.axis("off")

    for k in range(n):
        axs = fig.add_subplot(gs[k + 1, 0])
        spec = np.fft.fftshift(spatial[k], axes=0)  # center DC vertically
        axs.imshow(spec, cmap="magma", aspect="auto")
        axs.set_title(f"scale {k}: spatial DFT |F_s| (log)", fontsize=10)
        axs.set_xlabel("freq x")
        axs.set_ylabel("freq y")

        axt = fig.add_subplot(gs[k + 1, 1])
        amp = temporal[k]
        axt.stem(np.arange(len(amp)), amp)
        axt.set_title(f"scale {k}: temporal DFT |F_t|", fontsize=10)
        axt.set_xlabel("temporal freq bin")
        axt.set_ylabel("amplitude")
        axt.set_xticks(np.arange(len(amp)))

    fig.subplots_adjust(
        left=0.08, right=0.97, top=0.94, bottom=0.05, hspace=0.55, wspace=0.3
    )
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--surf-stride", type=int, default=10)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--no-dam", dest="dam", action="store_false")
    ap.add_argument(
        "--fourier",
        action="store_true",
        help="graph the spatial + temporal DFTs per scale (instead of 3D saliency)",
    )
    ap.add_argument("--out", default="saliency.gif")
    ap.set_defaults(dam=True)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    cfg = Config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, cfg, dev)

    ds = MOCIDDataset(cfg.val_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=False)
    window = pick_window(ds, a.frames, rng)
    seq = os.path.basename(os.path.dirname(ds.clips[window[0]][-1]["path"]))
    print(f"seq {seq}: {len(window)} consecutive frames starting at clip {window[0]}")

    H = W = cfg.IMG_SIZE[0]
    st = max(1, a.surf_stride)
    yy, xx = np.mgrid[0:H:st, 0:W:st]

    gif_frames = []

    if a.fourier:
        n_scales = 3
        for fi, idx in enumerate(window):
            clip, gts = collate_eval([ds[idx]])
            clip = clip.to(dev)
            hs, sp, tp = fourier_hooks(model)
            with torch.no_grad():
                outs = model(clip, use_dam=a.dam)
                dec = decode_outputs(outs, [8, 16, 32])
                box, scores = predict_image(dec[0], 1, a.conf, a.nms)
            for h in hs:
                h.remove()
            sp = _reduce_to(sp, n_scales)
            tp = _reduce_to(tp, n_scales)
            img = (
                (clip[0, -1].permute(1, 2, 0).cpu().numpy() * 255)
                .astype(np.uint8)
                .copy()
            )
            for (x1, y1, x2, y2), sc in zip(box.cpu().numpy(), scores.cpu().numpy()):
                cv2.rectangle(
                    img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1
                )
            for x1, y1, x2, y2 in gts[0].numpy():
                cv2.rectangle(
                    img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1
                )
            title = f"seq {seq} frame {fi+1}/{len(window)}  ({len(box)} det)"
            gif_frames.append(render_fourier(img, sp, tp, title))
            print(f"  frame {fi+1}/{len(window)}: spatial+temporal DFT")
        out = a.out if a.out.endswith(".gif") else "fourier.gif"
        imageio.mimsave(out, gif_frames, duration=1.0 / a.fps, loop=0)
        print(f"saved -> {out}  ({len(gif_frames)} frames)")
        return

    for fi, idx in enumerate(window):
        clip, gts = collate_eval([ds[idx]])
        clip = clip.to(dev)

        fh, fstore = fista_hooks(model)
        dh, dstore = dam_hooks(model)
        with torch.no_grad():
            outs = model(clip, use_dam=a.dam)
            dec = decode_outputs(outs, [8, 16, 32])
            box, scores = predict_image(dec[0], 1, a.conf, a.nms)
        for h in fh + dh:
            h.remove()

        # match the visualize_saliency.py reduction: build all FISTA + DAM maps,
        # then pick FISTA layer outputs by pairing against the DAM scale count
        # (last block of each equal-sized group).
        fista_all = [to_map(s, (H, W)) for s in fstore]
        dam_maps = [to_map(s, (H, W)) for s in dstore]
        if len(fista_all) >= 3 * len(dam_maps) and len(dam_maps) > 0:
            step = len(fista_all) // len(dam_maps)
            fista_maps = [fista_all[(i + 1) * step - 1] for i in range(len(dam_maps))]
        else:
            fista_maps = fista_all[: len(dam_maps)] if dam_maps else fista_all

        img = (clip[0, -1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
        for (x1, y1, x2, y2), sc in zip(box.cpu().numpy(), scores.cpu().numpy()):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(
                img,
                f"{sc:.2f}",
                (int(x1), max(0, int(y1) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
            )
        for x1, y1, x2, y2 in gts[0].numpy():
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

        title = f"seq {seq} frame {fi+1}/{len(window)}  ({len(box)} det)"
        gif_frames.append(render_frame(img, fista_maps, dam_maps, title, st, xx, yy))
        print(f"  frame {fi+1}/{len(window)}: {len(box)} detections")

    imageio.mimsave(a.out, gif_frames, duration=1.0 / a.fps, loop=0)
    print(f"saved -> {a.out}  ({len(gif_frames)} frames @ {a.fps} fps)")


if __name__ == "__main__":
    main()
