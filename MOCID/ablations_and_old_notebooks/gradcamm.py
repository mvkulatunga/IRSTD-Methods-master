#!/usr/bin/env python3
"""Grad-CAM family on the MOCID detection head for a (seeded) validation clip.

GRADIENT-based saliency: backprops the top objectness logit to a head conv feature.
--method builtin  -> our transparent hand-rolled Grad-CAM++ (no deps).
--method {gradcam,gradcam++,xgradcam,eigencam,layercam} -> via `pip install grad-cam`.

Self-contained: only imports mocid.py + eval.py (+ optionally pytorch-grad-cam).

Usage:
    python gradcampp.py --ckpt runs/DAUB/dam.pth --seed 0
    python gradcampp.py --ckpt runs/DAUB/dam.pth --seed 46 --frames 30 --threed
    python gradcampp.py --ckpt runs/DAUB/dam.pth --seed 46 --method layercam
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as _cm

try:
    import imageio.v2 as imageio
except Exception:
    import imageio

from mocid import MOCID, MOCIDDataset, Config
from eval import collate_eval, decode_outputs, predict_image, _strip_compile

# optional library backends
_LIB = {}
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, LayerCAM

    _LIB = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "layercam": LayerCAM,
    }
except Exception:
    pass

METHOD_CHOICES = ["builtin"] + list(_LIB) if _LIB else ["builtin"]


def load_model(ckpt, cfg, dev):
    model = MOCID(num_classes=1, num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]).to(dev)
    ck = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(_strip_compile(ck.get("model", ck)), strict=False)
    model.eval()
    # Grad-CAM needs a live graph; run the eager module to avoid compile detaching grads
    return getattr(model, "_orig_mod", model)


def find_head(model):
    for m in model.modules():
        if type(m).__name__ == "YOLOXHead":
            return m
    raise RuntimeError("YOLOXHead not found")


def _dam_hooks(model):
    """Hook DisplacementNet; per scale -> displacement energy (B,H,W)."""
    store = []

    def hook(mod, inp, out):
        for scale in out:  # (B,T,C,H,W) per scale
            disp = scale[:, :-1]  # refs (drop target)
            e = torch.linalg.vector_norm(disp, ord=2, dim=2)
            store.append(e.mean(1).detach())  # (B,H,W)

    return [
        m.register_forward_hook(hook)
        for m in model.modules()
        if type(m).__name__ == "DisplacementNet"
    ], store


def _dam_map(t, size, gamma=0.5):
    s = (
        F.interpolate(t.unsqueeze(1), size=size, mode="bilinear", align_corners=False)[
            0, 0
        ]
        .cpu()
        .numpy()
    )
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    s = np.clip((s - lo) / (hi - lo + 1e-8), 0, 1) ** gamma
    return (s - s.min()) / (s.max() - s.min() + 1e-8)


def _dam_hooks(model):
    """Hook DisplacementNet; per scale -> displacement energy (B,H,W)."""
    store = []

    def hook(mod, inp, out):
        for scale in out:  # (B,T,C,H,W) per scale
            disp = scale[:, :-1]
            e = torch.linalg.vector_norm(disp, ord=2, dim=2)
            store.append(e.mean(1).detach())

    hs = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if type(m).__name__ == "DisplacementNet"
    ]
    return hs, store


def _energy_map(t, size, gamma=0.5):
    s = t.unsqueeze(1)
    s = F.interpolate(s, size=size, mode="bilinear", align_corners=False)[0, 0]
    s = s.cpu().numpy()
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    s = np.clip((s - lo) / (hi - lo + 1e-8), 0, 1) ** gamma
    return (s - s.min()) / (s.max() - s.min() + 1e-8)


# ---------------------------------------------------------------- built-in CAM
def gradcampp_builtin(model, clip, use_dam, scale=0):
    """Hand-rolled Grad-CAM++. Returns (cam HxW in [0,1], head_outputs)."""
    target = find_head(model).reg_convs[scale]
    act = {}

    def fwd_hook(m, i, o):
        o.retain_grad()
        act["A"] = o

    h = target.register_forward_hook(fwd_hook)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        outs = model(clip, use_dam=use_dam)
        score = outs[scale][:, 4].max()  # top objectness -> the detection
        score.backward()
    h.remove()

    A = act.get("A")
    if A is None or A.grad is None:
        raise RuntimeError("no gradient captured -- model likely ran compiled.")
    A = A.detach()[0]
    G = act["A"].grad.detach()[0]
    G2 = G * G
    G3 = G2 * G
    denom = 2 * G2 + (A * G3).sum(dim=(1, 2), keepdim=True)
    alpha = G2 / torch.clamp(denom, min=1e-8)
    weights = (alpha * F.relu(G)).sum(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * A).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), outs


# ---------------------------------------------------------------- library CAM
class _ScaleWrapper(nn.Module):
    """Expose MOCID as a classifier-like model: forward(clip) -> chosen scale output."""

    def __init__(self, model, use_dam, scale):
        super().__init__()
        self.model = model
        self.use_dam = use_dam
        self.scale = scale

    def forward(self, clip):
        return self.model(clip, use_dam=self.use_dam)[self.scale]


class _ObjTarget:
    """Scalar the CAM explains: top objectness (channel 4). Gets one sample output."""

    def __call__(self, out):
        return out[4].max()


def gradcam_lib(method, model, clip, use_dam, scale):
    wrapper = _ScaleWrapper(model, use_dam, scale).eval()
    target_layer = find_head(model).reg_convs[scale]
    cam = _LIB[method](model=wrapper, target_layers=[target_layer])
    # the library reads input_tensor.shape[-2:] for the resize target, but our clip is
    # 5D (B,T,3,H,W) -> it grabs (3,H,W). Force the correct (H,W) instead.
    H, W = clip.shape[-2], clip.shape[-1]
    cam.get_target_width_height = lambda _inp: (W, H)
    targets = None if method == "eigencam" else [_ObjTarget()]
    g = cam(input_tensor=clip, targets=targets)[0]  # (H,W) in [0,1]
    try:
        cam.activations_and_grads.release()
    except Exception:
        pass
    return g


def compute(model, clip, a):
    """Returns (cam HxW numpy, box, scores, dam_maps) for the chosen --method.
    dam_maps is a list of per-scale DAM overlays (empty if --dam off)."""
    W = clip.shape[-1]
    hs, store = _dam_hooks(model) if a.dam else ([], [])
    if a.method == "builtin":
        cam, outs = gradcampp_builtin(model, clip, a.dam, a.scale)
    else:
        cam = gradcam_lib(a.method, model, clip, a.dam, a.scale)
        with torch.no_grad():
            outs = model(clip, use_dam=a.dam)
    for h in hs:
        h.remove()
    dam_maps = [_dam_map(s, (W, W)) for s in store]
    with torch.no_grad():
        dec = decode_outputs([o.detach() for o in outs], [8, 16, 32])
        box, scores = predict_image(dec[0], 1, a.conf, a.nms)
    return cam, box, scores, dam_maps


# ------------------------------------------------------------------ window utils
def _consecutive_window(ds, start, n):
    def seq_of(i):
        return os.path.basename(os.path.dirname(ds.clips[i][-1]["path"]))

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


def _pick_window(ds, n, rng, tries=200):
    for _ in range(tries):
        w = _consecutive_window(ds, rng.randrange(len(ds)), n)
        if len(w) == n:
            return w
    return _consecutive_window(ds, rng.randrange(len(ds)), n)


# ------------------------------------------------------------------ plotting
def _draw_det(img, box, scores, gts):
    det = img.copy()
    for (x1, y1, x2, y2), sc in zip(box.cpu().numpy(), scores.cpu().numpy()):
        cv2.rectangle(det, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.putText(
            det,
            f"{sc:.2f}",
            (int(x1), max(0, int(y1) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
        )
    for x1, y1, x2, y2 in gts[0].numpy():
        cv2.rectangle(det, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
    return det


def _boxes_patches(ax, box, scores, gts):
    for (x1, y1, x2, y2), sc in zip(box.cpu().numpy(), scores.cpu().numpy()):
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, edgecolor="lime", facecolor="none", lw=1.2
            )
        )
    for x1, y1, x2, y2 in gts[0].numpy():
        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                edgecolor="deepskyblue",
                facecolor="none",
                lw=1.2,
            )
        )


def _cam_overlay2d(ax, img, cam_up, box, scores, gts, cmap, ag, clevel, title):
    alpha_map = np.clip(cam_up, 0, 1) ** ag
    ax.imshow(img, cmap="gray")
    ax.imshow(cam_up, cmap=cmap, vmin=0, vmax=1, alpha=alpha_map)
    if cam_up.max() > clevel:
        ax.contour(cam_up, levels=[clevel], colors="white", linewidths=0.8)
    _boxes_patches(ax, box, scores, gts)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _cam_surface(ax, img, cam_up, box, scores, gts, st, cmap, clevel, title):
    H, W = cam_up.shape
    yy, xx = np.mgrid[0:H:st, 0:W:st]
    z = cam_up[::st, ::st]
    g = img.mean(2) / 255.0 if img.ndim == 3 else img / 255.0
    floor = _cm.gray(g[::st, ::st])
    floor[..., 3] = 0.45
    ax.plot_surface(
        xx,
        yy,
        np.zeros_like(z),
        facecolors=floor,
        shade=False,
        linewidth=0,
        antialiased=False,
        zorder=0,
    )
    ax.plot_surface(
        xx,
        yy,
        z,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidth=0,
        antialiased=True,
        rcount=z.shape[0],
        ccount=z.shape[1],
        alpha=0.95,
    )
    if z.max() > clevel:
        ax.contour(
            xx,
            yy,
            z,
            levels=[clevel],
            colors="white",
            linewidths=1.0,
            offset=0,
            zdir="z",
        )

    def rect(x1, y1, x2, y2, c):
        ax.plot(
            [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], [0] * 5, color=c, linewidth=1.4
        )

    for (x1, y1, x2, y2), sc in zip(box.cpu().numpy(), scores.cpu().numpy()):
        rect(x1, y1, x2, y2, "lime")
    for x1, y1, x2, y2 in gts[0].numpy():
        rect(x1, y1, x2, y2, "deepskyblue")
    ax.set_zlim(0, 1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("CAM")
    ax.view_init(elev=40, azim=-60)
    ax.invert_yaxis()


def render(img, cam_up, box, scores, gts, title, a, dam_maps=None, save=None):
    """Detection | (3D surface + 2D overlay). Returns RGB buffer; saves if `save`."""
    det = _draw_det(img, box, scores, gts)
    if a.threed:
        fig = plt.figure(figsize=(18, 5.5))
        ax0 = fig.add_subplot(1, 3, 1)
        ax0.imshow(det)
        ax0.set_title(title, fontsize=11)
        ax0.axis("off")
        ax1 = fig.add_subplot(1, 3, 2, projection="3d")
        _cam_surface(
            ax1,
            img,
            cam_up,
            box,
            scores,
            gts,
            a.surf_stride,
            a.cmap,
            a.contour,
            f"{a.method} 3D + boxes",
        )
        ax2 = fig.add_subplot(1, 3, 3)
        _cam_overlay2d(
            ax2,
            img,
            cam_up,
            box,
            scores,
            gts,
            a.cmap,
            a.alpha_gamma,
            a.contour,
            "2D overlay + boxes",
        )
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.08)
    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(det)
        ax[0].set_title(title, fontsize=11)
        ax[0].axis("off")
        alpha_map = np.clip(cam_up, 0, 1) ** a.alpha_gamma
        ax[1].imshow(img, cmap="gray", alpha=0.35)
        ax[1].imshow(cam_up, cmap=a.cmap, vmin=0, vmax=1, alpha=alpha_map)
        if cam_up.max() > a.contour:
            ax[1].contour(cam_up, levels=[a.contour], colors="white", linewidths=0.8)
        _boxes_patches(ax[1], box, scores, gts)
        ax[1].set_title(f"{a.method} + boxes", fontsize=11)
        ax[1].axis("off")
        _cam_overlay2d(
            ax[2],
            img,
            cam_up,
            box,
            scores,
            gts,
            a.cmap,
            a.alpha_gamma,
            a.contour,
            "overlay + boxes",
        )
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05)

    if dam_maps and getattr(a, "dam_overlay", False):
        # extra row: DAM per-scale 2D overlays on the target frame
        n = len(dam_maps)
        f2, ax2 = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            ax2 = [ax2]
        for k, m in enumerate(dam_maps):
            am = np.clip(m, 0, 1) ** a.alpha_gamma
            ax2[k].imshow(img, cmap="gray")
            ax2[k].imshow(m, cmap=a.cmap, vmin=0, vmax=1, alpha=am)
            if m.max() > a.contour:
                ax2[k].contour(m, levels=[a.contour], colors="white", linewidths=0.8)
            _boxes_patches(ax2[k], box, scores, gts)
            ax2[k].set_title(f"DAM scale {k} overlay", fontsize=11)
            ax2[k].axis("off")
        f2.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05)
        if save:
            f2.savefig(
                save.replace(".png", "_dam.png").replace(".gif", "_dam.gif"), dpi=130
            )
        f2.canvas.draw()
        dam_buf = np.asarray(f2.canvas.buffer_rgba())[..., :3].copy()
        plt.close(f2)
    else:
        dam_buf = None

    if save:
        fig.savefig(save, dpi=130)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    if dam_buf is not None:
        # stack CAM row on top of DAM row into one image (match widths)
        w = max(buf.shape[1], dam_buf.shape[1])

        def pad(im):
            if im.shape[1] == w:
                return im
            out = np.full((im.shape[0], w, 3), 255, np.uint8)
            out[:, : im.shape[1]] = im
            return out

        buf = np.vstack([pad(buf), pad(dam_buf)])
    return buf


def render_dam(img, dam_maps, box, scores, gts, title, a, save=None):
    """Detection + each DAM scale overlaid in 2D. Returns RGB buffer; saves if `save`."""
    det = _draw_det(img, box, scores, gts)
    n = 1 + len(dam_maps)
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        ax = [ax]
    ax[0].imshow(det)
    ax[0].set_title(title, fontsize=11)
    ax[0].axis("off")
    for k, m in enumerate(dam_maps):
        _cam_overlay2d(
            ax[k + 1],
            img,
            m,
            box,
            scores,
            gts,
            a.cmap,
            a.alpha_gamma,
            a.contour,
            f"DAM scale {k} overlay",
        )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05)
    if save:
        fig.savefig(save, dpi=130)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--method",
        default="builtin",
        choices=METHOD_CHOICES,
        help="builtin = hand-rolled Grad-CAM++; others need `pip install grad-cam`",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=0, help="FPN scale (0=stride8 finest)")
    ap.add_argument(
        "--frames", type=int, default=1, help="1=image; >1=sliding-window GIF"
    )
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--alpha-gamma", type=float, default=0.7)
    ap.add_argument("--contour", type=float, default=0.5)
    ap.add_argument("--threed", action="store_true")
    ap.add_argument("--surf-stride", type=int, default=8)
    ap.add_argument(
        "--dam-overlay",
        action="store_true",
        help="also draw DAM per-scale 2D overlays on the frame",
    )
    ap.add_argument("--no-dam", dest="dam", action="store_false")
    ap.add_argument(
        "--dam-maps",
        action="store_true",
        help="instead of Grad-CAM, overlay per-scale DAM attention in 2D",
    )
    ap.add_argument("--out", default="gradcampp.png")
    ap.set_defaults(dam=True)
    a = ap.parse_args()

    if a.method != "builtin" and not _LIB:
        raise SystemExit("`pip install grad-cam` to use --method " + a.method)

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg = Config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, cfg, dev)

    ds = MOCIDDataset(cfg.val_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=False)
    W = cfg.IMG_SIZE[0]

    # ---- DAM attention overlay mode (per scale, 2D) ----
    if a.dam_maps:

        def dam_clip(idx):
            clip, gts = collate_eval([ds[idx]])
            clip = clip.to(dev)
            hs, store = _dam_hooks(model)
            with torch.no_grad():
                outs = model(clip, use_dam=True)
                dec = decode_outputs([o.detach() for o in outs], [8, 16, 32])
                box, scores = predict_image(dec[0], 1, a.conf, a.nms)
            for h in hs:
                h.remove()
            maps = [_energy_map(s, (W, W)) for s in store]
            img = (
                (clip[0, -1].permute(1, 2, 0).cpu().numpy() * 255)
                .astype(np.uint8)
                .copy()
            )
            return img, maps, box, scores, gts

        if a.frames > 1:
            window = _pick_window(ds, a.frames, rng)
            seq = os.path.basename(os.path.dirname(ds.clips[window[0]][-1]["path"]))
            print(f"seq {seq}: {len(window)} frames, DAM overlay")
            gif = []
            for fi, idx in enumerate(window):
                img, maps, box, scores, gts = dam_clip(idx)
                gif.append(
                    render_dam(
                        img,
                        maps,
                        box,
                        scores,
                        gts,
                        f"seq {seq} frame {fi+1}/{len(window)}  ({len(box)} det)",
                        a,
                    )
                )
                print(f"  frame {fi+1}/{len(window)}: {len(box)} det")
            out = a.out if a.out.endswith(".gif") else "dam_overlay.gif"
            imageio.mimsave(out, gif, duration=1.0 / a.fps, loop=0)
            print(f"saved -> {out}")
            return
        idx = rng.randrange(len(ds))
        img, maps, box, scores, gts = dam_clip(idx)
        render_dam(
            img,
            maps,
            box,
            scores,
            gts,
            f"clip {idx}: pred(green)/GT(blue)",
            a,
            save=a.out,
        )
        print(f"saved -> {a.out}  ({len(box)} det, {len(maps)} DAM scales)")
        return

    if a.frames > 1:
        window = _pick_window(ds, a.frames, rng)
        seq = os.path.basename(os.path.dirname(ds.clips[window[0]][-1]["path"]))
        print(f"seq {seq}: {len(window)} frames, method={a.method}")
        gif = []
        for fi, idx in enumerate(window):
            clip, gts = collate_eval([ds[idx]])
            clip = clip.to(dev)
            cam, box, scores, dam_maps = compute(model, clip, a)
            cam_up = cv2.resize(cam, (W, W))
            img = (
                (clip[0, -1].permute(1, 2, 0).cpu().numpy() * 255)
                .astype(np.uint8)
                .copy()
            )
            title = f"seq {seq} frame {fi+1}/{len(window)}  ({len(box)} det)"
            gif.append(
                render(img, cam_up, box, scores, gts, title, a, dam_maps=dam_maps)
            )
            print(f"  frame {fi+1}/{len(window)}: {len(box)} det")
        out = a.out if a.out.endswith(".gif") else "gradcampp.gif"
        imageio.mimsave(out, gif, duration=1.0 / a.fps, loop=0)
        print(f"saved -> {out}  ({len(gif)} frames @ {a.fps} fps)")
        return

    idx = rng.randrange(len(ds))
    clip, gts = collate_eval([ds[idx]])
    clip = clip.to(dev)
    print(f"val clip idx {idx}/{len(ds)} (seed {a.seed}, method={a.method})")
    cam, box, scores, dam_maps = compute(model, clip, a)
    cam_up = cv2.resize(cam, (W, W))
    img = (clip[0, -1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
    render(
        img,
        cam_up,
        box,
        scores,
        gts,
        f"clip {idx}: pred(green)/GT(blue)",
        a,
        dam_maps=dam_maps,
        save=a.out,
    )
    print(f"saved -> {a.out}  ({len(box)} det)")


if __name__ == "__main__":
    main()
