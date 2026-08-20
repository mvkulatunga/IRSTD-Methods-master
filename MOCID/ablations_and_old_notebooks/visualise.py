#!/usr/bin/env python3
"""Visualize one (seeded-random) validation clip end-to-end:
  - predicted + GT boxes drawn on the target frame
  - FISTA high-frequency saliency (||f_hat||_2) at EVERY scale

Standalone: imports the model + dataset from mocid.py, loads a checkpoint.

Usage:
    python visualize_saliency.py --ckpt runs/DAUB/dam.pth --seed 0
    python visualize_saliency.py --ckpt runs/DAUB/fista.pth --no-dam --seed 42
"""

import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d)

from mocid import MOCID, MOCIDDataset, Config
from eval import collate_eval, decode_outputs, predict_image, _strip_compile


def load_model(ckpt_path, cfg, device):
    model = MOCID(num_classes=1, num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]).to(device)
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    missing = model.load_state_dict(_strip_compile(sd), strict=False)
    print(f"loaded {ckpt_path}  (stage={ck.get('stage','?')})")
    if missing.missing_keys:
        print(
            f"  [warn] {len(missing.missing_keys)} missing keys (e.g. {missing.missing_keys[:2]})"
        )
    model.eval()
    return model


def fista_saliency_hooks(model):
    """Hook every TemporalFISTA; recompute ||f_hat||_2 from its inputs.
    Returns (handles, store) where store fills with (B,C,H,W) tensors per scale."""
    store = []

    def hook(mod, inp, out):
        f_s = inp[0].float()  # (B,T,C,H,W)
        Ft = torch.fft.fft(f_s, dim=1, norm="ortho")
        Kt = torch.complex(mod.weight_real.float(), mod.weight_imag.float())
        f_hat = torch.fft.ifft(Ft * Kt, dim=1, norm="ortho").real
        norm = torch.linalg.vector_norm(f_hat, ord=2, dim=1)  # (B,C,H,W)  norm over T
        store.append(norm.detach())

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if type(m).__name__ == "TemporalFISTA"
    ]
    return handles, store


def dam_saliency_hook(model):
    """Hook DisplacementNet output; energy of the displacement feats (the T-1
    non-target slices), ||.||_2 over channels, meaned over reference frames."""
    store = []

    def hook(mod, inp, out):
        # out: list of (B, T, C, H, W) per scale; target is the last T-slice
        for scale in out:
            disp = scale[:, :-1]  # (B, T-1, C, H, W) refs
            e = torch.linalg.vector_norm(disp, ord=2, dim=2)  # (B, T-1, H, W) over C
            store.append(e.mean(1).detach())  # (B, H, W) mean over refs

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if type(m).__name__ == "DisplacementNet"
    ]
    return handles, store


def to_map(norm_bchw, size, gamma=0.5):
    """(B,C,H,W) -> contrast-stretched (H,W) numpy saliency at target size.
    Percentile clip (kills outliers that flatten everything) + gamma (lifts peaks)."""
    if norm_bchw.dim() == 4:
        s = norm_bchw.mean(1, keepdim=True)  # (B,C,H,W) -> mean C
    else:
        s = norm_bchw.unsqueeze(1)  # (B,H,W) -> (B,1,H,W)
    s = F.interpolate(s, size=size, mode="bilinear", align_corners=False)[0, 0]
    s = s.cpu().numpy()
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)  # robust range
    s = np.clip((s - lo) / (hi - lo + 1e-8), 0, 1)
    s = s**gamma  # <1 lifts mid/low peaks
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--no-dam", dest="dam", action="store_false")
    ap.add_argument("--out", default="saliency_clip.png")
    ap.add_argument(
        "--surf-stride",
        type=int,
        default=8,
        help="pixel step for the 3D surface (lower = finer, slower)",
    )
    ap.set_defaults(dam=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    cfg = Config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, cfg, dev)

    ds = MOCIDDataset(cfg.val_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=False)
    idx = random.randrange(len(ds))
    print(f"val clip idx {idx} / {len(ds)}  (seed {a.seed})")

    clip, gts = collate_eval([ds[idx]])
    clip = clip.to(dev)

    handles, sal_store = fista_saliency_hooks(model)
    dam_handles, dam_store = dam_saliency_hook(model)
    with torch.no_grad():
        outs = model(clip, use_dam=a.dam)
        dec = decode_outputs(outs, [8, 16, 32])
        box, scores = predict_image(dec[0], 1, a.conf, a.nms)
    for h in handles + dam_handles:
        h.remove()

    # target frame -> displayable RGB uint8
    tgt = clip[0, -1].permute(1, 2, 0).cpu().numpy()
    img = (tgt * 255).astype(np.uint8).copy()
    H, W = img.shape[:2]

    # draw boxes: green = pred, blue = GT
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
    print(
        f"{len(box)} detections above conf {a.conf}"
        + (f", top score {scores.max().item():.3f}" if len(box) else "")
    )

    # keep only the 3 FISTA LAYER outputs (last block of each layer) for a clean
    # FISTA-vs-DAM comparison; DAM produces one map per scale (3).
    fista_maps = [to_map(s, (H, W)) for s in sal_store]
    dam_maps = [to_map(s, (H, W)) for s in dam_store]
    print(f"{len(fista_maps)} FISTA block map(s), {len(dam_maps)} DAM scale map(s)")

    # pair DAM scales (3) with the matching FISTA layer outputs (every 3rd block)
    if len(fista_maps) >= 3 * len(dam_maps) and len(dam_maps) > 0:
        step = len(fista_maps) // len(dam_maps)
        fista_sel = [fista_maps[(i + 1) * step - 1] for i in range(len(dam_maps))]
    else:
        fista_sel = fista_maps[: len(dam_maps)] if dam_maps else fista_maps

    rows = [("FISTA", fista_sel)]
    if dam_maps and a.dam:
        rows.append(("DAM", dam_maps))
    ncol = 1 + max(len(r[1]) for r in rows)
    nrow = len(rows)

    st = max(1, a.surf_stride)
    yy, xx = np.mgrid[0:H:st, 0:W:st]
    fig = plt.figure(figsize=(5 * ncol, 5 * nrow))

    ax0 = fig.add_subplot(nrow, ncol, 1)
    ax0.imshow(det)
    ax0.set_title(f"clip {idx}: pred(green)/GT(blue)")
    ax0.axis("off")

    for ri, (name, maps) in enumerate(rows):
        for k, m in enumerate(maps):
            z = m[::st, ::st]
            ax = fig.add_subplot(nrow, ncol, ri * ncol + k + 2, projection="3d")
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
            ax.set_title(f"{name} saliency scale {k} (3D)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("energy")
            ax.view_init(elev=45, azim=-60)
            ax.invert_yaxis()

    import matplotlib.cm as cm, matplotlib.colors as mcolors

    sm = cm.ScalarMappable(cmap="jet", norm=mcolors.Normalize(0, 1))
    fig.colorbar(sm, ax=fig.axes[1:], shrink=0.5, pad=0.02, label="saliency (norm)")
    plt.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"saved -> {a.out}  (top row FISTA, bottom row DAM)")


if __name__ == "__main__":
    main()
