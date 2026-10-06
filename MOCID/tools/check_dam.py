"""Checks to run after every change to the DAM, before spending hours on a stage-2 run.

    python tools/check_dam.py --profile r0 --ckpt runs/server-r0/fista_best.pth [--ap]

The checkpoint must be a stage-1 checkpoint (DAM never trained), and --profile must
match the settings it was trained with, because the input normalisation differs.

Checks
  1. checkpoint   the DAM is at its initial weights (the checkpoint's disp.* are dropped,
                  as stage 2 does), so the handover check below tests the DAM's init
  2. parameters   stage-1 model, DAM and total vs the paper's Table 2 (9.45 / 3.60 / 13.05 M)
  3. handover     switching the DAM on at its initial weights leaves the model's output
                  unchanged, so stage 2 starts from the stage-1 model (CODE-REVIEW.md
                  finding 1). --ap also measures it as AP50 on the full validation set
  4. training     a stage-2 training step with the DAM on gives a finite loss and finite
                  gradients; reports how many DAM tensors receive a non-zero gradient
  5. alignment    each scanned token gets the scan parameters (B, C, delta) of its own
                  frame; not implemented yet (step 2 of the DAM work)

Exit code 0 if nothing FAILs.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # the MOCID folder

import torch
from torch.utils.data import DataLoader, Subset

from config import Config, setup_torch
from model import MOCID
from utils.data import MOCIDDataset, collate_eval, collate_train
from utils.eval import evaluate
from utils.utils import set_stage, strip_compile

PAPER_M = {"stage 1 (backbone + FPN + head)": 9.45, "DAM": 3.60, "total": 13.05}
results = []


def report(name, status, detail):
    results.append(status)
    print(f"  [{status:4s}] {name:12s} {detail}")


def rel_diff(a, b):
    return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default=None, help="settings profile the checkpoint was trained with")
    ap.add_argument("--ckpt", default="runs/server-r0/fista_best.pth", help="a stage-1 checkpoint")
    ap.add_argument("--clips", type=int, default=16, help="validation clips for the handover check")
    ap.add_argument("--tol", type=float, default=1e-3, help="relative tolerance for 'unchanged'")
    ap.add_argument("--ap", action="store_true", help="also compare AP50 on the full validation set")
    a = ap.parse_args()

    setup_torch()
    torch.manual_seed(0)
    cfg = Config(a.profile)
    dev = torch.device("cuda")
    print(f"profile {cfg.PROFILE} (input {cfg.NORMALISE}), checkpoint {a.ckpt}\n")

    model = MOCID(num_frames=cfg.T, img_size=cfg.IMG_SIZE[0])
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    # as stage 2 does (train.py, seed_from_fista_best): drop the checkpoint's disp.* and keep
    # the DAM at its own init, so a checkpoint saved with an older DAM layout still loads
    sd = {k: v for k, v in strip_compile(ck["model"]).items() if not k.startswith("disp.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    stray = [k for k in missing if not k.startswith("disp.")] + unexpected
    assert not stray, f"checkpoint does not match the model outside the DAM: {stray[:5]}"
    model = model.to(dev).eval()

    # 1. the DAM in the checkpoint is untouched
    untouched = all(
        torch.count_nonzero(blk.mid.weight) == 0
        and torch.equal(blk.out.weight[:, :, 0, 0].cpu(), torch.eye(blk.out.weight.shape[0]))
        for blk in model.disp.blocks
    )
    report(
        "checkpoint",
        "PASS" if untouched else "WARN",
        "DAM at its initial weights (mid = 0, out = identity)" if untouched
        else "DAM weights differ from the init; check 3 then tests those weights, not the init",
    )

    # 2. parameter counts
    n = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
    counts = {
        "stage 1 (backbone + FPN + head)": n(model) - n(model.disp),
        "DAM": n(model.disp),
        "total": n(model),
    }
    for k, v in counts.items():
        target = PAPER_M[k]
        off = 100 * (v - target) / target
        report("parameters", "PASS" if abs(off) <= 2 else "FAIL", f"{k}: {v:.3f} M vs paper {target:.2f} M ({off:+.1f}%)")

    # 3. handover: DAM off vs DAM on at init, on clips spread across all validation videos
    val = MOCIDDataset(cfg.val_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=False, norm=cfg.NORMALISE)
    idx = list(range(0, len(val), max(1, len(val) // a.clips)))[: a.clips]
    clips, _ = collate_eval([val[i] for i in idx])
    clips = clips.to(dev)
    with torch.no_grad():
        Ft, Fr = model.backbone(clips)
        feats = [torch.cat([Fr[k], Ft[k].unsqueeze(1)], dim=1) for k in range(3)]
        pool_off = model.pool(feats)
        pool_on = model.pool(model.disp([f.float() for f in feats]))
        out_off = model(clips, use_dam=False)
        out_on = model(clips, use_dam=True)
    d_pool = max(rel_diff(x, y) for x, y in zip(pool_off, pool_on))
    d_out = max(rel_diff(x, y) for x, y in zip(out_off, out_on))
    ok = d_pool < a.tol and d_out < a.tol
    report(
        "handover",
        "PASS" if ok else "FAIL",
        f"{len(idx)} clips: pooled features change {100*d_pool:.2f}%, head outputs {100*d_out:.2f}% "
        f"(tolerance {100*a.tol:.2f}%)",
    )
    if a.ap:
        loader = DataLoader(val, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_eval)
        ap_off, _ = evaluate(model, loader, dev, False, cfg.STRIDES, cfg.NUM_CLASSES)
        ap_on, _ = evaluate(model, loader, dev, True, cfg.STRIDES, cfg.NUM_CLASSES)
        report(
            "handover",
            "PASS" if abs(ap_on - ap_off) * 100 < 0.1 else "FAIL",
            f"AP50 on all {len(val)} validation frames: DAM off {100*ap_off:.2f}, DAM on {100*ap_on:.2f}",
        )

    # 4. one stage-2 training step
    train = MOCIDDataset(cfg.train_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=True, norm=cfg.NORMALISE)
    tclips, labels = collate_train([train[i] for i in (0, len(train) // 3, 2 * len(train) // 3, len(train) - 1)])
    set_stage(model, 2)
    model.train()
    model.backbone.eval()  # frozen in stage 2, as train.py does
    model.zero_grad(set_to_none=True)
    loss = model(tclips.to(dev), [l.to(dev) for l in labels], use_dam=True)
    loss.backward()
    dam = [(k, p) for k, p in model.disp.named_parameters()]
    finite = torch.isfinite(loss).item() and all(p.grad is None or torch.isfinite(p.grad).all() for _, p in dam)
    live = [k for k, p in dam if p.grad is not None and torch.count_nonzero(p.grad) > 0]
    report("training", "PASS" if finite else "FAIL", f"loss {loss.item():.3f}; loss and DAM gradients finite: {finite}")
    report(
        "training",
        "INFO",
        f"{len(live)} of {len(dam)} DAM tensors get a non-zero gradient at the first step"
        + ("" if len(live) == len(dam) else f" (e.g. only: {', '.join(sorted({k.split('.', 2)[2] for k in live}))})"),
    )

    # 5. token / parameter alignment
    report("alignment", "SKIP", "per-frame scan parameters are not implemented yet (step 2)")

    fails = results.count("FAIL")
    print(f"\n{fails} check(s) failed" if fails else "\nall checks passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
