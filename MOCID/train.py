import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import MOCID
from utils.utils import (
    ModelEMA,
    build_optimizer,
    filtered_load,
    load_checkpoint,
    log_eval,
    save_ckpt,
    set_stage,
    strip_compile,
)

from utils.data import MOCIDDataset, collate_eval, collate_train
from utils.eval import evaluate


def build_loaders(cfg):
    """cfg -> (train_loader, val_loader)."""
    train_ds = MOCIDDataset(
        cfg.train_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=True
    )
    val_ds = MOCIDDataset(cfg.val_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_train,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_eval,
    )
    return train_loader, val_loader


def run_stage(
    model,
    loader,
    val_loader,
    opt,
    sched,
    epochs,
    device,
    use_dam,
    tag,
    stage,
    out_dir=".",
    start_ep=0,
    ckpt_every=5,
    ckpt_path="ckpt_last.pth",
    final_path=None,
    best_path=None,
    eval_every=20,
    csv_path="eval_log.csv",
    do_eval=True,
    ema=None,
    track_best_after=0,
    strides=(8, 16, 32),
    num_classes=1,
):
    """One training stage; all artifacts land in out_dir. -> None."""
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, ckpt_path)
    csv_path = os.path.join(out_dir, csv_path)
    if final_path is not None:
        final_path = os.path.join(out_dir, final_path)
    if best_path is not None:
        best_path = os.path.join(out_dir, best_path)

    scaler = torch.amp.GradScaler("cuda")

    # carry the previous best forward so a resumed run doesn't overwrite it
    best_ap50 = -1.0
    if best_path is not None and os.path.exists(best_path):
        try:
            prev = torch.load(best_path, map_location="cpu")
            best_ap50 = prev.get("ap50", -1.0) or -1.0
            print(f"[{tag}] existing best AP50 = {best_ap50:.2f}")
        except Exception as e:
            print(f"[warn] could not read best ckpt ({e})")

    for ep in range(start_ep, epochs):
        model.train()
        if use_dam:
            model.backbone.eval()  # frozen FISTA: no BN stat updates

        tot = nb = 0
        pbar = tqdm(
            loader, desc=f"[{tag}] {ep+1}/{epochs}", leave=False, mininterval=15
        )
        for clip, labels in pbar:
            clip = clip.to(device, non_blocking=True)
            labels = [l.to(device, non_blocking=True) for l in labels]

            with torch.amp.autocast("cuda"):
                loss = model(clip, labels, use_dam=use_dam)

            if not torch.isfinite(loss):  # skip the step rather than poison the weights
                opt.zero_grad(set_to_none=True)
                del loss
                nb += 1
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # unscale before clipping so the norm is real
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=10.0
            )
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)

            tot += loss.item()
            nb += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{tot/nb:.4f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

        sched.step()
        avg = tot / max(nb, 1)
        print(
            f"[{tag}] {ep+1}/{epochs}  loss {avg:.4f}  lr {opt.param_groups[0]['lr']:.2e}"
        )

        if (ep + 1) % ckpt_every == 0 or (ep + 1) == epochs:
            save_ckpt(ckpt_path, model, opt, sched, ep, stage, tag, ema=ema)

        if do_eval and ((ep + 1) % eval_every == 0 or (ep + 1) == epochs):
            eval_target = ema.ema if ema is not None else model
            ap50, f1 = evaluate(
                eval_target,
                val_loader,
                device,
                use_dam,
                strides=list(strides),
                num_classes=num_classes,
            )
            # one row per eval: timestamp, tag, epoch, AP50, F1, mean epoch loss
            log_eval(
                csv_path,
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tag": tag,
                    "epoch": ep + 1,
                    "ap50": round(ap50 * 100, 2),
                    "f1": round(f1 * 100, 2),
                    "avg_loss": round(avg, 4),
                },
            )
            print(f"[{tag}] eval @ ep{ep+1}: AP50 {ap50*100:.2f}  F1 {f1*100:.2f}")

            # track_best_after avoids locking in an early high-LR AP50 spike
            track_ok = (ep + 1) > track_best_after
            if best_path is not None and track_ok and ap50 * 100 > best_ap50:
                best_ap50 = ap50 * 100
                # save the exact weights that were evaluated (EMA if present)
                save_ckpt(
                    best_path,
                    eval_target,
                    opt,
                    sched,
                    ep,
                    stage,
                    tag,
                    ap50=best_ap50,
                    ema=ema,
                )
                print(f"[{tag}] *** new best AP50 {best_ap50:.2f} -> {best_path}")
            elif best_path is not None and not track_ok:
                print(f"[{tag}] (best tracking starts after epoch {track_best_after})")

    if final_path is not None:
        deploy = ema.ema if ema is not None else model
        save_ckpt(final_path, deploy, opt, sched, epochs - 1, stage, tag, ema=ema)
        print(f"[{tag}] saved stage checkpoint -> {final_path}")


def seed_from_fista_best(model, out_dir):
    """Load the best stage-1 weights into the frozen backbone/pool/fpn/head. -> None."""
    # the backbone is frozen for all of stage 2, so it must start from the BEST
    # stage-1 weights rather than whatever the last epoch happened to leave behind
    for name, kind in (("fista_best.pth", "best"), ("fista.pth", "final (fallback)")):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu")
            # drop disp.* so the zero-init DAM branch survives untouched
            sd = {
                k: v
                for k, v in strip_compile(ck["model"]).items()
                if not k.startswith("disp.")
            }
            raw = getattr(model, "_orig_mod", model)
            dropped = filtered_load(raw, sd)
            print(
                f"[stage2] seeded from FISTA {kind}: {path}  "
                f"(disp.* kept at zero-init; {dropped} non-matching keys skipped)"
            )
            return
    print("[stage2] no FISTA checkpoint found to seed from — using current weights")


def train_mocid(
    cfg, device, tag="default", train_loader=None, val_loader=None, do_eval=True
):
    """Two-stage training with auto-resume from runs/<tag>/. -> the trained model."""
    model = MOCID(num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]).to(device)
    model = torch.compile(model)

    out_dir = os.path.join("runs", tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] artifacts -> {out_dir}/")

    if train_loader is None:
        train_loader, val_loader = build_loaders(cfg)

    # ---------------- stage 1: FISTA backbone ----------------
    print("=" * 25, "STAGE 1: FISTA", "=" * 25)
    set_stage(model, 1)
    opt, sched = build_optimizer(model, cfg, cfg.EPOCHS_SPTBACKBONE)
    ema = ModelEMA(model)
    s1_start = load_checkpoint(
        os.path.join(out_dir, "fista_last.pth"), model, opt, sched, ema
    )

    if s1_start < cfg.EPOCHS_SPTBACKBONE:
        run_stage(
            model,
            train_loader,
            val_loader,
            opt,
            sched,
            cfg.EPOCHS_SPTBACKBONE,
            device,
            use_dam=False,
            tag=f"{tag}-fista",
            stage=1,
            out_dir=out_dir,
            start_ep=s1_start,
            ckpt_path="fista_last.pth",
            final_path="fista.pth",
            best_path="fista_best.pth",
            csv_path="eval_log.csv",
            eval_every=cfg.EVAL_EVERY,
            do_eval=do_eval,
            ema=ema,
            track_best_after=cfg.TRACK_BEST_AFTER,
            strides=cfg.STRIDES,
            num_classes=cfg.NUM_CLASSES,
        )
    else:
        print(f"[train] stage 1 already complete ({s1_start} epochs)")

    # ---------------- stage 2: freeze FISTA, train DAM ----------------
    print("=" * 25, "STAGE 2: DAM", "=" * 25)
    set_stage(model, 2)

    # runs before ModelEMA so the shadow copy also starts from the best backbone
    seed_from_fista_best(model, out_dir)

    opt, sched = build_optimizer(model, cfg, cfg.EPOCHS_DAM, lr=cfg.LR_DAM)
    # fast EMA: a fresh disp branch must be tracked, not smoothed over ~10k updates
    ema = ModelEMA(model, decay=0.999, tau=300, freeze_backbone=True)
    s2_start = load_checkpoint(
        os.path.join(out_dir, "dam_last.pth"), model, opt, sched, ema
    )

    if s2_start < cfg.EPOCHS_DAM:
        run_stage(
            model,
            train_loader,
            val_loader,
            opt,
            sched,
            cfg.EPOCHS_DAM,
            device,
            use_dam=True,
            tag=f"{tag}-dam",
            stage=2,
            out_dir=out_dir,
            start_ep=s2_start,
            ckpt_path="dam_last.pth",
            final_path="dam.pth",
            best_path="dam_best.pth",
            csv_path="eval_log.csv",
            eval_every=cfg.EVAL_EVERY,
            do_eval=do_eval,
            ema=ema,
            strides=cfg.STRIDES,
            num_classes=cfg.NUM_CLASSES,
        )
    else:
        print(f"[train] stage 2 already complete ({s2_start} epochs)")

    return model
