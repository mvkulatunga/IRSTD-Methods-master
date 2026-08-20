import csv
import math
import os
from copy import deepcopy

import torch


class ModelEMA:
    """EMA of params and buffers; evaluate/checkpoint this instead of the live model."""

    def __init__(self, model, decay=0.9999, tau=2000, updates=0, freeze_backbone=False):
        raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
        self.ema = deepcopy(raw).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # warmup ramp
        self.freeze_backbone = freeze_backbone  # True only in stage 2
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Blend the live weights into the shadow copy. -> None."""
        raw = getattr(model, "_orig_mod", model)
        self.updates += 1
        d = self.decay(self.updates)
        msd = raw.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:
                continue
            mv = msd[k].detach().to(v.dtype)
            if self.freeze_backbone and k.startswith("backbone."):
                v.copy_(mv)  # frozen FISTA: track exactly, nothing to smooth
            else:
                v.mul_(d).add_(mv, alpha=1 - d)

    def state_dict(self):
        return self.ema.state_dict()


def strip_compile(sd):
    """state_dict -> same dict without torch.compile's '_orig_mod.' key prefix."""
    return {
        k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k: v
        for k, v in sd.items()
    }


def save_ckpt(path, model, opt, sched, ep, stage, tag, ap50=None, ema=None):
    """Write a checkpoint atomically (.tmp then rename). -> None."""
    torch.save(
        {
            "model": strip_compile(model.state_dict()),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": ep,
            "stage": stage,
            "tag": tag,
            "ap50": ap50,
            "rng_cpu": torch.get_rng_state().cpu(),
            "rng_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()],
            "ema": ema.state_dict() if ema is not None else None,
        },
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)


def filtered_load(module, sd):
    """Load only shape-matching keys. -> count of dropped source keys."""
    tgt = module.state_dict()
    filt = {k: v for k, v in sd.items() if k in tgt and v.shape == tgt[k].shape}
    module.load_state_dict(filt, strict=False)
    return len(sd) - len(filt)


def load_checkpoint(path, model, opt=None, sched=None, ema=None):
    """Resume-load a checkpoint. -> epoch to resume from, or 0 if the file is absent."""
    if not os.path.exists(path):
        return 0
    ck = torch.load(path, map_location="cpu")
    raw = getattr(model, "_orig_mod", model)

    # shape-filtered so stale or re-architected disp.* params are skipped
    dropped = filtered_load(raw, strip_compile(ck["model"]))
    if dropped:
        print(f"[resume] {path}: skipped {dropped} incompatible key(s)")

    if opt is not None and ck.get("opt"):
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as e:
            print(f"[resume] optimizer state not restored ({e})")
    if sched is not None and ck.get("sched"):
        try:
            sched.load_state_dict(ck["sched"])
        except Exception as e:
            print(f"[resume] scheduler state not restored ({e})")
    if ema is not None and ck.get("ema"):
        filtered_load(ema.ema, ck["ema"])  # weights only; the decay ramp restarts

    nxt = ck.get("epoch", -1) + 1
    print(f"[resume] loaded {path}  -> resume at epoch {nxt}")
    return nxt


def build_optimizer(model, cfg, epochs, lr=None):
    """-> (SGD+Nesterov over trainable params, LambdaLR with warmup + cosine decay)."""
    lr = cfg.LR_INIT if lr is None else lr
    params = [p for p in model.parameters() if p.requires_grad]  # skips frozen stages

    opt = torch.optim.SGD(
        params,
        lr=lr,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
        nesterov=True,
    )

    warmup = cfg.WARMUP_EPOCHS

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / (epochs - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (cfg.MIN_LR / lr) + (1 - cfg.MIN_LR / lr) * cosine

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def set_stage(model, stage):
    """stage 1: train backbone+pool+fpn+head. stage 2: freeze backbone, train disp. -> None."""
    for p in model.parameters():
        p.requires_grad_(False)

    if stage == 1:
        train = [model.backbone, model.pool, model.fpn, model.head]
    elif stage == 2:
        train = [
            m
            for m in [getattr(model, "disp", None), model.pool, model.fpn, model.head]
            if m is not None
        ]
    else:
        raise ValueError(stage)

    for m in train:
        for p in m.parameters():
            p.requires_grad_(True)


def count_params_m(model, include_dam):
    """-> parameter count in millions; include_dam=False excludes the disp.* branch."""
    if include_dam:
        n = sum(p.numel() for p in model.parameters())
    else:
        n = sum(
            p.numel()
            for name, p in model.named_parameters()
            if not name.startswith("disp.")
        )
    return n / 1e6


def log_eval(csv_path, row):
    """Append a dict row to a CSV, writing the header on first use. -> None."""
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
