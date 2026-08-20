import argparse
from pathlib import Path
from copy import deepcopy

import yaml
import torch
import ultralytics.nn.tasks
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from components import CoordAtt, TYRISTDetectionLoss

ultralytics.nn.tasks.CoordAtt = CoordAtt
DetectionModel.init_criterion = lambda self: TYRISTDetectionLoss(self)

# Model configs anchored to this file so they load regardless of CWD.
# (Run from the repo root anyway, so the data yaml's ../../datasets resolves.)
BASE = Path(__file__).resolve().parent

COCO_INIT  = "yolo12n.pt"
STAGE1_CFG = str(BASE / "configs/models/stage1_base.yaml")          # 4-head, NO CA
STAGE2_CFG = str(BASE / "configs/models/full_training_graph.yaml")  # 4-head, WITH CA (18-20)
OPTIMIZER, LR0, BATCH, DEVICE = "AdamW", 1e-4, 4, 0
PROJECT = "TY-RIST_Project"
WEIGHTS_DIR = Path("weights")

# Per-regime resolution + base epochs (Sec 4.3). Stage-2 CA fine-tune is 100 for both.
SINGLE_FRAME = dict(imgsz=640, stage1=200, stage2=100)
MULTI_FRAME  = dict(imgsz=512, stage1=100, stage2=100)
MULTIFRAME_TAGS = {"itsdt_15k", "irdst"}

# Stage-2 layer map: 0-8 backbone | 9-17 FPN | 18-20 CA | 21-29 PAN | 30 Detect
FREEZE_BACKBONE_NECK = list(range(0, 18)) + list(range(21, 30))
CA_SHIFT, STAGE1_HEAD_IDX = 3, 27

# Dataset-specific deployment trim (Sec 5.1-5.2):
#   NUDT + ITSDT-15k + IRDST -> P2 only ; NUAA + cross-dataset -> P2 + P3
PRUNE_TARGET = {
    "nuaa_sirst": "p2p3", "nudt_sirst": "p2", "combined_sirst": "p2p3",
    "itsdt_15k": "p2",    "irdst": "p2",
}
TRIM_CFG = {
    "p2":   str(BASE / "configs/models/irstd_15k_nudt_sirst.yaml"),   # Detect [[20]]
    "p2p3": str(BASE / "configs/models/nuaa_sirst_inference.yaml"),   # Detect [[20, 23]]
}
FULL_DETECT_IDX = 30
KEEP_SCALES = {"p2": [0], "p2p3": [0, 1]}


def hparams(tag):
    return MULTI_FRAME if tag in MULTIFRAME_TAGS else SINGLE_FRAME


def best_path(stage, tag, project=PROJECT):
    """Deterministic location of a stage's best.pt, so any stage runs alone."""
    return f"runs/detect/{project}/{stage}_{tag}/weights/best.pt"


def transfer_backbone_neck(stage1_ckpt, stage2_model):
    """Overlay Stage-1 backbone+neck into the Stage-2 (with-CA) graph; Stage-1
    layers >=18 shift +3 over the inserted CA blocks; Stage-1 Detect is dropped."""
    ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    src = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
    remapped = {}
    for k, v in src.items():
        parts = k.split("."); idx = int(parts[1])
        if idx == STAGE1_HEAD_IDX:
            continue
        parts[1] = str(idx + CA_SHIFT if idx >= 18 else idx)
        remapped[".".join(parts)] = v
    tgt = stage2_model.state_dict()
    matched = {k: v for k, v in remapped.items() if k in tgt and v.shape == tgt[k].shape}
    stage2_model.load_state_dict(matched, strict=False)
    print(f"  ✓ transferred {len(matched)} backbone+neck tensors from Stage 1")


def prune_and_save(stage2_ckpt, prune_kind, out_path):
    """Structurally trim Stage-2 weights into the smaller deployment graph."""
    d = yaml.safe_load(open(TRIM_CFG[prune_kind])); d["scale"] = "n"
    trimmed = DetectionModel(deepcopy(d), ch=3, nc=1, verbose=False)
    trim_detect_idx = len(trimmed.model) - 1

    ckpt = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)
    src = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
    tgt = trimmed.state_dict()
    keep = set(KEEP_SCALES[prune_kind])

    matched = {}
    for k, v in src.items():
        idx = int(k.split(".")[1])
        if idx < FULL_DETECT_IDX:
            if k in tgt and tgt[k].shape == v.shape:
                matched[k] = v
        elif idx == FULL_DETECT_IDX:
            parts = k.split("."); parts[1] = str(trim_detect_idx)
            if parts[2] in ("cv2", "cv3") and int(parts[3]) not in keep:
                continue
            nk = ".".join(parts)
            if nk in tgt and tgt[nk].shape == v.shape:
                matched[nk] = v

    trimmed.load_state_dict(matched, strict=False)
    print(f"  ✓ pruned to {prune_kind}: {len(matched)}/{len(tgt)} tensors transferred")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": trimmed.half(), "epoch": -1, "optimizer": None,
                "train_args": {}, "date": "", "version": ""}, out_path)
    print(f"  ✓ saved deployment model -> {out_path}")


# ======================================================================
#  STAGES — call/comment independently in __main__. imgsz/epochs auto-set
#  per dataset (single- vs multi-frame); CLI flags override.
# ======================================================================
def run_stage1(data_cfg, tag, epochs=None, batch=None, imgsz=None, project=PROJECT):
    hp = hparams(tag)
    epochs, imgsz, batch = epochs or hp["stage1"], imgsz or hp["imgsz"], batch or BATCH
    print(f"\n>>> Stage 1: base, no CA ({epochs} epochs, imgsz={imgsz})")
    m1 = YOLO(STAGE1_CFG, task="detect"); m1.load(COCO_INIT)
    m1.train(data=data_cfg, epochs=epochs, imgsz=imgsz, optimizer=OPTIMIZER,
             lr0=LR0, batch=batch, device=DEVICE,
             project=project, name=f"Stage1_{tag}", exist_ok=True)


def run_stage2(data_cfg, tag, epochs=None, batch=None, imgsz=None,
               project=PROJECT, stage1_ckpt=None):
    hp = hparams(tag)
    epochs, imgsz, batch = epochs or hp["stage2"], imgsz or hp["imgsz"], batch or BATCH
    stage1_ckpt = stage1_ckpt or best_path("Stage1", tag, project)
    print(f"\n>>> Stage 2: add CA, freeze backbone+neck, fine-tune ({epochs} epochs, imgsz={imgsz})")
    print(f"    (Stage-1 weights: {stage1_ckpt})")
    m2 = YOLO(STAGE2_CFG, task="detect"); m2.load(COCO_INIT)
    transfer_backbone_neck(stage1_ckpt, m2.model)
    m2.train(data=data_cfg, epochs=epochs, imgsz=imgsz, optimizer=OPTIMIZER,
             lr0=LR0, batch=batch, device=DEVICE, freeze=FREEZE_BACKBONE_NECK,
             project=project, name=f"Stage2_{tag}", exist_ok=True)


def run_stage3(tag, stage2_ckpt=None, project=PROJECT):
    stage2_ckpt = stage2_ckpt or best_path("Stage2", tag, project)
    if not Path(stage2_ckpt).is_file():
        raise FileNotFoundError(
            f"Stage-2 checkpoint not found: {stage2_ckpt}\n"
            f"Run Stage 2 first, or pass --ckpt /path/to/best.pt")
    kind = PRUNE_TARGET[tag]
    print(f"\n>>> Stage 3: prune -> {kind}\n    (pruning checkpoint: {stage2_ckpt})")
    out = WEIGHTS_DIR / f"tyrist_{tag}_{kind}.pt"
    prune_and_save(stage2_ckpt, kind, out)


def parse_args():
    p = argparse.ArgumentParser(
        description="TY-RIST: stage1 -> CA fine-tune -> prune. imgsz/epochs auto-set "
                    "per dataset (single/multi-frame). Comment stage calls to run subsets.")
    p.add_argument("--data", required=True,
                   help="e.g. configs/data/nuaa_sirst.yaml or configs/data/itsdt_15k.yaml")
    p.add_argument("--stage1-epochs", type=int, default=None, help="override base epochs")
    p.add_argument("--stage2-epochs", type=int, default=None, help="override CA fine-tune epochs")
    p.add_argument("--batch", type=int, default=None, help="override batch (paper=4)")
    p.add_argument("--imgsz", type=int, default=None, help="override image size")
    p.add_argument("--project", type=str, default=PROJECT,
                   help=f"output project dir (default {PROJECT}); e.g. SMOKE_TEST to isolate")
    p.add_argument("--ckpt", default=None,
                   help="Stage-2 best.pt for Stage 3 (default: the standard run path)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    tag = Path(a.data).stem
    if tag not in PRUNE_TARGET:
        raise ValueError(f"No prune target for '{tag}'. Add it to PRUNE_TARGET.")
    regime = "multi-frame" if tag in MULTIFRAME_TAGS else "single-frame"
    print(f"\n===== TY-RIST: {a.data}  [{regime}, prune->{PRUNE_TARGET[tag]}] =====")

    run_stage1(a.data, tag, a.stage1_epochs, a.batch, a.imgsz, a.project)
    run_stage2(a.data, tag, a.stage2_epochs, a.batch, a.imgsz, a.project)
    run_stage3(tag, a.ckpt, a.project)

    print(f"✓ Done: {a.data}\n")