import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import Config, get_device, setup_torch
from utils.data import MOCIDDataset, collate_eval
from utils.eval import evaluate
from model import MOCID
from train import train_mocid
from utils.utils import count_params_m, strip_compile


def parse_args():
    """-> parsed CLI namespace."""
    ap = argparse.ArgumentParser(description="MOCID train / eval / param count")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="two-stage training with auto-resume")
    p_train.add_argument("--tag", default="default")
    p_train.add_argument("--no-eval", dest="do_eval", action="store_false")
    p_train.set_defaults(do_eval=True)

    p_eval = sub.add_parser("eval", help="AP50 / best-F1 on the val split")
    p_eval.add_argument("--ckpt", default="ckpt_last.pth")
    p_eval.add_argument("--dam", dest="dam", action="store_true")
    p_eval.add_argument("--no-dam", dest="dam", action="store_false")
    p_eval.add_argument("--conf", type=float, default=1e-3)
    p_eval.add_argument("--nms", type=float, default=0.65)
    p_eval.set_defaults(dam=None)

    p_params = sub.add_parser("params", help="parameter counts with and without DAM")
    p_params.add_argument("--ckpt", default=None)

    return ap.parse_args()


def build_model(cfg, device):
    """-> MOCID on device."""
    return MOCID(
        num_classes=cfg.NUM_CLASSES, num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]
    ).to(device)


def load_weights(model, ckpt_path):
    """-> the training stage recorded in the checkpoint (default 2)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(strip_compile(ckpt["model"]))
    return ckpt.get("stage", 2)


def print_params(model):
    """Print param counts for both configurations. -> None."""
    print(f"Params  .+FISTA (no DAM)      : {count_params_m(model, False):.2f} M")
    print(f"Params  .+FISTA+DAM (MOCID)   : {count_params_m(model, True):.2f} M")


def cmd_eval(args, cfg, device):
    """Load a checkpoint and report AP50 / best-F1. -> None."""
    model = build_model(cfg, device)
    stage = load_weights(model, args.ckpt)

    # --dam/--no-dam overrides; otherwise the branch follows the checkpoint's stage
    use_dam = args.dam if args.dam is not None else (stage == 2)
    print(f"loaded {args.ckpt}  (stage={stage})  ->  use_dam={use_dam}")
    print_params(model)

    val_ds = MOCIDDataset(cfg.val_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=False)
    assert len(val_ds) > 0, f"Val set empty — check {cfg.val_path} (cwd={os.getcwd()})"
    loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_eval,
    )

    ap50, f1 = evaluate(
        model,
        loader,
        device,
        use_dam,
        cfg.STRIDES,
        cfg.NUM_CLASSES,
        conf_thr=args.conf,
        nms_thr=args.nms,
    )
    row = ".+FISTA+DAM (MOCID)" if use_dam else ".+FISTA"
    print(f"\n=== {row} ===")
    print(f"AP50 : {ap50 * 100:.2f}")
    print(f"F1   : {f1 * 100:.2f}   (best over PR sweep)")


def main():
    setup_torch()
    cfg = Config()
    device = get_device()
    args = parse_args()

    if args.cmd == "train":
        train_mocid(cfg, device, tag=args.tag, do_eval=args.do_eval)
    elif args.cmd == "eval":
        cmd_eval(args, cfg, device)
    elif args.cmd == "params":
        model = build_model(cfg, device)
        if args.ckpt:
            load_weights(model, args.ckpt)
        print_params(model)


if __name__ == "__main__":
    main()
