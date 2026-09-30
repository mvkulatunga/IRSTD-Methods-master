import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import Config, get_device, setup_torch
from utils.data import MOCIDDataset, collate_eval, collate_train
from utils.eval import evaluate
from model import MOCID
from base import MOCIDBase
from base_fista import MOCIDBaseFISTA
from train import train_mocid, train_single_stage
from utils.utils import count_params_m, recalibrate_bn, strip_compile

# the paper's Table 2 rows: Base, +FISTA, and the full model (+FISTA+DAM, two stages)
MODELS = {"MOCID": MOCID, "MOCIDBase": MOCIDBase, "MOCIDBaseFISTA": MOCIDBaseFISTA}
LABELS = {"MOCIDBase": "base", "MOCIDBaseFISTA": "basefista"}  # eval_log.csv row tags


def parse_args():
    """-> parsed CLI namespace."""
    ap = argparse.ArgumentParser(description="MOCID train / eval / param count")
    ap.add_argument(
        "--profile",
        default=None,
        help="settings profile from config.PROFILES, e.g. r0 (default: MOCID_PROFILE, else none)",
    )
    ap.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="MOCID",
        help="MOCID (two stages: FISTA backbone, then DAM), or the one-stage ablation "
        "models MOCIDBase / MOCIDBaseFISTA (default: MOCID)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="training with auto-resume from runs/<tag>/")
    p_train.add_argument("--tag", default="default")
    p_train.add_argument("--no-eval", dest="do_eval", action="store_false")
    p_train.add_argument(
        "--stage1-from",
        default=None,
        help="skip stage 1 and train stage 2 (DAM) from this stage-1 checkpoint",
    )
    p_train.set_defaults(do_eval=True)

    p_eval = sub.add_parser("eval", help="AP50 / best-F1 on the val split")
    p_eval.add_argument("--ckpt", default="ckpt_last.pth")
    p_eval.add_argument("--dam", dest="dam", action="store_true")
    p_eval.add_argument("--no-dam", dest="dam", action="store_false")
    p_eval.add_argument("--conf", type=float, default=1e-3)
    p_eval.add_argument("--nms", type=float, default=0.65)
    p_eval.add_argument(
        "--perseq",
        action="store_true",
        help="print a per-video recall/confidence breakdown instead of just the aggregate",
    )
    p_eval.add_argument(
        "--recal-bn",
        type=int,
        default=0,
        metavar="N",
        help="recompute BatchNorm statistics from N training batches before scoring "
        "(training now does this before every evaluation; use it for older checkpoints)",
    )
    p_eval.set_defaults(dam=None)

    p_params = sub.add_parser("params", help="parameter counts with and without DAM")
    p_params.add_argument("--ckpt", default=None)

    return ap.parse_args()


def build_model(cfg, device, name="MOCID"):
    """-> the model called name (a key of MODELS) on device."""
    return MODELS[name](
        num_classes=cfg.NUM_CLASSES, num_frames=cfg.T, img_size=cfg.IMG_SIZE[0]
    ).to(device)


def load_weights(model, ckpt_path):
    """-> the training stage recorded in the checkpoint (default 2)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(strip_compile(ckpt["model"]))
    return ckpt.get("stage", 2)


def print_params(model, name="MOCID"):
    """Print param counts (both configurations for MOCID). -> None."""
    if name != "MOCID":
        print(f"Params  {name:22s}: {count_params_m(model, True):.2f} M")
        return
    print(f"Params  .+FISTA (no DAM)      : {count_params_m(model, False):.2f} M")
    print(f"Params  .+FISTA+DAM (MOCID)   : {count_params_m(model, True):.2f} M")


def cmd_eval(args, cfg, device):
    """Load a checkpoint and report AP50 / best-F1. -> None."""
    model = build_model(cfg, device, args.model)
    stage = load_weights(model, args.ckpt)

    # --dam/--no-dam overrides; otherwise the branch follows the checkpoint's stage.
    # The ablation models have no DAM, so it stays off for them.
    use_dam = args.dam if args.dam is not None else (stage == 2)
    use_dam = use_dam and args.model == "MOCID"
    print(f"loaded {args.ckpt}  ({args.model}, stage={stage})  ->  use_dam={use_dam}")
    print_params(model, args.model)

    if args.recal_bn > 0:
        # stage 2 froze the backbone, so only the layers after it are recomputed
        names = None
        if stage == 2:
            names = {
                n for n, m in model.named_modules()
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and not n.startswith("backbone.")
            }
        train_ds = MOCIDDataset(
            cfg.train_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=True, norm=cfg.NORMALISE
        )
        loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_train
        )
        recalibrate_bn(model, loader, device, use_dam, args.recal_bn, names)
        model.eval()
        print(f"BatchNorm statistics recomputed from {args.recal_bn} training batches")

    val_ds = MOCIDDataset(
        cfg.val_path, T=cfg.T, img_size=cfg.IMG_SIZE, is_train=False, norm=cfg.NORMALISE
    )
    assert len(val_ds) > 0, f"Val set empty — check {cfg.val_path} (cwd={os.getcwd()})"

    if args.model == "MOCID":
        row = ".+FISTA+DAM (MOCID)" if use_dam else ".+FISTA (MOCID, DAM off)"
    else:
        row = args.model
    print(f"\n=== {row} ===")

    if args.perseq:
        from utils.perseq import collate_eval_perseq, perseq_breakdown, print_perseq_table

        loader = DataLoader(
            val_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_eval_perseq,
        )
        summary, per_seq = perseq_breakdown(
            model, loader, device, use_dam, cfg.STRIDES, cfg.NUM_CLASSES,
            conf_thr=args.conf, nms_thr=args.nms,
        )
        print_perseq_table(summary, per_seq)
        return

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
    print(f"AP50 : {ap50 * 100:.2f}")
    print(f"F1   : {f1 * 100:.2f}   (best over PR sweep)")


def main():
    setup_torch()
    args = parse_args()
    cfg = Config(args.profile)
    device = get_device()
    print(f"[config] profile: {cfg.PROFILE}  (LR {cfg.LR_INIT} -> {cfg.MIN_LR}, "
          f"input {cfg.NORMALISE}, decay on all params: {cfg.DECAY_ALL}, "
          f"BatchNorm recomputed before eval: {cfg.BN_RECAL_BATCHES} batches)")

    if args.cmd == "train":
        if args.model == "MOCID":
            train_mocid(
                cfg, device, tag=args.tag, do_eval=args.do_eval,
                stage1_ckpt=args.stage1_from,
            )
        else:
            if args.stage1_from:
                raise SystemExit(f"--stage1-from is for MOCID's stage 2; {args.model} has one stage")
            train_single_stage(
                cfg, device, MODELS[args.model], LABELS[args.model], tag=args.tag,
                do_eval=args.do_eval,
            )
    elif args.cmd == "eval":
        cmd_eval(args, cfg, device)
    elif args.cmd == "params":
        model = build_model(cfg, device, args.model)
        if args.ckpt:
            load_weights(model, args.ckpt)
        print_params(model, args.model)


if __name__ == "__main__":
    main()
