import os

import torch


class Config:
    T = 5
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 4

    # LR_INIT follows SSTNet's linear batch-size scaling rule (which the paper's
    # split/training convention follows): base 0.01 is defined for batch 64, so
    # at our batch of 4 that's 0.01 * 4/64 = 6.25e-4. Running 0.01 unscaled (16x
    # too high) is the prime suspect behind the Base/`.+FISTA` mid-training
    # collapse (EXPERIMENTS.md: Base and R0 runs) -- MIN_LR is scaled by the same
    # factor so the warmup/cosine shape (in ratio terms) is unchanged.
    LR_INIT = 6.25e-4
    MIN_LR = 6.25e-6
    LR_DAM = 1e-3  # stage-2 LR; 1e-2 diverges a fresh Mamba branch -- unreviewed, see PLAN.md
    WARMUP_EPOCHS = 6
    MOMENTUM = 0.937
    WEIGHT_DECAY = 5e-4
    EPOCHS_SPTBACKBONE = 100
    EPOCHS_DAM = 100
    EVAL_EVERY = 2
    # 0 = track "best AP50" from the first eval. Previously 40, on the assumption
    # that an early spike is a fluke -- but both Base and R0 showed a genuine
    # good early epoch (~epoch 4-8) get overwritten by mid-training instability
    # with no checkpoint saved to fall back on. Track from the start so a good
    # early result is never silently lost again.
    TRACK_BEST_AFTER = 0

    STRIDES = [8, 16, 32]
    NUM_CLASSES = 1

    # ---- dataset paths ---------------------------------------------------- #
    # Override from the environment (Colab / CI). MOCID_DATASET selects a
    # built-in split pair under MOCID_DATA_ROOT; MOCID_TRAIN_PATH /
    # MOCID_VAL_PATH override the annotation files directly. Default is DAUB
    # (the reproduction targets the DAUB ablation ladder first).
    DATASET = os.environ.get("MOCID_DATASET", "DAUB").upper()
    DATA_ROOT = os.environ.get("MOCID_DATA_ROOT", "../../datasets")

    _SPLITS = {
        "DAUB": ("DAUB_mocid/train_DAUB.txt", "DAUB_mocid/val_DAUB.txt"),
        "IRDST": ("IRDST_mocid/train_IRDST.txt", "IRDST_mocid/val_IRDST.txt"),
    }
    assert DATASET in _SPLITS, (
        f"MOCID_DATASET must be one of {list(_SPLITS)}, got {DATASET!r}"
    )

    train_path = os.environ.get(
        "MOCID_TRAIN_PATH", os.path.join(DATA_ROOT, _SPLITS[DATASET][0])
    )
    val_path = os.environ.get(
        "MOCID_VAL_PATH", os.path.join(DATA_ROOT, _SPLITS[DATASET][1])
    )


def get_device():
    """-> torch.device, cuda when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_torch():
    """Global torch settings applied once at process start. -> None."""
    torch.set_float32_matmul_precision("high")


# VMamba checkout that provides the selective-scan kernel used by dam.py
VMAMBA_PATH = os.environ.get("VMAMBA_PATH", "/content/VMamba")
