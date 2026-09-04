import os

import torch


class Config:
    T = 5
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 4

    LR_INIT = 0.01
    MIN_LR = 1e-4
    LR_DAM = 1e-3  # stage-2 LR; 1e-2 diverges a fresh Mamba branch
    WARMUP_EPOCHS = 6
    MOMENTUM = 0.937
    WEIGHT_DECAY = 5e-4
    EPOCHS_SPTBACKBONE = 100
    EPOCHS_DAM = 100
    EVAL_EVERY = 2
    TRACK_BEST_AFTER = 40

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
