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

    train_path = "../../datasets/IRDST_mocid/train_IRDST.txt"
    val_path = "../../datasets/IRDST_mocid/val_IRDST.txt"


def get_device():
    """-> torch.device, cuda when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_torch():
    """Global torch settings applied once at process start. -> None."""
    torch.set_float32_matmul_precision("high")


# VMamba checkout that provides the selective-scan kernel used by dam.py
VMAMBA_PATH = os.environ.get(
    "VMAMBA_PATH", "/home/thor/Programming/IRSTD/methods/VMamba"
)
