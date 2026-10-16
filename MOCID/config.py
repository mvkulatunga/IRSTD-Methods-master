import os
import sys

import torch


# ---- lab server defaults -------------------------------------------------- #
# On the lab server (pl-lawr7615) the dependencies, DAUB splits and a place for run
# outputs are in shared folders, so main.py runs with no environment set up. On any
# other machine these folders don't exist and the previous defaults apply; environment
# variables (MOCID_TRAIN_PATH, MOCID_VAL_PATH, VMAMBA_PATH, PYTHONPATH) override both.
SERVER_DIR = "/srv/proj-mamba/mocid-baseline"
SERVER_RUNS = "/srv/proj-mamba/runs"
ON_SERVER = os.path.isdir(SERVER_DIR)
_SERVER_SPLITS = {
    "DAUB": (
        f"{SERVER_DIR}/splits/daub_train_server.txt",
        f"{SERVER_DIR}/splits/daub_val_server.txt",
    ),
}


# Named settings profiles, selected with `main.py --profile <name>` or MOCID_PROFILE=<name>.
# Each entry overrides the Config defaults below. Several can be combined, applied left to
# right: `--profile r0,wdfix,freeze`. Dict values (DAM_OPTS) are merged, not replaced.
PROFILES = {
    # R0: the settings of the team's first full two-stage run (commit b82300f,
    # results/R0), which the team uses as the reference for new work. Differs from
    # the defaults in the unscaled LR, /255-only input, weight decay on every
    # parameter, and best-checkpoint tracking from epoch 40 in stage 1.
    "r0": {
        "LR_INIT": 0.01,
        "MIN_LR": 1e-4,
        "TRACK_BEST_AFTER": 40,
        "NORMALISE": "255",
        "DECAY_ALL": True,
        # not an R0 setting: evaluation-only, and part of the team standard since 1 Oct 2026
        "BN_RECAL_BATCHES": 300,
    },
    # Add-ons, for use after r0. wdfix: no weight decay on 1-D parameters or on parameters
    # marked _no_weight_decay (the fix from results/Base-fixedwd-imagenet), keeping R0's
    # /255 input so R0's stage-1 checkpoint still fits.
    "wdfix": {"DECAY_ALL": False},
    # freeze: stage 2 trains the DAM only, as the paper describes
    "freeze": {"STAGE2_TRAIN_NECK_HEAD": False},
    # MambaIR DAM options (components/dam.py DAMBlock). A checkpoint must be loaded with
    # the same ones it was trained with.
    "ln": {"DAM_OPTS": {"norm": "layer"}},  # per-pixel LayerNorm, as MambaIR
    "dtbias": {"DAM_OPTS": {"dt_bias": True}},  # Mamba-initialised delta bias
    "noout": {"DAM_OPTS": {"out_proj": False}},  # no final projection, as MambaIR
    "cabstd": {"DAM_OPTS": {"cab": "standard"}},  # MambaIR's standard CAB (+~2.1 M)
}


class Config:
    T = 5
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 4

    # LR_INIT follows SSTNet's linear batch-size scaling rule (which the paper's
    # split/training convention follows): base 0.01 is defined for batch 64, so
    # at our batch of 4 that's 0.01 * 4/64 = 6.25e-4. Kept for fidelity to SSTNet,
    # not as a fix: 0.01 unscaled trains stably once weight decay is scoped
    # correctly (EXPERIMENTS.md, Base-fixedwd-imagenet). MIN_LR is scaled by the
    # same factor so the warmup/cosine shape (in ratio terms) is unchanged.
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
    # input normalisation: "imagenet" = /255 then ImageNet mean/std; "255" = /255 only
    NORMALISE = "imagenet"
    # False = no weight decay on 1-D parameters (BatchNorm, biases) or on parameters marked
    # _no_weight_decay (Mamba's A_log/D, the DAM's identity-initialised out); True = decay all
    DECAY_ALL = False
    # stage 2: also train the FPN and head (True, the repo's original behaviour) or the DAM only
    STAGE2_TRAIN_NECK_HEAD = True
    # keyword options for the DAM blocks (components/dam.py); {} = the default block
    DAM_OPTS = {}
    # before every evaluation, recompute the evaluated (EMA) model's BatchNorm statistics
    # from this many training batches (utils.recalibrate_bn); 0 = off. Evaluation only:
    # training is unaffected. Without it, R0-profile runs can show a false collapse.
    BN_RECAL_BATCHES = 300

    STRIDES = [8, 16, 32]
    NUM_CLASSES = 1

    # ---- dataset paths ---------------------------------------------------- #
    # Override from the environment (Colab / CI). MOCID_DATASET selects a
    # built-in split pair under MOCID_DATA_ROOT; MOCID_TRAIN_PATH /
    # MOCID_VAL_PATH override the annotation files directly. Default is DAUB
    # (the reproduction targets the DAUB ablation ladder first); on the lab server
    # the default is the shared DAUB split, unless MOCID_DATA_ROOT is set.
    DATASET = os.environ.get("MOCID_DATASET", "DAUB").upper()
    DATA_ROOT = os.environ.get("MOCID_DATA_ROOT", "../../datasets")

    _SPLITS = {
        "DAUB": ("DAUB_mocid/train_DAUB.txt", "DAUB_mocid/val_DAUB.txt"),
        "IRDST": ("IRDST_mocid/train_IRDST.txt", "IRDST_mocid/val_IRDST.txt"),
    }
    assert DATASET in _SPLITS, (
        f"MOCID_DATASET must be one of {list(_SPLITS)}, got {DATASET!r}"
    )
    if ON_SERVER and DATASET in _SERVER_SPLITS and "MOCID_DATA_ROOT" not in os.environ:
        _default_splits = _SERVER_SPLITS[DATASET]
    else:
        # no generator here: inside a class body it can't see DATA_ROOT (NameError)
        _default_splits = (
            os.path.join(DATA_ROOT, _SPLITS[DATASET][0]),
            os.path.join(DATA_ROOT, _SPLITS[DATASET][1]),
        )

    train_path = os.environ.get("MOCID_TRAIN_PATH", _default_splits[0])
    val_path = os.environ.get("MOCID_VAL_PATH", _default_splits[1])

    def __init__(self, profile=None):
        """profile: comma-separated keys of PROFILES, or None for MOCID_PROFILE (default: none)."""
        self.PROFILE = profile or os.environ.get("MOCID_PROFILE", "default")
        if self.PROFILE == "default":
            return
        for name in self.PROFILE.split(","):
            if name not in PROFILES:
                raise ValueError(f"unknown profile {name!r}; choose from {list(PROFILES)}")
            for k, v in PROFILES[name].items():
                if isinstance(v, dict):
                    v = {**getattr(self, k), **v}
                setattr(self, k, v)


def get_device():
    """-> torch.device, cuda when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_torch():
    """Global torch settings applied once at process start. -> None."""
    torch.set_float32_matmul_precision("high")


# VMamba checkout that provides the selective-scan kernel used by dam.py
VMAMBA_PATH = os.environ.get(
    "VMAMBA_PATH", f"{SERVER_DIR}/deps/VMamba" if ON_SERVER else "/content/VMamba"
)


def add_dependency_paths():
    """Make VMamba importable for dam.py, and on the lab server also the prebuilt
    selective-scan CUDA kernel and fvcore (which VMamba imports). -> None."""
    paths = [VMAMBA_PATH]
    if ON_SERVER:
        paths = [f"{SERVER_DIR}/deps/ss-kernel", f"{SERVER_DIR}/deps/python-deps"] + paths
    for p in paths:
        if p not in sys.path:
            sys.path.append(p)
