import torch
import torch.nn as nn

try:
    from detectors.registry import register_module
except ImportError:  # outside the mamba-experiments harness (e.g. main.py): nothing to register with
    def register_module(cls):
        return cls

from components.components import FPN, FISTALayer, YOLOXHead
from components.yolox.darknet import CSPDarknet
from utils.losses import YOLOLoss


class ClipFISTALayer(nn.Module):
    """
    FISTALayer on frame-folded input: (B*T,C,H,W) -> (B*T,C,H,W).

    CSPDarknet runs every frame as a separate image, with the frames folded into the batch.
    FISTA attends across time, so this unfolds the frame axis, applies the layer and folds it
    back. That lets it sit inside a CSPDarknet stage without changing CSPDarknet's forward.
    """

    def __init__(self, channels, frames, height, width, n_blocks):
        super().__init__()
        self.frames = frames
        self.layer = FISTALayer(channels, frames, height, width, n_blocks)

    def forward(self, x):
        BT, C, H, W = x.shape
        y = self.layer(x.view(BT // self.frames, self.frames, C, H, W))
        return y.reshape(BT, C, H, W)


@register_module
class MOCIDBaseFISTA(nn.Module):
    """
    MOCID ablation 2, Base + FISTA: the paper's spatio-temporal backbone with an FPN and the
    YOLOX head, and no DAM. Sized to the paper's "+FISTA" row (Table 2, 9.45 M parameters).

    Backbone: the paper retains "the first two layers of CSPDarknet and replace[s] the last three
    spatial layers with three FISTA layers". Here the first two layers are YOLOX-S's Focus stem
    and dark2, shared with MOCIDBase (base.py). Each of dark3, dark4 and dark5 keeps its stride-2
    downsampling conv, and the rest of the stage (the CSP stack, and in dark5 also the SPP block)
    is replaced by a FISTA layer of the same width.

    Neck: a plain top-down FPN, as the paper specifies for MOCID ("Feature Pyramid Network (Lin et
    al. 2017)"). MOCIDBase uses YOLOX's PAFPN, so the neck changes between the two ablation rows,
    as it does in the paper: its Base is stock YOLOX-S (8.94 M), and its 9.45 M for +FISTA is only
    reachable with the plain FPN and without SPP.

    Every frame of the clip runs through the backbone. At dark3, dark4 and dark5 the FISTA layer
    mixes information across the frames, and the enhanced features of all frames feed the next
    stage, so temporal context accumulates as in the paper. Only the target frame's features go
    on to the FPN and head, as F_T does in the paper's Fig. 2.

    Arguments:
    - num_classes (int): number of foreground classes.
    - num_frames (int): number of frames T in each input clip (references + target).
    - img_size (int): size of the input frames (assumes a square shape). The FISTA spectral
      filters are sized to it, so inputs must match.
    - depth (float): YOLOX depth multiplier for the retained layers; 0.33 for YOLOX-S.
    - width (float): YOLOX width multiplier; 0.50 for YOLOX-S.
    - n_blocks (tuple of int): FISTA blocks per stage (dark3, dark4, dark5); MOCID uses (4, 4, 1).
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_frames: int = 5,
        img_size: int = 512,
        depth: float = 0.33,
        width: float = 0.50,
        n_blocks: tuple = (4, 4, 1),
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.img_size = img_size
        self.depth = depth
        self.width = width
        self.n_blocks = tuple(n_blocks)

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """
        Build the YOLOX-S CSPDarknet, replace everything after the downsampling conv in dark3,
        dark4 and dark5 with a FISTA layer, then add the FPN, detection head and loss.
        """
        self.backbone = CSPDarknet(self.depth, self.width)
        base = int(self.width * 64)
        ch = [base * 4, base * 8, base * 16]  # dark3/4/5 widths: 128/256/512 for YOLOX-S

        for name, c, stride, n in zip(("dark3", "dark4", "dark5"), ch, (8, 16, 32), self.n_blocks):
            stage = getattr(self.backbone, name)
            hw = self.img_size // stride
            downsample = stage[0]  # the stage's stride-2 BaseConv
            setattr(
                self.backbone,
                name,
                nn.Sequential(downsample, ClipFISTALayer(c, self.num_frames, hw, hw, n)),
            )

        self.fpn = FPN(ch)
        self.head = YOLOXHead(
            self.num_classes, width=self.width, in_channels=[256, 512, 1024]
        )
        self.loss_fn = YOLOLoss(self.num_classes, fp16=False, strides=[8, 16, 32])

    def _init_weights(self):
        """
        YOLOX's init_yolo: BatchNorm eps 1e-3 and momentum 0.03 in the backbone. The FPN and head
        BaseConvs already use these, and the FISTA layers' own parameters (spectral filters, base
        kernels) keep their initialisation.
        """
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03

    def forward(self, clip, labels=None, use_dam=False):
        """
        Forward function to evaluate the Base + FISTA output.

        clip (B,T,3,H,W), target frame last -> raw head outputs, or the scalar loss when labels
        are given. `use_dam` is accepted so MOCID's training loop can call this model unchanged;
        there is no DAM here.
        """
        B, T = clip.shape[:2]
        assert T == self.num_frames, f"clip has {T} frames, model built for {self.num_frames}"

        feats = self.backbone(clip.flatten(0, 1))  # all frames, folded into the batch
        target = [
            feats[f].view(B, T, *feats[f].shape[1:])[:, -1]
            for f in ("dark3", "dark4", "dark5")
        ]
        outs = self.head(self.fpn(target))

        if labels is not None:
            with torch.amp.autocast("cuda", enabled=False):  # loss in fp32
                return self.loss_fn([o.float() for o in outs], labels)
        return outs
