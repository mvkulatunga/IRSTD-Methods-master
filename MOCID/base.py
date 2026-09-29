import torch
import torch.nn as nn

from detectors.registry import register_module

from components.components import YOLOXHead
from components.yolox.yolo_pafpn import YOLOPAFPN
from utils.losses import YOLOLoss


@register_module
class MOCIDBase(nn.Module):
    """
    MOCID ablation 1, Base: the spatial-only YOLOX-S detector, with no FISTA and no DAM.

    CSPDarknet (YOLOX-S) -> PAFPN -> YOLOX head, run on the target frame only. This is the
    paper's "Base" row (Table 2, 8.94 M parameters) and the model behind
    results/Base-fixedwd-imagenet (88.83 AP50); checkpoints from that run load into it with
    strict=True. MOCIDBaseFISTA (base_fista.py) is this model with its last three CSP stages
    replaced by FISTA layers.

    It takes the same clip input as MOCID, (B,T,3,H,W) with the target frame last, and uses
    only the target frame, so all the ablation models can share one data pipeline. A single
    frame (B,3,H,W) is also accepted.

    Arguments:
    - num_classes (int): number of foreground classes.
    - num_frames (int): accepted for config compatibility with MOCID; unused.
    - img_size (int): accepted for config compatibility with MOCID; unused (fully convolutional).
    - depth (float): YOLOX depth multiplier; 0.33 for YOLOX-S.
    - width (float): YOLOX width multiplier; 0.50 for YOLOX-S.
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_frames: int = 5,
        img_size: int = 512,
        depth: float = 0.33,
        width: float = 0.50,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.img_size = img_size
        self.depth = depth
        self.width = width

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """
        Initialise the YOLOX-S backbone + PAFPN (as one module, `neck`, whose `.backbone` is the
        CSPDarknet), the detection head and the loss.
        """
        self.neck = YOLOPAFPN(depth=self.depth, width=self.width)
        self.head = YOLOXHead(
            self.num_classes, width=self.width, in_channels=[256, 512, 1024]
        )
        self.loss_fn = YOLOLoss(self.num_classes, fp16=False, strides=[8, 16, 32])

    def _init_weights(self):
        """
        YOLOX's init_yolo: BatchNorm eps 1e-3 and momentum 0.03 in the backbone and neck. The
        head's BaseConvs already use these, and YOLOXHead sets its own bias priors.
        """
        for m in self.neck.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03

    def forward(self, clip, labels=None, use_dam=False):
        """
        Forward function to evaluate the Base output.

        clip (B,T,3,H,W) or (B,3,H,W) -> raw head outputs, or the scalar loss when labels are
        given. `use_dam` is accepted so MOCID's training loop can call this model unchanged;
        there is no DAM here.
        """
        x = clip[:, -1] if clip.dim() == 5 else clip  # target frame
        outs = self.head(self.neck(x))

        if labels is not None:
            with torch.amp.autocast("cuda", enabled=False):  # loss in fp32
                return self.loss_fn([o.float() for o in outs], labels)
        return outs
