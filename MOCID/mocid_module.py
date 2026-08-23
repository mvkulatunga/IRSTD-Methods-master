import torch
import torch.nn as nn

from detectors.registry import register_module

from components.components import (
    FPN,
    SpatioTemporalBackbone,
    TemporalPooling,
    YOLOXHead,
)
from components.dam import DisplacementNet
from utils.losses import YOLOLoss


@register_module
class MOCID(nn.Module):
    """
    MOCID: motion-compensated infrared small target detector.

    A CSPDarknet stem with FISTA-based spatio-temporal stages (SpatioTemporalBackbone)
    extracts per-frame features from a clip; a Displacement-Aware Mamba module (DAM)
    aligns reference frames to the target frame, temporal max-pooling collapses the
    clip, and an FPN + YOLOX head produce per-scale box/obj/cls predictions.

    Arguments:
    - num_classes (int): number of foreground classes.
    - num_frames (int): number of frames T in each input clip (references + target).
    - img_size (int): size of the input frames provided to the model (assumes a square shape).
    - base_channels (int): channel multiplier for the CSPDarknet stem/FISTA stages.
    - d_state (int): SSM state dimension used by the DAM's selective scan.
    - expand (int): channel expansion factor inside each DAM block.
    - theta (float): central-difference weighting used by the DAM's 3D-CDC.
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_frames: int = 5,
        img_size: int = 512,
        base_channels: int = 16,
        d_state: int = 32,
        expand: int = 1,
        theta: float = 0.7,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.img_size = img_size
        self.base_channels = base_channels
        self.d_state = d_state
        self.expand = expand
        self.theta = theta

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """
        Initialise the backbone, DAM, temporal pooling, FPN, detection head and loss.
        """
        ch = [self.base_channels * 8, self.base_channels * 16, self.base_channels * 32]

        self.backbone = SpatioTemporalBackbone(
            3, self.base_channels, self.num_frames, self.img_size
        )
        self.pool = TemporalPooling()
        self.disp = DisplacementNet(
            ch, d_state=self.d_state, expand=self.expand, theta=self.theta
        )
        self.fpn = FPN(ch)
        # width=0.5 halves the doubled in_channels (target + pooled motion) back to ch
        self.head = YOLOXHead(
            self.num_classes, width=0.5, in_channels=[c * 2 for c in ch]
        )
        self.loss_fn = YOLOLoss(self.num_classes, fp16=False, strides=[8, 16, 32])

    def _init_weights(self):
        """
        No-op: DAMBlock, YOLOXHead and TIDS each set their own weights in __init__
        (zero-init residual paths, YOLOX bias priors, log-spaced A_log for the Mamba
        scan) and a blanket re-init here would overwrite and destabilise them.
        """
        pass

    def forward(self, clip, labels=None, use_dam=True):
        """
        Forward function to evaluate the MOCID output.

        clip (B,T,3,H,W) -> raw head outputs, or the scalar loss when labels are given.
        """
        Ft, Fr_list = self.backbone(clip)

        # rebuild per-scale (B,T,C,H,W) volumes with the target frame last
        feats_by_scale = [
            torch.cat([Fr_list[k], Ft[k].unsqueeze(1)], dim=1) for k in range(3)
        ]
        F_T = Ft

        if use_dam:
            with torch.autocast("cuda", enabled=False):  # scan is fp16-unstable
                motion_features = self.disp([f.float() for f in feats_by_scale])
        else:
            motion_features = feats_by_scale

        F_f = self.pool(motion_features)
        for t in F_f:
            assert torch.isfinite(t).all(), "non-finite in F_f (DAM output)"

        outs = self.head(self.fpn(F_T, F_f))

        if labels is not None:
            with torch.amp.autocast("cuda", enabled=False):  # loss in fp32
                return self.loss_fn([o.float() for o in outs], labels)
        return outs
