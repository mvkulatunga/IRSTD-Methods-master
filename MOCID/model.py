import torch
import torch.nn as nn

from components.components import (
    FPN,
    SpatioTemporalBackbone,
    TemporalPooling,
    YOLOXHead,
)
from components.dam import DisplacementNet
from utils.losses import YOLOLoss


class MOCID(nn.Module):
    """Backbone -> DAM -> temporal pooling -> FPN -> YOLOX head."""

    def __init__(
        self, num_classes=1, num_frames=5, img_size=512, base_channels=16, d_state=32
    ):
        super().__init__()
        ch = [base_channels * 8, base_channels * 16, base_channels * 32]
        self.backbone = SpatioTemporalBackbone(3, base_channels, num_frames, img_size)
        self.pool = TemporalPooling()
        self.disp = DisplacementNet(ch, d_state=d_state, expand=1, theta=0.7)
        self.fpn = FPN(ch)
        # width=0.5 halves the doubled in_channels back to ch
        self.head = YOLOXHead(num_classes, width=0.5, in_channels=[c * 2 for c in ch])
        self.loss_fn = YOLOLoss(num_classes, fp16=False, strides=[8, 16, 32])

    def forward(self, clip, labels=None, use_dam=True):
        """clip (B,T,3,H,W) -> raw head outputs, or the scalar loss when labels are given."""
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
