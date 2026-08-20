import torch
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import bbox2dist


# ============================ CoordAtt ===================================
class CoordAtt(nn.Module):
    """Coordinate Attention (Hou et al., 2021) with lazy channel inference,
    so one YAML works at any width scale. Use `[]` as args in the YAML."""
    def __init__(self, c1=None, reduction=32):
        super().__init__()
        self.reduction = reduction
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.act = nn.Hardswish()
        self._built = False
        if c1 is not None:
            self._build(c1)

    def _build(self, c1):
        mip = max(8, c1 // self.reduction)
        self.conv1 = nn.Conv2d(c1, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, c1, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, c1, 1, bias=False)
        self._built = True

    def forward(self, x):
        if not self._built:
            self._build(x.shape[1])
            self.to(x.device, x.dtype)
        _, _, H, W = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_h * a_w


# ===================== NWD regression loss (Eqs 1-3) =====================
class NWDBboxLoss(BboxLoss):
    """Replaces CIoU with Normalized Gaussian Wasserstein Distance; DFL kept."""
    def __init__(self, reg_max, c_constant=17.0):
        super().__init__(reg_max)          # builds self.dfl_loss
        self.c = c_constant

    # FIX: Added *args, **kwargs to safely absorb extra positional arguments passed by newer Ultralytics versions
    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask, *args, **kwargs):
        
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        pb, tb = pred_bboxes[fg_mask], target_bboxes[fg_mask]          # xyxy
        cx1, cy1 = (pb[..., 0] + pb[..., 2]) / 2, (pb[..., 1] + pb[..., 3]) / 2
        w1,  h1  =  pb[..., 2] - pb[..., 0],       pb[..., 3] - pb[..., 1]
        cx2, cy2 = (tb[..., 0] + tb[..., 2]) / 2, (tb[..., 1] + tb[..., 3]) / 2
        w2,  h2  =  tb[..., 2] - tb[..., 0],       tb[..., 3] - tb[..., 1]

        loc = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
        shp = ((w1 - w2) / 2) ** 2 + ((h1 - h2) / 2) ** 2
        w2_dist = torch.clamp(loc + shp, min=0.0)                      # W2^2  (Eq 2)
        nwd = torch.exp(-torch.sqrt(w2_dist + 1e-7) / self.c)          # NWD   (Eq 3)
        loss_box = ((1.0 - nwd).unsqueeze(-1) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            tgt_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                     tgt_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)
            
        return loss_box, loss_dfl

class TYRISTDetectionLoss(v8DetectionLoss):
    """v8 detection loss with NWD box regression (C=17, Table 4)."""
    def __init__(self, model, c_constant=17.0):
        super().__init__(model)
        self.bbox_loss = NWDBboxLoss(self.reg_max, c_constant).to(self.device)