from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    """IoU / GIoU / CIoU regression loss on cxcywh boxes."""

    def __init__(self, reduction="none", loss_type="iou"):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """pred, target (N,4) cxcywh -> (N,) loss, or scalar if reduced."""
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # intersection over union on the corner-converted boxes
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=1)  # 0 when boxes are disjoint
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            # penalise by the empty area of the smallest enclosing box
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "ciou":
            # GIoU plus a center-distance term and an aspect-ratio term
            center_distance = torch.sum(
                torch.pow((pred[:, :2] - target[:, :2]), 2), axis=-1
            )
            enclose_mins = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            enclose_maxes = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(br))
            enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
            ciou = iou - 1.0 * center_distance / torch.clamp(enclose_diagonal, min=1e-6)
            v = (4 / (torch.pi**2)) * torch.pow(
                (
                    torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-6))
                    - torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-6))
                ),
                2,
            )
            alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
            loss = 1 - (ciou - alpha * v).clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class YOLOLoss(nn.Module):
    """YOLOX loss: SimOTA label assignment + IoU / obj / cls terms."""

    def __init__(self, num_classes, fp16, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(strides)  # cached per-scale cell grids
        self.fp16 = fp16

    @torch.compiler.disable
    def forward(self, inputs, labels=None):
        """inputs: 3 x (B,C,H,W) raw grids, labels: list of (n_gt,5) -> scalar loss."""
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []

        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        return self.get_losses(
            x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1)
        )

    def get_output_and_grid(self, output, k, stride):
        """(B,C,H,W) raw grid -> ((B,HW,C) pixel-space preds, (1,HW,2) cell grid)."""
        grid = self.grids[k]
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:  # rebuild only on shape change
            yv, xv = torch.meshgrid(
                [torch.arange(hsize), torch.arange(wsize)], indexing="ij"
            )
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k] = grid
        grid = grid.view(1, -1, 2)

        output = output.flatten(start_dim=2).permute(0, 2, 1)

        # cell offset + grid position -> pixels; wh clamped before exp to avoid inf
        xy = (output[..., :2] + grid.type_as(output)) * stride
        wh = torch.exp(torch.clamp(output[..., 2:4], max=20.0)) * stride
        rest = output[..., 4:]
        return torch.cat([xy, wh, rest], dim=-1), grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        """Assign GT to anchors and reduce to a single loss. -> scalar tensor."""
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1).type_as(outputs)
        y_shifts = torch.cat(y_shifts, 1).type_as(outputs)
        expanded_strides = torch.cat(expanded_strides, 1).type_as(outputs)

        cls_targets, reg_targets, obj_targets, fg_masks = [], [], [], []
        num_fg = 0.0

        # assignment is per image: SimOTA needs each image's own GT set
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])
            if num_gt == 0:  # background-only image
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx][..., :4].type_as(outputs)
                gt_classes = labels[batch_idx][..., 4].type_as(outputs)

                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bbox_preds[batch_idx],
                    cls_preds[batch_idx],
                    obj_preds[batch_idx],
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                )
                num_fg += num_fg_img

                # cls target is soft: one-hot scaled by the matched IoU
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ).float() * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        # obj is scored over all anchors; iou/cls only over the foreground ones
        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum()
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        self.last_parts = (
            float(reg_weight * loss_iou / num_fg),
            float(loss_obj / num_fg),
            float(loss_cls / num_fg),
            float(num_fg),
        )
        return loss / num_fg

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        cls_preds_per_image,
        obj_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
    ):
        """SimOTA matching -> (matched classes, fg_mask, matched ious, gt inds, num_fg)."""
        # stage 1: geometric prefilter (inside a GT box or its center region)
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if num_in_boxes_anchor == 0:  # nothing survived the prefilter
            return (
                gt_classes.new_zeros((0,), dtype=torch.long),
                gt_bboxes_per_image.new_zeros((total_num_anchors,), dtype=torch.bool),
                gt_bboxes_per_image.new_zeros((0,), dtype=torch.float),
                gt_classes.new_zeros((0,), dtype=torch.long),
                0,
            )

        pair_wise_ious = self.bboxes_iou(
            gt_bboxes_per_image, bboxes_preds_per_image, False
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # stage 2: classification cost; forced to fp32 because BCE underflows in fp16
        with torch.cuda.amp.autocast(enabled=False) if self.fp16 else nullcontext():
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
                * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid()
            )
            gt_cls_per_image = (
                F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
            )
            _cls = torch.nan_to_num(cls_preds_.sqrt_(), nan=0.0).clamp_(0.0, 1.0)
            pair_wise_cls_loss = F.binary_cross_entropy(
                _cls, gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        # anchors outside the center region are made prohibitively expensive
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center).float()
        )

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = (
            self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        )
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        """(Na,4), (Nb,4) -> (Na,Nb) IoU; xyxy=False treats inputs as cxcywh."""
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )
            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        center_radius=2.5,
    ):
        """-> (anchors kept (A,) bool, in-box AND in-center mask (num_gt, kept) bool)."""
        expanded_strides_per_image = expanded_strides[0]

        # anchor centers in pixel space
        x_centers_per_image = (
            ((x_shifts[0] + 0.5) * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        y_centers_per_image = (
            ((y_shifts[0] + 0.5) * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        # test 1: anchor center falls inside the GT box
        gt_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        bbox_deltas = torch.stack(
            [
                x_centers_per_image - gt_l,
                y_centers_per_image - gt_t,
                gt_r - x_centers_per_image,
                gt_b - y_centers_per_image,
            ],
            2,
        )
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # test 2: anchor center falls in a fixed-radius square around the GT center
        c_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        c_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        c_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        c_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        center_deltas = torch.stack(
            [
                x_centers_per_image - c_l,
                y_centers_per_image - c_t,
                c_r - x_centers_per_image,
                c_b - y_centers_per_image,
            ],
            2,
        )
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # keep anchors passing either test; the AND mask drives the cost penalty
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """-> (num_fg, matched classes, matched ious, matched gt indices)."""
        matching_matrix = torch.zeros_like(cost)

        # each GT takes k anchors, where k = sum of its top-10 IoUs
        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(
            topk_ious.sum(1).int(), min=3, max=pair_wise_ious.size(1)
        )
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        # an anchor claimed by several GTs goes to the cheapest one
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # narrow the prefilter mask

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
