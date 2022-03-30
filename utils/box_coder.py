import numpy as np
import torch


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.

    Args:
        encode() function:
            ex_rois: positive anchors: [xc, yc, w, h, angle(radian)]
            gt_rois: positive anchor ground-truth box: [xc, yc, h, w, angle(radian)]

        decode() function:
            boxes: anchors: [xc, yc, w, h, angle(radian)]
            deltas: offset: [xc_offset, yc_offset, h_offset, w_offset, angle_offset(radian)]
    """
    def __init__(self, means=(0., 0., 0., 0., 0.), stds=(0.1, 0.1, 0.1, 0.1, 0.05)):
        self.means = means
        self.stds = stds

    def encode(self, ex_rois, gt_rois):
        ex_widths = ex_rois[:, 2]
        ex_heights = ex_rois[:, 3]
        ex_widths = torch.clamp(ex_widths, min=1)
        ex_heights = torch.clamp(ex_heights, min=1)
        ex_ctr_x = ex_rois[:, 0]
        ex_ctr_y = ex_rois[:, 1]
        ex_thetas = ex_rois[:, 4]

        gt_widths = gt_rois[:, 3]
        gt_heights = gt_rois[:, 2]
        gt_widths = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)
        gt_ctr_x = gt_rois[:, 0]
        gt_ctr_y = gt_rois[:, 1]
        gt_thetas = gt_rois[:, 4]

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths  # t_x = (x - x_a) / w_a
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights  # t_y = (y - y_a) / h_a
        targets_dw = torch.log(gt_widths / ex_widths)  # t_w = log(w / w_a)
        targets_dh = torch.log(gt_heights / ex_heights)  # t_h = log(h / h_a)
        targets_dt = gt_thetas - ex_thetas

        targets = torch.stack(
            (targets_dx, targets_dy, targets_dh, targets_dw, targets_dt), dim=1)

        means = targets.new_tensor(self.means).unsqueeze(0)
        stds = targets.new_tensor(self.stds).unsqueeze(0)
        targets = targets.sub_(means).div_(stds)
        return targets

    def decode(self, boxes, deltas):
        means = deltas.new_tensor(self.means).view(1, 1, -1).repeat(1, deltas.size(1), 1)
        stds = deltas.new_tensor(self.stds).view(1, 1, -1).repeat(1, deltas.size(1), 1)
        denorm_deltas = deltas * stds + means

        dx = denorm_deltas[:, :, 0]
        dy = denorm_deltas[:, :, 1]
        dh = denorm_deltas[:, :, 2]
        dw = denorm_deltas[:, :, 3]
        dt = denorm_deltas[:, :, 4]

        widths = boxes[:, :, 2]
        heights = boxes[:, :, 3]
        widths = torch.clamp(widths, min=1)
        heights = torch.clamp(heights, min=1)
        ctr_x = boxes[:, :, 0]
        ctr_y = boxes[:, :, 1]
        thetas = boxes[:, :, 4]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        pred_t = thetas + dt

        pred_boxes = torch.stack([
            pred_ctr_x,
            pred_ctr_y,
            pred_h,
            pred_w,
            pred_t], dim=2)
        return pred_boxes
