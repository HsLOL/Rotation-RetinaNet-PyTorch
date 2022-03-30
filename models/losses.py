import torch.nn as nn
from utils.utils import bbox_overlaps
from utils.bbox_transforms import *
from utils.box_coder import BoxCoder
from utils.rotation_overlaps.rbox_overlaps import rbox_overlaps
import matplotlib.pyplot as plt


class IntegratedLoss(nn.Module):
    def __init__(self, params):
        super(IntegratedLoss, self).__init__()
        loss_dict = params.loss
        self.alpha = loss_dict['cls']['alpha']
        self.gamma = loss_dict['cls']['gamma']
        func = loss_dict['reg']['type']

        assign_dict = params.assigner
        self.pos_iou_thr = assign_dict['pos_iou_thr']
        self.neg_iou_thr = assign_dict['neg_iou_thr']
        self.min_pos_iou = assign_dict['min_pos_iou']
        self.low_quality_match = assign_dict['low_quality_match']

        self.box_coder = BoxCoder()

        if func == 'smooth':
            self.criteron = smooth_l1_loss
            print(f'[Info]: Using {func} Loss.')

    def forward(self, classifications, regressions, anchors, annotations, image_names):
        cls_losses = []
        reg_losses = []
        batch_size = classifications.shape[0]
        device = classifications[0].device
        for j in range(batch_size):
            image_name = image_names[j]
            anchor = anchors[j]  # [xc, yc, w, h, angle(radian)]
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]  # [xc_offset, yc_offset, h_offset, w_offset, angle_offset]
            bbox_annotation = annotations[j, :, :]  # [xc, yc, h, w, angle(radian)]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]
            num_gt = len(bbox_annotation)
            if bbox_annotation.shape[0] == 0:
                cls_losses.append(torch.tensor(0).float().cuda(device=device))
                reg_losses.append(torch.tensor(0).float().cuda(device=device))
                continue
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            # get minimum circumscribed rectangle of the rotated ground-truth box and
            # calculate the horizontal overlaps between minimum circumscribed rectangles and anchor boxes

            horizontal_overlaps = bbox_overlaps(
                anchor.clone(),  # generate anchor data copy
                obb2hbb_oc(bbox_annotation[:, :-1]))

            # obb_rect = [xc, yc, h, w, angle(radian)]
            ious = rbox_overlaps(
                swap_axis(anchor[:, :]).cpu().numpy(),
                bbox_annotation[:, :-1].cpu().numpy(),
                horizontal_overlaps.cpu().numpy(),
                thresh=1e-1
            )

            if not torch.is_tensor(ious):
                ious = torch.from_numpy(ious).cuda(device=device)

            iou_max, iou_argmax = torch.max(ious, dim=1)

            positive_indices = torch.ge(iou_max, self.pos_iou_thr)

            if self.low_quality_match is True:
                max_gt, argmax_gt = ious.max(dim=0)
                for idx in range(num_gt):
                    if max_gt[idx] >= self.min_pos_iou:
                        positive_indices[argmax_gt[idx]] = 1

            # calculate classification loss
            cls_targets = (torch.ones(classification.shape) * -1).cuda(device=device)
            cls_targets[torch.lt(iou_max, self.neg_iou_thr), :] = 0
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[iou_argmax, :]
            cls_targets[positive_indices, :] = 0
            cls_targets[positive_indices, assigned_annotations[positive_indices, 5].long()] = 1
            alpha_factor = torch.ones(cls_targets.shape).cuda(device=device) * self.alpha
            alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            # bin_cross_entropy = -(cls_targets * torch.log(classification + 1e-6) + (1.0 - cls_targets) * torch.log(
            #     1.0 - classification + 1e-6))
            bin_cross_entropy = -(cls_targets * torch.log(classification) + (1.0 - cls_targets) * torch.log(
                1.0 - classification))
            cls_loss = focal_weight * bin_cross_entropy
            cls_loss = torch.where(torch.ne(cls_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda(device=device))
            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # calculate regression loss
            if positive_indices.sum() > 0:
                all_rois = anchor[positive_indices, :]
                gt_boxes = assigned_annotations[positive_indices, :]
                reg_targets = self.box_coder.encode(all_rois, gt_boxes)
                reg_loss = self.criteron(regression[positive_indices, :], reg_targets)
                reg_losses.append(reg_loss)
            else:
                reg_losses.append(torch.tensor(0).float().cuda(device=device))
        loss_cls = torch.stack(cls_losses).mean(dim=0, keepdim=True)
        loss_reg = torch.stack(reg_losses).mean(dim=0, keepdim=True)
        return loss_cls, loss_reg


def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True,
                   weight=None):
    """https://github.com/facebookresearch/maskrcnn-benchmark"""
    diff = torch.abs(inputs - targets)
    if weight is None:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
    else:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        ) * weight.max(1)[0].unsqueeze(1).repeat(1,5)
    if size_average:
        return loss.mean()
    return loss.sum()
