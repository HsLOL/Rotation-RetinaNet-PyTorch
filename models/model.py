import torch
import torch.nn as nn
from models.anchors import Anchors
from models.fpn import FPN, LastLevelP6_P7
from models import resnet
from models.heads import CLSBranch, REGBranch, CLSHead, REGHead
from models.losses import IntegratedLoss
from utils.utils import clip_boxes
from utils.box_coder import BoxCoder
from utils.rotation_nms.cpu_nms import cpu_nms
import math
import cv2
import numpy as np


class RetinaNet(nn.Module):
    def __init__(self, params):
        super(RetinaNet, self).__init__()
        self.num_class = len(params.classes)
        self.num_regress = 5
        self.anchor_generator = Anchors(params)
        self.num_anchors = self.anchor_generator.num_anchors
        self.pretrained = params.backbone['pretrained']
        self.init_backbone(params.backbone['type'])
        self.cls_branch_num_stacked = params.head['num_stacked']
        self.rotation_nms_thr = params.rotation_nms_thr
        self.score_thr = params.score_thr

        self.fpn = FPN(
            in_channel_list=self.fpn_in_channels,
            out_channels=256,
            top_blocks=LastLevelP6_P7(in_channels=256,
                                      out_channels=256,
                                      init_method=params.neck['extra_conv_init_method']),  # in_channels: 1) 2048 on C5, 2) 256 on P5
            init_method=params.neck['init_method'])

        self.cls_branch = CLSBranch(
            in_channels=256,
            feat_channels=256,
            num_stacked=self.cls_branch_num_stacked,
            init_method=params.head['cls_branch_init_method']
        )

        self.cls_head = CLSHead(
            feat_channels=256,
            num_anchors=self.num_anchors,
            num_classes=self.num_class
        )

        self.reg_branch = REGBranch(
            in_channels=256,
            feat_channels=256,
            num_stacked=self.cls_branch_num_stacked,
            init_method=params.head['reg_branch_init_method']
        )

        self.reg_head = REGHead(
            feat_channels=256,
            num_anchors=self.num_anchors,
            num_regress=self.num_regress  # x, y, w, h, angle
        )

        self.loss = IntegratedLoss(params)

        self.box_coder = BoxCoder()

    def init_backbone(self, backbone):
        if backbone == 'resnet34':
            print(f'[Info]: Use Backbone is {backbone}.')
            self.backbone = resnet.resnet34(pretrained=self.pretrained)
            self.fpn_in_channels = [128, 256, 512]

        elif backbone == 'resnet50':
            print(f'[Info]: Use Backbone is {backbone}.')
            self.backbone = resnet.resnet50(pretrained=self.pretrained)
            self.fpn_in_channels = [512, 1024, 2048]

        elif backbone == 'resnet101':
            print(f'[Info]: Use Backbone is {backbone}.')
            self.backbone = resnet.resnet101(pretrained=self.pretrained)
            self.fpn_in_channels = [512, 1024, 2048]

        elif backbone == 'resnet152':
            print(f'[Info]: Use Backbone is {backbone}.')
            self.backbone = resnet.resnet101(pretrained=self.pretrained)
            self.fpn_in_channels = [512, 1024, 2048]
        else:
            raise NotImplementedError

        del self.backbone.avgpool
        del self.backbone.fc

    def backbone_output(self, imgs):
        feature = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(imgs)))
        c2 = self.backbone.layer1(self.backbone.maxpool(feature))
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return [c3, c4, c5]

    def forward(self, images, annots=None, image_names=None, test_conf=None):
        anchors_list, offsets_list = [], []
        original_anchors, num_level_anchors = self.anchor_generator(images)
        anchors_list.append(original_anchors)

        features = self.fpn(self.backbone_output(images))

        cls_score = torch.cat([self.cls_head(self.cls_branch(feature)) for feature in features], dim=1)
        bbox_pred = torch.cat([self.reg_head(self.reg_branch(feature), with_deform=False)
                              for feature in features], dim=1)

        # get the predicted bboxes
        # predicted_boxes = torch.cat(
        #     [self.box_coder.decode(anchors_list[-1][index], bbox_pred[index]).unsqueeze(0)
        #      for index in range(len(bbox_pred))], dim=0).detach()

        if self.training:
            # Max IoU Assigner with Focal Loss and Smooth L1 loss
            loss_cls, loss_reg = self.loss(cls_score,  # cls_score with all levels
                                           bbox_pred,  # bbox_pred with all levels
                                           anchors_list[-1],
                                           annots,
                                           image_names)

            return loss_cls, loss_reg

        else:  # for model eval()
            return self.decoder(images, anchors_list[-1], cls_score, bbox_pred,
                                thresh=self.score_thr, nms_thresh=self.rotation_nms_thr, test_conf=test_conf)

    def decoder(self, ims, anchors, cls_score, bbox_pred,
                thresh=0.6, nms_thresh=0.1, test_conf=None):
        """
        Args:
            thresh: equal to score_thr.
            nms_thresh: nms_thr.
            test_conf: equal to thresh.
        """
        if test_conf is not None:
            thresh = test_conf
        bboxes = self.box_coder.decode(anchors, bbox_pred)  # bboxes: [pred_xc, pred_yc, pred_h, pred_w, pred_angle(radian)]
        # bboxes = clip_boxes(bboxes, ims)
        scores = torch.max(cls_score, dim=2, keepdim=True)[0]
        keep = (scores >= thresh)[0, :, 0]
        if keep.sum() == 0:
            return [torch.zeros(1), torch.zeros(1), torch.zeros(1, 5)]
        scores = scores[:, keep, :]
        anchors = anchors[:, keep, :]
        cls_score = cls_score[:, keep, :]
        bboxes = bboxes[:, keep, :]

        # NMS
        anchors_nms_idx = cpu_nms(torch.cat([bboxes, scores], dim=2)[0, :, :].cpu().detach().numpy(), nms_thresh)
        nms_scores, nms_class = cls_score[0, anchors_nms_idx, :].max(dim=1)
        output_boxes = torch.cat([
            bboxes[0, anchors_nms_idx, :],
            anchors[0, anchors_nms_idx, :]],
            dim=1
        )
        return [nms_scores, nms_class, output_boxes]

    def freeze_bn(self):
        """Set BN.eval(), BN is in the model's Backbone. """
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                # is only used to make the bn.running_mean and running_var not change in training phase.
                layer.eval()

                # freeze the bn.weight and bn.bias which are two learnable params in BN Layer.
                # layer.weight.requires_grad = False
                # layer.bias.requires_grad = False
