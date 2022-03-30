import torch
import torch.nn as nn
from utils.utils import kaiming_init, constant_init, normal_init
import math

init_method_list = ['random_init', 'kaiming_init', 'xavier_init', 'normal_init']


class CLSBranch(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 init_method=None):
        super(CLSBranch, self).__init__()

        assert init_method is not None, f'init_method in class CLSBranch needs to be set.'
        assert init_method in init_method_list, f'init_method in class CLSBranch is wrong.'

        self.convs = nn.ModuleList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            # <Method1>: Conv(wo bias) + BN + Relu()
            # self.convs.append(nn.Conv2d(chns, feat_channels, 3, 1, 1, bias=False))  # conv_weight -> bn -> relu()
            # self.convs.append(nn.BatchNorm2d(feat_channels, affine=True))  # add BN layer
            # self.convs.append(nn.ReLU(inplace=True))
        # self.init_weights()

            # <Method2>: Conv(bias) + Relu() and using kaiming_init_weight / mmdet_init_weight
            self.convs.append(nn.Conv2d(chns, feat_channels, 3, 1, 1, bias=True))  # conv with bias -> relu()
            self.convs.append(nn.ReLU(inplace=True))

        if init_method is 'kaiming_init':
            self.kaiming_init_weights()
        if init_method is 'normal_init':
            self.mmdet_init_weights()

    def mmdet_init_weights(self):
        print('[Info]: Using mmdet_init_weights() {normal_init} to init Cls Branch.')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1, bias=0)

    def kaiming_init_weights(self):
        print('[Info]: Using kaiming_init_weights() to init Cls Branch.')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, a=0, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1, bias=0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class CLSHead(nn.Module):
    def __init__(self,
                 feat_channels,
                 num_anchors,
                 num_classes):
        super(CLSHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.head = nn.Conv2d(self.feat_channels, self.num_anchors * self.num_classes, 3, 1, 1)  # with bias
        self.head_init_weights()

    def head_init_weights(self):
        print('[Info]: Using RetinaNet Paper Init Method to init Cls Head.')
        prior = 0.01
        self.head.weight.data.fill_(0)
        self.head.bias.data.fill_(-math.log((1.0 - prior) / prior))

    def forward(self, x):
        x = torch.sigmoid(self.head(x))
        x = x.permute(0, 2, 3, 1)
        n, h, w, c = x.shape
        x = x.reshape(n, h, w, self.num_anchors, self.num_classes)
        return x.reshape(x.shape[0], -1, self.num_classes)


class REGBranch(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 init_method=None):
        super(REGBranch, self).__init__()

        assert init_method is not None, f'init_method in class RegBranch needs to be set.'
        assert init_method in init_method_list, f'init_method in class RegBranch is wrong.'

        self.convs = nn.ModuleList()

        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels

            # <Method1>: Conv(wo bias) + BN + Relu()
            # self.convs.append(nn.Conv2d(chns, feat_channels, 3, 1, 1, bias=False))  # conv_weight -> bn -> relu()
            # self.convs.append(nn.BatchNorm2d(feat_channels, affine=True))
            # self.convs.append(nn.ReLU(inplace=True))
        # self.init_weights()

            # <Method2>: Conv(bias) + Relu() and using kaiming_init_weight / mmdet_init_weight
            self.convs.append(nn.Conv2d(chns, feat_channels, 3, 1, 1, bias=True))  # conv with bias -> relu()
            self.convs.append(nn.ReLU(inplace=True))
        if init_method is 'kaiming_init':
            self.kaiming_init_weights()
        if init_method is 'normal_init':
            self.mmdet_init_weights()

    def mmdet_init_weights(self):
        print('[Info]: Using mmdet_init_weights() {normal_init} to init Reg Branch.')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1, bias=0)

    def kaiming_init_weights(self):
        print('[Info]: Using kaiming_init_weights() to init Reg Branch.')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, a=0, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1, bias=0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class REGHead(nn.Module):
    def __init__(self,
                 feat_channels,
                 num_anchors,
                 num_regress):
        super(REGHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_regress = num_regress
        self.feat_channels = feat_channels
        self.head = nn.Conv2d(self.feat_channels, self.num_anchors * self.num_regress, 3, 1, 1)  # with bias
        self.mmdet_init_weights()

    def mmdet_init_weights(self):
        print('[Info]: Using mmdet_init_weights() {normal_init} to init Reg Head.')
        normal_init(self.head, mean=0, std=0.01, bias=0)

    def forward(self, x, with_deform=False):
        x = self.head(x)
        if with_deform is False:
            x = x.permute(0, 2, 3, 1)
            return x.reshape(x.shape[0], -1, self.num_regress)
        else:
            return x
