import torch.nn.functional as F
from torch import nn
from utils.utils import kaiming_init, xavier_init

init_method_list = ['random_init', 'kaiming_init', 'xavier_init', 'normal_init']


class FPN(nn.Module):
    def __init__(self,
                 in_channel_list,
                 out_channels,
                 top_blocks,
                 init_method=None):
        """
        Args:
            out_channels(int): number of channels of the FPN feature.
            top_blocks(nn.Module or None): if provided, an extra op will be
                performed on the FPN output, and the result will extend the result list.
            init_method: which method to init lateral_conv and fpn_conv.
                         kaiming_init: kaiming_init()
                         xavier_init: xavier_init()
                         random_init: PyTorch_init()
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        self.init_method = init_method
        print('[Info]: ===== Neck Using FPN =====')

        assert init_method is not None, f'init_method in class FPN needs to be set.'
        assert init_method in init_method_list, f'init_method in class FPN is wrong.'
        if init_method is 'kaiming_init':
            print('[Info]: Using kaiming_init() to init lateral_conv and fpn_conv.')
        if init_method is 'xavier_init':
            print('[Info]: Using xavier_init() to init lateral_conv and fpn_conv.')
        if init_method is 'random_init':
            print('[Info]: Using PyTorch_init() to init lateral_conv and fpn_conv.')

        for idx, in_channels in enumerate(in_channel_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue

            # lateral conv  1x1
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)  # with bias, without BN Layer
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1, 1)  # with bias, without BN Layer

            if self.init_method is 'kaiming_init':
                kaiming_init(inner_block_module, a=0, nonlinearity='relu')
                kaiming_init(layer_block_module, a=0, nonlinearity='relu')

            if self.init_method is 'xavier_init':
                xavier_init(inner_block_module, gain=1, bias=0, distribution='uniform')
                xavier_init(layer_block_module, gain=1, bias=0, distribution='uniform')

            # if self.init_method is 'random_init':
            #     Don't do anything

            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)

            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x : feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
            They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = F.interpolate(
                last_inner, size=
                    (int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])),
                mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6_P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        else:
            raise NotImplementedError

        return tuple(results)


class LastLevelP6_P7(nn.Module):
    """This module is used in RetinaNet to generate extra layers, P6 and P7.
    Args:
        init_method: which method to init P6_conv and P7_conv,
                     support methods: kaiming_init:kaiming_init,
                                      xavier_init: xavier_init,
                                      random_init: PyTorch_init
    """
    def __init__(self, in_channels,
                 out_channels,
                 init_method=None):
        super(LastLevelP6_P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)  # with bias without BN Layer
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)  # with bias without BN Layer

        assert init_method is not None, f'init_method in class LastLevelP6_P7 needs to be set.'
        assert init_method in init_method_list, f'init_method in class LastLevelP6_P7 is wrong.'

        if init_method is 'kaiming_init':
            print('[Info]: Using kaiming_init() to init P6_conv and P7_conv')
            for layer in [self.p6, self.p7]:
                kaiming_init(layer, a=0, nonlinearity='relu')

        if init_method is 'xavier_init':
            print('[Info]: Using xavier_init() to init P6_conv and P7_conv')
            for layer in [self.p6, self.p7]:
                xavier_init(layer, gain=1, bias=0, distribution='uniform')

        if init_method is 'random_init':
            print('[Info]: Using PyTorch_init() to init P6_conv and P7_conv')
            # Don't do anything

        self.use_p5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_p5 else c5
        p6 = self.p6(x)
        p7 = self.p7(p6)
        return [p6, p7]
