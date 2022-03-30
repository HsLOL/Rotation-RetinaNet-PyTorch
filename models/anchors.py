import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self,
                 params=None,
                 pyramid_levels=None,
                 strides=None,
                 rotations=None):
        super(Anchors, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.strides = strides
        self.base_size = params.base_size
        self.ratios = params.ratios
        self.scales = params.scales
        self.rotations = rotations

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.base_size = params.base_size
        self.ratios = params.ratios
        self.scales = np.array([2**(i / 3) for i in range(params.scales_per_octave)])
        self.rotations = np.array([params.angle / 180 * np.pi])

        self.num_anchors = len(self.scales) * len(self.ratios) * len(self.rotations)

        print(f'[Info]: anchor ratios: {self.ratios}\tanchor scales: {self.scales}\tbase_size: {self.base_size}\t'
              f'angle: {self.rotations}')
        print(f'[Info]: number of anchors: {self.num_anchors}')

    @staticmethod
    def generate_anchors(base_size, ratios, scales, rotations):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.

        anchors: [xc, yc, w, h, angle(radian)]
        """
        num_anchors = len(ratios) * len(scales) * len(rotations)
        # initialize output anchors
        anchors = np.zeros((num_anchors, 5))
        # scale base_size
        anchors[:, 2:4] = base_size * np.tile(scales, (2, len(ratios) * len(rotations))).T
        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]
        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales) * len(rotations)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales) * len(rotations))
        # add rotations
        anchors[:, 4] = np.tile(np.repeat(rotations, len(scales)), (1, len(ratios))).T[:, 0]
        # # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        # anchors[:, 0:3:2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        # anchors[:, 1:4:2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors  # [x_ctr, y_ctr, w, h, angle(radian)]

    @staticmethod
    def shift(shape, stride, anchors):
        shift_x = np.arange(0, shape[1]) * stride
        shift_y = np.arange(0, shape[0]) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            np.zeros(shift_x.ravel().shape), np.zeros(shift_y.ravel().shape),
            np.zeros(shift_x.ravel().shape)
        )).transpose()
        # add A anchors (1, A, 5) to
        # cell K shifts (K, 1, 5) to get
        # shift anchors (K, A, 5)
        # reshape to (K*A, 5) shifted anchors
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 5)) + shifts.reshape((1, K, 5)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 5))
        return all_anchors

    def forward(self, images):
        image_shape = np.array(images.shape[2:])
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 5)).astype(np.float32)
        num_level_anchors = []
        for idx, p in enumerate(self.pyramid_levels):
            base_anchors = self.generate_anchors(
                base_size=self.base_size * self.strides[idx],
                ratios=self.ratios,
                scales=self.scales,
                rotations=self.rotations)
            shifted_anchors = self.shift(image_shapes[idx], self.strides[idx], base_anchors)
            num_level_anchors.append(shifted_anchors.shape[0])
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)
        all_anchors = np.tile(all_anchors, (images.size(0), 1, 1))
        all_anchors = torch.from_numpy(all_anchors.astype(np.float32))
        if torch.is_tensor(images) and images.is_cuda:
            device = images.device
            all_anchors = all_anchors.cuda(device=device)
        return all_anchors, torch.from_numpy(np.array(num_level_anchors)).cuda(device=device)


if __name__ == '__main__':
    from train import Params
    params = Params('/home/fzh/Pictures/Rotation-RetinaNet-PyTorch/configs/retinanet_r50_fpn_hrsc.yml')
    anchors = Anchors(params)
    feature_map_sizes = [(128, 128), (64, 64), (32, 32), (16, 16), (8, 8)]
    for level_idx in range(5):
        # print(f'# ============================base_anchor{level_idx}========================================= #')
        base_anchor = anchors.generate_anchors(
            base_size=anchors.base_size * anchors.strides[level_idx],
            ratios=anchors.ratios,
            scales=anchors.scales,
            rotations=anchors.rotations
        )
        # print(base_anchor)
        print(f'# ============================shift_anchor{level_idx}========================================= #')
        shift_anchor = anchors.shift(feature_map_sizes[level_idx], anchors.strides[level_idx], base_anchor)
        print(shift_anchor)
