import torch
import numpy as np
from utils.utils import Rescale, Normalize, Reshape
from torchvision.transforms import Compose
import cv2
import matplotlib.pyplot as plt
import math
from utils.bbox_transforms import poly2obb_np


class Collater(object):
    def __init__(self, scales, keep_ratio=False, multiple=32):
        self.scales = scales
        self.keep_ratio = keep_ratio
        self.multiple = multiple

    def __call__(self, batch):
        scales = int(np.floor(float(self.scales) / self.multiple) * self.multiple)
        rescale = Rescale(target_size=scales, keep_ratio=self.keep_ratio)
        transform = Compose([Normalize(), Reshape(unsqueeze=False)])

        images = [sample['image'] for sample in batch]
        bboxes = [sample['boxes'] for sample in batch]
        image_names = [sample['imagename'] for sample in batch]

        max_height, max_width = -1, -1

        for index in range(len(batch)):
            im, _ = rescale(images[index])
            height, width = im.shape[0], im.shape[1]
            max_height = height if height > max_height else max_height
            max_width = width if width > max_width else max_width

        padded_ims = torch.zeros(len(batch), 3, max_height, max_width)

        # ready to save the openCV format info [xc, yc, w, h, theta, class_index]
        num_params = 6
        max_num_boxes = max(bbox.shape[0] for bbox in bboxes)
        padded_boxes = torch.ones(len(batch), max_num_boxes, num_params) * -1

        for i in range(len(batch)):
            im, bbox = images[i], bboxes[i]

            # rescale the image
            im, im_scale = rescale(im)
            height, width = im.shape[0], im.shape[1]
            padded_ims[i, :, :height, :width] = transform(im)  # transform is similar to the pipeline in mmdet

            # rescale the bounding box
            oc_bboxes = []
            labels = []
            for single in bbox:

                # rescale the bounding box
                single[0::2] *= im_scale[0]
                single[1::2] *= im_scale[1]

                # polygons to the opencv format, opencv version > 4.5.1
                oc_bbox = poly2obb_np(single[:-1], 'oc')  # oc_bbox: [xc, yc, h, w, angle(radian)]
                assert 0 < oc_bbox[4] <= np.pi / 2
                oc_bboxes.append(np.array(oc_bbox, dtype=np.float32))
                labels.append(single[-1])

            if bbox.shape[0] != 0:
                padded_boxes[i, :bbox.shape[0], :-1] = torch.from_numpy(np.array(oc_bboxes))
                padded_boxes[i, :bbox.shape[0], -1] = torch.from_numpy(np.array(labels))

            # # visualize rescale result
            # vis_im = images[i]
            # vis_im, _ = rescale(vis_im)
            # for gt_bbox in oc_bboxes:
            #     xc, yc, h, w, ag = gt_bbox[:5]
            #     print(f'GT Annotation: xc:{xc} yc:{yc} h:{h} w:{w} ag:{ag}')
            #     wx, wy = -w / 2 * math.sin(ag), w / 2 * math.cos(ag)
            #     hx, hy = h / 2 * math.cos(ag), h / 2 * math.sin(ag)
            #     p1 = (xc - wx - hx, yc - wy - hy)
            #     p2 = (xc - wx + hx, yc - wy + hy)
            #     p3 = (xc + wx + hx, yc + wy + hy)
            #     p4 = (xc + wx - hx, yc + wy - hy)
            #     ps = np.int0(np.array([p1, p2, p3, p4]))
            #     cv2.drawContours(vis_im, [ps], -1, [0, 255, 0], thickness=2)
            # plt.imshow(vis_im)
            # plt.title(image_names[i])
            # plt.show()

        return {'image': padded_ims, 'bboxes': padded_boxes, 'image_name': image_names}
