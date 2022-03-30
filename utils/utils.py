import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import random
import torch.nn as nn


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        deterministic is set True if use torch.backends.cudnn.deterministic
        Default is False.
    """
    print(f'[Info]: Set random seed to {seed}, deterministic: {deterministic}.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)

    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def pretty_print(num_params, units=None, precision=2):
    if units is None:
        if num_params // 10**6 > 0:
            print(f'[Info]: Model Params = {str(round(num_params / 10**6, precision))}' + ' M')
        elif num_params // 10**3:
            print(f'[Info]: Model Params = {str(round(num_params / 10**3, precision))}' + ' k')
        else:
            print(f'[Info]: Model Params = {str(num_params)}')


def count_param(model, units=None, precision=2):
    """Count Params"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pretty_print(num_params)


def show_args(args):
    print('=============== Show Args ===============')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))


def clip_boxes(boxes, ims):
    _, _, h, w = ims.shape
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=w)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=h)
    return boxes


# (num_boxes, 5)  xyxya
def min_area_square(rboxes):
    w = rboxes[:, 2] - rboxes[:, 0]
    h = rboxes[:, 3] - rboxes[:, 1]
    ctr_x = rboxes[:, 0] + w * 0.5
    ctr_y = rboxes[:, 1] + h * 0.5
    s = torch.max(w, h)
    return torch.stack((
        ctr_x - s * 0.5, ctr_y - s * 0.5,
        ctr_x + s * 0.5, ctr_y + s * 0.5),
        dim=1
    )


def rbox_overlaps(boxes, query_boxes, indicator=None, thresh=1e-1):
    # rewrited by cython
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    a_tt = boxes[:, 4]
    a_ws = boxes[:, 2] - boxes[:, 0]
    a_hs = boxes[:, 3] - boxes[:, 1]
    a_xx = boxes[:, 0] + a_ws * 0.5
    a_yy = boxes[:, 1] + a_hs * 0.5

    b_tt = query_boxes[:, 4]
    b_ws = query_boxes[:, 2] - query_boxes[:, 0]
    b_hs = query_boxes[:, 3] - query_boxes[:, 1]
    b_xx = query_boxes[:, 0] + b_ws * 0.5
    b_yy = query_boxes[:, 1] + b_hs * 0.5

    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = b_ws[k] * b_hs[k]
        for n in range(N):
            if indicator is not None and indicator[n, k] < thresh:
                continue
            ua = a_ws[n] * a_hs[n] + box_area
            rtn, contours = cv2.rotatedRectangleIntersection(
                ((a_xx[n], a_yy[n]), (a_ws[n], a_hs[n]), a_tt[n]),
                ((b_xx[k], b_yy[k]), (b_ws[k], b_hs[k]), b_tt[k])
            )
            if rtn == 1:
                ia = cv2.contourArea(contours)
                overlaps[n, k] = ia / (ua - ia)
            elif rtn == 2:
                ia = np.minimum(ua - box_area, box_area)
                overlaps[n, k] = ia / (ua - ia)
    return overlaps


def bbox_overlaps(boxes, query_boxes):
    """Calculate the horizontal overlaps

    Args:
        boxes: [xc, yc, w, h, angle]
        query_boxes: [xc, yc, w, h, pi/2]
    """
    if not isinstance(boxes, float):   # apex
        boxes = boxes.float()

    # convert the [xc, yc, w, h, angle] to [x1, y1, x2, y2, angle]
    query_boxes[:, 0] = query_boxes[:, 0] - query_boxes[:, 2] / 2
    query_boxes[:, 1] = query_boxes[:, 1] - query_boxes[:, 3] / 2
    query_boxes[:, 2] = query_boxes[:, 0] + query_boxes[:, 2]
    query_boxes[:, 3] = query_boxes[:, 1] + query_boxes[:, 3]

    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    area = (query_boxes[:, 2] - query_boxes[:, 0]) * \
           (query_boxes[:, 3] - query_boxes[:, 1])
    iw = torch.min(torch.unsqueeze(boxes[:, 2], dim=1), query_boxes[:, 2]) - \
         torch.max(torch.unsqueeze(boxes[:, 0], 1), query_boxes[:, 0])
    ih = torch.min(torch.unsqueeze(boxes[:, 3], dim=1), query_boxes[:, 3]) - \
         torch.max(torch.unsqueeze(boxes[:, 1], 1), query_boxes[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    return intersection / ua


def rescale(im, target_size, max_size, keep_ratio, multiple=32):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if keep_ratio:
        # scale method 1:
        # scale the shorter side to target size by the constraint of the max size
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im_scale_x = np.floor(im.shape[1] * im_scale / multiple) * multiple / im.shape[1]
        im_scale_y = np.floor(im.shape[0] * im_scale / multiple) * multiple / im.shape[0]
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])

        # scale method 2:
        # scale the longer side to target size
        # im_scale = float(target_size) / float(im_size_max)
        # im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # im_scale = np.array([im_scale, im_scale, im_scale, im_scale])

    else:
        target_size = int(np.floor(float(target_size) / multiple) * multiple)
        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])
        im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
    return im, im_scale


class Rescale(object):
    def __init__(self, target_size, keep_ratio):
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.max_size = 2000  # for scale method 1

    def __call__(self, image):
        im, im_scale = rescale(image, target_size=self.target_size, max_size=self.max_size,
                               keep_ratio=self.keep_ratio)
        return im, im_scale


class Normalize(object):
    def __init__(self):
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # mean / std

    def __call__(self, im):
        im = self._transform(im)
        return im


class Reshape(object):
    def __init__(self, unsqueeze=True):
        self._unsqueeze = unsqueeze
        return

    def __call__(self, ims):
        if not torch.is_tensor(ims):
            ims = torch.from_numpy(ims.transpose((2, 0, 1)))
        if self._unsqueeze:
            ims = ims.unsqueeze(0)
        return ims

