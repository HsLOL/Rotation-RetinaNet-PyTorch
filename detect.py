import numpy as np
import torch
from torchvision.transforms import Compose
from utils.utils import Rescale, Normalize, Reshape
from utils.rotation_nms.cpu_nms import cpu_nms
from utils.bbox_transforms import *


def im_detect(model, src, target_sizes, params, use_gpu=True, conf=None, device=None):
    if isinstance(target_sizes, int):
        target_sizes = [target_sizes]
    if len(target_sizes) == 1:
        return single_scale_detect(model, src, target_size=target_sizes[0], params=params,
                                   use_gpu=use_gpu, conf=conf, device=device)


def single_scale_detect(model, src, target_size, params=None,
                        use_gpu=True, conf=None, device=None):
    im, im_scales = Rescale(target_size=target_size, keep_ratio=params.keep_ratio)(src)
    im = Compose([Normalize(), Reshape(unsqueeze=True)])(im)
    if use_gpu and torch.cuda.is_available():
        model, im = model.cuda(device=device), im.cuda(device=device)
    with torch.no_grad():  # bboxes: [x, y, x, y, a, a_x, a_y, a_x, a_y, a_a]
        scores, classes, boxes = model(im, test_conf=conf)
    scores = scores.data.cpu().numpy()
    classes = classes.data.cpu().numpy()
    boxes = boxes.data.cpu().numpy()

    # convert oc format to polygon for rescale predict box coordinate
    predicted_bboxes = []
    for idx in range(len(boxes)):
        single_box = boxes[idx]  # single box: [pred_xc, pred_yc, pred_h, pred_w, pred_angle(radian)]
        single_box = np.array([[single_box[0], single_box[1], single_box[2], single_box[3], single_box[4], 0]],
                              dtype=np.float32)  # add extra score 0
        predicted_polygon = obb2poly_np_oc(single_box)[0, :-1].astype(np.float32)
        predicted_polygon[0::2] /= im_scales[0]
        predicted_polygon[1::2] /= im_scales[1]
        predicted_bbox = poly2obb_np(predicted_polygon, 'oc')  # polygon 2 rbboxes (oc format: [xc, yc, h, w, angle(radian)]
        predicted_bboxes.append(predicted_bbox)

    if boxes.shape[1] > 5:
        # [pred_xc, pred_yc, pred_h, pred_w, pred_angle(radian),
        # anchor_xc, anchor_yc, anchor_w, anchor_h, anchor_angle(radian)]
        boxes[:, 5:9] = boxes[:, 5:9] / im_scales
    scores = np.reshape(scores, (-1, 1))
    classes = np.reshape(classes, (-1, 1))
    for id in range(len(predicted_bboxes)):
        boxes[id, :5] = predicted_bboxes[id]
    cls_dets = np.concatenate([classes, scores, boxes], axis=1)
    keep = np.where(classes < model.num_class)[0]
    return cls_dets[keep, :]
    # cls, score, x,y,w,h,a,   a_x,a_y,a_w,a_h,a_a
