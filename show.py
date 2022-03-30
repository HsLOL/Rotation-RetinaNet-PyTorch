import argparse
from models.model import RetinaNet
import os
import cv2
import torch
from detect import im_detect
import numpy as np
import matplotlib.pyplot as plt
import math


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--config_file', type=str, default='./configs/retinanet_r50_fpn_ssdd.yml')
    parser.add_argument('--target_sizes', type=list, default=[512], help='the size of the input image.')
    parser.add_argument('--chkpt', type=str, default='best/best.pth', help='the chkpt file name')
    parser.add_argument('--result_path', type=str, default='show_result', help='the relative path for saving'
                                                                               'ori pic and predicted pic')
    parser.add_argument('--score_thresh', type=float, default=0.05, help='score threshold')
    parser.add_argument('--pic_name', type=str, default='demo6.jpg', help='relative path')
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    return args


def plot_box(image, coord, label_index=None, score=None, color=None, line_thickness=None):
    bbox_color = [226, 43, 138] if color is None else color
    text_color = [255, 255, 255]
    line_thickness = 1 if line_thickness is None else line_thickness
    xc, yc, h, w, ag = coord[:5]
    wx, wy = -w / 2 * math.sin(ag), w / 2 * math.cos(ag)
    hx, hy = h / 2 * math.cos(ag), h / 2 * math.sin(ag)
    p1 = (xc - wx - hx, yc - wy - hy)
    p2 = (xc - wx + hx, yc - wy + hy)
    p3 = (xc + wx + hx, yc + wy + hy)
    p4 = (xc + wx - hx, yc + wy - hy)
    ps = np.int0(np.array([p1, p2, p3, p4]))
    cv2.drawContours(image, [ps], -1, bbox_color, thickness=3)
    if label_index is not None:
        label_text = params.classes[label_index]
        label_text += '|{:.02f}'.format(score)
        font = cv2.FONT_HERSHEY_COMPLEX
        text_size = cv2.getTextSize(label_text, font, fontScale=0.25, thickness=line_thickness)
        text_width = text_size[0][0]
        text_height = text_size[0][1]
        try:
            cv2.rectangle(image, (int(xc), int(yc) - text_height -2),
                  (int(xc) + text_width, int(yc) + 3), (0, 128, 0), -1)
            cv2.putText(image, label_text, (int(xc), int(yc)), font, 0.25, text_color, thickness=1)
        except:
            print(f'{coord} is wrong!')


def show_pred_box(args, params):
    # create folder
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    model = RetinaNet(params)
    chkpt_path = os.path.join(params.output_path, 'checkpoints', args.chkpt)
    chkpt = torch.load(chkpt_path, map_location='cpu')
    print(f"The current model training {chkpt['epoch']} epoch(s)")
    print(f"The current model mAP: {chkpt['best_fitness']} based on test_conf={params.score_thr} & nms_thr={params.nms_thr}")

    model.load_state_dict(chkpt['model'])
    model.cuda(device=args.device)
    model.eval()

    image = cv2.cvtColor(cv2.imread(os.path.join(args.result_path, args.pic_name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    dets = im_detect(model,
                     image,
                     target_sizes=args.target_sizes,
                     params=params,
                     use_gpu=True,
                     conf=args.score_thresh,
                     device=args.device)

    # dets: list[class_index,  0
    #            score,        1
    #            pred_xc, pred_yc, pred_w, pred_h, pred_angle(radian),  2 - 6
    #            anchor_xc, anchor_yc, anchor_w, anchor_h, anchor_angle(radian)] 7 - 11
    for det in dets:
        cls_index = int(det[0])
        score = float(det[1])
        pred_box = det[2:7]
        anchor = det[7:12]

        # plot predict box
        plot_box(image, coord=pred_box, label_index=cls_index, score=score, color=None,
                 line_thickness=4)

        # plot which anchor to create predict box
        # plot_box(image, coord=anchor, color=[0, 0, 255])

    plt.imsave(os.path.join(args.result_path, f"{args.pic_name.split('.')[0]}_predict.png"), image)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    from train import Params

    args = get_args()
    params = Params(args.config_file)
    if args.score_thresh != params.score_thr:
        print('[Info]: score_thresh is not equal to cfg.score_thr')
    params.backbone['pretrained'] = False
    show_pred_box(args, params)
