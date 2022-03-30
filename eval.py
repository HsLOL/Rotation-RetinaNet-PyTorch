import os
from detect import im_detect
import shutil
from tqdm import tqdm
from utils.map import eval_mAP
from utils.bbox_transforms import *


# evaluate by rotation detection result
def evaluate(model=None,
             target_size=None,
             test_path=None,
             conf=None,
             device=None,
             mode=None,
             params=None):
    evaluate_dir = 'voc_evaluate'
    _dir = mode + '_evaluate'
    out_dir = os.path.join(params.output_path, evaluate_dir, _dir, 'detection-results')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Step1. Collect detect result for per image or get predict result
    for image_name in tqdm(os.listdir(os.path.join(params.data_path, mode, 'images'))):
        image_path = os.path.join(params.data_path, mode, 'images', image_name)
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model=model,
                         src=image,
                         params=params,
                         target_sizes=target_size,
                         use_gpu=True,
                         conf=conf,  # score threshold
                         device=device)

        # Step2. Write per image detect result into per txt file
        # line = cls_name score x1 y1 x2 y2 x3 y3 x4 y4
        img_ext = image_name.split('.')[-1]
        with open(os.path.join(out_dir, image_name.replace(img_ext, 'txt')), 'w') as f:
            for det in dets:
                cls_ind = int(det[0])
                cls_socre = det[1]
                rbox = det[2:7]  # [xc, yc, h, w, angle(radian)]

                if np.isnan(rbox[0]) or np.isnan(rbox[1]) or np.isnan(rbox[2]) or np.isnan(rbox[3]) or np.isnan(rbox[4]):
                    line = ''
                else:
                    # add extra score
                    rbbox = np.array([[rbox[0], rbox[1], rbox[2], rbox[3], rbox[4], 0]], dtype=np.float32)
                    polygon = obb2poly_np(rbbox, 'oc')[0, :-1].astype(np.float32)
                    line = str(params.classes[cls_ind]) + ' ' + str(cls_socre) + ' ' + str(polygon[0]) + ' ' + str(polygon[1]) +\
                        ' ' + str(polygon[2]) + ' ' + str(polygon[3]) + ' ' + str(polygon[4]) + ' ' + str(polygon[5]) +\
                        ' ' + str(polygon[6]) + ' ' + str(polygon[7]) + '\n'
                f.write(line)

    # Step3. Calculate Precision, Recall, mAP, plot PR Curve
    mAP, Precision, Recall = eval_mAP(gt_root_dir=params.data_path,
                                      test_path=test_path,  # test_path = ground-truth
                                      eval_root_dir=os.path.join(params.output_path, evaluate_dir, _dir),
                                      use_07_metric=False,
                                      thres=0.5)  # rotation nms threshold
    print(f'mAP: {mAP}\tPrecision: {Precision}\tRecall: {Recall}')
    return mAP, Precision, Recall


if __name__ == '__main__':
    import argparse
    import torch
    from train import Params
    import time
    from models.model import RetinaNet

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--Dataset', type=str, default='SSDD')
    parser.add_argument('--config_file', type=str, default='./configs/retinanet_r50_fpn_ssdd.yml')
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--chkpt', type=str, default='best/best.pth', help='the checkpoint file of the trained model.')
    parser.add_argument('--score_thr', type=float, default=0.05)

    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--FPS', type=bool, default=False, help='Check the FPS of the Model.')  # todo: Ready to Support
    args = parser.parse_args()
    params = Params(args.config_file)
    params.backbone['pretrained'] = False
    model = RetinaNet(params)

    checkpoint = os.path.join(params.output_path, 'checkpoints', args.chkpt)

    # from checkpoint load model weight file
    # model weight
    chkpt = torch.load(checkpoint, map_location='cpu')
    pth = chkpt['model']
    model.load_state_dict(pth)
    model.cuda(device=args.device)

    """The following codes is used to Debug eval() function."""
    if args.evaluate:
        model.eval()
        mAP, Precision, Recall = evaluate(
            model=model,
            target_size=[args.target_size],
            test_path='ground-truth',
            conf=args.score_thr,  # score threshold
            device=args.device,
            mode='test',
            params=params)
        print(f'mAP: {mAP}\nPrecision: {Precision}\nRecall: {Recall}\n')

    """The following codes are used to calculate FPS of model."""
    if args.FPS:
        times = 50  # 50 is enough to balance some additional times for IO
        image_path = os.path.join(params.data_path, args.single_image)
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        model.eval()
        t1 = time.time()
        for _ in range(times):
            dets = im_detect(model=model,
                             image=image,
                             target_sizes=[args.target_size],
                             use_gpu=True,
                             conf=0.25,
                             device=args.device,
                             params=params)
        t2 = time.time()
        tact_time = (t2 - t1) / times
        print(f'{tact_time} seconds, {1 / tact_time} FPS, Batch_size = 1')
