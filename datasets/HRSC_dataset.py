import os
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils.bbox_transforms import *
import cv2
from utils.augment import *


class HRSCDataset(data.Dataset):
    def __init__(self, root_path, set_name, augment=False, classes=None):
        self.root_path = root_path
        self.set_name = set_name
        self.augment = augment
        self.image_lists = self._load_image_names()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        if self.augment is True:
            print(f'[Info]: Using the data augmentation.')
        else:
            print(f'[Info]: Not using the data augmentation.')

    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, index):
        imagename = self.image_lists[index]
        img_path = os.path.join(self.root_path, self.set_name, "images", imagename)
        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        roidb = self._load_annotation(imagename)
        gt_inds = np.where(roidb['gt_classes'] != 0)[0]
        num_gt = len(roidb['boxes'])
        gt_boxes = np.zeros((len(gt_inds), 9), dtype=np.float32)  # [x1,y1,x2,y2,x3,y3,x4,y4,class_index]
        if num_gt:
            # get the bboxes and classes info from the self._load_annotation() result.
            bboxes = roidb['boxes'][gt_inds, :]
            classes = roidb['gt_classes'][gt_inds] - 1

            # perform the data augmentation
            if self.augment is True:
                transforms = Augment([
                    HSV(0.5, 0.5, p=0.5),
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5)
                ])
                image, bboxes = transforms(image, bboxes)
            gt_boxes[:, :-1] = bboxes

            for i, bbox in enumerate(bboxes):
                gt_boxes[i, 8] = classes[i]

        return {'image': image, 'boxes': gt_boxes, 'imagename': imagename}

    def _load_image_names(self):
        return os.listdir(os.path.join(self.root_path, self.set_name, 'images'))

    def _load_annotation(self, imagename):
        filename = os.path.join(self.root_path, self.set_name, "Annotations", imagename.replace('jpg', 'xml'))
        boxes, gt_classes = [], []
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            objects = content.split('<HRSC_Object>')
            info = objects.pop(0)
            for obj in objects:
                cls_id = obj[obj.find('<Class_ID>') + 10: obj.find('</Class_ID>')]
                cx = float(eval(obj[obj.find('<mbox_cx>') + 9: obj.find('</mbox_cx>')]))
                cy = float(eval(obj[obj.find('<mbox_cy>') + 9: obj.find('</mbox_cy>')]))
                w = float(eval(obj[obj.find('<mbox_w>') + 8: obj.find('</mbox_w>')]))
                h = float(eval(obj[obj.find('<mbox_h>') + 8: obj.find('</mbox_h>')]))
                angle = float(obj[obj.find('<mbox_ang>') + 10: obj.find('</mbox_ang>')])  # radian

                # add extra score parameter to use obb2poly_up
                bbox = np.array([[cx, cy, w, h, angle, 0]], dtype=np.float32)
                polygon = obb2poly_np(bbox, 'le90')[0, :-1].astype(np.float32)
                boxes.append(polygon)
                label_index = 1
                gt_classes.append(label_index)
        return {'boxes': np.array(boxes), 'gt_classes': np.array(gt_classes)}


if __name__ == '__main__':
    hrsc = HRSCDataset(root_path='/data/fzh/HRSC/',
                       set_name='train',
                       augment=True,
                       classes=['ship', ])
    for idx in range(len(hrsc)):
        a = hrsc[idx]
        bboxes = a['boxes']  # polygon format [x1, y1, x2, y2, x3, y3, x4, y4]
        img = a['image']
        image_name = a['imagename']
        for gt_bbox in bboxes:
            ps = gt_bbox[:-1].reshape(1, 4, 2).astype(np.int32)
            cv2.drawContours(img, [ps], -1, [0, 255, 0], thickness=2)
        plt.imshow(img)
        plt.title(image_name)
        plt.show()
