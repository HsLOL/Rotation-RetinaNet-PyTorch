import os
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils.bbox_transforms import *
import cv2
import xml.etree.ElementTree as ET
from utils.augment import *


class SSDDataset(data.Dataset):
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
                    # HSV(0.5, 0.5, p=0.5),
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
        infile = open(os.path.join(filename))
        tree = ET.parse(infile)
        root = tree.getroot()
        for obj in root.iter('object'):
            rbox = obj.find('rotated_bndbox')
            x1 = float(rbox.find('x1').text)
            y1 = float(rbox.find('y1').text)
            x2 = float(rbox.find('x2').text)
            y2 = float(rbox.find('y2').text)
            x3 = float(rbox.find('x3').text)
            y3 = float(rbox.find('y3').text)
            x4 = float(rbox.find('x4').text)
            y4 = float(rbox.find('y4').text)
            polygon = np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.int32)
            boxes.append(polygon)
            label_index = 1
            gt_classes.append(label_index)
        return {'boxes': np.array(boxes), 'gt_classes': np.array(gt_classes)}


if __name__ == '__main__':
    rssdd = SSDDataset(root_path='/data/fzh/RSSDD/',
                       set_name='train',
                       augment=True,
                       classes=['ship', ])
    for idx in range(len(rssdd)):
        idx = 0
        a = rssdd[idx]
        bboxes = a['boxes']
        img = a['image']
        for gt_bbox in bboxes:

            ps = gt_bbox[:-1].reshape(-1, 4, 2).astype(np.int32)
            cv2.drawContours(img, [ps], -1, [0, 255, 0], thickness=2)

        plt.imshow(img)
        plt.show()