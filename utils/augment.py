import random
import cv2
import numpy as np


class HorizontalFlip(object):
    """
    Args:
        p: the probability of the horizontal flip
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bboxes):
        """
        Args:
            image: array([C, H, W])
            bboxes: array (N, 8) :[[x1, y1, x2, y2, x3, y3, x4, y4] ... ]
        """
        if random.random() < self.p:
            h, w, _ = image.shape
            image = np.array(np.fliplr(image))
            for idx, single_box in enumerate(bboxes):
                bboxes[idx, 0::2] = w - single_box[0::2]
        return image, bboxes


class VerticalFlip(object):
    """
    Args:
        p: the probability of the vertical flip
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bboxes):
        """
        Args:
            image: array([C, H, W])
            bboxes: list (N, 9) :[[x1, y1, x2, y2, x3, y3, x4, y4, class_index] ... ]
        """
        if random.random() < self.p:
            h, w, _ = image.shape
            image = np.array(np.flipud(image))
            for idx, single_box in enumerate(bboxes):
                bboxes[idx, 1::2] = h - single_box[1::2]
        return image, bboxes


class HSV(object):
    def __init__(self, saturation=0, brightness=0, p=0.):
        self.saturation = saturation
        self.brightness = brightness
        self.p = p

    def __call__(self, image, bboxes, mode=None):
        if random.random() < self.p:
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value
            a = random.uniform(-1, 1) * self.saturation + 1
            b = random.uniform(-1, 1) * self.brightness + 1
            S *= a
            V *= b
            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)
        return image, bboxes


class Augment(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxes):
        for transform in self.transforms:
            image, bboxes = transform(image, bboxes)
        return image, bboxes
