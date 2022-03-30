"""This script is used to convert .bmp format to .jpg format."""

from PIL import Image
from tqdm import tqdm
import shutil
import os
import cv2


class Convert(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.convert_image_folder = 'images'
        self.image_folder = 'AllImages'
        self._mkdir()

    def _mkdir(self):
        self.train_image_path = os.path.join(self.root_path, 'train', self.convert_image_folder)
        self.val_image_path = os.path.join(self.root_path, 'test', self.convert_image_folder)

        if not os.path.exists(self.train_image_path):
            os.makedirs(self.train_image_path)

        if not os.path.exists(self.val_image_path):
            os.makedirs(self.val_image_path)

    def convert(self, set_name):
        image_lists = os.listdir(os.path.join(self.root_path, set_name, self.image_folder))
        for single_image in image_lists:
            image = cv2.imread(os.path.join(self.root_path, set_name, self.image_folder, single_image))
            converted_single_image = single_image.replace('bmp', 'jpg')
            cv2.imwrite(os.path.join(self.root_path, set_name, self.convert_image_folder, converted_single_image),
                        image)


if __name__ == '__main__':
    convert = Convert(root_path='/home/fzh/Data/HRSC/')
    convert.convert(set_name='test')

