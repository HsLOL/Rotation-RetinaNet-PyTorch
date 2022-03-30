"""This script is used to convert xml format to txt format for HRSC Dataset for evaluation."""

import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


class Convert(object):
    def __init__(self, xml_path, txt_path, image_path):
        self.xml_path = xml_path
        self.txt_path = txt_path
        self.image_path = image_path
        self.xml_lists = os.listdir(xml_path)
        self._makedir()

    def _makedir(self):
        if not os.path.exists(self.txt_path):
            os.makedirs(self.txt_path)

    def _readXml(self, single_xml):
        with open(os.path.join(self.xml_path, single_xml), 'r', encoding='utf-8-sig') as f:
            content = f.read()
            objects = content.split('<HRSC_Object>')
            info = objects.pop(0)

            results = []
            for obj in objects:
                cls_name = 'ship'
                cx = round(eval(obj[obj.find('<mbox_cx>') + 9: obj.find('</mbox_cx>')]))
                cy = round(eval(obj[obj.find('<mbox_cy>') + 9: obj.find('</mbox_cy>')]))
                w = round(eval(obj[obj.find('<mbox_w>') + 8: obj.find('</mbox_w>')]))
                h = round(eval(obj[obj.find('<mbox_h>') + 8: obj.find('</mbox_h>')]))
                angle = eval(obj[obj.find('<mbox_ang>') + 10: obj.find('</mbox_ang>')]) / math.pi * 180
                rbox = np.array([cx, cy, w, h, angle])
                quad_box = rbox_2_quad(rbox, 'xywha').squeeze()
                line = cls_name + ' ' + str(quad_box[0]) + ' ' + str(quad_box[1]) + ' ' + str(quad_box[2]) + ' ' +\
                    str(quad_box[3]) + ' ' + str(quad_box[4]) + ' ' + str(quad_box[5]) + ' ' + str(quad_box[6]) +\
                    ' ' + str(quad_box[7]) + '\n'
                results.append(line)
        return results

    def writeTxt(self):
        for single_xml in self.xml_lists:
            lines = self._readXml(single_xml)
            txt_file = single_xml.replace('xml', 'txt')
            with open(os.path.join(self.txt_path, txt_file), 'w') as f:
                for single_line in lines:
                    f.write(single_line)

    def plotgt(self):
        for single_xml in self.xml_lists:
            single_image = single_xml.replace('xml', 'jpg')
            image = cv2.cvtColor(cv2.imread(os.path.join(self.image_path, single_image), cv2.IMREAD_COLOR),
                                 cv2.COLOR_BGR2RGB)
            lines = self._readXml(single_xml)
            for single_line in lines:
                single_line = single_line.strip().split(' ')
                box = np.array(list(map(float, single_line[1:])))
                cv2.polylines(image, [box.reshape(-1, 2).astype(np.int32)], True, (255, 0, 0), 3)
            plt.imshow(image)
            plt.show()


if __name__ == '__main__':
    convert = Convert(xml_path='/data/fzh/HRSC/train/Annotations/',
                      txt_path='/data/fzh/HRSC/train/train-ground-truth/',
                      image_path='/data/fzh/HRSC/train/images/')

    convert.writeTxt()
