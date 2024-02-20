
import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.fcos import FCOS
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import DecodeBox


class Fcos(object):
    _defaults = {
        "model_path": '/home/miivii/wuwei-project/grassland-project/brandt-vole-hole/FCOS-inference/model_data/best_epoch_weights.pth',
        "classes_path"      : '/home/miivii/wuwei-project/grassland-project/brandt-vole-hole/FCOS-inference/model_data/hole_classes.txt',

        "input_shape"       : [640, 640],
        "strides"           : [8, 16, 32, 64, 128],
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "letterbox_image"   : True,
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.bbox_util                      = DecodeBox(self.strides)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()
        
        # show_config(**self._defaults)

    def generate(self, onnx=False):
        self.net = FCOS(self.num_classes)
        total = sum([param.nelement() for param in self.net.parameters()])
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        if not os.path.exists(map_out_path):
            os.makedirs(map_out_path)
        f = open(os.path.join(map_out_path, image_id[:-4] + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs, self.input_shape)
            results = self.bbox_util.non_max_suppression(outputs, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                               
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(round(left, 2)), str(round(top, 2)), str(round(right, 2)), str(round(bottom, 2))))

        f.close()
        return 
