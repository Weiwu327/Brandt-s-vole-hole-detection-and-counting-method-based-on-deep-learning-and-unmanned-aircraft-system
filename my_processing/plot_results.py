# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from draw_box_utils import draw_objs


def plot_results(img_file,integration_dict):
    # read class_indict
    label_json_path = '/home/miivii/wuwei-project/grassland-project/brandt-vole-hole/FCOS-inference/model_data/hole_classes.json'
    assert os.path.exists(label_json_path), "class json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r', encoding='utf-8') as f:
        class_dict = json.load(f)
    category_index = {str(v): str(k) for k, v in class_dict.items()}

    original_img = Image.open(img_file)
    info_boxes = np.array(integration_dict['boxes'])
    info_classes = np.array(integration_dict['classes'])
    info_scores = np.array(integration_dict['scores'])

    plot_img = draw_objs(original_img,
                         info_boxes,
                         info_classes,
                         info_scores,
                         category_index=category_index,
                         box_thresh=0.1,
                         line_thickness=3,
                         font='arial.ttf',  # font='arial.ttf'
                         font_size=20)
    plt.imshow(plot_img)
    plt.show()
    # 保存预测的图片结果
    # plot_img.save(save_path + json_id + '.jpg')


