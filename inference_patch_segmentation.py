from PIL import Image
from utils.utils import get_classes
from my_processing.im2subimgs import segment
from my_processing.img2voc import load_image_file
from my_processing.integration import integrate
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from my_processing.draw_box_utils import draw_objs
from fcos import Fcos
import argparse
import glob
import os.path as osp
import imgviz
import labelme
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--input_path', default='/home/miivii/wuwei-project/grassland-project/brandt-vole-hole/FCOS-inference/input/0/DJI_20220410124835_0003.JPG', type=str, help='path to input images')
parser.add_argument('--input_path', default='../VOCdevkit/tif_img', type=str, help='path to input images')
parser.add_argument('--output_path', default='./output/tifimg_results/', type=str, help='path to output results')
args = parser.parse_args()

if __name__ == "__main__":
    classes_path    = '/home/miivii/wuwei-project/grassland-project/brandt-vole-hole/FCOS-inference/model_data/hole_classes.txt'
    MINOVERLAP      = 0.5
    map_vis         = False
    input_path  = args.input_path
    output_path    = args.output_path

    # image_ids = os.listdir(input_path)
    # for image_path in tqdm(image_ids):
    #     image_id = image_path.split('/')[-1]
    #     TEMP_FILES_PATH = os.path.join('/home/miivii/wuwei-project/grassland-project/brandt-vole-hole/FCOS-inference/temp/.'+ image_id[:-4])
    #     subimgs_path = TEMP_FILES_PATH + '/imgs/'
    #     voc_subimgs_path = TEMP_FILES_PATH + '/voc_imgs/'
    #     txt_path = TEMP_FILES_PATH + '/txt/'
    #     if not os.path.exists(TEMP_FILES_PATH):
    #         os.makedirs(TEMP_FILES_PATH)
    #     if not os.path.exists(subimgs_path):
    #         os.makedirs(subimgs_path)
    #     if not os.path.exists(voc_subimgs_path):
    #         os.makedirs(voc_subimgs_path)
    #     if not os.path.exists(txt_path):
    #         os.makedirs(txt_path)

        # image_path = os.path.join(input_path, image_path)
        # segment(image_path,subimgs_path)

        # for filename in glob.glob(osp.join(subimgs_path, "*.jpg")):
        #     imageData = load_image_file(filename)
        #     img = labelme.utils.img_data_to_arr(imageData)
        #     imgviz.io.imsave(filename.replace('imgs', 'voc_imgs'), img)

    class_names, _ = get_classes(classes_path)

    fcos = Fcos(confidence=0.1, nms_iou=0.5)
    for image_path in glob.glob(osp.join(input_path, "*.jpg")):
        image_id2 = image_path.split('/')[-1]
        image = Image.open(image_path)
        fcos.get_map_txt(image_id2, image, class_names, output_path)
        # integration = integrate(txt_path)
        # json_path = output_path + image_id[:-4] + '.json'
        # with open(json_path, 'w') as ff:
        #     json.dump(integration, ff, indent=4, ensure_ascii=False)
        ###########################################
        # shutil.rmtree(TEMP_FILES_PATH)
        ###################plot####################
        # label_json_path = '/home/miivii/wuwei-project/grassland-project/brandt-vole-hole/FCOS-inference/model_data/hole_classes.json'
        # assert os.path.exists(label_json_path), "class json file {} dose not exist.".format(label_json_path)
        # with open(label_json_path, 'r', encoding='utf-8') as f:
        #     class_dict = json.load(f)
        # category_index = {str(v): str(k) for k, v in class_dict.items()}
        # original_img = Image.open(input_path)
        # info_boxes = np.array(integration['boxes'])
        # info_classes = np.array(integration['classes'])
        # info_scores = np.array(integration['scores'])
        # plot_img = draw_objs(original_img,
        #                      info_boxes,
        #                      info_classes,
        #                      info_scores,
        #                      category_index=category_index,
        #                      box_thresh=0.62,
        #                      line_thickness=3,
        #                      font='arial.ttf',  # font='arial.ttf'
        #                      font_size=20)
        # plt.imshow(plot_img)
        # # plt.show()
        # save_path = './output/'
        # plot_img.save(save_path + image_id[:-4] + '.jpg')