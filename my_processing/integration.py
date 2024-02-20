import json
import os
from glob import glob


def integrate(txt_root):
    boxes = []
    classes = []
    scores = []
    results = []
    col = 1708 - 87
    row = 1160 - 85
    txt_list = glob(os.path.join(txt_root, '*.txt'))
    txt_id = txt_root.split('/')[-2]
    # print(txt_id)

    for txt_path in txt_list:
        i = int(txt_path[-7])
        j = int(txt_path[-5])
        f = open(txt_path)
        info = f.readlines()
        if len(info) == 0:
            a = 0
            # print("没有目标！")
        else:
            box = []
            score = []
            cls = []
            result = []
            for p in range(len(info)):
                infop = info[p].split()
                left = float(infop[-4]) + j * col
                top = float(infop[-3]) + i * row
                right = float(infop[-2]) + j * col
                bottom = float(infop[-1]) + i * row
                sco = float(infop[-5])
                if infop[-6] == 'hole':
                    cl = int(1)
                else:
                    cl = int(0)
                # print(cl)
                box += [[left, top, right, bottom]]  # left, top, right, bottom
                score += [sco]
                cls += [cl]
                result += [[sco, round(left,2), round(top,2), round(right,2), round(bottom,2)]]
            # print(box, score, cls)
        boxes += box
        scores += score
        classes += cls
        results += result
        f.close()

    # if len(results) == 0:
    #     print("null")
    # else:
    #     print(results)

    integration_dict = {"img_name": txt_id[7:] + '.jpg', "boxes": boxes, "classes": classes, "scores": scores}

    return integration_dict

