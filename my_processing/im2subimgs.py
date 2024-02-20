import cv2


def segment(im_path,subimgs_path):
    img = cv2.imread(im_path)
    seg_col = 0
    for i in range(5):
        if i != 0:
            seg_col = seg_col - 85
        seg_row = 0
        for j in range(5):
            if j != 0:
                seg_row = seg_row - 87
            part = img[seg_col:(seg_col+1160), seg_row:(seg_row+1708)]
            savepath = subimgs_path + str(i) + '_' + str(j) + '.jpg'
            cv2.imwrite(savepath, part)
            seg_row = seg_row + 1708
        seg_col = seg_col + 1160
