import os
from mmdet.core import draw_boxes_only, backward_convert, forward_convert
import numpy as np
import glob
import cv2

def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_scores = []
    text_tags = []
    # if not os.path.exists(p):
    #     return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        for line in f.readlines():
            #label = 'text'
            # x1, y1, x2, y2, x3, y3, x4, y4, label, difficult= line.split(' ')[0:10]
            # score = 1.0
            x1, y1, x2, y2, x3, y3, x4, y4, score, label = line.split(' ')[0:10]
            # name, score, x1, y1, x2, y2, x3, y3, x4, y4 = line.split(' ')[0:10]
            #print(label)
            text_polys.append([round(float(x1)), round(float(y1)),
                               round(float(x2)), round(float(y2)),
                               round(float(x3)), round(float(y3)),
                               round(float(x4)), round(float(y4))])
            text_scores.append(float(score))
            text_tags.append(label)

        return np.array(text_polys, dtype=np.int32), np.array(text_scores, dtype=np.float),\
               np.array(text_tags, dtype=np.str)

def load_result_8points(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    n=0
    text_polys = []
    text_scores = []
    text_names = []
    # if not os.path.exists(p):
    #     return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        for line in f.readlines():
            n=n+1
            # if line == '\n':
            #     continue
            try:
                name, score, x1, y1, x2, y2, x3, y3, x4, y4 = line.split(' ')[0:10]
            except:
                print(line)
                print(n)
                exit()
            text_polys.append([round(float(x1)), round(float(y1)),
                               round(float(x2)), round(float(y2)),
                               round(float(x3)), round(float(y3)),
                               round(float(x4)), round(float(y4))])
            text_scores.append(float(score))
            text_names.append(name)
        return np.array(text_polys, dtype=np.int32), np.array(text_scores, dtype=np.float),\
               np.array(text_names, dtype=np.str)

def show_result(img,
                r_bboxes,
                scores,
                score_thr=0.2):
    if scores.size == 0:
        return img
    # r_bboxes = np.vstack(boxes)
    # draw bounding boxes
    indices = scores >= score_thr
    # detected_scores = scores[indices,-1]
    detected_boxes = r_bboxes[indices,:8]
    img_show = draw_boxes_only(img, boxes=detected_boxes, method=1)
    return img_show

def draw_boxes_and_save():
    # txt_path = '/home/nieguangtao/programing/mmdetection/work_dirs/HRSC/fcos_r101_fpn_rotate/det_img_result/'
    # img_path = '/home/nieguangtao/programing/mmdetection/data/HRSC/images/'
    txt_path = '/home/nieguangtao/programing/mmdetection/work_dirs/fcos_test/det_img_result/'
    # txt_path = '/home/nieguangtao/programing/mmdetection/work_dirs/fcos_r101_fpn_rotate_addP2/det_img_result'
    # txt_path = '/home/nieguangtao/dataset/train_overlap200/labelTxt'
    img_path = '/home/nieguangtao/programing/mmdetection/data/DOTA_train/images/'
    if not os.path.exists(os.path.join(txt_path, '../saveimgs_0_6')):
        os.makedirs(os.path.join(txt_path, '../saveimgs_0_6'))
    # txts = os.listdir(txt_path)
    # txts.remove('imgs')
    txts = []
    with open('./mytools/fail.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            txts.append(line)
    for count, t in enumerate(txts):
        t = t + '.txt'                                                     #attention
        boxes, scores, labels = load_annoataion(os.path.join(txt_path, t))
        img_name = t.replace('.txt', '.png')
        save_name = t.replace('.txt', '.jpg')
        img = cv2.imread(os.path.join(img_path, img_name))
        if img is None:
            continue
        img = show_result(img, boxes, scores, score_thr=0.4)
        # cv2.imshow("show", img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(txt_path, '../saveimgs_0_6/', save_name) ,img)
        if count % 10 == 0:
            print(count)

def change_boxes_format():
    txt_path = '/home/nieguangtao/programing/mmdetection/work_dirs/HRSC/fcos_r101_fpn_rotate/det_cls_result/'
    filename = txt_path + 'l1/comp4_det_test_'
    #txts = os.listdir(txt_path)
    txts = ['Task1_ship.txt',]
    result_lines = []
    for count, t in enumerate(txts):
        boxes, scores, names = load_result_8points(os.path.join(txt_path, t))
        boxes = backward_convert(boxes, with_label=False)
        for box, score, name in zip (boxes, scores, names):
            line = ('{} {} {} {} {} {} {}'.format(name, score,
                          box[0], box[1], box[2], box[3], box[4]))
            result_lines.append(line)
        with open(filename + t.split('_')[-1], 'w') as f:
            for line in result_lines:
                f.write(line + '\n')
if __name__ == "__main__":
    draw_boxes_and_save()
    # change_boxes_format()
