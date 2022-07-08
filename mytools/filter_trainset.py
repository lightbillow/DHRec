import os
import numpy as np
import cv2

def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    # if not os.path.exists(p):
    #     return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        for line in f.readlines():
            #label = 'text'
            x1, y1, x2, y2, x3, y3, x4, y4 ,label= line.split(' ')[0:9]
            #print(label)
            text_polys.append([float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)])
            text_tags.append(label)

        return np.array(text_polys, dtype=np.int32), np.array(text_tags, dtype=np.str)


if __name__ == "__main__":
    txt_path = 'data/DOTA_train/labelTxt/'
    train_list_file = 'data/DOTA_train/val.txt'
    save_list_file = 'data/DOTA_train/val_filter.txt'
    validline = []
    totalnum = 0
    with open(train_list_file, 'r') as f:
        for line in f.readlines():
            totalnum += 1
            txt_file_path = txt_path + line.strip('\n') + '.txt'
            boxes, labels = load_annoataion(txt_file_path)
            if len(boxes) > 0:
                validline.append(line)
            else:
                print("Ignore the file %s with no boxes", line)
    print("The total %d ----> After filter %d", totalnum, len(validline))
    with open(save_list_file, 'w') as f:
        for line in validline:
            f.write(line)