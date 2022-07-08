import numpy as np
import matplotlib.pyplot as plt
import os
from DOTA import DOTA
import dota_utils as util
import scipy.io as sio
import shutil

def filter_anns(anns, cls):
    results=[]
    for ann in anns:
        if ann['name'] == cls:
            results.append(ann)
    return results

def from_one_package(package_name, label):
    imgids = package_name.getImgIds(catNms=[label])
    result = []
    for imgid in imgids:
        anns = package_name.loadAnns(imgId=imgid)
        result = result + filter_anns(anns, label)
        # trainset.showAnns(anns, imgid, 2)

    output = np.zeros((len(result), 10))
    i = 0
    for imgid in imgids:
        anns = package_name.loadAnns(imgId=imgid)
        anns = filter_anns(anns, label)
        for ann in anns:
            output[i, 0] = str(imgid[1:])
            if label == 'ship':
                output[i, 1] = 0
            elif label == 'plane':
                output[i, 1] = 1
            else:
                print('wrong')
            temp = ann['poly']
            output[i, 2:4] = temp[0]
            output[i, 4:6] = temp[1]
            output[i, 6:8] = temp[2]
            output[i, 8:10] = temp[3]
            i = i + 1
    return output

def filter_multifile(datenames):
    filterdatanames = []
    for name in datenames:
        if name not in filterdatanames:
            filterdatanames.append(name)
    return filterdatanames

# valset = DOTA('/home/nieguangtao/dataset/DOTA/val')
# val_ship = from_one_package(valset, 'ship')
# trainset = DOTA('/home/nieguangtao/dataset/DOTA/train')
# train_ship = from_one_package(trainset, 'ship')
# valset = DOTA('/home/nieguangtao/dataset/DOTA/val')
# val_plane = from_one_package(valset, 'plane')
# trainset = DOTA('/home/nieguangtao/dataset/DOTA/train')
# train_plane = from_one_package(trainset, 'plane')
#
# final_result = np.concatenate((train_ship, val_ship, train_plane, val_plane),axis=0)
# sio.savemat('/home/nieguangtao/dataset/dota_ship_plane.mat', {'groundtruth': final_result})
src = '/home/nieguangtao/dataset/DOTA/val/images/'
dst = '/home/nieguangtao/dataset/dota_sp/image/'
valset = DOTA('/home/nieguangtao/dataset/DOTA/val')
imgids_val = valset.getImgIds(catNms=['ship'])
imgids_val = imgids_val + valset.getImgIds(catNms=['plane'])
print("Total val images is ", len(imgids_val))
imgids_val = filter_multifile(imgids_val)
print("Total filter images is ", len(imgids_val))
for file in imgids_val:
    srcfile = src + file + '.png'
    dstfile = dst + str(int(file[1:])) + '.png'
    shutil.copy(srcfile, dstfile)

src = '/home/nieguangtao/dataset/DOTA/train/images/'
trainset = DOTA('/home/nieguangtao/dataset/DOTA/train')
imgids_train = trainset.getImgIds(catNms=['ship'])
imgids_train = imgids_train + trainset.getImgIds(catNms=['plane'])
print("Total train images is ", len(imgids_train))
imgids_train = filter_multifile(imgids_train)
print("Total filter images is ", len(imgids_train))
for file in imgids_train:
    srcfile = src + file + '.png'
    dstfile = dst + str(int(file[1:])) + '.png'
    shutil.copy(srcfile, dstfile)








