# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import xml.etree.ElementTree as ET
import os
#TODO: finish it
#import cPickle
import numpy as np
import matplotlib.pyplot as plt
import polyiou
from multiprocessing import Pool
from functools import partial
from mmdet.core import draw_boxes_only
import cv2

def parse_gt(filename):
    """

    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def show_the_results(image_id, classname, BB, BBGT, tp, miss, confidence, thresh=0.3):
    img_path = r'/home/nieguangtao/dataset/DOTA/val/images/{:s}.png'
    image = img_path.format(image_id)
    BB_TP = BB[(tp == 1) * (confidence >= thresh), :]
    BB_FP = BB[(tp == 0) * (confidence >= thresh), :]
    BB_MISS = BBGT[np.array(miss)==False]

    img_array = cv2.imread(image)
    img_array = draw_boxes_only(img_array, BB_TP, method=1, color=(0,255,0)) #green success
    img_array = draw_boxes_only(img_array, BB_FP, method=1, color=(255, 0, 0)) #blue wrong
    img_array = draw_boxes_only(img_array, BB_MISS, method=1, color=(0, 0, 255))  # red miss
    cv2.imwrite('./results/' + image_id + '_' + classname + '.jpg', img_array)

    show_scale = 1000.0 / np.max(img_array.shape)
    img_array = cv2.resize(img_array, (0,0), fx=show_scale, fy=show_scale, interpolation=cv2.INTER_LINEAR)
    cv2.putText(img_array,
                classname,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                [0, 0, 0], 2)
    cv2.imshow("show", img_array)
    cv2.waitKey(0)
    print("done!")

def show_groundtruth(image_id, BBGT):
    img_path = r'/home/nieguangtao/dataset/DOTA/val/images/{:s}.png'
    image = img_path.format(image_id)

    img_array = cv2.imread(image)
    img_array = draw_boxes_only(img_array, BBGT, method=1, color=(0,255,0)) #green success
    cv2.imwrite('./results/groundtruth/' + image_id + '.jpg', img_array)

    # show_scale = 1000.0 / np.max(img_array.shape)
    # img_array = cv2.resize(img_array, (0,0), fx=show_scale, fy=show_scale, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("show", img_array)
    # cv2.waitKey(0)
    print("done!")

def show_all_groundtruth(annopath, imagesetfile):
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_gt(annopath.format(imagename))
    for imagename in imagenames:
        R = [obj for obj in recs[imagename]]
        bbox = np.array([x['bbox'] for x in R])
        show_groundtruth(imagename, bbox)

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    ap_singleimg = {}
    print('eval ' + classname)
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    #print('imagenames: ', imagenames)
    #if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))
        #if i % 100 == 0:
         #   print ('Reading annotation for {:d}/{:d}'.format(
          #      i + 1, len(imagenames)) )
        # save
        #print ('Saving cached annotations to {:s}'.format(cachefile))
        #with open(cachefile, 'w') as f:
         #   cPickle.dump(recs, f)
    #else:
        # load
        #with open(cachefile, 'r') as f:
         #   recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    # npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        # npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    confidence = confidence[sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs

    single_class_det = {}

    for i, image_id in  enumerate(image_ids):
        if not image_id in single_class_det:
            single_class_det[image_id] = {}
            single_class_det[image_id]['bbox_index'] = []
        single_class_det[image_id]['bbox_index'].append(i)

    choose = ['P1508', 'P1390', 'P1179', 'P1397', 'P2791', 'P1184', 'P0168', 'P1384', 'P2271', 'P1154']
    for image_id in single_class_det:
        if not image_id in choose:
            continue
        nd = len(single_class_det[image_id]['bbox_index'])
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        npos = sum(~class_recs[image_id]['difficult'])
        for d in range(nd):
            R = class_recs[image_id]
            bb = BB[single_class_det[image_id]['bbox_index'][d], :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            ## compute det bb with each BBGT

            if BBGT.size > 0:
                # compute overlaps
                # intersection

                def calcoverlaps(BBGT, bb):
                    overlaps = []
                    for index, GT in enumerate(BBGT):

                        overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT[index]), polyiou.VectorDouble(bb))
                        overlaps.append(overlap)
                    return overlaps
                overlaps = calcoverlaps(BBGT, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = True
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall

        # print('check fp:', fp)
        # print('check tp', tp)

        show_the_results(image_id, classname, BB[single_class_det[image_id]['bbox_index']], BBGT,
                         tp, R['det'], confidence[single_class_det[image_id]['bbox_index']])

        # print('npos num:', npos)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        ap_singleimg[image_id] = ap
        # return rec, prec, ap
    return ap_singleimg

def single_voc_eval_warp(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=ovthresh,
             use_07_metric=use_07_metric)
    return ap
def main():

    # detpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_testnms_c_extension_0.1/comp4_det_test_{:s}.txt'
    # annopath = r'/home/dingjian/evaluation_task1/testset/wordlabel-utf-8/{:s}.txt'
    # imagesetfile = r'/home/dingjian/evaluation_task1/testset/testset.txt'
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'turntable', 'harbor', 'swimming-pool', 'helicopter']

    detpath = r'PATH_TO_BE_CONFIGURED/Task1_{:s}.txt'
    annopath = r'PATH_TO_BE_CONFIGURED/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = r'PATH_TO_BE_CONFIGURED/valset.txt'

    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)

        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
       # plt.show()
    map = map/len(classnames)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)

def eval_DOTA_Task1(detpath, annopath, imagesetfile):

    # detpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_testnms_c_extension_0.1/comp4_det_test_{:s}.txt'
    # annopath = r'/home/dingjian/evaluation_task1/testset/wordlabel-utf-8/{:s}.txt'
    # imagesetfile = r'/home/dingjian/evaluation_task1/testset/testset.txt'
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'turntable', 'harbor', 'swimming-pool', 'helicopter']

    # detpath = r'PATH_TO_BE_CONFIGURED/Task1_{:s}.txt'
    # annopath = r'PATH_TO_BE_CONFIGURED/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    # imagesetfile = r'PATH_TO_BE_CONFIGURED/valset.txt'
    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

    classaps = []
    map = 0
    # TODO: change it to pool
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)

        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
        # plt.savefig
        #plt.show()
    map = map/len(classnames)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)
    # with open(detpath + '/mAP.txt', 'w') as f_out:
    #     f_out.write('mAP: ' + str(map) + '\n')
    #     f_out.write('classaps: ' + str(classaps))
    return map, classaps

def eval_HRSC_L1(detpath, annopath, imagesetfile):

    # detpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_testnms_c_extension_0.1/comp4_det_test_{:s}.txt'
    # annopath = r'/home/dingjian/evaluation_task1/testset/wordlabel-utf-8/{:s}.txt'
    # imagesetfile = r'/home/dingjian/evaluation_task1/testset/testset.txt'
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'turntable', 'harbor', 'swimming-pool', 'helicopter']

    # detpath = r'PATH_TO_BE_CONFIGURED/Task1_{:s}.txt'
    # annopath = r'PATH_TO_BE_CONFIGURED/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    # imagesetfile = r'PATH_TO_BE_CONFIGURED/valset.txt'
    classnames = ['ship']
    classaps = []
    map = 0
    # TODO: change it to pool
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)

        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
        # plt.savefig
        #plt.show()
    map = map/len(classnames)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)
    # with open(detpath + '/mAP.txt', 'w') as f_out:
    #     f_out.write('mAP: ' + str(map) + '\n')
    #     f_out.write('classaps: ' + str(classaps))
    return map, classaps


def eval_vehicle(detpath, annopath, imagesetfile):

    # detpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_testnms_c_extension_0.1/comp4_det_test_{:s}.txt'
    # annopath = r'/home/dingjian/evaluation_task1/testset/wordlabel-utf-8/{:s}.txt'
    # imagesetfile = r'/home/dingjian/evaluation_task1/testset/testset.txt'
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'turntable', 'harbor', 'swimming-pool', 'helicopter']

    # detpath = r'PATH_TO_BE_CONFIGURED/Task1_{:s}.txt'
    # annopath = r'PATH_TO_BE_CONFIGURED/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    # imagesetfile = r'PATH_TO_BE_CONFIGURED/valset.txt'
    classnames = ['vehicle']
    classaps = []
    map = 0
    # TODO: change it to pool
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)

        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
        # plt.savefig
        #plt.show()
    map = map/len(classnames)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)
    # with open(detpath + '/mAP.txt', 'w') as f_out:
    #     f_out.write('mAP: ' + str(map) + '\n')
    #     f_out.write('classaps: ' + str(classaps))
    return map, classaps

def eval_DOTA_Task1_multi_process(detpath, annopath, imagesetfile):

    # detpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_testnms_c_extension_0.1/comp4_det_test_{:s}.txt'
    # annopath = r'/home/dingjian/evaluation_task1/testset/wordlabel-utf-8/{:s}.txt'
    # imagesetfile = r'/home/dingjian/evaluation_task1/testset/testset.txt'
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'turntable', 'harbor', 'swimming-pool', 'helicopter']
    # detpath = r'PATH_TO_BE_CONFIGURED/Task1_{:s}.txt'
    # annopath = r'PATH_TO_BE_CONFIGURED/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    # imagesetfile = r'PATH_TO_BE_CONFIGURED/valset.txt'
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'] #, 'container-crane']
    classnames = ['helicopter']
    # pool = Pool(80)
    pool = Pool(1)
    classaps = []
    mAP = 0
    # TODO: change it to pool
    eval_fn = partial(single_voc_eval_warp, detpath, annopath, imagesetfile, ovthresh=0.5, use_07_metric=True)
    aps = pool.map(eval_fn, classnames)
    # for classname in classnames:
    #     print('classname:', classname)
    #     rec, prec, ap = voc_eval(detpath,
    #          annopath,
    #          imagesetfile,
    #          classname,
    #          ovthresh=0.5,
    #          use_07_metric=True)
    #     map = map + ap
    #     #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
    #     print('ap: ', ap)
    #     classaps.append(ap)
    #
    #     # umcomment to show p-r curve of each category
    #     # plt.figure(figsize=(8,4))
    #     # plt.xlabel('recall')
    #     # plt.ylabel('precision')
    #     # plt.plot(rec, prec)
    #     # plt.savefig
    #     #plt.show()

    single_AP = {}
    for i in range(len(classnames)):
        print('classname:', classnames[i])
        for img_id, ap_value in aps[i].items():
            if not img_id in single_AP:
                single_AP[img_id] = -np.ones([15])
            single_AP[img_id][i] = ap_value

    # for i in range(len(classnames)):
    #     print('classname:', classnames[i])
    #     mAP = mAP + aps[i]
    #     print('ap: ', aps[i])
    # mAP = mAP/len(classnames)
    # print('map:', mAP)
    # classaps = 100*np.array(aps)
    # print('classaps: ', classaps)
    with open(detpath[:-14] + 'AP_per_image1.txt', 'w') as f_out:
        for img_id, ap_value in single_AP.items():
            line = "%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f"\
                   % (img_id, ap_value[0], ap_value[1], ap_value[2], ap_value[3], ap_value[4], ap_value[5], ap_value[6],
                      ap_value[7], ap_value[8], ap_value[9], ap_value[10], ap_value[11], ap_value[12], ap_value[13], ap_value[14])

            f_out.write(line + '\n')
            # f_out.write('classaps: ' + str(classaps))
    # return mAP, classaps

if __name__ == '__main__':
    # detpath = os.path.join(r'/data1/nieguangtao/programing/RoITransformer_DOTA/output/rcnn/DOTA/trainset/resnet_v1_101_dota_RoITransformer_trainval_rcnn_end2end/val/Task1_results_0.1_nms') + '/Task1_{:s}.txt'
    # detpath = os.path.join(r'/data1/nieguangtao/programing/RoITransformer_DOTA/output/fpn/DOTA/trainset/resnet_v1_101_dota_rotbox_light_head_RoITransformer_trainval_fpn_end2end/val/Task1_results_0.1_nms') + '/Task1_{:s}.txt'
    detpath = os.path.join(r'/home/nieguangtao/programing/new/mmdetection/work_dirs/fcos/Res101_caffe_deformable_train3/det_cls_result_nms') + '/Task1_{:s}.txt'
    annopath = r'/home/nieguangtao/dataset/DOTA/val/labelTxt/{:s}.txt'
    imagesetfile = r'/home/nieguangtao/dataset/DOTA/val.txt'
    eval_DOTA_Task1_multi_process(detpath, annopath, imagesetfile)
    # show_all_groundtruth(annopath, imagesetfile)
