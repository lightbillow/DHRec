import cv2
import numpy as np
import torch
import warnings
import shapely
from shapely.geometry import Polygon

def sort_convert(rect):
    c_x = rect[0]
    c_y = rect[1]
    offset_x = rect[2] / 2
    offset_y = rect[3] / 2
    angle = rect[4] * np.pi / 180

    off_1 = np.stack([-offset_x, -offset_y], axis=-1)
    off_2 = np.stack([-offset_x, offset_y], axis=-1)
    off_3 = np.stack([offset_x, offset_y], axis=-1)
    off_4 = np.stack([offset_x, -offset_y], axis=-1)

    off = np.stack([off_1, off_2, off_3, off_4], axis=0)
    x = off[:,0] * np.cos(angle) - off[:,1] * np.sin(angle) + c_x
    y = off[:,0] * np.sin(angle) + off[:,1] * np.cos(angle) + c_y
    # x = x.clip(0, 800)
    # y = y.clip(0, 800)
    return [x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3]]

def match(true_sort, gt):
    # gt1 = gt.reshape([4,2])
    # gt1 = gt1[::-1, :]
    # gt1 = gt1.reshape(-1)
    gt1 = gt.reshape([4,2])[::-1, :].reshape(-1)
    gt = np.hstack([gt, gt])
    gt1 = np.hstack([gt1,gt1])
    all = []
    for i in range(4):
        all.append(gt[i*2:i*2+8])
        all.append(gt1[i * 2:i * 2 + 8])
    all = np.vstack(all)
    compare = np.sum(abs(all-true_sort),1)
    index = np.argmin(compare)
    return all[index]


def forward_convert(ground_truth, coordinate, with_label=True, with_qbb=True):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    boxes = []
    if with_label:
        for gt, rect in zip(ground_truth, coordinate):
            # box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            # box = np.reshape(box, [-1, ])
            true_sort = sort_convert(rect)
            if with_qbb:
                box = match(true_sort, gt)
            else:
                box = true_sort
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for gt, rect in zip(ground_truth, coordinate):
            # box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            # box = np.reshape(box, [-1, ])
            true_sort = sort_convert(rect)
            if with_qbb:
                box = match(true_sort, gt)
            else:
                box = true_sort
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    return np.array(boxes, dtype=np.float32)


def backward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            # if np.abs(theta) > 45:
            #     temp = w
            #     w = h
            #     h = temp
            #     theta = theta + 90
            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            # if w==0 or h==0:
            #     warnings.warn('Skip the groundtruth with area equal to 0!')
            #     continue
            # if np.abs(theta) > 45:
            #     temp = w
            #     w = h
            #     h = temp
            #     theta = theta + 90
            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)

def get_horizen_minAreaRectangle(bbox):
    x_list = bbox[:,0:8:2]
    y_list = bbox[:,1:8:2]
    y_max = np.max(y_list, axis=1)
    y_min = np.min(y_list, axis=1)
    x_max = np.max(x_list, axis=1)
    x_min = np.min(x_list, axis=1)
    h_bbox = np.vstack((x_min, y_min, x_max, y_max)).T

    return h_bbox

def get_horizen_from_Rotated(proposal_list, use_cpu=False):
    if use_cpu:
        proposal_list = proposal_list.cpu().numpy()
        proposal_list = forward_convert(proposal_list, with_label=False)
        proposal_list = get_horizen_minAreaRectangle(proposal_list)
        proposal_list = torch.from_numpy(proposal_list).to('cuda')
    else:
        proposal_list = my_angle_convert(proposal_list)
    return proposal_list

def area_factor_compute(coordinates):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """
    obliquity_factors = []
    direction_factors = []
    for rect in coordinates:
        # box = np.int0(rect)
        box = rect
        x_all = np.sort(box[0:8:2])
        y_all = np.sort(box[1:8:2])
        # line_y = 0.75*y_all[0] + 0.25*y_all[3]
        # line_y = y_all[1] + 0.1
        # x_c = x_all.sum() / 4
        # line_x = x_all[1] + 0.1
        # y_c = y_all.sum() / 4
        # left_retangle = np.array([x_all[0], y_all[0], x_all[0], line_y, x_c, line_y, x_c, y_all[0]])
        # right_retangle = np.array([x_all[3], y_all[0], x_all[3], line_y, x_c, line_y, x_c, y_all[0]])

        # left_retangle = np.array([x_all[1]-0.1, y_all[0], x_all[1]-0.1, line_y, x_c, line_y, x_c, y_all[0]])
        # right_retangle = np.array([x_all[2]+0.1, y_all[0], x_all[2]+0.1, line_y, x_c, line_y, x_c, y_all[0]])

        # top_retangle = np.array([x_all[0], y_all[1]-0.1, line_x, y_all[1]-0.1, line_x, y_c, x_all[0], y_c])
        # bottom_retangle = np.array([x_all[0], y_all[2]+0.1, line_x, y_all[2]+0.1, line_x, y_c, x_all[0], y_c])

        h_box = np.array([x_all[0], y_all[0], x_all[0], y_all[3], x_all[3], y_all[3], x_all[3], y_all[0]])
        
        box = Polygon(box.reshape([4, 2])).convex_hull
        h_box = Polygon(h_box.reshape([4, 2])).convex_hull

        obliquity_factor = box.area / (h_box.area + 1e-6)
        err = box.area * 0.05

        # err = 1 if (y_all[1] - y_all[0] + 1) < 2 else 0
        # err = max(y_all[1] - y_all[0] , 1) * (x_all[2] - x_all[1]) * obliquity_factor * 2 if obliquity_factor > 0.9 else 0
        triangle_left = np.array([x_all[0], y_all[1], x_all[1], y_all[0], x_all[3], y_all[1]])
        triangle_right = np.array([x_all[0], y_all[1], x_all[2], y_all[0], x_all[3], y_all[1]])
        triangle_left = Polygon(triangle_left.reshape([3, 2])).convex_hull
        triangle_right = Polygon(triangle_right.reshape([3, 2])).convex_hull
        area1 = box.intersection(triangle_left).area + err
        area2 = box.intersection(triangle_right).area + err
        direction_factor1 = area1 / max((area1 + area2), 1e-6)

        # err = 1 if (x_all[1] - x_all[0]) < 2 else 0
        # err = max(x_all[1] - x_all[0] , 1) * (y_all[2] - y_all[1]) * obliquity_factor * 2 if obliquity_factor > 0.9 else 0
        triangle_top = np.array([x_all[1], y_all[0], x_all[0], y_all[1], x_all[1], y_all[3]])
        triangle_bottom = np.array([x_all[1], y_all[0], x_all[0], y_all[2], x_all[1], y_all[3]])
        triangle_top = Polygon(triangle_top.reshape([3, 2])).convex_hull
        triangle_bottom = Polygon(triangle_bottom.reshape([3, 2])).convex_hull
        area1 = box.intersection(triangle_top).area + err
        area2 = box.intersection(triangle_bottom).area + err
        direction_factor2 = area1 / max((area1 + area2), 1e-6)
        
        direction_factor = [direction_factor1, direction_factor2]
        obliquity_factors.append(obliquity_factor)
        direction_factors.append(direction_factor)
        # if ((obliquity_factor>0.5) & (direction_factor1>0.5) & (direction_factor2>0.5)):
        #     print(rect)
        #     print(obliquity_factor, direction_factor1, direction_factor2)
    return np.array(obliquity_factors, dtype=np.float32), np.array(direction_factors, dtype=np.float32)