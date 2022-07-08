import os.path as osp
import xml.etree.ElementTree as ET
import tempfile

import mmcv
import numpy as np

from .builder import DATASETS
from .xml_style import XMLDataset

import os
import sys
from mmdet.core import rbox2poly_single


@DATASETS.register_module()
class SKU110KRDataset(XMLDataset):

    CLASSES = ( '0',)

    def __init__(self, **kwargs):
        super(SKU110KRDataset, self).__init__(**kwargs)
        self.max_groundtruth = 5000
        self.min_size = 1

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        self.img_ids = img_ids
        for img_id in img_ids:
            filename = 'images/{}.jpg'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                               '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def get_horizen_minAreaRectangle(self, bbox):
        x_list = bbox[0:8:2]
        y_list = bbox[1:8:2]
        y_max = max(y_list)
        y_min = min(y_list)
        x_max = max(x_list)
        x_min = min(x_list)
        h_bbox = [x_min, y_min, x_max, y_max]
        return h_bbox

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        r_bboxes = []
        h_bboxes = []
        labels = []
        r_bboxes_ignore = []
        h_bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.cat2label[name]
            # difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            r_bbox = [
                int(bnd_box.find('x').text),
                int(bnd_box.find('y').text),
                int(bnd_box.find('w').text),
                int(bnd_box.find('h').text),
                float(bnd_box.find('theta').text),
            ]
            r_bbox = rbox2poly_single(r_bbox)
            h_bbox = self.get_horizen_minAreaRectangle(r_bbox)
            ignore = obj.find('difficult')
            if self.min_size:
                assert not self.test_mode
                w = h_bbox[2] - h_bbox[0]
                h = h_bbox[3] - h_bbox[1]
                if min(h_bbox[0], h_bbox[1]) <= 1:
                    ignore = True if min(w,h) < 10 else False
                    # ignore = True
                if w <= self.min_size or h <= self.min_size:
                    ignore = True
                x_all = np.sort(np.array(r_bbox)[0:8:2])
                y_all = np.sort(np.array(r_bbox)[1:8:2])
                if (x_all[2] - x_all[0] < 1) or (y_all[2] - y_all[0] < 1) or (x_all[3] - x_all[1] < 1) or (y_all[3] - y_all[1] < 1):
                    ignore = True
            if ignore:
                r_bboxes_ignore.append(r_bbox)
                h_bboxes_ignore.append(h_bbox)
                labels_ignore.append(label)
            else:
                r_bboxes.append(r_bbox)
                h_bboxes.append(h_bbox)
                labels.append(label)
        if not r_bboxes:
            r_bboxes = np.zeros((0, 8))
            h_bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            r_bboxes = np.array(r_bboxes, ndmin=2) - 1
            h_bboxes = np.array(h_bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not r_bboxes_ignore:
            r_bboxes_ignore = np.zeros((0, 8))
            h_bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            r_bboxes_ignore = np.array(r_bboxes_ignore, ndmin=2) - 1
            h_bboxes_ignore = np.array(h_bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=r_bboxes[:self.max_groundtruth].astype(np.float32),
            # h_bboxes=h_bboxes[:self.max_groundtruth].astype(np.float32),
            labels=labels[:self.max_groundtruth].astype(np.int64),
            bboxes_ignore=r_bboxes_ignore.astype(np.float32),
            # h_bboxes_ignore=h_bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def format_results(self, results, txtfile_prefix=None):
        """Format the results to txt (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            txtfile_prefix (str | None): The prefix of txt files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving txt/png files when txtfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if txtfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            txtfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2txt(results, txtfile_prefix)

        return result_files, tmp_dir

    def det2txt(self, results):
        txt_results_dict = {}
        for idx in range(len(self)):  # num images
            img_id = self.img_ids[idx]
            result = results[idx]
            txt_results_dict[img_id] = []
            for label in range(len(result)):  # num classes
                bboxes = result[label]
                for i in range(bboxes.shape[0]):  # num boxes per class
                    score = float(bboxes[i, -1])
                    if bboxes.shape[-1] <8:
                        coordinate = rbox2poly_single(bboxes[i, :5]).astype('int64')
                    else:
                        coordinate = bboxes[i, :8].astype('int64')
                    category = self.CLASSES[label]
                    if not self.validate_clockwise_points(coordinate):
                        coordinate = coordinate.reshape([4, 2])[::-1, :].reshape(-1)
                    line = ('{} {} {} {} {} {} {} {} {} {}'.format(coordinate[0], coordinate[1], coordinate[2],
                                                                   coordinate[3], coordinate[4], coordinate[5],
                                                                   coordinate[6], coordinate[7],
                                                                   score,
                                                                   category))
                    txt_results_dict[img_id].append(line)
        return txt_results_dict

    def det2txt_cls(self, results):
        txt_results_dict = {}
        for i in self.CLASSES:
            txt_results_dict[i] = []
        for idx in range(len(self)):  # num images
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):  # num classes
                bboxes = result[label]
                for i in range(bboxes.shape[0]):  # num boxes per class
                    coordinate = bboxes[i, :8].astype('int64')
                    score = float(bboxes[i, -1])
                    category = self.CLASSES[label]
                    if not self.validate_clockwise_points(coordinate):
                        coordinate = coordinate.reshape([4, 2])[::-1, :].reshape(-1)
                    line = ('{} {} {} {} {} {} {} {} {} {}'.format(img_id, score,
                                                                   coordinate[0], coordinate[1], coordinate[2],
                                                                   coordinate[3], coordinate[4], coordinate[5],
                                                                   coordinate[6], coordinate[7],
                                                                   ))
                    txt_results_dict[category].append(line)
        return txt_results_dict

    def writetxtlines(self, filename, lines):
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line + '\n')

    def writetxtfiles(self, fileroot, dict):
        for filename, lines in dict.items():
            self.writetxtlines(fileroot + '/' + filename + '.txt', lines)

    def writetxtfiles_cls(self, fileroot, dict):
        for filename, lines in dict.items():
            self.writetxtlines(fileroot + '/Task1_' + filename + '.txt', lines)

    def results2txt(self, results, rootdir):
        fileroot = os.path.join(rootdir, 'det_img_result')
        if not os.path.exists(fileroot):
            os.makedirs(fileroot)
        if isinstance(results[0], list):
            txt_results_dict = self.det2txt(results)
            self.writetxtfiles(fileroot, txt_results_dict)

        fileroot = os.path.join(rootdir, 'det_cls_result')
        if not os.path.exists(fileroot):
            os.makedirs(fileroot)
        if isinstance(results[0], list):
            txt_results_dict = self.det2txt_cls(results)
            self.writetxtfiles_cls(fileroot, txt_results_dict)
        else:
            raise TypeError('invalid type of results')

    def validate_clockwise_points(self, points):
        """
        Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
        """

        if len(points) != 8:
            raise Exception("Points list not valid." + str(len(points)))

        point = [
            [int(points[0]), int(points[1])],
            [int(points[2]), int(points[3])],
            [int(points[4]), int(points[5])],
            [int(points[6]), int(points[7])]
        ]
        edge = [
            (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
            (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
            (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
            (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])
        ]

        summatory = edge[0] + edge[1] + edge[2] + edge[3];
        if summatory > 0:
            return False
        else:
            return True