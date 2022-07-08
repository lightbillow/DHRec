import json
import numpy as np
import os.path as osp
import tempfile
import mmcv
import os
import sys
sys.path.insert(0,'./mytools')
from dota_kit.ResultMerge_multi_process import mergebypoly
from mmdet.core import rbox2poly_single
from .custom import CustomDataset
from .builder import DATASETS


@DATASETS.register_module
class DotaOBBDataset(CustomDataset):
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    def load_annotations(self, ann_file):
        '''
        load annotations from .json ann_file
        '''
        # self.cat2label = {
        #     cat_id: i + 1
        #     for i, cat_id in enumerate(self.CLASSES)
        # }
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.CLASSES)}
        with open(ann_file, 'r') as f_ann_file:
            self.data_dicts = json.load(f_ann_file)
            self.img_infos = []
            self.img_names = []
            for data_dict in self.data_dicts:
                img_info = {}
                img_info['filename'] = data_dict['filename']
                img_info['height'] = data_dict['height']
                img_info['width'] = data_dict['width']
                img_info['id'] = data_dict['id']
                self.img_infos.append(img_info)
                self.img_names.append(data_dict['filename'])
        return self.img_infos

    def get_ann_info(self, idx):
        ann_dict = self.data_dicts[idx]['annotations']
        ann = {}
        bboxes = np.array(ann_dict['bboxes'])
        bboxes_ignore = np.array(ann_dict['bboxes_ignore'])
        if not bboxes.any():
            bboxes = np.zeros((0, 8))
            labels = np.zeros((0, ))
        if not bboxes_ignore.any():
            bboxes_ignore = np.zeros((0, 8))
            labels_ignore = np.zeros((0, ))
        ann['bboxes'] = bboxes.astype(np.float32) - 1
        ann['bboxes_ignore'] = bboxes_ignore.astype(np.float32) - 1
        if len(ann_dict['labels']):
            ann['labels'] = np.array([self.cat2label[label]
                                      for label in ann_dict['labels']]).astype(np.int64)
        else:
            ann['labels'] = labels.astype(np.int64)

        if len(ann_dict['labels_ignore']):
            ann['labels_ignore'] = np.array(
                [self.cat2label[label] for label in ann_dict['labels_ignore']]).astype(np.int64)
        else:
            ann['labels_ignore'] = labels_ignore.astype(np.int64)

        return ann

    def get_cat_ids(self, idx):
        ann_dict = self.data_dicts[idx]['annotations']
        if len(ann_dict['labels']):
            return [self.cat2label[label] for label in ann_dict['labels']]
        else:
            return []

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

            nms_path = os.path.join(rootdir, 'det_cls_result_nms')
            if not os.path.exists(nms_path):
                os.mkdir(nms_path)
            mergebypoly(fileroot, nms_path)
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