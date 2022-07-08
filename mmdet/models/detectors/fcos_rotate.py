from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import draw_boxes_with_label_and_scores, draw_boxes_only, tensor2imgs, bbox2result
import cv2
import numpy as np
import mmcv


@DETECTORS.register_module()
class FCOS_Rotate(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOS_Rotate, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)


    def forward_train(self,
                      img,
                      img_metas,
                      r_gt_bboxes_five,
                      r_gt_bboxes_eight,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # self.show_groundtruth(img, r_gt_bboxes_eight)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, r_gt_bboxes_five,
             r_gt_bboxes_eight, gt_labels, gt_bboxes_ignore)
        return losses

    # def simple_test(self, img, img_metas,
    #                 r_gt_bboxes_five,
    #                 r_gt_bboxes_eight,
    #                 gt_labels,
    #                 rescale=False):
    #     """Test function without test time augmentation.
    #
    #     Args:
    #         imgs (list[torch.Tensor]): List of multiple images
    #         img_metas (list[dict]): List of image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.
    #
    #     Returns:
    #         np.ndarray: proposals
    #     """
    #     # print(img_metas[0]['filename'])
    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #
    #     # Use Groundtruth
    #     # self.show_groundtruth(img, r_gt_bboxes_eight[0])
    #     cls_scores, bbox_preds, reg_confidences, cls_scores_init = outs[:4]
    #     assert len(cls_scores) == len(bbox_preds) == len(reg_confidences) == len(cls_scores_init)
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     all_level_points = self.bbox_head.get_points(featmap_sizes, bbox_preds[0].dtype,
    #                                        bbox_preds[0].device)
    #
    #     bbox_preds_nostride = []                      # no * stride
    #     for bbox_pred, stride in zip(bbox_preds, self.bbox_head.strides):
    #         bbox_preds_nostride.append(bbox_pred / stride)
    #
    #     # score=iou
    #     bbox_targets_weight, bbox_targets, labels, labels_score, labels_weight = \
    #         self.bbox_head.get_targets(all_level_points, bbox_preds_nostride, r_gt_bboxes_five[0], r_gt_bboxes_eight[0],
    #                                             gt_labels[0], img_metas, None)
    #
    #     # # score=iou, bbox=gt
    #     # bbox_targets_weight, bbox_targets, _, _, _ = \
    #     #     self.bbox_head.get_targets(all_level_points, bbox_preds_nostride, r_gt_bboxes_five[0], r_gt_bboxes_eight[0],
    #     #                                         gt_labels[0], img_metas, None)
    #     #
    #     # bbox_targets_reshape = []                     # no * stride
    #     # for bbox_target, bbox_pred, stride in zip(bbox_targets, bbox_preds, self.bbox_head.strides):
    #     #     bbox_targets_reshape.append(bbox_target.permute(1, 0).view_as(bbox_pred)+ 0.001)
    #     # _, _, labels, labels_score, labels_weight = \
    #     #     self.bbox_head.get_targets(all_level_points, bbox_targets_reshape, r_gt_bboxes_five[0], r_gt_bboxes_eight[0],
    #     #                                         gt_labels[0], img_metas, None)
    #     #
    #     # bbox_targets_reshape = []                     # the bbox target of the true size *stride
    #     # for bbox_target, bbox_pred, stride in zip(bbox_targets, bbox_preds, self.bbox_head.strides):
    #     #     bbox_targets_reshape.append(bbox_target.permute(1, 0).view_as(bbox_pred) * stride)
    #
    #     # self.bbox_head.show_featuremap_levels(bbox_targets_weight, 'label_gt_init', type='labels')
    #     # self.bbox_head.show_featuremap_levels(labels, 'label_gt_refine', type='labels')
    #     # self.bbox_head.show_featuremap_levels(labels_score, 'iou_score')
    #
    #     label_targets = []
    #     for label, label_score, cls_score \
    #             in zip(labels, labels_score, cls_scores):
    #         bin_label = label_score.new_full((label.size(0), 15), 0)
    #         inds =((label >= 0) & (label < 15)).nonzero().squeeze(1)
    #         if inds.numel() > 0:
    #             bin_label[inds, label[inds]] = label_score[inds]
    #         label_targets.append(bin_label.float().permute(1,0).view_as(cls_score))
    #
    #     # outs = (label_targets, bbox_preds, reg_confidences, cls_scores_init)
    #     # outs = (label_targets, bbox_targets_reshape, reg_confidences, cls_scores_init)
    #
    #     # self.bbox_head.show_featuremap_levels(label_targets, 'cls_score')
    #
    #
    #     bbox_list = self.bbox_head.get_bboxes(
    #         *outs, img_metas, rescale=rescale)
    #
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in bbox_list
    #     ]
    #     return bbox_results[0]

    def show_groundtruth(self,
                    data,
                    r_groundtruths):
        # img_norm_cfg = dict(
        #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        img_norm_cfg = dict(
            mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
        img_tensor = data
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        for img, r_groundtruth in zip(imgs, r_groundtruths):
            r_groundtruth = r_groundtruth.cpu().numpy()
            # r_groundtruth = forward_convert(r_groundtruth, with_label=False)
            img_show = draw_boxes_only(img, r_groundtruth, method=1)
            cv2.imshow("gt", img_show)
            cv2.waitKey(0)

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        indices = bboxes[:,-1] >= score_thr
        detected_scores = bboxes[indices, -1]
        detected_boxes = bboxes[indices, :8]
        detected_categories = labels[indices]
        img_show = draw_boxes_with_label_and_scores(img,
                                                    boxes=detected_boxes,
                                                    labels=detected_categories,
                                                    scores=detected_scores,
                                                    method=1,
                                                    labelname=self.CLASSES)
        # img_show = draw_boxes_only(img_show, boxes=detected_boxes, method=1)
        cv2.imshow("show", img_show)
        cv2.waitKey(0)
        if not (show or out_file):
            return img