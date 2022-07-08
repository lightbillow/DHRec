import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.ops import DeformConv2d
from mmcv.cnn import Scale, normal_init, ConvModule, bias_init_with_prob
import numpy as np

from mmdet.core import (cornerbias2rbbox, force_fp32, multi_apply, multiclass_rotate_eight_nms, multiclass_nms_keep,
                        anchor_target_rotated, anchor_inside_flags, build_assigner, build_sampler, multiclass_nms_keep_withgt,
                        images_to_levels, unmap)
from ..builder import HEADS, build_loss
# from .anchor_free_head_rotate import AnchorFreeHead_Rotate
from .anchor_head_rotate import AnchorHeadRotate

INF = 1e8


@HEADS.register_module()
class ATSSDHRecHeadRotate(AnchorHeadRotate):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 deform_sigma=2/3,
                 dcn_on_last_conv=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_factor=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):

        self.stacked_convs = stacked_convs
        self.deform_sigma = deform_sigma
        self.norm_cfg = norm_cfg
        self.basesize = kwargs['anchor_generator'].octave_base_scale
        self.dcn_on_last_conv = dcn_on_last_conv

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            **kwargs)
        self.loss_factor = build_loss(loss_factor)
        self.loss_centerness = build_loss(loss_centerness)
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)


    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = None
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg))
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg))
        self.reg = nn.Conv2d(self.feat_channels, 11, 3, padding=1)
        self.cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)

        self.reg_confidence = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(self.deform_sigma) for _ in self.anchor_generator.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reg, std=0.01)
        normal_init(self.cls, std=0.01, bias=bias_cls)
        normal_init(self.reg_confidence, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): Centerss for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        # dcn_base_offset = self.dcn_base_offset.type_as(x)
        reg_feat = x
        cls_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        cls_score = self.cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.reg(reg_feat)
        reg_confidence = self.reg_confidence(reg_feat)
        return cls_score, bbox_pred, reg_confidence

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'reg_confidences'))
    def loss(self,
             cls_scores,
             bbox_preds,
             reg_confidences,
             r_gt_bboxes_five,
             r_gt_bboxes_eight,
             gt_labels,
             img_metas,
             obliquity_factors=None,
             direction_factors=None,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        anchors, labels_init, bbox_targets, _, labels_score, _ = self.get_targets(anchor_list, valid_flag_list,
                                                bbox_preds, r_gt_bboxes_five, r_gt_bboxes_eight, gt_labels,
                                                obliquity_factors, direction_factors, img_metas, gt_bboxes_ignore)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 11)
            for bbox_pred in bbox_preds
        ]
        flatten_reg_confidence = [
            reg_confidence.permute(0, 2, 3, 1).reshape(-1)
            for reg_confidence in reg_confidences
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_reg_confidence = torch.cat(flatten_reg_confidence)

        flatten_anchors = torch.cat(anchors)
        flatten_labels_init = torch.cat(labels_init)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_labels_score = torch.cat(labels_score)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels_init >= 0)
                    & (flatten_labels_init < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls_init = self.loss_cls(
            flatten_cls_scores, flatten_labels_init,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_reg_confidence = flatten_reg_confidence[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_reg_confidence_targets = flatten_labels_score[pos_inds]
            pos_anchors = flatten_anchors[pos_inds]

            pos_decode_bbox_pred = self.bbox_coder.decode_bbox(
                pos_anchors, pos_bbox_preds[..., :8])
            pos_decode_bbox_targets = self.bbox_coder.decode_bbox(
                pos_anchors, pos_bbox_targets[..., :8])

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred, pos_decode_bbox_targets,
                avg_factor = (num_pos * 3 + num_imgs),
            )
            loss_factor = self.loss_factor(
                pos_bbox_preds[..., 8:],
                pos_bbox_targets[..., 8:],
                avg_factor = (num_pos + num_imgs),
            )

            loss_reg_confidence = self.loss_centerness(pos_reg_confidence,
                                                       pos_reg_confidence_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_factor = pos_bbox_preds.sum()
            loss_reg_confidence = pos_reg_confidence.sum()

        return dict(
            loss_cls_init = loss_cls_init,
            loss_bbox=loss_bbox,
            loss_factor = loss_factor,
            loss_reg_confidence=loss_reg_confidence
        )

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'reg_confidences'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   reg_confidences,
                   img_metas,
                   cfg=None,
                   rescale=None,
                   fast=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # cls_scores = cls_scores_init
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=cls_scores[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            reg_confidence_list = [
                reg_confidences[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 reg_confidence_list,
                                                 mlvl_anchors, img_shape,
                                                 scale_factor, cfg, rescale, fast)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           reg_confidences,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           fast=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        channel = bbox_preds[0].shape[0]
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        if fast:
            mlvl_filter_anchors = []
            mlvl_bboxes_pred = []
            mlvl_bboxes_nms = []
        else:
            mlvl_bboxes = []
        mlvl_scores = []
        mlvl_reg_confidences = []
        for cls_score, bbox_pred, reg_confidence, anchors in zip(
                cls_scores, bbox_preds, reg_confidences, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            reg_confidence = reg_confidence.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, channel)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # max_scores, _ = scores.max(dim=1)
                max_scores, _ = (scores * reg_confidence[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                reg_confidence = reg_confidence[topk_inds]
            if fast:
                bboxes_nms = self.bbox_coder.decode_nms(anchors, bbox_pred)
                mlvl_filter_anchors.append(anchors)
                mlvl_bboxes_pred.append(bbox_pred)
                mlvl_bboxes_nms.append(bboxes_nms)
            else:
                bboxes = self.bbox_coder.decode(anchors, bbox_pred, err=0.3) + 1
                mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_reg_confidences.append(reg_confidence)
        
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_reg_confidences = torch.cat(mlvl_reg_confidences)
        if fast:
            mlvl_bboxes_pred = torch.cat(mlvl_bboxes_pred)
            mlvl_filter_anchors = torch.cat(mlvl_filter_anchors)
            mlvl_bboxes_nms = torch.cat(mlvl_bboxes_nms)
            det_anchors, det_bboxes_pred, det_labels, det_scores = multiclass_nms_keep(
                mlvl_bboxes_nms,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_reg_confidences,
                multi_anchors=mlvl_filter_anchors,
                multi_bboxpred=mlvl_bboxes_pred,
            )
            det_bboxes = self.bbox_coder.decode(det_anchors, det_bboxes_pred, err=0.3) + 1
            if rescale:
                det_bboxes /= det_bboxes.new_tensor(np.hstack([scale_factor, scale_factor]))
            det_bboxes = torch.cat([det_bboxes, det_scores[:, None]], dim=1)
            
            return det_bboxes, det_labels
        else:
            mlvl_bboxes = torch.cat(mlvl_bboxes)
            if rescale:
                mlvl_bboxes /= mlvl_bboxes.new_tensor(np.hstack([scale_factor, scale_factor]))
            # return mlvl_bboxes, mlvl_scores, mlvl_reg_confidences
            det_bboxes, det_labels = multiclass_rotate_eight_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_reg_confidences
            )
            return det_bboxes, det_labels

    def get_targets(self, anchor_list, valid_flag_list, bbox_preds,
                    r_gt_bboxes_five_list, r_gt_bboxes_eight_list, gt_labels_list,
                    obliquity_factors_list, direction_factors_list, img_metas, gt_bboxes_ignore_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        num_levels = len(num_level_anchors)

        bbox_preds_list = self.get_refine_anchors(
            bbox_preds, anchor_list, img_metas, device=bbox_preds[0].device)

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
            bbox_preds_list[i] = torch.cat(bbox_preds_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        ################ ATSS assigner ####################### 
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_bbox_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             r_gt_bboxes_eight_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             obliquity_factors_list,
             direction_factors_list,
             img_metas,
             label_channels=self.cls_out_channels,
             unmap_outputs=True)
        anchors_list, labels_init_list, bbox_targets_list = all_anchors, all_labels, all_bbox_targets

        ################ IoU prediction ####################### 
        # With ATSS assigner, we only use labels_score_list as the iou score (centerness) branch
        (labels_list, labels_score_list, labels_weight_list, _, _) = anchor_target_rotated(
            bbox_preds_list,
            valid_flag_list,
            r_gt_bboxes_eight_list,
            img_metas,
            0,
            1,
            self.train_cfg.cls_cfg,
            # self.test_cfg.odm_cfg,
            gt_labels_list=gt_labels_list,
            label_channels=self.cls_out_channels,
            sampling=False)

        # split to per img, per level
        labels_list = [labels.split(num_level_anchors, 0) for labels in labels_list]
        labels_score_list = [labels_score.split(num_level_anchors, 0) for labels_score in labels_score_list]
        labels_weight_list = [labels_weight.split(num_level_anchors, 0) for labels_weight in labels_weight_list]
        bbox_targets_list = [
            bbox_targets.split(num_level_anchors, 0)
            for bbox_targets in bbox_targets_list
        ]
        labels_init_list = [labels.split(num_level_anchors, 0) for labels in labels_init_list]
        anchors_list = [anchors.split(num_level_anchors, 0) for anchors in anchors_list]

        # concat per level image
        concat_lvl_anchors = []
        concat_lvl_labels = []
        concat_lvl_labels_score = []
        concat_lvl_labels_weight = []
        concat_lvl_bbox_targets = []
        concat_lvl_labels_init = []
        for i in range(num_levels):
            concat_lvl_anchors.append(
                torch.cat([anchors[i] for anchors in anchors_list]))
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_labels_score.append(
                torch.cat([labels_score[i] for labels_score in labels_score_list]))
            concat_lvl_labels_weight.append(
                torch.cat([labels_weight[i] for labels_weight in labels_weight_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            concat_lvl_labels_init.append(
                torch.cat([labels[i] for labels in labels_init_list]))
            concat_lvl_bbox_targets.append(bbox_targets)
        # return concat_lvl_labels, concat_lvl_labels_weight, concat_lvl_bbox_targets, concat_lvl_labels_init
        return concat_lvl_anchors, concat_lvl_labels_init, concat_lvl_bbox_targets, concat_lvl_labels, concat_lvl_labels_score, concat_lvl_labels_weight

    def _get_bbox_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           obliquity_factor,
                           direction_factor,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        if len(gt_bboxes) == 0:
            return flat_anchors, \
                   gt_labels.new_full((flat_anchors.shape[0],), self.background_label), \
                   anchors.new_full((flat_anchors.shape[0],), 1.0, dtype=torch.float), \
                   gt_bboxes.new_zeros(flat_anchors.shape[0], 11), \
                   gt_bboxes.new_zeros(flat_anchors.shape[0], 11), \
                   None, None

        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros([num_valid_anchors, 11], dtype=torch.float)
        bbox_weights = anchors.new_zeros([num_valid_anchors, 11], dtype=torch.float)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :8] = pos_bbox_targets
            bbox_targets[pos_inds, 8] = obliquity_factor[sampling_result.pos_assigned_gt_inds]
            bbox_targets[pos_inds, -2:] = direction_factor[sampling_result.pos_assigned_gt_inds]
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    def get_refine_anchors(self,
                           bbox_preds,
                           anchors,
                           img_metas,
                           is_train=True,
                           device='cuda'):
        num_imgs = len(img_metas)
        num_levels = len(bbox_preds)
        channel = bbox_preds[0][0].shape[0]
        refine_anchors_list = []
        for img_id, img_meta in enumerate(img_metas):
            mlvl_refine_anchors = []
            for i in range(num_levels):
                bbox_pred = bbox_preds[i][img_id].detach().permute(1, 2, 0).reshape(-1, channel)
                anchors_perlevel = anchors[img_id][i]
                refine_anchors_perlevel = self.bbox_coder.decode(anchors_perlevel, bbox_pred, err=0.3)
                mlvl_refine_anchors.append(refine_anchors_perlevel)
            refine_anchors_list.append(mlvl_refine_anchors)
        return refine_anchors_list

