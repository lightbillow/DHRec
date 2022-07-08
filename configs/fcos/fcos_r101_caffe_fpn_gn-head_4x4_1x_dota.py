_base_ = './fcos_r50_caffe_fpn_gn-head_4x4_1x_dota.py'
model = dict(
    pretrained='open-mmlab://detectron/resnet101_caffe',
    # pretrained='torchvision://resnet101',
    backbone=dict(
        depth=101,
        # norm_cfg=dict(type='BN', requires_grad=True),
        # norm_eval=True,
        # style='pytorch'
    ),
    bbox_head = dict(
        # norm_cfg=None,
        # regress_ranges=((-1, 64), (48, 128), (96, 256), (192, 512), (384, 1e8)),
        # regress_ranges=((-1, 32), (24, 64), (48, 128), (96, 256), (192, 1e8)),
        regress_ranges=((-1, 32), (32, 64), (64, 128), (128, 256), (256, 1e8)),
        stacked_convs=4,
        loss_cls_init=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    )
)
train_cfg = dict(
    odm_cfg=dict(
        anchor_target_type='obb_obb_rbox_overlap',
        anchor_inside_type='center',
        assigner=dict(
            type='MaxIoUAssignerRbox',
            pos_iou_thr=0.3,
            neg_iou_thr=0.3,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
# test_cfg = dict(
#     odm_cfg=dict(
#         anchor_target_type='obb_obb_rbox_overlap',
#         anchor_inside_type='center',
#         assigner=dict(
#             type='MaxIoUAssignerRbox',
#             pos_iou_thr=0.3,
#             neg_iou_thr=0.3,
#             min_pos_iou=0.5,
#             ignore_iof_thr=-1),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False))
log_config = dict(
    interval=50,)
total_epochs = 12
work_dir = './work_dirs/fcos/Res101_caffe_deformable_train13'
# resume_from = 'work_dirs/fcos/Res101_caffe_deformable_train12/epoch_12.pth'
# load_from = 'work_dirs/fcos/Res101_caffe_deformable_train12/epoch_12.pth'