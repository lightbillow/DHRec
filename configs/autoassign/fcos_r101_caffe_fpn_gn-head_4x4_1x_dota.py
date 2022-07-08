_base_ = '../fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_dota.py'
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
        type='AutoAssignMPHeadRotate',
        # norm_cfg=None,
        stacked_convs=4,
        center_sampling=False,
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)
train_cfg = dict(
    odm_cfg=dict(
        anchor_target_type='obb_obb_rbox_overlap',
        anchor_inside_type='center',
        assigner=dict(
            type='MaxIoUAssignerRbox',
            pos_iou_thr=0.3,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
log_config = dict(
    interval=50,)
total_epochs = 12
work_dir = './work_dirs/autoassign/Res101_caffe_deformable_train1'
# resume_from = 'work_dirs/fcos/Res101_caffe_deformable_train12/epoch_12.pth'
load_from = 'work_dirs/fcos/Res101_caffe_deformable_train12/epoch_12.pth'