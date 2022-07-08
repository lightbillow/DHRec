_base_ = [
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='ATSSRotate',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSHeadRotate',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        # dcn_on_last_conv=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYXYBBoxCoder',
            target_means=0,
            target_stds=1),
        loss_cls_init=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    ))
# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssignerRbox', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
    cls_cfg = dict(
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
        debug=False)
)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.1),
    max_per_img=1000)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
log_config = dict(
    interval=50,)
total_epochs = 12
work_dir = './work_dirs/atss/Res50_baseline_five_params_1x_dota'
