_base_ = [
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize_Rotate', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip_Rotate', flip_ratio=0.5, direction='horizontal'),
    # dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomRotate', rate=0.5, angles=[90, 180, 270], auto_bound=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='MeanRecGenerator', resort=True, with_h_bbox=True, with_factor=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'r_gt_bboxes_eight', 'r_gt_bboxes_five', 'gt_labels', 'obliquity_factors', 'direction_factors']),
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
        # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSDHRecHeadRotate',
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
            type='DeltaXYWHRBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1, 1, 1, 1]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_factor=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
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
    nms=dict(type='nms', iou_threshold=0.4), # Fast rotated NMS
    # nms=dict(type='nms', iou_threshold=0.1), # Original rotated NMS
    max_per_img=1000)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(dataset=dict(pipeline=train_pipeline)),
)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(
#     lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(
#     policy='step',
#     warmup='constant',
#     # warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[8, 11])
log_config = dict(
    interval=50,)
total_epochs = 24
work_dir = './work_dirs/atss/Res50_DHRec_2x_dota'