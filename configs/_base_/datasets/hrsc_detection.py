# dataset settings
dataset_type = 'HRSCDataset'
data_root = '/home/nieguangtao/programing/mmdetection/data/HRSC/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize_Rotate', img_scale=(800, 512), keep_ratio=True),
    dict(type='RandomFlip_Rotate', flip_ratio=0.5, direction='horizontal'),
    # dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    # dict(type='RandomRotate', rate=0.5, angles=[90, 180, 270], auto_bound=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='MeanRecGenerator', resort=True, with_h_bbox=True, with_factor=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'r_gt_bboxes_eight', 'r_gt_bboxes_five', 'gt_labels', 'obliquity_factors', 'direction_factors']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug_Rotate',
        img_scale=(800, 512),
        flip=False,
        # img_rotate=[90, 180, 270],
        transforms=[
            dict(type='Resize_Rotate', img_scale=(800, 512), keep_ratio=True),
            dict(type='RandomFlip_Rotate'),
            dict(type='RandomRotate', auto_bound=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], \
                                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
                                 'scale_factor', 'flip', 'flip_direction', 'rotate_angle', 'img_norm_cfg')),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'ImageSets/trainval.txt',
            ],
            img_prefix=data_root,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/trainval.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))