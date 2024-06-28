_base_ = '../_base_/default_runtime.py'

# 修改数据集相关设置

data_root = '/home/kb535/lss/data/LITS20177/part1/coco/'
backend_args = None

# model settings
model = dict(
    type='enhanceBoxInst',
    data_preprocessor=dict(
        type='BoxInstDataPreprocessor',
        # mean=[123.675, 116.28, 103.53],
        # std=[58.395, 57.12, 57.375],
        mean=[50.5, 50.5, 50.5],
        std=[80.5, 80.5, 80.5],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        mask_stride=4,
        pairwise_size=3,
        pairwise_dilation=2,
        pairwise_color_thresh=0.3,
        bottom_pixels_removed=10),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='enhanceBoxInstBboxHead',
        num_params=593 +256+32,
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    mask_head=dict(
        type='enhanceBoxInstMaskHead',
        num_layers=3,
        feat_channels=16,
        size_of_interest=8,
        mask_out_stride=4,
        topk_masks_per_img=64,
        mask_feature_head=dict(
            in_channels=256,
            feat_channels=128,
            start_level=0,
            end_level=2,
            out_channels=16,
            mask_stride=8,
            num_stacked_convs=4,
            norm_cfg=dict(type='BN', requires_grad=True)),
        loss_mask=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            eps=5e-6,
            loss_weight=1.0)),
    # model training and testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr=0.5))


# Align with Detectron2
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333,800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
    ),
    dict(type='Resize', scale=(1333,800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
            data_root=data_root,
            ann_file='annotations/instances_train2014.json',
            data_prefix=dict(img='train2014/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)
    )
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_val2014.json',
        data_prefix=dict(img='val2014/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_test2014.json',
        data_prefix=dict(img='test2014/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2014.json',
    metric=['bbox', 'segm'],
    # format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test2014.json',
    metric=['bbox', 'segm'],
    # format_only=False,
    backend_args=backend_args)

max_epochs = 200
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
    ]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=False, base_batch_size=16)

find_unused_parameters = True