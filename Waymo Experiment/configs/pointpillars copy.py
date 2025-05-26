# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.32, 0.32, 6]
model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        max_num_points=20,
        point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4],
        voxel_size=voxel_size,
        max_voxels=(32000, 32000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4],
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[468, 468]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345],
                    [-74.88, -74.88, -0.1188, 74.88, 74.88, -0.1188],
                    [-74.88, -74.88, 0, 74.88, 74.88, 0]],
            sizes=[
                [4.73, 2.08, 1.77],  # car
                [1.81, 0.84, 1.77],  # cyclist
                [0.91, 0.84, 1.74]  # pedestrian
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(  # car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
                dict(  # cyclist
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                dict(  # pedestrian
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=4096,
            nms_thr=0.25,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=500)))

# dataset settings
# D5 in the config name means the whole dataset is divided into 5 folds
# We only use one fold for efficient experiments
dataset_type = 'WaymoDataset'
# data_root = 'data/waymo/kitti_format/'
data_root = '/mnt/data/ataparia/LidarTraining/night-day-frames/'
# file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://waymo_data/'))

class_names = ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root, # + 'training/',
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4],
        # file_client_args=file_client_args
    ))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        # file_client_args=file_client_args
        ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        # file_client_args=file_client_args
        ),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        # file_client_args=file_client_args
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            # dict(
            #     type='DefaultFormatBundle3D',
            #     class_names=class_names,
            #     with_label=False),
            # dict(type='Collect3D', keys=['points'])
            dict(
                type='Pack3DDetInputs',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        # file_client_args=file_client_args
    ),
    # dict(
    #     type='DefaultFormatBundle3D',
    #     class_names=class_names,
    #     with_label=False),
    # dict(type='Collect3D', keys=['points'])
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root + 'training/',
            ann_file=data_root + 'waymo_infos_train.pkl',
            split='training',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            # load one frame every five frames
            load_interval=1)),
# data settings
# data = dict(train=dict(dataset=dict(load_interval=1)))
    val=dict(
        type=dataset_type,
        data_root=data_root + 'training/',
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root + 'testing/',
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))


evaluation = dict(interval=1, pipeline=eval_pipeline)

# optimizer
# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[20, 23])
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)

checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'




train_dataloader = dict(
    dataset=dict(
        type='WaymoDataset',
        data_root=data_root + 'training/',
        ann_file=data_root + 'waymo_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
        load_interval=1),
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_size=2,
    num_workers=4
)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01)
)
