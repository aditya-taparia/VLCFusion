_base_ = [
    '_base_/schedules/schedule_2x.py',
    '_base_/default_runtime.py',
]
custom_imports = dict(
    imports=['configs.cbam_ensemble_vlm.cbam_ensemble_vlm'], allow_failed_imports=False)

model = dict(
    type='CBAMEnsembleVLM',
    lidar_model_path='/mnt/data/ataparia/mmdet_lidar/work_dirs/second/epoch_100.pth',
    lidar_model_cfg_path='configs/second_detr_ensemble.py',
    rgb_model_path='trained_models/rgb_fulldata/checkpoint-190000',
    train_jsonl_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/configs/cbam_ensemble_vlm/vlm_conditions/night_day_training.jsonl',
    val_jsonl_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/configs/cbam_ensemble_vlm/vlm_conditions/night_day_validation.jsonl',
    test_jsonl_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/configs/cbam_ensemble_vlm/vlm_conditions/day_testing.jsonl',
    # test_jsonl_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/configs/cbam_ensemble_vlm/vlm_conditions/dawn_dusk_testing.jsonl',
    # test_jsonl_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/configs/cbam_ensemble_vlm/vlm_conditions/night_day_testing.jsonl',
    backbone=dict(
        type='SECOND',
        in_channels=384,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-76.8, -51.2, -0.0345, 76.8, 51.2, -0.0345],
                    [-76.8, -51.2, 0, 76.8, 51.2, 0],
                    [-76.8, -51.2, -0.1188, 76.8, 51.2, -0.1188]],
            sizes=[
                [4.73, 2.08, 1.77],  # car
                [0.91, 0.84, 1.74],  # pedestrian
                [1.81, 0.84, 1.77]  # cyclist
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
        assigner=[
            dict(  # car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            dict(  # pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            dict(  # cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=4096,
        nms_thr=0.25,
        score_thr=0.1,
        min_bbox_size=0,
        max_num=500)
)

dataset_type = 'WaymoDataset'
data_root = '/mnt/data/ataparia/LidarTraining/day-frames/'
# data_root = '/mnt/data/ataparia/LidarTraining/dawn_dusk-frames/'
# data_root = '/mnt/data/ataparia/LidarTraining/night-day-frames/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
metainfo = dict(classes=class_names)

point_cloud_range = [-76.8, -51.2, -2, 76.8, 51.2, 4]
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args, data_root=data_root+'training/image_0'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
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
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'img_path'],
        meta_keys=['box_type_3d', 'sample_idx', 'context_name', 'timestamp'])
]

val_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args, data_root=data_root+'training/image_0'),
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
        ]),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'img_path'],
        meta_keys=['box_type_3d', 'sample_idx', 'context_name', 'timestamp'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args, data_root=data_root+'testing/image_0'),
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
        ]),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'img_path'],
        meta_keys=['box_type_3d', 'sample_idx', 'context_name', 'timestamp'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='waymo_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne', img='training/image_0'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            # load one frame every fifth frames
            load_interval=5,
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', img='training/image_0'),
        ann_file='waymo_infos_val.pkl',
        pipeline=val_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='testing/velodyne', img='testing/image_0'),
        ann_file='waymo_infos_test.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)
train_cfg = dict(type='EpochBasedTrainLoop', val_interval=1, max_epochs=50)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01)
)
val_cfg = dict(type='ValLoop')
val_evaluator = dict(
    type='WaymoMetric',
    waymo_bin_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/gt_validation.bin',    # data_root + 'waymo_infos_val.pkl'
)
test_cfg = dict(type='TestLoop')
# test_evaluator = dict(
#     type='KittiMetric',
#     ann_file=data_root + 'waymo_infos_test.pkl',
#     default_cam_key='CAM_FRONT',
#     # waymo_bin_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/gt_dawn_dusk_testing.bin',
# )
test_evaluator = dict(
    type='WaymoMetric',
    # waymo_bin_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/gt_dawn_dusk_testing.bin',
    waymo_bin_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/gt_day_testing.bin',
    # waymo_bin_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/gt_testing.bin',
)
