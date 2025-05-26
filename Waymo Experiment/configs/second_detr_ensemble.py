_base_ = [
    '_base_/models/hv_second_secfpn_waymo.py',
    '_base_/datasets/waymoD5-3d-3class.py',
    '_base_/schedules/schedule_2x.py',
    '_base_/default_runtime.py',
]

dataset_type = 'WaymoDataset'
data_root = '/mnt/data/ataparia/LidarTraining/night-day-frames/'
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
    dict(type='LoadImageFromFile', backend_args=backend_args, data_root=data_root),
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

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args, data_root=data_root),
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
            # load one frame every five frames
            load_interval=5,
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', img='training/image_0'),
        ann_file='waymo_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
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
train_cfg = dict(type='EpochBasedTrainLoop', val_interval=1, max_epochs=100)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01)
)
val_cfg = dict(type='ValLoop')
# val_evaluator = dict(
#     type='KittiMetric',
#     # data_prefix=dict(pts='training/velodyne'),
#     ann_file=data_root + 'waymo_infos_val.pkl',
#     # ann_file=data_root + 'nuscenes_infos_val.pkl',
#     metric='bbox',
#     backend_args=backend_args)
val_evaluator = dict(
    type='WaymoMetric',
    waymo_bin_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/gt_validation.bin',    # data_root + 'waymo_infos_val.pkl'
)
test_cfg = dict(type='TestLoop')
test_evaluator = dict(
    type='WaymoMetric',
    waymo_bin_file='/home/local/ASURITE/ataparia/snap/snapd-desktop-integration/current/Desktop/mmdet3d/gt_testing.bin',
)
