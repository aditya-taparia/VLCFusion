# %%
import numpy as np
import torch

from tqdm import tqdm

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR messages -> Supress tensorflow warnings
from os import path as osp

import random
import cv2

import matplotlib.pyplot as plt
import json

import tensorflow as tf
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection

from io import BytesIO
import copy

import utils.utils as utils
from utils.lidar_box3d import LiDARInstance3DBoxes
from utils.box_3d_mode import Box3DMode

from concurrent.futures import ProcessPoolExecutor
from PIL import Image

# %%
def setup_seed(seed=0, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# %%
variation = 'night'
type_ = 'validation'

data_path = '/mnt/data/ataparia/waymo_perception_dataset_v1_4_3'
save_path = f'/mnt/data/ataparia/LidarTraining/{variation}-frames'

# Create the save directory
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/training', exist_ok=True)
os.makedirs(f'{save_path}/validation', exist_ok=True)
os.makedirs(f'{save_path}/testing', exist_ok=True)

all_sf_files = {
    'dawn_dusk': [], 
    'day': [], 
    'night': []
}

# %%
train_files = os.listdir(os.path.join(data_path, 'training'))
train_files = [f for f in train_files if f.endswith('.tfrecord')]
print('Number of training files:', len(train_files))

val_files = os.listdir(os.path.join(data_path, 'validation'))
val_files = [f for f in val_files if f.endswith('.tfrecord')]
print('Number of validation files:', len(val_files))

for f in tqdm(train_files, desc="Processing files in training"):
    filename = os.path.join(data_path, 'training', f)
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frame_stats = frame.context.stats
        if frame_stats.location == 'location_sf':
            time_of_day = frame_stats.time_of_day
            if time_of_day == 'Night':
                time_of_day = 'night'
            elif time_of_day == 'Day':
                time_of_day = 'day'
            elif time_of_day == 'Dawn/Dusk':
                time_of_day = 'dawn_dusk'   
            all_sf_files[time_of_day].append(filename)
        break

# Validation split
for f in tqdm(val_files, desc="Processing files in validation"):
    filename = os.path.join(data_path, 'validation', f)
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frame_stats = frame.context.stats
        if frame_stats.location == 'location_sf':
            time_of_day = frame_stats.time_of_day
            if time_of_day == 'Night':
                time_of_day = 'night'
            elif time_of_day == 'Day':
                time_of_day = 'day'
            elif time_of_day == 'Dawn/Dusk':
                time_of_day = 'dawn_dusk'            
            all_sf_files[time_of_day].append(filename)
        break

# %%
setup_seed(42)

# %%
video_to_frames = []

files = all_sf_files[variation]

for f in tqdm(files, desc="Processing files"):
    filename = f
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        video_to_frames.append(frame)

# %%
random.shuffle(video_to_frames)

# %%
print(f'Number of {variation} files:', len(video_to_frames))

# %%
train_files = video_to_frames[:int(0.8*len(video_to_frames))]
val_files = video_to_frames[int(0.8*len(video_to_frames)):int(0.9*len(video_to_frames))]
test_files = video_to_frames[int(0.9*len(video_to_frames)):]

print('Number of train files:', len(train_files))
print('Number of val files:', len(val_files))
print('Number of test files:', len(test_files))

# %%
INDEX_LENGTH = 7
MAX_SWEEPS = 1

# keep the order defined by the official protocol
CAM_LIST = [
    '_FRONT',
    '_FRONT_LEFT',
    '_FRONT_RIGHT',
    '_SIDE_LEFT',
    '_SIDE_RIGHT',
]
LIDAR_LIST = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
TYPE_LIST = [
    'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
]

# MMDetection3D unified camera keys & class names
CAMERA_TYPES = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_SIDE_LEFT',
    'CAM_SIDE_RIGHT',
]
SELECTED_WAYMO_CLASSES = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
INFO_MAP = {
    'training': '_infos_train.pkl',
    'validation': '_infos_val.pkl',
    'testing': '_infos_test.pkl'
}

# %%
# Make data folders
image_save_path = os.path.join(save_path, type_, 'image_')
pointcloud_save_path = os.path.join(save_path, type_, 'velodyne')

os.makedirs(pointcloud_save_path, exist_ok=True)
for i in range(5):
    os.makedirs(f"{image_save_path}{str(i)}", exist_ok=True)

# %%
def save_image(frame, frame_idx, image_save_dir):
    """Parse and save the images in jpg format.

    Args:
        frame (:obj:`Frame`): Open dataset frame proto.
        frame_idx (int): Current frame index.
    """
    for img in frame.images:
        img_path = f'{image_save_dir}{str(img.name - 1)}/' + \
            f'{str(frame_idx).zfill(INDEX_LENGTH)}.jpg'
        with open(img_path, 'wb') as fp:
            fp.write(img.image)

# %%
def save_lidar(frame, point_cloud_save_dir, frame_idx):
    
    def convert_range_image_to_point_cloud(frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0,
                                           filter_no_label_zone_points=True):
        """Convert range images to point cloud.

        Args:
            frame (:obj:`Frame`): Open dataset frame.
            range_images (dict): Mapping from laser_name to list of two
                range images corresponding with two returns.
            camera_projections (dict): Mapping from laser_name to list of two
                camera projections corresponding with two returns.
            range_image_top_pose (:obj:`Transform`): Range image pixel pose for
                top lidar.
            ri_index (int, optional): 0 for the first return,
                1 for the second return. Default: 0.

        Returns:
            tuple[list[np.ndarray]]: (List of points with shape [N, 3],
                camera projections of points with shape [N, 6], intensity
                with shape [N, 1], elongation with shape [N, 1], points'
                position in the depth map (element offset if points come from
                the main lidar otherwise -1) with shape[N, 1]). All the
                lists have the length of lidar numbers (5).
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        intensity = []
        elongation = []
        mask_indices = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == open_dataset.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            if filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

            mask_index = tf.where(range_image_mask)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(
                tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, mask_index)
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            mask_index)
            intensity.append(intensity_tensor.numpy())

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             mask_index)
            elongation.append(elongation_tensor.numpy())
            if c.name == 1:
                mask_index = (ri_index * range_image_mask.shape[0] +
                              mask_index[:, 0]
                              ) * range_image_mask.shape[1] + mask_index[:, 1]
                mask_index = mask_index.numpy().astype(elongation[-1].dtype)
            else:
                mask_index = np.full_like(elongation[-1], -1)

            mask_indices.append(mask_index)

        return points, cp_points, intensity, elongation, mask_indices
    
    """Parse and save the lidar data in psd format.

    Args:
        frame (:obj:`Frame`): Open dataset frame proto.
        file_idx (int): Current file index.
        frame_idx (int): Current frame index.
    """
    range_images, camera_projections, seg_labels, range_image_top_pose = \
        parse_range_image_and_camera_projection(frame)

    if range_image_top_pose is None:
        # the camera only split doesn't contain lidar points.
        return
    # First return
    points_0, cp_points_0, intensity_0, elongation_0, mask_indices_0 = \
        convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0
        )
    points_0 = np.concatenate(points_0, axis=0)
    intensity_0 = np.concatenate(intensity_0, axis=0)
    elongation_0 = np.concatenate(elongation_0, axis=0)
    mask_indices_0 = np.concatenate(mask_indices_0, axis=0)

    # Second return
    points_1, cp_points_1, intensity_1, elongation_1, mask_indices_1 = \
        convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1
        )
    points_1 = np.concatenate(points_1, axis=0)
    intensity_1 = np.concatenate(intensity_1, axis=0)
    elongation_1 = np.concatenate(elongation_1, axis=0)
    mask_indices_1 = np.concatenate(mask_indices_1, axis=0)

    points = np.concatenate([points_0, points_1], axis=0)
    intensity = np.concatenate([intensity_0, intensity_1], axis=0)
    elongation = np.concatenate([elongation_0, elongation_1], axis=0)
    mask_indices = np.concatenate([mask_indices_0, mask_indices_1], axis=0)

    # timestamp = frame.timestamp_micros * np.ones_like(intensity)

    # concatenate x,y,z, intensity, elongation, timestamp (6-dim)
    point_cloud = np.column_stack(
        (points, intensity, elongation, mask_indices))

    pc_path = f'{point_cloud_save_dir}/' + \
        f'{str(frame_idx).zfill(INDEX_LENGTH)}.bin'
    point_cloud.astype(np.float32).tofile(pc_path)

# %%
def cart_to_homo(self, mat):
    """Convert transformation matrix in Cartesian coordinates to
    homogeneous format.

    Args:
        mat (np.ndarray): Transformation matrix in Cartesian.
            The input matrix shape is 3x3 or 3x4.

    Returns:
        np.ndarray: Transformation matrix in homogeneous format.
            The matrix shape is 4x4.
    """
    ret = np.eye(4)
    if mat.shape == (3, 3):
        ret[:3, :3] = mat
    elif mat.shape == (3, 4):
        ret[:3, :] = mat
    else:
        raise ValueError(mat.shape)
    return ret

def gather_instance_info(frame, cam_sync=False):
    """Generate instances and cam_sync_instances infos.

    For more details about infos, please refer to:
    https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html
    """  # noqa: E501
    filter_empty_3dboxes = True
    id_to_bbox = dict()
    id_to_name = dict()
    for labels in frame.projected_lidar_labels:
        name = labels.name
        for label in labels.labels:
            # TODO: need a workaround as bbox may not belong to front cam
            bbox = [
                label.box.center_x - label.box.length / 2,
                label.box.center_y - label.box.width / 2,
                label.box.center_x + label.box.length / 2,
                label.box.center_y + label.box.width / 2
            ]
            id_to_bbox[label.id] = bbox
            id_to_name[label.id] = name - 1

    group_id = 0
    instance_infos = []
    for obj in frame.laser_labels:
        instance_info = dict()
        bounding_box = None
        name = None
        id = obj.id
        for proj_cam in CAM_LIST:
            if id + proj_cam in id_to_bbox:
                bounding_box = id_to_bbox.get(id + proj_cam)
                name = id_to_name.get(id + proj_cam)
                break

        # NOTE: the 2D labels do not have strict correspondence with
        # the projected 2D lidar labels
        # e.g.: the projected 2D labels can be in camera 2
        # while the most_visible_camera can have id 4
        if cam_sync:
            if obj.most_visible_camera_name:
                name = CAM_LIST.index(
                    f'_{obj.most_visible_camera_name}')
                box3d = obj.camera_synced_box
            else:
                continue
        else:
            box3d = obj.box

        if bounding_box is None or name is None:
            name = 0
            bounding_box = [0.0, 0.0, 0.0, 0.0]

        my_type = TYPE_LIST[obj.type]

        if my_type not in SELECTED_WAYMO_CLASSES:
            continue
        else:
            label = SELECTED_WAYMO_CLASSES.index(my_type)

        if filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
            continue

        group_id += 1
        instance_info['group_id'] = group_id
        instance_info['camera_id'] = name
        instance_info['bbox'] = bounding_box
        instance_info['bbox_label'] = label

        height = box3d.height
        width = box3d.width
        length = box3d.length

        # NOTE: We save the bottom center of 3D bboxes.
        x = box3d.center_x
        y = box3d.center_y
        z = box3d.center_z - height / 2

        rotation_y = box3d.heading

        instance_info['bbox_3d'] = np.array(
            [x, y, z, length, width, height,
                rotation_y]).astype(np.float32).tolist()
        instance_info['bbox_label_3d'] = label
        instance_info['num_lidar_pts'] = obj.num_lidar_points_in_box

        # if self.save_track_id:
        #     instance_info['track_id'] = obj.id
        instance_infos.append(instance_info)
    return instance_infos

def gather_cam_instance_info(self, instances: dict, images: dict):
    """Generate cam_instances infos.

    For more details about infos, please refer to:
    https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html
    """  # noqa: E501
    cam_instances = dict()
    for cam_type in self.camera_types:
        lidar2cam = np.array(images[cam_type]['lidar2cam'])
        cam2img = np.array(images[cam_type]['cam2img'])
        cam_instances[cam_type] = []
        for instance in instances:
            cam_instance = dict()
            gt_bboxes_3d = np.array(instance['bbox_3d'])
            # Convert lidar coordinates to camera coordinates
            gt_bboxes_3d = LiDARInstance3DBoxes(
                gt_bboxes_3d[None, :]).convert_to(
                    Box3DMode.CAM, lidar2cam, correct_yaw=True)
            corners_3d = gt_bboxes_3d.corners.numpy()
            corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
            in_camera = np.argwhere(corners_3d[2, :] > 0).flatten()
            corners_3d = corners_3d[:, in_camera]
            # Project 3d box to 2d.
            corner_coords = utils.view_points(corners_3d, cam2img,
                                        True).T[:, :2].tolist()

            # Keep only corners that fall within the image.
            # TODO: imsize should be determined by the current image size
            # CAM_FRONT: (1920, 1280)
            # CAM_FRONT_LEFT: (1920, 1280)
            # CAM_SIDE_LEFT: (1920, 886)
            final_coords = utils.post_process_coords(
                corner_coords,
                imsize=(images['CAM_FRONT']['width'],
                        images['CAM_FRONT']['height']))

            # Skip if the convex hull of the re-projected corners
            # does not intersect the image canvas.
            if final_coords is None:
                continue
            else:
                min_x, min_y, max_x, max_y = final_coords

            cam_instance['bbox'] = [min_x, min_y, max_x, max_y]
            cam_instance['bbox_label'] = instance['bbox_label']
            cam_instance['bbox_3d'] = gt_bboxes_3d.numpy().squeeze(
            ).astype(np.float32).tolist()
            cam_instance['bbox_label_3d'] = instance['bbox_label_3d']

            center_3d = gt_bboxes_3d.gravity_center.numpy()
            center_2d_with_depth = utils.points_cam2img(
                center_3d, cam2img, with_depth=True)
            center_2d_with_depth = center_2d_with_depth.squeeze().tolist()

            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if center_2d_with_depth[2] <= 0:
                continue
            cam_instance['center_2d'] = center_2d_with_depth[:2]
            cam_instance['depth'] = center_2d_with_depth[2]

            # TODO: Discuss whether following info is necessary
            cam_instance['bbox_3d_isvalid'] = True
            cam_instance['velocity'] = -1
            cam_instances[cam_type].append(cam_instance)

    return cam_instances

# %%
def create_waymo_info_file(frame, frame_idx, file_infos, test_mode, save_cam_sync_instances, save_cam_instances):
    r"""Generate waymo train/val/test infos.

    For more details about infos, please refer to:
    https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html
    """  # noqa: E501
    frame_infos = dict()

    # Gather frame infos
    sample_idx = f'{str(frame_idx).zfill(INDEX_LENGTH)}'
    frame_infos['sample_idx'] = int(sample_idx)
    frame_infos['timestamp'] = frame.timestamp_micros
    frame_infos['ego2global'] = np.array(frame.pose.transform).reshape(
        4, 4).astype(np.float32).tolist()
    frame_infos['context_name'] = frame.context.name
    frame_infos['stats'] = {
        'location': frame.context.stats.location,
        'time_of_day': frame.context.stats.time_of_day,
        'weather': frame.context.stats.weather,
    }

    # Gather camera infos
    frame_infos['images'] = dict()
    # waymo front camera to kitti reference camera
    T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                    [1.0, 0.0, 0.0]])
    camera_calibs = []
    Tr_velo_to_cams = []
    for camera in frame.context.camera_calibrations:
        # extrinsic parameters
        T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
            4, 4)
        T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
        Tr_velo_to_cam = \
            cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
        Tr_velo_to_cams.append(Tr_velo_to_cam)

        # intrinsic parameters
        camera_calib = np.zeros((3, 4))
        camera_calib[0, 0] = camera.intrinsic[0]
        camera_calib[1, 1] = camera.intrinsic[1]
        camera_calib[0, 2] = camera.intrinsic[2]
        camera_calib[1, 2] = camera.intrinsic[3]
        camera_calib[2, 2] = 1
        camera_calibs.append(camera_calib)

    for i, (cam_key, camera_calib, Tr_velo_to_cam) in enumerate(zip(CAMERA_TYPES, camera_calibs, Tr_velo_to_cams)):
        cam_infos = dict()
        cam_infos['img_path'] = str(sample_idx) + '.jpg'
        # NOTE: frames.images order is different
        for img in frame.images:
            if img.name == i + 1:
                width, height = Image.open(BytesIO(img.image)).size
        cam_infos['height'] = height
        cam_infos['width'] = width
        cam_infos['lidar2cam'] = Tr_velo_to_cam.astype(np.float32).tolist()
        cam_infos['cam2img'] = camera_calib.astype(np.float32).tolist()
        cam_infos['lidar2img'] = (camera_calib @ Tr_velo_to_cam).astype(
            np.float32).tolist()
        frame_infos['images'][cam_key] = cam_infos

    # Gather lidar infos
    lidar_infos = dict()
    lidar_infos['lidar_path'] = str(sample_idx) + '.bin'
    lidar_infos['num_pts_feats'] = 6
    frame_infos['lidar_points'] = lidar_infos

    # Gather lidar sweeps and camera sweeps infos
    # TODO: Add lidar2img in image sweeps infos when we need it.
    # TODO: Consider merging lidar sweeps infos and image sweeps infos.
    lidar_sweeps_infos, image_sweeps_infos = [], []
    for prev_offset in range(-1, -MAX_SWEEPS - 1, -1):
        prev_lidar_infos = dict()
        prev_image_infos = dict()
        if frame_idx + prev_offset >= 0:
            prev_frame_infos = file_infos[prev_offset]
            prev_lidar_infos['timestamp'] = prev_frame_infos['timestamp']
            prev_lidar_infos['ego2global'] = prev_frame_infos['ego2global']
            prev_lidar_infos['lidar_points'] = dict()
            lidar_path = prev_frame_infos['lidar_points']['lidar_path']
            prev_lidar_infos['lidar_points']['lidar_path'] = lidar_path
            lidar_sweeps_infos.append(prev_lidar_infos)

            prev_image_infos['timestamp'] = prev_frame_infos['timestamp']
            prev_image_infos['ego2global'] = prev_frame_infos['ego2global']
            prev_image_infos['images'] = dict()
            for cam_key in CAMERA_TYPES:
                prev_image_infos['images'][cam_key] = dict()
                img_path = prev_frame_infos['images'][cam_key]['img_path']
                prev_image_infos['images'][cam_key]['img_path'] = img_path
            image_sweeps_infos.append(prev_image_infos)
    if lidar_sweeps_infos:
        frame_infos['lidar_sweeps'] = lidar_sweeps_infos
    if image_sweeps_infos:
        frame_infos['image_sweeps'] = image_sweeps_infos

    if not test_mode:
        # Gather instances infos which is used for lidar-based 3D detection
        frame_infos['instances'] = gather_instance_info(frame)
        # Gather cam_sync_instances infos which is used for image-based
        # (multi-view) 3D detection.
        if save_cam_sync_instances:
            frame_infos['cam_sync_instances'] = gather_instance_info(
                frame, cam_sync=True)
        # Gather cam_instances infos which is used for image-based
        # (monocular) 3D detection (optional).
        # TODO: Should we use cam_sync_instances to generate cam_instances?
        if save_cam_instances:
            frame_infos['cam_instances'] = gather_cam_instance_info(
                copy.deepcopy(frame_infos['instances']),
                frame_infos['images'])
    file_infos.append(frame_infos)

# %%
def convert_one_frame(args, test_mode=False, save_cam_sync_instances=True, save_cam_instances=True):
    """
    Process one frame: save the images and lidar point cloud,
    create and return the info dictionary.
    
    Args:
        args (tuple): A tuple containing (frame, frame_idx)
    
    Returns:
        dict: the frame information (from create_waymo_info_file)
    """
    
    frame, frame_idx = args
    
    save_image(frame, frame_idx, image_save_path)
    save_lidar(frame, pointcloud_save_path, frame_idx)
    
    local_file_infos = []
    create_waymo_info_file(frame, frame_idx, local_file_infos, test_mode=test_mode,
                             save_cam_sync_instances=save_cam_sync_instances, save_cam_instances=save_cam_instances)
    
    if local_file_infos:
        return local_file_infos[0]
    else:
        return None

# %%
num_workers = 8
file_infos = []

files = train_files

if type_ == 'testing':
    files = test_files
elif type_ == 'validation':
    files = val_files

args_list = list(zip(files, range(len(files))))

# %%
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for result in tqdm(executor.map(convert_one_frame, args_list), total=len(args_list),
                           desc="Processing frames in parallel"):
            if result is not None:
                file_infos.append(result)

# %%
import pickle
data_list = []

for data_info in file_infos:
    data_list.extend(data_info)

metainfo = dict()
metainfo['dataset'] = 'waymo'
metainfo['version'] = 'waymo_v1.4'
metainfo['info_version'] = 'mmdet3d_v1.4'
waymo_infos = dict(data_list=data_list, metainfo=metainfo)

filenames = osp.join(save_path, f'waymo{INFO_MAP[type_]}')
with open(filenames, 'wb') as f:
    pickle.dump(waymo_infos, f)