
import numpy as np
import torch

from tqdm import tqdm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR messages -> Supress tensorflow warnings
from os import path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

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


def setup_seed(seed=0, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


variation1 = 'day'
# variation2 = 'day'
# type_ = 'validation'

data_path = '/mnt/data/ataparia/waymo_perception_dataset_v1_4_3'
save_path = f'/mnt/data/ataparia/LidarTraining/{variation1}-frames'

# Create the save directory
os.makedirs(save_path, exist_ok=True)
# os.makedirs(f'{save_path}/training', exist_ok=True)
# os.makedirs(f'{save_path}/validation', exist_ok=True)
# os.makedirs(f'{save_path}/testing', exist_ok=True)

all_sf_files = {
    'dawn_dusk': [], 
    'day': [], 
    'night': []
}


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


setup_seed(42)


video_to_frames = []

files1 = all_sf_files[variation1]

for f in tqdm(files1, desc="Processing files"):
    filename = f
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        video_to_frames.append(frame)


random.shuffle(video_to_frames)


print(f'Number of {variation1} files:', len(video_to_frames))

train_files1 = video_to_frames[:int(0.8*len(video_to_frames))]
# train_files1 = []
val_files1 = video_to_frames[int(0.8*len(video_to_frames)):int(0.9*len(video_to_frames))]
# val_files1 = []
test_files1 = video_to_frames[int(0.9*len(video_to_frames)):]
# test_files1 = video_to_frames

print('Variation:', variation1)
print('Number of train files:', len(train_files1))
print('Number of val files:', len(val_files1))
print('Number of test files:', len(test_files1))


# video_to_frames = []

# files2 = all_sf_files[variation2]

# for f in tqdm(files2, desc="Processing files"):
#     filename = f
#     dataset = tf.data.TFRecordDataset(filename, compression_type='')
#     for data in dataset:
#         frame = open_dataset.Frame()
#         frame.ParseFromString(bytearray(data.numpy()))
        
#         video_to_frames.append(frame)
    
# random.shuffle(video_to_frames)

# print(f'Number of {variation2} files:', len(video_to_frames))

# train_files2 = video_to_frames[:int(0.8*len(video_to_frames))]
# val_files2 = video_to_frames[int(0.8*len(video_to_frames)):int(0.9*len(video_to_frames))]
# test_files2 = video_to_frames[int(0.9*len(video_to_frames)):]

# print('Variation:', variation2)
# print('Number of train files:', len(train_files2))
# print('Number of val files:', len(val_files2))
# print('Number of test files:', len(test_files2))

# split_frames = {
#     'training': train_files1 + train_files2,
#     'validation': val_files1 + val_files2,
#     'testing': test_files1 + test_files2
# }

split_frames = {
    'training': train_files1,
    'validation': val_files1,
    'testing': test_files1
}


from os import path as osp
from mmengine import print_log

from create_gt_database import GTDatabaseCreater, create_groundtruth_database
from update_infos_to_v2 import update_pkl_infos


version = 'v1.4'
max_sweeps = 1
workers = 0
extra_tag = 'waymo'


def waymo_data_prep(frame_dict,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=1,
                    only_gt_database=False,
                    save_senor_data=True,
                    skip_cam_instances_infos=False):
    """Prepare waymo dataset. There are 3 steps as follows:

    Step 1. Extract camera images and lidar point clouds from waymo raw
        data in '*.tfreord' and save as kitti format.
    Step 2. Generate waymo train/val/test infos and save as pickle file.
    Step 3. Generate waymo ground truth database (point clouds within
        each 3D bounding box) for data augmentation in training.
    Steps 1 and 2 will be done in Waymo2KITTI, and step 3 will be done in
    GTDatabaseCreater.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default to 10. Here we store ego2global information of these
            frames for later use.
        only_gt_database (bool, optional): Whether to only generate ground
            truth database. Default to False.
        save_senor_data (bool, optional): Whether to skip saving
            image and lidar. Default to False.
        skip_cam_instances_infos (bool, optional): Whether to skip
            gathering cam_instances infos in Step 2. Default to False.
    """
    import waymo_converter as waymo

    if version == 'v1.4':
        splits = [
            'training', 
            'validation', 
            'testing'
            # , 'testing_3d_camera_only_detection'
        ]
    elif version == 'v1.4-mini':
        splits = ['training', 'validation']
    else:
        raise NotImplementedError(f'Unsupported Waymo version {version}!')
    # out_dir = osp.join(out_dir, 'kitti_format')

    if not only_gt_database:
        for i, split in enumerate(splits):
            # load_dir = osp.join(root_path, 'waymo_format', split)
            frames = frame_dict[split]
            if frames.__len__() == 0:
                continue
            
            if split == 'validation' or split == 'training':
                continue
            
            if split == 'validation':
                save_dir = osp.join(out_dir, 'training')
            else:
                save_dir = osp.join(out_dir, split)
            converter = waymo.Waymo2KITTI(
                # load_dir,
                frames,
                save_dir,
                prefix=str(i),
                workers=workers,
                test_mode=(split
                           in ['testing_3d_camera_only_detection']),
                info_prefix=info_prefix,
                max_sweeps=max_sweeps,
                split=split,
                save_senor_data=save_senor_data,
                save_cam_instances=not skip_cam_instances_infos)
            converter.convert()
            if split == 'validation':
                converter.merge_trainval_infos()

        from waymo_converter import create_ImageSets_img_ids
        create_ImageSets_img_ids(out_dir, splits)


    # GTDatabaseCreater(
    #     'WaymoDataset',
    #     out_dir,
    #     info_prefix,
    #     f'{info_prefix}_infos_train.pkl',
    #     relative_path=False,
    #     with_mask=False,
    #     num_worker=workers).create()

    print_log('Successfully preparing Waymo Open Dataset')


waymo_data_prep(
            frame_dict=split_frames,
            info_prefix=extra_tag,
            version=version,
            out_dir=save_path,
            workers=workers,
            save_senor_data=True,
            max_sweeps=max_sweeps)
