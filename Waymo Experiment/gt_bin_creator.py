from glob import glob
from os.path import exists, join

import mmengine
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

tf.enable_eager_execution()

import os
import random
import numpy as np
import torch
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU


class gt_bin_creator:
    """waymo gt.bin creator.

    Support create gt.bin from tfrecords and gt_subset.bin from gt.bin
    """

    # yapf: disable
    def __init__(
            self,
            ann_file,
            data_root,
            split,  # ('training','validation')
            waymo_bin_file=None,
            load_interval=1,
            for_cam_only_challenge=True,
            file_client_args: dict = dict(backend='disk')):
        # yapf: enable
        self.ann_file = ann_file
        self.waymo_bin_file = waymo_bin_file
        self.data_root = data_root
        self.split = split
        self.load_interval = load_interval
        self.for_cam_only_challenge = for_cam_only_challenge
        self.file_client_args = file_client_args
        self.waymo_tfrecords_dir = join(self.data_root, self.split)
        
        self.variation1 = 'day' # 'dawn_dusk', 'day', 'night'
        # self.variation2 = 'day'
        
        if self.waymo_bin_file is None:
            # self.waymo_bin_file = join(self.data_root,
            #                            'gt_{}.bin'.format(self.split))
            self.waymo_bin_file = 'gt_{}_{}_2.bin'.format(self.variation1, self.split)

    def get_target_timestamp(self):
        data_infos = mmengine.load(
            self.ann_file)['data_list'][::self.load_interval]
        self.timestamp = set()
        for info in data_infos:
            self.timestamp.add(info['timestamp'])
    
    def setup_seed(self, seed=0, deterministic = True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def create_subset(self):
        self.create_whole()
        name = 'gt_{}_subset_{}.bin'.format(self.split, self.load_interval)
        subset_path = join(self.data_root, name)

        if exists(subset_path):
            print(f'file {subset_path} exists. Skipping create_subset')
        else:
            print(f'Can not find {subset_path}, creating a new one...')
            objs = metrics_pb2.Objects()
            objs.ParseFromString(open(self.waymo_bin_file, 'rb').read())
            self.get_target_timestamp()
            objs_subset = metrics_pb2.Objects()
            prog_bar = mmengine.ProgressBar(len(objs.objects))
            for obj in objs.objects:
                prog_bar.update()
                if obj.frame_timestamp_micros not in self.timestamp:
                    continue
                if self.for_cam_only_challenge and \
                   obj.object.type == label_pb2.Label.TYPE_SIGN:
                    continue

                objs_subset.objects.append(obj)

            open(subset_path, 'wb').write(objs_subset.SerializeToString())
            print(f'Saved subset bin file to {subset_path}'
                  f'It has {len(objs_subset.objects)} objects.')

        return subset_path

    def create_whole(self):
        if exists(self.waymo_bin_file):
            print(f'file {self.waymo_bin_file} exists. Skipping create_whole')
        else:
            print(f'Can not find {self.waymo_bin_file}, creating a new one...')
            self.get_file_names()
            mapping = self.waymo_tfrecord_pathnames

            objs = metrics_pb2.Objects()
            frame_num = 0
            prog_bar = mmengine.ProgressBar(len(mapping))
            for tfnames, frame_idx in mapping:
                dataset = tf.data.TFRecordDataset(tfnames, compression_type='')
                
                for i, data in enumerate(dataset):
                    if i == frame_idx:
                        frame_num += 1
                        frame = open_dataset.Frame()
                        frame.ParseFromString(bytearray(data.numpy()))
                        
                        for label in frame.laser_labels:
                            if self.for_cam_only_challenge and (
                                    label.type == 3
                                    or label.camera_synced_box.ByteSize() == 0
                                    or label.num_lidar_points_in_box < 1):
                                continue
                            
                            new_obj = metrics_pb2.Object()
                            new_obj.frame_timestamp_micros = frame.timestamp_micros
                            new_obj.object.CopyFrom(label)
                            new_obj.context_name = frame.context.name
                            objs.objects.append(new_obj)
                        break
                    
                prog_bar.update()

            open(self.waymo_bin_file, 'wb').write(objs.SerializeToString())
            print(f'Saved groudtruth bin file to {self.waymo_bin_file}\n\
                    It has {len(objs.objects)} objects in {frame_num} frames.')

        return self.waymo_bin_file

    def get_file_names(self):
        """Get file names of waymo raw data."""
        # if 'path_mapping' in self.file_client_args:
        #     for path in self.file_client_args['path_mapping'].keys():
        #         if path in self.waymo_tfrecords_dir:
        #             self.waymo_tfrecords_dir = \
        #                 self.waymo_tfrecords_dir.replace(
        #                     path, self.file_client_args['path_mapping'][path])
        #     from petrel_client.client import Client
        #     client = Client()
        #     contents = client.list(self.waymo_tfrecords_dir)
        #     self.waymo_tfrecord_pathnames = list()
        #     for content in sorted(list(contents)):
        #         if content.endswith('tfrecord'):
        #             self.waymo_tfrecord_pathnames.append(
        #                 join(self.waymo_tfrecords_dir, content))
        # else:
        # self.waymo_tfrecord_pathnames = sorted(
        #     glob(join(self.waymo_tfrecords_dir, '*.tfrecord')))
        # print(len(self.waymo_tfrecord_pathnames), 'tfrecords found.')
        
        # Create record, frame pairs
        all_sf_files = {
            'dawn_dusk': [], 
            'day': [], 
            'night': []
        }
        train_files = os.listdir(os.path.join(self.data_root, 'training'))
        train_files = [f for f in train_files if f.endswith('.tfrecord')]
        print('Number of training files:', len(train_files))
        
        val_files = os.listdir(os.path.join(self.data_root, 'validation'))
        val_files = [f for f in val_files if f.endswith('.tfrecord')]
        print('Number of validation files:', len(val_files))
        
        for f in tqdm(train_files, desc="Processing files in training"):
            filename = os.path.join(self.data_root, 'training', f)
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
        
        for f in tqdm(val_files, desc="Processing files in validation"):
            filename = os.path.join(self.data_root, 'validation', f)
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
        
        self.setup_seed(42)
        
        video_to_frames1 = []
        frame_mapping1 = []
        
        files1 = all_sf_files[self.variation1]
        
        for f in tqdm(files1, desc="Processing files"):
            filename = f
            dataset = tf.data.TFRecordDataset(filename, compression_type='')
            for frame_idx, data in enumerate(dataset):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                
                video_to_frames1.append(frame)
                frame_mapping1.append((filename, frame_idx))
        
        combined1 = list(zip(video_to_frames1, frame_mapping1))
        random.shuffle(combined1)
        video_to_frames1, frame_mapping1 = zip(*combined1)
        
        num_frames1 = len(video_to_frames1)
        train_indices1 = list(range(0, int(0.8*num_frames1)))
        val_indices1 = list(range(int(0.8*num_frames1), int(0.9*num_frames1)))
        test_indices1 = list(range(int(0.9*num_frames1), num_frames1))
        # train_indices1 = []
        # val_indices1 = []
        # test_indices1 = list(range(0, int(num_frames1)))
        
        train_mapping1 = [frame_mapping1[i] for i in train_indices1]
        val_mapping1 = [frame_mapping1[i] for i in val_indices1]
        test_mapping1 = [frame_mapping1[i] for i in test_indices1]
        
        # video_to_frames2 = []
        # frame_mapping2 = []
        
        # files2 = all_sf_files[self.variation2]
        
        # for f in tqdm(files2, desc="Processing files"):
        #     filename = f
        #     dataset = tf.data.TFRecordDataset(filename, compression_type='')
        #     for frame_idx, data in enumerate(dataset):
        #         frame = open_dataset.Frame()
        #         frame.ParseFromString(bytearray(data.numpy()))
                
        #         video_to_frames2.append(frame)
        #         frame_mapping2.append((filename, frame_idx))
        
        # combined2 = list(zip(video_to_frames2, frame_mapping2))
        # random.shuffle(combined2)
        # video_to_frames2, frame_mapping2 = zip(*combined2)
        
        # num_frames2 = len(video_to_frames2)
        # train_indices2 = list(range(0, int(0.8*num_frames2)))
        # val_indices2 = list(range(int(0.8*num_frames2), int(0.9*num_frames2)))
        # test_indices2 = list(range(int(0.9*num_frames2), num_frames2))
        
        # train_mapping2 = [frame_mapping2[i] for i in train_indices2]
        # val_mapping2 = [frame_mapping2[i] for i in val_indices2]
        # test_mapping2 = [frame_mapping2[i] for i in test_indices2]
        
        # split_mapping = {
        #     'training': train_mapping1 + train_mapping2,
        #     'validation': val_mapping1 + val_mapping2,
        #     'testing': test_mapping1 + test_mapping2
        # }
        split_mapping = {
            'training': train_mapping1,
            'validation': val_mapping1,
            'testing': test_mapping1
        }
        
        self.waymo_tfrecord_pathnames = split_mapping[self.split]
        
        print(len(self.waymo_tfrecord_pathnames), 'tfrecords found.')

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--ann_file',
        default='./data/waymo_dev1x/kitti_format/waymo_infos_val.pkl')
    parser.add_argument(
        '--data_root', default='./data/waymo_dev1x/waymo_format')
    parser.add_argument(
        '--split', default='validation')  # ('training','validation')
    # parser.add_argument('waymo_bin_file')
    parser.add_argument('--load_interval', type=int, default=1)
    parser.add_argument('--for_cam_only_challenge', default=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # test = './data/waymo_dev1x/waymo_format/gt.bin'
    creator = gt_bin_creator(args.ann_file, args.data_root, args.split, None,
                             args.load_interval, args.for_cam_only_challenge)
    waymo_bin_file = creator.create_whole()
    print(waymo_bin_file)
    # waymo_subset_bin_file = creator.create_subset()
    # print(waymo_bin_file, waymo_subset_bin_file)
    # breakpoint()
    
    # creator = Create_gt_bin(args.data_root, None, args.split,
    #                         args.for_cam_only_challenge)
    # waymo_bin_file = creator.create_and_save()
    # print(waymo_bin_file)


if __name__ == '__main__':
    main()
"""
Usage:
python tools/dataset_converters/waymo_gtbin_creator.py \
    --ann_file ./data/waymo_dev1x/kitti_format/waymo_infos_val.pkl \
    --data_root ./data/waymo_dev1x/waymo_format \
    --split validation \
    --load_interval 1 \
    --for_cam_only_challenge True
"""