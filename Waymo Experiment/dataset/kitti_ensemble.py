import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from utils import read_pickle, read_points, bbox_camera2lidar
from dataset import point_range_filter, data_augment


class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class KittiEnsemble(Dataset):

    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2,
        'Sign': 3,
    }
    
    # CLASSES_TO_NAMES = {
    #     'PEDESTRIAN': 'Pedestrian',
    #     'CYCLIST': 'Cyclist',
    #     'VEHICLE': 'Car',
    #     'SIGN': 'Sign',
    # }
    CLASSES_TO_NAMES = {
        'pedestrian': 'Pedestrian',
        'cyclist': 'Cyclist',
        'vehicle': 'Car',
        'sign' : 'Sign',
    }

    def __init__(self, data_root_1, data_root_2, split, pts_prefix='lidar_reduced', root=None):
        assert split in ['train', 'val', 'trainval', 'test', 'validation']
        
        self.root = root
        self.data_root_1 = data_root_1
        self.data_root_2 = data_root_2
        
        self.split = split
        self.pts_prefix = pts_prefix
        
        # waymo_dbinfos_{split} for variation 1
        self.data_infos_1 = read_pickle(os.path.join(data_root_1, f'waymo_infos_{split}.pkl'))
        self.sorted_ids_1 = list(self.data_infos_1.keys())
        
        # Length of the first variation
        self.l1 = len(self.sorted_ids_1)
        
        db_infos_1 = read_pickle(os.path.join(data_root_1, f'waymo_dbinfos_{split}.pkl'))
        db_infos_1 = self.filter_db(db_infos_1)

        db_sampler_1 = {}
        for cat_name in self.CLASSES_TO_NAMES:
            db_sampler_1[self.CLASSES_TO_NAMES[cat_name]] = BaseSampler(db_infos_1[cat_name], shuffle=True)
        
        self.data_aug_config_1 = dict(
            db_sampler=dict(
                db_sampler=db_sampler_1,
                sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10) #, Sign=5)
                ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
                ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
                ), 
            point_range_filter=[0, -39.68, -3, 69.12, 39.68, 1],
            object_range_filter=[0, -39.68, -3, 69.12, 39.68, 1]             
        )
        
        # waymo_dbinfos_{split} for variation 2
        self.data_infos_2 = read_pickle(os.path.join(data_root_2, f'waymo_infos_{split}.pkl'))
        self.sorted_ids_2 = list(self.data_infos_2.keys())
        
        # Length of the second variation
        self.l2 = len(self.sorted_ids_2)
        
        db_infos_2 = read_pickle(os.path.join(data_root_2, f'waymo_dbinfos_{split}.pkl'))
        db_infos_2 = self.filter_db(db_infos_2)
        
        db_sampler_2 = {}
        for cat_name in self.CLASSES_TO_NAMES:
            db_sampler_2[self.CLASSES_TO_NAMES[cat_name]] = BaseSampler(db_infos_2[cat_name], shuffle=True)
            
        self.data_aug_config_2 = dict(
            db_sampler=dict(
                db_sampler=db_sampler_2,
                sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10) #, Sign=5)
                ),
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
                ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
                ),
            point_range_filter=[0, -39.68, -3, 69.12, 39.68, 1], # [-74.88, -74.88, -2, 74.88, 74.88, 4]
            object_range_filter=[0, -39.68, -3, 69.12, 39.68, 1]
        )

    def remove_dont_care(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info['name']) if name != 'DontCare' and name != 'Sign'] # Remove 'SIGN' for now
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        
        keep_ids = [i for i, difficulty in enumerate(annos_info['difficulty']) if difficulty != -1]
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item['difficulty'] != -1]

        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10, Sign=5)
        filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10, Sign=5)
        for cat in self.CLASSES_TO_NAMES:
            filter_thr = filter_thrs[self.CLASSES_TO_NAMES[cat]]
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
        
        return db_infos

    def __getitem__(self, index):
        
        if index < self.l1:        
            data_info = self.data_infos_1[self.sorted_ids_1[index]]
            image_info, calib_info, annos_info = \
                data_info['image'], data_info['calib'], data_info['annos']
        
            # point cloud input
            lidar_path = data_info['lidar_path'].replace('lidar', self.pts_prefix)
            if self.root is None:
                pts_path = os.path.join(lidar_path)
            else:
                pts_path = os.path.join(self.root, lidar_path)
            pts = read_points(pts_path)
            
            # calib input: for bbox coordinates transformation between Camera and Lidar.
            # because 
            tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
            r0_rect = calib_info['R0_rect'].astype(np.float32)

            # Updating annotations input
            annos_info['name'] = np.array(
                ['Car' if name == 'VEHICLE' else name for name in annos_info['name']]
            )
            annos_info['name'] = np.array(
                ['Pedestrian' if name == 'PEDESTRIAN' else name for name in annos_info['name']]
            )
            annos_info['name'] = np.array(
                ['Cyclist' if name == 'CYCLIST' else name for name in annos_info['name']]
            )
            annos_info['name'] = np.array(
                ['Sign' if name == 'SIGN' else name for name in annos_info['name']]
            )

            annos_info = self.remove_dont_care(annos_info)
            annos_name = annos_info['name']
            annos_location = annos_info['location']
            annos_dimension = annos_info['dimensions']
            rotation_y = annos_info['rotation_y']
            gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
            gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
            gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
            data_dict = {
                'pts': pts,
                'gt_bboxes_3d': gt_bboxes_3d,
                'gt_labels': np.array(gt_labels), 
                'gt_names': annos_name,
                'difficulty': annos_info['difficulty'],
                'image_info': image_info,
                'calib_info': calib_info,
                'image_path': data_info['image']['image_path'],
            }

            if self.split in ['train', 'trainval', 'val']:
                data_dict = data_augment(self.CLASSES, self.CLASSES_TO_NAMES, self.data_root_1, data_dict, self.data_aug_config_1)
            else:
                data_dict = point_range_filter(data_dict, point_range=self.data_aug_config_1['point_range_filter'])
        else:
            index = index - self.l1
            
            data_info = self.data_infos_2[self.sorted_ids_2[index]]
            image_info, calib_info, annos_info = \
                data_info['image'], data_info['calib'], data_info['annos']
        
            # point cloud input
            lidar_path = data_info['lidar_path'].replace('lidar', self.pts_prefix)
            if self.root is None:
                pts_path = os.path.join(lidar_path)
            else:
                pts_path = os.path.join(self.root, lidar_path)
            pts = read_points(pts_path)
            
            # calib input: for bbox coordinates transformation between Camera and Lidar.
            # because
            tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
            r0_rect = calib_info['R0_rect'].astype(np.float32)

            # Updating annotations input
            annos_info['name'] = np.array(
                ['Car' if name == 'VEHICLE' else name for name in annos_info['name']]
            )
            annos_info['name'] = np.array(
                ['Pedestrian' if name == 'PEDESTRIAN' else name for name in annos_info['name']]
            )
            annos_info['name'] = np.array(
                ['Cyclist' if name == 'CYCLIST' else name for name in annos_info['name']]
            )
            annos_info['name'] = np.array(
                ['Sign' if name == 'SIGN' else name for name in annos_info['name']]
            )

            annos_info = self.remove_dont_care(annos_info)
            annos_name = annos_info['name']
            annos_location = annos_info['location']
            annos_dimension = annos_info['dimensions']
            rotation_y = annos_info['rotation_y']
            gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
            gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect)
            gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
            data_dict = {
                'pts': pts,
                'gt_bboxes_3d': gt_bboxes_3d,
                'gt_labels': np.array(gt_labels), 
                'gt_names': annos_name,
                'difficulty': annos_info['difficulty'],
                'image_info': image_info,
                'calib_info': calib_info,
                'image_path': data_info['image']['image_path'],
            }

            if self.split in ['train', 'trainval']:
                data_dict = data_augment(self.CLASSES, self.CLASSES_TO_NAMES, self.data_root_2, data_dict, self.data_aug_config_2)
            else:
                data_dict = point_range_filter(data_dict, point_range=self.data_aug_config_2['point_range_filter'])

        return data_dict

    def __len__(self):
        return len(self.data_infos_1) + len(self.data_infos_2)
 

# if __name__ == '__main__':
    
#     kitti_data = Kitti(data_root='/mnt/ssd1/lifa_rdata/det/kitti', 
#                        split='train')
#     kitti_data.__getitem__(9)