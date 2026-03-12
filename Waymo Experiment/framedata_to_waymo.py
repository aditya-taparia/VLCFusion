# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    from waymo_open_dataset.protos.metrics_pb2 import Objects
except ImportError:
    Objects = None
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')

from typing import List

import mmengine
from mmengine import print_log


class Framedata2Waymo(object):
    """Predictions to Waymo converter. The format of prediction results could
    be original format or kitti-format.

    This class serves as the converter to change predictions from KITTI to
    Waymo format.

    Args:
        results (list[dict]): Prediction results.
        waymo_results_save_dir (str): Directory to save converted predictions
            in waymo format (.bin files).
        waymo_results_final_path (str): Path to save combined
            predictions in waymo format (.bin file), like 'a/b/c.bin'.
        num_workers (str): Number of parallel processes. Defaults to 4.
    """

    def __init__(self,
                 gt: List[dict],
                 waymo_gt_final_path: str,
                 classes: dict,
                 num_workers: int = 4):
        self.gt = gt
        self.waymo_gt_final_path = waymo_gt_final_path
        self.classes = classes
        self.num_workers = num_workers

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

    def convert_one(self, frame_idx: int):
        """Convert action for single file. It read the metainfo from the
        preprocessed file offline and will be faster.

        Args:
            res_idx (int): The indices of the results.
        """
        sample_idx = self.gt[frame_idx]['sample_idx']
        if len(self.gt[frame_idx]['labels_3d']) > 0:
            objects = self.parse_objects_from_origin(
                self.gt[frame_idx], self.gt[frame_idx]['context_name'],
                self.gt[frame_idx]['timestamp'])
        else:
            print(sample_idx, 'not found.')
            objects = metrics_pb2.Objects()

        return objects

    def parse_objects_from_origin(self, frame: dict, contextname: str,
                                  timestamp: str) -> Objects:
        """Parse obejcts from the original prediction results.

        Args:
            result (dict): The original prediction results.
            contextname (str): The ``contextname`` of sample in waymo.
            timestamp (str): The ``timestamp`` of sample in waymo.

        Returns:
            metrics_pb2.Objects: The parsed object.
        """
        lidar_boxes = frame['bboxes_3d']
        # scores = frame['scores_3d']
        labels = frame['labels_3d']
        num_lidar_pts = frame['num_lidar_pts']

        objects = metrics_pb2.Objects()
        for lidar_box, label, pts in zip(lidar_boxes, labels, num_lidar_pts):
            if pts < 1:
                continue
            
            # Parse one object
            box = label_pb2.Label.Box()
            height = lidar_box[5]
            heading = lidar_box[6]

            box.center_x = lidar_box[0]
            box.center_y = lidar_box[1]
            box.center_z = lidar_box[2] + height / 2
            box.length = lidar_box[3]
            box.width = lidar_box[4]
            box.height = height
            box.heading = heading

            object = metrics_pb2.Object()
            object.object.box.CopyFrom(box)
            object.object.num_lidar_points_in_box = pts
            class_name = self.classes[label]
            object.object.type = self.k2w_cls_map[class_name]
            object.score = 0.5
            object.context_name = contextname
            object.frame_timestamp_micros = timestamp
            objects.objects.append(object)

        return objects

    def convert(self):
        """Convert action."""
        print_log('Start converting ...', logger='current')

        # TODO: use parallel processes.
        # objects_list = mmengine.track_parallel_progress(
        #     self.convert_one, range(len(self)), self.num_workers)

        objects_list = mmengine.track_progress(self.convert_one,
                                               range(len(self)))

        combined = metrics_pb2.Objects()
        for objects in objects_list:
            for o in objects.objects:
                combined.objects.append(o)

        with open(self.waymo_gt_final_path, 'wb') as f:
            f.write(combined.SerializeToString())

    def __len__(self):
        """Length of the filename list."""
        return len(self.gt)
