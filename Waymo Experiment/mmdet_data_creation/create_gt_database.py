# Copyright (c) OpenMMLab. All rights reserved.
import pickle
from os import path as osp

import mmcv
import mmengine
import numpy as np
from mmcv.ops import roi_align
from mmdet.evaluation import bbox_overlaps
from mmengine import print_log, track_iter_progress
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

# from mmdet3d.datasets.transforms import LoadPointsFromFile
from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops as box_np_ops



# ####################################################################################################
# # Re-register the transforms with mmengine
# ####################################################################################################

# from mmengine.registry import TRANSFORMS
# from mmdet3d.structures.bbox_3d import get_box_type
# from mmcv.transforms.base import BaseTransform
# from typing import List, Optional, Union
# from mmengine.fileio import get
# from mmdet3d.structures.points import BasePoints, get_points_type
# from mmdet.datasets.transforms import LoadAnnotations

# @TRANSFORMS.register_module()
# class LoadPointsFromFile(BaseTransform):
#     """Load Points From File.

#     Required Keys:

#     - lidar_points (dict)

#         - lidar_path (str)

#     Added Keys:

#     - points (np.float32)

#     Args:
#         coord_type (str): The type of coordinates of points cloud.
#             Available options includes:

#             - 'LIDAR': Points in LiDAR coordinates.
#             - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
#             - 'CAMERA': Points in camera coordinates.
#         load_dim (int): The dimension of the loaded points. Defaults to 6.
#         use_dim (list[int] | int): Which dimensions of the points to use.
#             Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
#             or use_dim=[0, 1, 2, 3] to use the intensity dimension.
#         shift_height (bool): Whether to use shifted height. Defaults to False.
#         use_color (bool): Whether to use color features. Defaults to False.
#         norm_intensity (bool): Whether to normlize the intensity. Defaults to
#             False.
#         norm_elongation (bool): Whether to normlize the elongation. This is
#             usually used in Waymo dataset.Defaults to False.
#         backend_args (dict, optional): Arguments to instantiate the
#             corresponding backend. Defaults to None.
#     """

#     def __init__(self,
#                  coord_type: str,
#                  load_dim: int = 6,
#                  use_dim: Union[int, List[int]] = [0, 1, 2],
#                  shift_height: bool = False,
#                  use_color: bool = False,
#                  norm_intensity: bool = False,
#                  norm_elongation: bool = False,
#                  backend_args: Optional[dict] = None) -> None:
#         self.shift_height = shift_height
#         self.use_color = use_color
#         if isinstance(use_dim, int):
#             use_dim = list(range(use_dim))
#         assert max(use_dim) < load_dim, \
#             f'Expect all used dimensions < {load_dim}, got {use_dim}'
#         assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

#         self.coord_type = coord_type
#         self.load_dim = load_dim
#         self.use_dim = use_dim
#         self.norm_intensity = norm_intensity
#         self.norm_elongation = norm_elongation
#         self.backend_args = backend_args

#     def _load_points(self, pts_filename: str) -> np.ndarray:
#         """Private function to load point clouds data.

#         Args:
#             pts_filename (str): Filename of point clouds data.

#         Returns:
#             np.ndarray: An array containing point clouds data.
#         """
#         try:
#             pts_bytes = get(pts_filename, backend_args=self.backend_args)
#             points = np.frombuffer(pts_bytes, dtype=np.float32)
#         except ConnectionError:
#             mmengine.check_file_exist(pts_filename)
#             if pts_filename.endswith('.npy'):
#                 points = np.load(pts_filename)
#             else:
#                 points = np.fromfile(pts_filename, dtype=np.float32)

#         return points

#     def transform(self, results: dict) -> dict:
#         """Method to load points data from file.

#         Args:
#             results (dict): Result dict containing point clouds data.

#         Returns:
#             dict: The result dict containing the point clouds data.
#             Added key and value are described below.

#                 - points (:obj:`BasePoints`): Point clouds data.
#         """
#         pts_file_path = results['lidar_points']['lidar_path']
#         points = self._load_points(pts_file_path)
#         points = points.reshape(-1, self.load_dim)
#         points = points[:, self.use_dim]
#         if self.norm_intensity:
#             assert len(self.use_dim) >= 4, \
#                 f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
#             points[:, 3] = np.tanh(points[:, 3])
#         if self.norm_elongation:
#             assert len(self.use_dim) >= 5, \
#                 f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
#             points[:, 4] = np.tanh(points[:, 4])
#         attribute_dims = None

#         if self.shift_height:
#             floor_height = np.percentile(points[:, 2], 0.99)
#             height = points[:, 2] - floor_height
#             points = np.concatenate(
#                 [points[:, :3],
#                  np.expand_dims(height, 1), points[:, 3:]], 1)
#             attribute_dims = dict(height=3)

#         if self.use_color:
#             assert len(self.use_dim) >= 6
#             if attribute_dims is None:
#                 attribute_dims = dict()
#             attribute_dims.update(
#                 dict(color=[
#                     points.shape[1] - 3,
#                     points.shape[1] - 2,
#                     points.shape[1] - 1,
#                 ]))

#         points_class = get_points_type(self.coord_type)
#         points = points_class(
#             points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
#         results['points'] = points

#         return results

#     def __repr__(self) -> str:
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__ + '('
#         repr_str += f'shift_height={self.shift_height}, '
#         repr_str += f'use_color={self.use_color}, '
#         repr_str += f'backend_args={self.backend_args}, '
#         repr_str += f'load_dim={self.load_dim}, '
#         repr_str += f'use_dim={self.use_dim})'
#         repr_str += f'norm_intensity={self.norm_intensity})'
#         repr_str += f'norm_elongation={self.norm_elongation})'
#         return repr_str



# @TRANSFORMS.register_module()
# class LoadAnnotations3D(LoadAnnotations):
#     """Load Annotations3D.

#     Load instance mask and semantic mask of points and
#     encapsulate the items into related fields.

#     Required Keys:

#     - ann_info (dict)

#         - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
#           :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
#           3D ground truth bboxes. Only when `with_bbox_3d` is True
#         - gt_labels_3d (np.int64): Labels of ground truths.
#           Only when `with_label_3d` is True.
#         - gt_bboxes (np.float32): 2D ground truth bboxes.
#           Only when `with_bbox` is True.
#         - gt_labels (np.ndarray): Labels of ground truths.
#           Only when `with_label` is True.
#         - depths (np.ndarray): Only when
#           `with_bbox_depth` is True.
#         - centers_2d (np.ndarray): Only when
#           `with_bbox_depth` is True.
#         - attr_labels (np.ndarray): Attribute labels of instances.
#           Only when `with_attr_label` is True.

#     - pts_instance_mask_path (str): Path of instance mask file.
#       Only when `with_mask_3d` is True.
#     - pts_semantic_mask_path (str): Path of semantic mask file.
#       Only when `with_seg_3d` is True.
#     - pts_panoptic_mask_path (str): Path of panoptic mask file.
#       Only when both `with_panoptic_3d` is True.

#     Added Keys:

#     - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
#       :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
#       3D ground truth bboxes. Only when `with_bbox_3d` is True
#     - gt_labels_3d (np.int64): Labels of ground truths.
#       Only when `with_label_3d` is True.
#     - gt_bboxes (np.float32): 2D ground truth bboxes.
#       Only when `with_bbox` is True.
#     - gt_labels (np.int64): Labels of ground truths.
#       Only when `with_label` is True.
#     - depths (np.float32): Only when
#       `with_bbox_depth` is True.
#     - centers_2d (np.ndarray): Only when
#       `with_bbox_depth` is True.
#     - attr_labels (np.int64): Attribute labels of instances.
#       Only when `with_attr_label` is True.
#     - pts_instance_mask (np.int64): Instance mask of each point.
#       Only when `with_mask_3d` is True.
#     - pts_semantic_mask (np.int64): Semantic mask of each point.
#       Only when `with_seg_3d` is True.

#     Args:
#         with_bbox_3d (bool): Whether to load 3D boxes. Defaults to True.
#         with_label_3d (bool): Whether to load 3D labels. Defaults to True.
#         with_attr_label (bool): Whether to load attribute label.
#             Defaults to False.
#         with_mask_3d (bool): Whether to load 3D instance masks for points.
#             Defaults to False.
#         with_seg_3d (bool): Whether to load 3D semantic masks for points.
#             Defaults to False.
#         with_bbox (bool): Whether to load 2D boxes. Defaults to False.
#         with_label (bool): Whether to load 2D labels. Defaults to False.
#         with_mask (bool): Whether to load 2D instance masks. Defaults to False.
#         with_seg (bool): Whether to load 2D semantic masks. Defaults to False.
#         with_bbox_depth (bool): Whether to load 2.5D boxes. Defaults to False.
#         with_panoptic_3d (bool): Whether to load 3D panoptic masks for points.
#             Defaults to False.
#         poly2mask (bool): Whether to convert polygon annotations to bitmasks.
#             Defaults to True.
#         seg_3d_dtype (str): String of dtype of 3D semantic masks.
#             Defaults to 'np.int64'.
#         seg_offset (int): The offset to split semantic and instance labels from
#             panoptic labels. Defaults to None.
#         dataset_type (str): Type of dataset used for splitting semantic and
#             instance labels. Defaults to None.
#         backend_args (dict, optional): Arguments to instantiate the
#             corresponding backend. Defaults to None.
#     """

#     def __init__(self,
#                  with_bbox_3d: bool = True,
#                  with_label_3d: bool = True,
#                  with_attr_label: bool = False,
#                  with_mask_3d: bool = False,
#                  with_seg_3d: bool = False,
#                  with_bbox: bool = False,
#                  with_label: bool = False,
#                  with_mask: bool = False,
#                  with_seg: bool = False,
#                  with_bbox_depth: bool = False,
#                  with_panoptic_3d: bool = False,
#                  poly2mask: bool = True,
#                  seg_3d_dtype: str = 'np.int64',
#                  seg_offset: int = None,
#                  dataset_type: str = None,
#                  backend_args: Optional[dict] = None) -> None:
#         super().__init__(
#             with_bbox=with_bbox,
#             with_label=with_label,
#             with_mask=with_mask,
#             with_seg=with_seg,
#             poly2mask=poly2mask,
#             backend_args=backend_args)
#         self.with_bbox_3d = with_bbox_3d
#         self.with_bbox_depth = with_bbox_depth
#         self.with_label_3d = with_label_3d
#         self.with_attr_label = with_attr_label
#         self.with_mask_3d = with_mask_3d
#         self.with_seg_3d = with_seg_3d
#         self.with_panoptic_3d = with_panoptic_3d
#         self.seg_3d_dtype = eval(seg_3d_dtype)
#         self.seg_offset = seg_offset
#         self.dataset_type = dataset_type

#     def _load_bboxes_3d(self, results: dict) -> dict:
#         """Private function to move the 3D bounding box annotation from
#         `ann_info` field to the root of `results`.

#         Args:
#             results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

#         Returns:
#             dict: The dict containing loaded 3D bounding box annotations.
#         """

#         results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
#         return results

#     def _load_bboxes_depth(self, results: dict) -> dict:
#         """Private function to load 2.5D bounding box annotations.

#         Args:
#             results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

#         Returns:
#             dict: The dict containing loaded 2.5D bounding box annotations.
#         """

#         results['depths'] = results['ann_info']['depths']
#         results['centers_2d'] = results['ann_info']['centers_2d']
#         return results

#     def _load_labels_3d(self, results: dict) -> dict:
#         """Private function to load label annotations.

#         Args:
#             results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

#         Returns:
#             dict: The dict containing loaded label annotations.
#         """

#         results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
#         return results

#     def _load_attr_labels(self, results: dict) -> dict:
#         """Private function to load label annotations.

#         Args:
#             results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

#         Returns:
#             dict: The dict containing loaded label annotations.
#         """
#         results['attr_labels'] = results['ann_info']['attr_labels']
#         return results

#     def _load_masks_3d(self, results: dict) -> dict:
#         """Private function to load 3D mask annotations.

#         Args:
#             results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

#         Returns:
#             dict: The dict containing loaded 3D mask annotations.
#         """
#         pts_instance_mask_path = results['pts_instance_mask_path']

#         try:
#             mask_bytes = get(
#                 pts_instance_mask_path, backend_args=self.backend_args)
#             pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
#         except ConnectionError:
#             mmengine.check_file_exist(pts_instance_mask_path)
#             pts_instance_mask = np.fromfile(
#                 pts_instance_mask_path, dtype=np.int64)

#         results['pts_instance_mask'] = pts_instance_mask
#         # 'eval_ann_info' will be passed to evaluator
#         if 'eval_ann_info' in results:
#             results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
#         return results

#     def _load_semantic_seg_3d(self, results: dict) -> dict:
#         """Private function to load 3D semantic segmentation annotations.

#         Args:
#             results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

#         Returns:
#             dict: The dict containing the semantic segmentation annotations.
#         """
#         pts_semantic_mask_path = results['pts_semantic_mask_path']

#         try:
#             mask_bytes = get(
#                 pts_semantic_mask_path, backend_args=self.backend_args)
#             # add .copy() to fix read-only bug
#             pts_semantic_mask = np.frombuffer(
#                 mask_bytes, dtype=self.seg_3d_dtype).copy()
#         except ConnectionError:
#             mmengine.check_file_exist(pts_semantic_mask_path)
#             pts_semantic_mask = np.fromfile(
#                 pts_semantic_mask_path, dtype=np.int64)

#         if self.dataset_type == 'semantickitti':
#             pts_semantic_mask = pts_semantic_mask.astype(np.int64)
#             pts_semantic_mask = pts_semantic_mask % self.seg_offset
#         # nuScenes loads semantic and panoptic labels from different files.

#         results['pts_semantic_mask'] = pts_semantic_mask

#         # 'eval_ann_info' will be passed to evaluator
#         if 'eval_ann_info' in results:
#             results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
#         return results

#     def _load_panoptic_3d(self, results: dict) -> dict:
#         """Private function to load 3D panoptic segmentation annotations.

#         Args:
#             results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

#         Returns:
#             dict: The dict containing the panoptic segmentation annotations.
#         """
#         pts_panoptic_mask_path = results['pts_panoptic_mask_path']

#         try:
#             mask_bytes = get(
#                 pts_panoptic_mask_path, backend_args=self.backend_args)
#             # add .copy() to fix read-only bug
#             pts_panoptic_mask = np.frombuffer(
#                 mask_bytes, dtype=self.seg_3d_dtype).copy()
#         except ConnectionError:
#             mmengine.check_file_exist(pts_panoptic_mask_path)
#             pts_panoptic_mask = np.fromfile(
#                 pts_panoptic_mask_path, dtype=np.int64)

#         if self.dataset_type == 'semantickitti':
#             pts_semantic_mask = pts_panoptic_mask.astype(np.int64)
#             pts_semantic_mask = pts_semantic_mask % self.seg_offset
#         elif self.dataset_type == 'nuscenes':
#             pts_semantic_mask = pts_semantic_mask // self.seg_offset

#         results['pts_semantic_mask'] = pts_semantic_mask

#         # We can directly take panoptic labels as instance ids.
#         pts_instance_mask = pts_panoptic_mask.astype(np.int64)
#         results['pts_instance_mask'] = pts_instance_mask

#         # 'eval_ann_info' will be passed to evaluator
#         if 'eval_ann_info' in results:
#             results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
#             results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
#         return results

#     def _load_bboxes(self, results: dict) -> None:
#         """Private function to load bounding box annotations.

#         The only difference is it remove the proceess for
#         `ignore_flag`

#         Args:
#             results (dict): Result dict from :obj:`mmcv.BaseDataset`.

#         Returns:
#             dict: The dict contains loaded bounding box annotations.
#         """

#         results['gt_bboxes'] = results['ann_info']['gt_bboxes']

#     def _load_labels(self, results: dict) -> None:
#         """Private function to load label annotations.

#         Args:
#             results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

#         Returns:
#             dict: The dict contains loaded label annotations.
#         """
#         results['gt_bboxes_labels'] = results['ann_info']['gt_bboxes_labels']

#     def transform(self, results: dict) -> dict:
#         """Function to load multiple types annotations.

#         Args:
#             results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

#         Returns:
#             dict: The dict containing loaded 3D bounding box, label, mask and
#             semantic segmentation annotations.
#         """
#         results = super().transform(results)
#         if self.with_bbox_3d:
#             results = self._load_bboxes_3d(results)
#         if self.with_bbox_depth:
#             results = self._load_bboxes_depth(results)
#         if self.with_label_3d:
#             results = self._load_labels_3d(results)
#         if self.with_attr_label:
#             results = self._load_attr_labels(results)
#         if self.with_panoptic_3d:
#             results = self._load_panoptic_3d(results)
#         if self.with_mask_3d:
#             results = self._load_masks_3d(results)
#         if self.with_seg_3d:
#             results = self._load_semantic_seg_3d(results)
#         return results

#     def __repr__(self) -> str:
#         """str: Return a string that describes the module."""
#         indent_str = '    '
#         repr_str = self.__class__.__name__ + '(\n'
#         repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
#         repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
#         repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
#         repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
#         repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
#         repr_str += f'{indent_str}with_panoptic_3d={self.with_panoptic_3d}, '
#         repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
#         repr_str += f'{indent_str}with_label={self.with_label}, '
#         repr_str += f'{indent_str}with_mask={self.with_mask}, '
#         repr_str += f'{indent_str}with_seg={self.with_seg}, '
#         repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
#         repr_str += f'{indent_str}poly2mask={self.poly2mask})'
#         repr_str += f'{indent_str}seg_offset={self.seg_offset})'

#         return repr_str
    
# ####################################################################################################
    


def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def _parse_coco_ann_info(ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        if ann['area'] <= 0:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get('iscrowd', False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_masks_ann.append(ann['segmentation'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(
        bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair
    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = (
        torch.arange(num_pos,
                     device=device).to(dtype=pos_proposals.dtype)[:, None])
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks).to(device).index_select(
            0, pos_assigned_gt_inds).to(dtype=rois.dtype))
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    targets = (
        roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1))
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1:y1 + h, x1:x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1:y1 + h, x1:x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks


def create_groundtruth_database(dataset_class_name,
                                data_path,
                                info_prefix,
                                info_path=None,
                                mask_anno_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None,
                                with_mask=False):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
    """
    print(f'Create GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path)
    if dataset_class_name == 'KittiDataset':
        backend_args = None
        dataset_cfg.update(
            modality=dict(
                use_lidar=True,
                use_camera=with_mask,
            ),
            data_prefix=dict(
                pts='training/velodyne_reduced', img='training/image_2'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    backend_args=backend_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    backend_args=backend_args)
            ])

    elif dataset_class_name == 'NuScenesDataset':
        dataset_cfg.update(
            use_valid_flag=True,
            data_prefix=dict(
                pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP'),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5),
                dict(
                    type='LoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])

    elif dataset_class_name == 'WaymoDataset':
        backend_args = None
        dataset_cfg.update(
            test_mode=False,
            data_prefix=dict(
                pts='training/velodyne', img='', sweeps='training/velodyne'),
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=6,
                    use_dim=6,
                    backend_args=backend_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    backend_args=backend_args)
            ])

    dataset = DATASETS.build(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f'{info_prefix}_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path,
                                     f'{info_prefix}_dbinfos_train.pkl')
    mmengine.mkdir_or_exist(database_save_path)
    all_db_infos = dict()
    if with_mask:
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0]
            file2id.update({info['file_name']: i})

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        data_info = dataset.get_data_info(j)
        example = dataset.pipeline(data_info)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        names = [dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        if with_mask:
            # prepare masks
            gt_boxes = annos['gt_bboxes']
            img_path = osp.split(example['img_info']['filename'])[-1]
            if img_path not in file2id.keys():
                print(f'skip image {img_path} for empty mask')
                continue
            img_id = file2id[img_path]
            kins_annIds = coco.getAnnIds(imgIds=img_id)
            kins_raw_info = coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos['img_shape'][:2]
            gt_masks = [
                _poly2mask(mask, h, w) for mask in kins_ann_info['masks']
            ]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info['bboxes'], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = (bbox_iou.max(axis=0) > 0.5)

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos['img'])

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f'{info_prefix}_gt_database', filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            if with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + '.png'
                mask_patch_path = abs_filepath + '.mask.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if with_mask:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


class GTDatabaseCreater:
    """Given the raw data, generate the ground truth database. This is the
    parallel version. For serialized version, please refer to
    `create_groundtruth_database`

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask_anno_path (str, optional): Path of the mask_anno.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        with_mask (bool, optional): Whether to use mask.
            Default: False.
        num_worker (int, optional): the number of parallel workers to use.
            Default: 8.
    """

    def __init__(self,
                 dataset_class_name,
                 data_path,
                 info_prefix,
                 info_path=None,
                 mask_anno_path=None,
                 used_classes=None,
                 database_save_path=None,
                 db_info_save_path=None,
                 relative_path=True,
                 add_rgb=False,
                 lidar_only=False,
                 bev_only=False,
                 coors_range=None,
                 with_mask=False,
                 num_worker=8) -> None:
        self.dataset_class_name = dataset_class_name
        self.data_path = data_path
        self.info_prefix = info_prefix
        self.info_path = info_path
        self.mask_anno_path = mask_anno_path
        self.used_classes = used_classes
        self.database_save_path = database_save_path
        self.db_info_save_path = db_info_save_path
        self.relative_path = relative_path
        self.add_rgb = add_rgb
        self.lidar_only = lidar_only
        self.bev_only = bev_only
        self.coors_range = coors_range
        self.with_mask = with_mask
        self.num_worker = num_worker
        self.pipeline = None

    def create_single(self, input_dict):
        group_counter = 0
        single_db_infos = dict()
        example = self.pipeline(input_dict)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        names = [
            self.dataset.metainfo['classes'][i] for i in annos['gt_labels_3d']
        ]
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        if self.with_mask:
            # prepare masks
            gt_boxes = annos['gt_bboxes']
            img_path = osp.split(example['img_info']['filename'])[-1]
            if img_path not in self.file2id.keys():
                print(f'skip image {img_path} for empty mask')
                return single_db_infos
            img_id = self.file2id[img_path]
            kins_annIds = self.coco.getAnnIds(imgIds=img_id)
            kins_raw_info = self.coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos['img_shape'][:2]
            gt_masks = [
                _poly2mask(mask, h, w) for mask in kins_ann_info['masks']
            ]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info['bboxes'], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = (bbox_iou.max(axis=0) > 0.5)

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos['img'])

        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(self.database_save_path, filename)
            rel_filepath = osp.join(f'{self.info_prefix}_gt_database',
                                    filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            if self.with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + '.png'
                mask_patch_path = abs_filepath + '.mask.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (self.used_classes is None) or names[i] in self.used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if self.with_mask:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in single_db_infos:
                    single_db_infos[names[i]].append(db_info)
                else:
                    single_db_infos[names[i]] = [db_info]

        return single_db_infos

    def create(self):
        print_log(
            f'Create GT Database of {self.dataset_class_name}',
            logger='current')
        dataset_cfg = dict(
            type=self.dataset_class_name,
            data_root=self.data_path,
            ann_file=self.info_path)
        if self.dataset_class_name == 'KittiDataset':
            backend_args = None
            dataset_cfg.update(
                test_mode=False,
                data_prefix=dict(
                    pts='training/velodyne_reduced', img='training/image_2'),
                modality=dict(
                    use_lidar=True,
                    use_depth=False,
                    use_lidar_intensity=True,
                    use_camera=self.with_mask,
                ),
                pipeline=[
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=4,
                        use_dim=4,
                        backend_args=backend_args),
                    dict(
                        type='LoadAnnotations3D',
                        with_bbox_3d=True,
                        with_label_3d=True,
                        backend_args=backend_args)
                ])

        elif self.dataset_class_name == 'NuScenesDataset':
            dataset_cfg.update(
                use_valid_flag=True,
                data_prefix=dict(
                    pts='samples/LIDAR_TOP', img='',
                    sweeps='sweeps/LIDAR_TOP'),
                pipeline=[
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=5,
                        use_dim=5),
                    dict(
                        type='LoadPointsFromMultiSweeps',
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True),
                    dict(
                        type='LoadAnnotations3D',
                        with_bbox_3d=True,
                        with_label_3d=True)
                ])

        elif self.dataset_class_name == 'WaymoDataset':
            backend_args = None
            dataset_cfg.update(
                test_mode=False,
                data_prefix=dict(
                    pts='training/velodyne',
                    img='',
                    sweeps='training/velodyne'),
                modality=dict(
                    use_lidar=True,
                    use_depth=False,
                    use_lidar_intensity=True,
                    use_camera=False,
                ),
                pipeline=[
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=6,
                        use_dim=6,
                        backend_args=backend_args),
                    dict(
                        type='LoadAnnotations3D',
                        with_bbox_3d=True,
                        with_label_3d=True,
                        backend_args=backend_args)
                ])

        self.dataset = DATASETS.build(dataset_cfg)
        self.pipeline = self.dataset.pipeline
        if self.database_save_path is None:
            self.database_save_path = osp.join(
                self.data_path, f'{self.info_prefix}_gt_database')
        if self.db_info_save_path is None:
            self.db_info_save_path = osp.join(
                self.data_path, f'{self.info_prefix}_dbinfos_train.pkl')
        mmengine.mkdir_or_exist(self.database_save_path)
        if self.with_mask:
            self.coco = COCO(osp.join(self.data_path, self.mask_anno_path))
            imgIds = self.coco.getImgIds()
            self.file2id = dict()
            for i in imgIds:
                info = self.coco.loadImgs([i])[0]
                self.file2id.update({info['file_name']: i})

        def loop_dataset(i):
            input_dict = self.dataset.get_data_info(i)
            input_dict['box_type_3d'] = self.dataset.box_type_3d
            input_dict['box_mode_3d'] = self.dataset.box_mode_3d
            return input_dict

        if self.num_worker == 0:
            multi_db_infos = mmengine.track_progress(
                self.create_single,
                ((loop_dataset(i)
                  for i in range(len(self.dataset))), len(self.dataset)))
        else:
            multi_db_infos = mmengine.track_parallel_progress(
                self.create_single,
                ((loop_dataset(i)
                  for i in range(len(self.dataset))), len(self.dataset)),
                self.num_worker,
                chunksize=1000)
        print_log('Make global unique group id', logger='current')
        group_counter_offset = 0
        all_db_infos = dict()
        for single_db_infos in track_iter_progress(multi_db_infos):
            group_id = -1
            for name, name_db_infos in single_db_infos.items():
                for db_info in name_db_infos:
                    group_id = max(group_id, db_info['group_id'])
                    db_info['group_id'] += group_counter_offset
                if name not in all_db_infos:
                    all_db_infos[name] = []
                all_db_infos[name].extend(name_db_infos)
            group_counter_offset += (group_id + 1)

        for k, v in all_db_infos.items():
            print_log(f'load {len(v)} {k} database infos', logger='current')

        print_log(f'Saving GT database infos into {self.db_info_save_path}')
        with open(self.db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
