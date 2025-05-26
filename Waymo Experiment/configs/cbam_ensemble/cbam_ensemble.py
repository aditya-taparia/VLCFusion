import mmengine
from mmdet3d.apis import init_model
from copy import deepcopy
import numpy as np

from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type

import os
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from mmengine.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.models.detectors.single_stage import SingleStage3DDetectorCustom, SingleStage3DDetector
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList


from mmcv.cnn import build_conv_layer, build_norm_layer

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
)
import json

# Channel Attention Module
class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output

# Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 

# Convolutional Block Attention Module
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x

class ConcatHeadWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of channels of the fused features (e.g., LiDAR + RGB).
            out_channels (int): Number of output channels after the head.
        """
        super(ConcatHeadWithCBAM, self).__init__()
        self.cbam = CBAM(in_channels, r=2)
        self.norm = nn.BatchNorm2d(in_channels)
        # First convolution to reduce dimensions (or blend features)
        self.conv1 = build_conv_layer(dict(type='Conv2d'), in_channels, out_channels, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Additional processing layers.
        self.conv2 = build_conv_layer(dict(type='Conv2d'), out_channels, out_channels, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = build_conv_layer(dict(type='Conv2d'), out_channels, out_channels, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(out_channels)

    def forward(self, fused_features):
        """
        Args:
            fused_features: Tensor of shape [B, in_channels, H, W]
        Returns:
            Processed feature tensor.
        """
        x = self.cbam(fused_features)
        x = self.norm(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x

@MODELS.register_module()
class CBAMEnsemble(Base3DDetector):
    """SingleStage3DDetector.

    This class serves as a base class for single-stage 3D detectors which
    directly and densely predict 3D bounding boxes on the output features
    of the backbone+neck.

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """
    def __init__(self,
                lidar_model_path: str,
                lidar_model_cfg_path: str,
                rgb_model_path: str,
                train_jsonl_file:str = './vlm_conditions/night_day_training.jsonl',
                val_jsonl_file:str = './vlm_conditions/night_day_validation.jsonl',
                test_jsonl_file:str = './vlm_conditions/night_day_test.jsonl',
                backbone: ConfigType = None,
                neck: OptConfigType = None,
                bbox_head: OptConfigType = None,
                train_cfg: OptConfigType = None,
                test_cfg: OptConfigType = None,
                data_preprocessor: OptConfigType = None,
                init_cfg: OptMultiConfig = None,
                ) -> None:
        
        # Load the LiDAR model
        lidar_model = init_model(lidar_model_cfg_path, lidar_model_path)
        
        super().__init__(
            data_preprocessor = lidar_model.data_preprocessor, init_cfg=init_cfg,
        )
        
        # Load the RGB model
        IMAGE_SIZE = 480
        categories_to_tgttype = {
            0: 'pedestrian',
            1: 'cyclist', 
            2: 'vehicle',
        }
        id2label = categories_to_tgttype
        label2id = {v: k for k, v in id2label.items()}
        config = AutoConfig.from_pretrained(
            rgb_model_path,
            label2id=label2id,
            id2label=id2label,
        )
        rgb_model = AutoModelForObjectDetection.from_pretrained(
            rgb_model_path, 
            config=config,
            ignore_mismatched_sizes=True,
        )
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # rgb_model = rgb_model.to(device)
        
        # print(train_cfg)
        
        # Initialize the models
        self.rgb_backbone = rgb_model.model.backbone.to(self.device)
        self.rgb_input_projection = rgb_model.model.input_projection.to(self.device)
        
        for module in [
            self.rgb_backbone,
            self.rgb_input_projection,
        ]:
            for param in module.parameters():
                param.requires_grad = False
        
        self.rgb_image_processor = AutoImageProcessor.from_pretrained(
            rgb_model_path,
            do_resize=True,
            size={"max_height": IMAGE_SIZE, "max_width": IMAGE_SIZE},
            do_pad=True,
            pad_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
            use_fast=True,
        )
        
        # lidar_model = init_model(lidar_model_cfg_path, lidar_model_path)
        self.lidar_data_preprocessor = lidar_model.data_preprocessor
        self.lidar_backbone = lidar_model.backbone
        self.lidar_voxel_encoder = lidar_model.voxel_encoder
        self.lidar_middle_encoder = lidar_model.middle_encoder
        
        for module in [
            self.lidar_data_preprocessor,
            self.lidar_backbone,
            self.lidar_voxel_encoder,
            self.lidar_middle_encoder,
        ]:
            for param in module.parameters():
                param.requires_grad = False
        
        # Required params
        cfg = lidar_model.cfg
        self.backbone = lidar_model.backbone
        self.neck = lidar_model.neck
        
        # self.bbox_head = lidar_model.bbox_head
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        conv_transpose_layer = build_conv_layer(
            dict(type='ConvTranspose2d', bias=False),
            in_channels=256,
            out_channels=256,
            kernel_size=(6, 16),
            stride=(11, 16),
            padding=0,
        )
        
        self.rgb_transform_layer = nn.Sequential(
            conv_transpose_layer,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            build_conv_layer(dict(type='Conv2d'), 256, 256, 1,),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            build_conv_layer(dict(type='Conv2d'), 256, 256, 1,),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # self.concat_head = nn.Sequential(
        #     build_conv_layer(dict(type='Conv2d'), 643, 384, 1,), # 384 (LiDAR) + 256 (RGB) + 3 (Conditions)
        #     nn.ReLU(inplace=True),
        #     SelfAttention2D(384),
        #     build_conv_layer(dict(type='Conv2d'), 384, 384, 1,),
        #     nn.ReLU(inplace=True),
        #     # SelfAttention2D(384),
        #     build_conv_layer(dict(type='Conv2d'), 384, 384, 1,),
        #     nn.ReLU(inplace=True),
        #     # SelfAttention2D(384),
        # )
        self.concat_head = ConcatHeadWithCBAM(640, 384) # 384 (LiDAR) + 256 (RGB) and 3 (Conditions)
        
        self.train_conditions_dict = {}
        self.train_jsonl_file = train_jsonl_file
        self.val_conditions_dict = {}
        self.val_jsonl_file = val_jsonl_file
        self.test_conditions_dict = {}
        self.test_jsonl_file = test_jsonl_file
        
        self.load_conditions()
    
    def load_conditions(self):
        with open(self.train_jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_path = data['image_path']
                conditions = data['conditions']
                self.train_conditions_dict[image_path] = conditions
        
        with open(self.val_jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_path = data['image_path']
                conditions = data['conditions']
                self.val_conditions_dict[image_path] = conditions
        
        with open(self.test_jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_path = data['image_path']
                conditions = data['conditions']
                self.test_conditions_dict[image_path] = conditions
        
        print(f'Loaded {len(self.train_conditions_dict)} conditions.')
        print(f'Loaded {len(self.val_conditions_dict)} conditions.')
        print(f'Loaded {len(self.test_conditions_dict)} conditions.')
    
    def get_conditions(self, img_paths:str):
        conditions = []
        img_paths = img_paths[0]
        for img_path in img_paths:
            
            # Check based on file name. File name starting with 0 is train, 1 is val, 2 is test
            file_name = os.path.basename(img_path)
            if file_name[0] == '0':
                conditions.append(self.train_conditions_dict[img_path])
            elif file_name[0] == '1':
                conditions.append(self.val_conditions_dict[img_path])
            elif file_name[0] == '2':
                conditions.append(self.test_conditions_dict[img_path])
            
            # conditions.append(self.conditions_dict[img_path])
        return conditions
    
    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs_dict)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses
    
    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        x = self.extract_feat(batch_inputs_dict)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions
    
    def _forward(self,
                 batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs_dict)
        results = self.bbox_head.forward(x)
        return results

    def batched_tensor_to_pil(self, batched_tensor: torch.Tensor):
        to_pil = ToPILImage()
        pil_images = []
        for img_tensor in batched_tensor:
            img = to_pil(img_tensor.cpu())
            pil_images.append(img)
        return pil_images
    
    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features from points."""
        
        # print(batch_inputs_dict)
        
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.lidar_voxel_encoder(voxel_dict['voxels'],
                                        voxel_dict['num_points'],
                                        voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        lidar_x = self.lidar_middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        
        # Random tensor for RGB
        # print(batch_inputs_dict['imgs'].shape)
        batched_tensor_imgs = batch_inputs_dict['imgs']
        
        
        # # Make RGB inputs to 0 for every batch image
        # batch_size, num_channels, height, width = batched_tensor_imgs.shape
        # empty_image = torch.zeros((batch_size, num_channels, height, width), dtype=torch.uint8)
        # batched_tensor_imgs = empty_image.to(lidar_x.device)
        
        
        batched_pil_imgs = self.batched_tensor_to_pil(batched_tensor_imgs)
        
        rgb_inputs = self.rgb_image_processor(batched_pil_imgs, return_tensors="pt").to(lidar_x.device)
        features, object_queries_list = self.rgb_backbone(**rgb_inputs)
        feature_map, mask = features[-1]
        projected_feature_map = self.rgb_input_projection(feature_map) # (batch_size, 256, 15, 15)
        
        # rgb_x = torch.randn(batch_size, 256, 15, 15).to(lidar_x.device)
        rgb_x = projected_feature_map
        # Transform the RGB features
        rgb_x = self.rgb_transform_layer(rgb_x)
        
        # Get conditions
        batched_conditions = self.get_conditions(batch_inputs_dict['img_paths'])
        conditions_tensor_list = []
        for cond in batched_conditions:
            cond_tensor = torch.tensor([1.0 if c else 0.0 for c in cond], dtype=torch.float, device=self.device)
            H, W = lidar_x.size(2), lidar_x.size(3)
            cond_tensor = cond_tensor.view(-1, 1, 1).expand(-1, H, W)
            conditions_tensor_list.append(cond_tensor)
        
        conditions_tensor = torch.stack(conditions_tensor_list, dim=0)
        
        # Concatenate the features
        x = torch.cat([lidar_x, rgb_x], dim=1)
        x = self.concat_head(x)
        
        # Combine the features
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x