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

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention2D, self).__init__()
        # Reduce channel dimension for query and key projections
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        # Value projection keeps the same number of channels
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        # Learnable scaling parameter (initialized to 0 so the network can start by preserving the original signal)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape [B, C, H, W]
        Returns:
            out: Tensor of same shape as x after applying self-attention
        """
        B, C, H, W = x.size()
        # Compute query, key, and value projections and flatten the spatial dimensions
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # shape: [B, H*W, C//8]
        key = self.key_conv(x).view(B, -1, H * W)                         # shape: [B, C//8, H*W]
        energy = torch.bmm(query, key)                                      # shape: [B, H*W, H*W]
        attention = self.softmax(energy)                                    # shape: [B, H*W, H*W]
        value = self.value_conv(x).view(B, -1, H * W)                        # shape: [B, C, H*W]
        
        # Compute weighted sum of values and reshape back to [B, C, H, W]
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)
        
        # Apply residual connection with scaling parameter
        out = self.gamma * out + x
        return out


class ConditionCrossAttention2D(nn.Module):
    def __init__(self, feat_channels, cond_channels, key_channels=None):
        """
        Args:
            feat_channels (int): Number of channels in the fused features (keys/values).
            cond_channels (int): Number of channels in the condition tensor (query).
            key_channels (int, optional): Reduced channel dimension for the query/key projections.
                                          Defaults to feat_channels // 8.
        """
        super(ConditionCrossAttention2D, self).__init__()
        if key_channels is None:
            key_channels = feat_channels // 8

        # Project conditions into query space.
        self.query_conv = nn.Conv2d(cond_channels, key_channels, kernel_size=1)
        # Project fused features into key space.
        self.key_conv = nn.Conv2d(feat_channels, key_channels, kernel_size=1)
        # Project fused features into value space.
        self.value_conv = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling parameter

    def forward(self, features, conditions):
        """
        Args:
            features: Tensor of shape [B, feat_channels, H, W] (keys/values)
            conditions: Tensor of shape [B, cond_channels, H, W] (query)
        Returns:
            Tensor of shape [B, feat_channels, H, W] with cross-attended features.
        """
        B, C, H, W = features.size()
        # Compute query from conditions.
        query = self.query_conv(conditions).view(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, key_channels]
        # Compute key from features.
        key = self.key_conv(features).view(B, -1, H * W)                         # [B, key_channels, H*W]
        # Compute attention map.
        energy = torch.bmm(query, key)                                           # [B, H*W, H*W]
        attention = self.softmax(energy)
        # Compute value from features.
        value = self.value_conv(features).view(B, -1, H * W)                     # [B, feat_channels, H*W]
        # Apply attention to value.
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)      # [B, feat_channels, H, W]
        # Residual connection.
        out = self.gamma * out + features
        return out

class ConcatHeadWithCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels):
        """
        Args:
            in_channels (int): Number of channels of the fused features (e.g., LiDAR + RGB).
            out_channels (int): Number of output channels after the head.
            cond_channels (int): Number of channels in the condition tensor.
        """
        super(ConcatHeadWithCrossAttention, self).__init__()
        # First convolution to reduce dimensions (or blend features)
        self.conv1 = build_conv_layer(dict(type='Conv2d'), in_channels, out_channels, kernel_size=1)
        # Insert cross attention here.
        self.cross_attn = ConditionCrossAttention2D(
            feat_channels=out_channels, 
            cond_channels=cond_channels
        )
        # self.cross_attn2 = ConditionCrossAttention2D(
        #     feat_channels=out_channels, 
        #     cond_channels=cond_channels
        # )
        # self.cross_attn3 = ConditionCrossAttention2D(
        #     feat_channels=out_channels, 
        #     cond_channels=cond_channels
        # )
        self.relu = nn.ReLU(inplace=True)
        # Additional processing layers.
        self.conv2 = build_conv_layer(dict(type='Conv2d'), out_channels, out_channels, kernel_size=1)
        self.conv3 = build_conv_layer(dict(type='Conv2d'), out_channels, out_channels, kernel_size=1)

    def forward(self, fused_features, conditions):
        """
        Args:
            fused_features: Tensor of shape [B, in_channels, H, W]
            conditions: Tensor of shape [B, cond_channels, H, W]
        Returns:
            Processed feature tensor.
        """
        x = self.conv1(fused_features)
        x = self.cross_attn(x, conditions)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.cross_attn2(x, conditions)
        x = self.relu(x)
        x = self.conv3(x)
        # x = self.cross_attn3(x, conditions)
        x = self.relu(x)
        return x

class ConcatHeadWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of channels of the fused features (e.g., LiDAR + RGB).
            out_channels (int): Number of output channels after the head.
        """
        super(ConcatHeadWithCBAM, self).__init__()
        self.cbam = CBAM(in_channels, r=2)
        # First convolution to reduce dimensions (or blend features)
        self.conv1 = build_conv_layer(dict(type='Conv2d'), in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        # Additional processing layers.
        self.conv2 = build_conv_layer(dict(type='Conv2d'), out_channels, out_channels, kernel_size=1)
        self.conv3 = build_conv_layer(dict(type='Conv2d'), out_channels, out_channels, kernel_size=1)

    def forward(self, fused_features):
        """
        Args:
            fused_features: Tensor of shape [B, in_channels, H, W]
        Returns:
            Processed feature tensor.
        """
        x = self.cbam(fused_features)
        # print(x.shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x

class LearnableAlignVLM(nn.Module):
    def __init__(self, lidar_dim=384, img_dim=256, embed_dim=256, out_dim=192):
        super().__init__()
        self.q_embed = nn.Linear(lidar_dim, embed_dim)
        self.k_embed = nn.Linear(img_dim, embed_dim)
        self.v_embed = nn.Linear(img_dim, embed_dim)
        self.out_fc = nn.Linear(embed_dim, out_dim)
        self.fusion_fc = nn.Linear(lidar_dim + out_dim, lidar_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, lidar_feat, img_feat):
        """
        Args:
            lidar_feat: [B, C_lidar, H, W] tensor
            img_feat: [B, C_img, H, W] tensor (must be aligned in BEV grid)
        Returns:
            Fused LiDAR features: [B, C_lidar, H, W]
        """
        B, C_lidar, H, W = lidar_feat.shape
        C_img = img_feat.shape[1]

        # Flatten spatial dimensions
        lidar_flat = lidar_feat.permute(0, 2, 3, 1).reshape(B, -1, C_lidar)  # [B, HW, C_lidar]
        img_flat = img_feat.permute(0, 2, 3, 1).reshape(B, -1, C_img)        # [B, HW, C_img]

        # Project Q, K, V
        Q = self.q_embed(lidar_flat)     # [B, HW, D]
        K = self.k_embed(img_flat)       # [B, HW, D]
        V = self.v_embed(img_flat)       # [B, HW, D]

        # Compute dot-product attention for each position (1:1)
        affinity = torch.sum(Q * K, dim=-1, keepdim=True) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32, device=Q.device))  # [B, HW, 1]
        weights = F.softmax(affinity, dim=1)  # [B, HW, 1]
        weights = self.dropout(weights)

        # Weighted sum (since 1:1, this is just attention modulated V)
        v_attn = weights * V  # [B, HW, D]

        # Project and fuse
        v_attn_proj = self.out_fc(v_attn)                    # [B, HW, fused_channels]
        fused = torch.cat([lidar_flat, v_attn_proj], dim=-1) # [B, HW, C_lidar + fused]
        fused = self.fusion_fc(fused)                        # [B, HW, C_lidar]

        # Reshape back to BEV grid
        fused = fused.view(B, H, W, C_lidar).permute(0, 3, 1, 2)  # [B, C_lidar, H, W]
        return fused

@MODELS.register_module()
class LearnableAlignVLMEnsemble(Base3DDetector):
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
        
        # Load the conditions
        self.train_conditions_dict = {}
        self.train_jsonl_file = train_jsonl_file
        self.val_conditions_dict = {}
        self.val_jsonl_file = val_jsonl_file
        self.test_conditions_dict = {}
        self.test_jsonl_file = test_jsonl_file
        self.n_conditions = 10 # 3
        
        self.load_conditions()
        
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
            nn.ReLU(inplace=True),
            build_conv_layer(dict(type='Conv2d'), 256, 256, 1,),
            nn.ReLU(inplace=True),
            build_conv_layer(dict(type='Conv2d'), 256, 256, 1,),
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
        
        # self.concat_head = ConcatHeadWithCBAM(640, 384) # 384 (LiDAR) + 256 (RGB) and 3 (Conditions)
        self.learnable_align = LearnableAlignVLM(lidar_dim=384, img_dim=256+self.n_conditions, embed_dim=256+self.n_conditions, out_dim=192)
    
    def load_conditions(self):
        if self.n_conditions == 10:
            indices_to_sample = np.array([31, 16, 9, 40, 27, 6, 3, 23, 10, 1])
        elif self.n_conditions == 3:
            indices_to_sample = np.array([1, 2, 3])
        
        with open(self.train_jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_path = data['image_path']
                conditions = data['conditions']
                conditions = np.array(conditions)[indices_to_sample - 1]
                self.train_conditions_dict[image_path] = conditions
        
        with open(self.val_jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_path = data['image_path']
                conditions = data['conditions']
                conditions = np.array(conditions)[indices_to_sample - 1]
                self.val_conditions_dict[image_path] = conditions
        
        with open(self.test_jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_path = data['image_path']
                conditions = data['conditions']
                conditions = np.array(conditions)[indices_to_sample - 1]
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
        # x = torch.cat([lidar_x, rgb_x, conditions_tensor], dim=1)
        # x = self.concat_head(x)
        # x = torch.cat([lidar_x, rgb_x], dim=1)
        # x = self.concat_head(x)
        # x = self.concat_head(x, conditions_tensor)
        # x = lidar_x
        
        rgb_x = torch.cat([rgb_x, conditions_tensor], dim=1)
        
        x = self.learnable_align(lidar_x, rgb_x)
        
        # Combine the features
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x