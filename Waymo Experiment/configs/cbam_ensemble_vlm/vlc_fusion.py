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
import torchvision.transforms.functional as TF

from mmengine.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
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


# =============================================================================
# Building blocks from cross_cbam_dit_v11_utils
# =============================================================================

def _num_groups(channels: int, desired: int = 8) -> int:
    for g in range(min(desired, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, r: int = 2):
        super().__init__()
        mid = max(1, channels // max(1, r))
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=True),
            nn.SiLU(),
            nn.Linear(mid, channels, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return w * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, stride=1, dilation=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return w * x


class CBAM(nn.Module):
    def __init__(self, channels: int, r: int = 2):
        super().__init__()
        self.cam = ChannelAttention(channels, r)
        self.sam = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sam(self.cam(x))


class FiLMModulation(nn.Module):
    def __init__(self, in_channels, cond_dim, is_extra_scale: bool = True):
        super(FiLMModulation, self).__init__()
        self.is_extra_scale = is_extra_scale
        if is_extra_scale:
            self.linear = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, in_channels * 3, bias=True),
            )
        else:
            self.linear = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, in_channels * 2, bias=True),
            )

    def forward(self, x, cond):
        film_params = self.linear(cond)
        if self.is_extra_scale:
            gamma, beta, alpha = film_params.chunk(3, dim=1)
        else:
            gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)
        if self.is_extra_scale:
            alpha = alpha.view(-1, x.size(1), 1, 1)
            return gamma, beta, alpha
        else:
            return gamma, beta


class MultiHeadFusionBottleneck(nn.Module):
    """SKNet / DynamicConv style: multiple fusion heads, input-dependent weights."""
    def __init__(self, in_channels: int, out_channels: int, hidden_ratio: int = 4):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
        ])
        hidden = max(16, in_channels // hidden_ratio)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, len(self.branches), bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [b(x) for b in self.branches]
        y = torch.stack(branch_outs, dim=1)
        alpha = torch.softmax(self.router(x), dim=-1)
        y = (y * alpha[:, :, None, None, None]).sum(dim=1)
        return y


class VLCBlock(nn.Module):
    """
    GroupNorm -> FiLM(cond) -> CBAM -> residual,
    followed by a conditioned FFN block.
    """
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        r: int = 2,
        if_ffn: bool = True,
        ffn_factor: int = 4,
        ffn_kernel_size: int = 3,
        num_groups: int = 8,
    ):
        super().__init__()
        ng = _num_groups(channels, num_groups)
        self.norm = nn.GroupNorm(ng, channels, affine=False)
        self.film = FiLMModulation(channels, cond_dim)
        self.cbam = CBAM(channels, r=r)

        self.if_ffn = if_ffn
        if if_ffn:
            self.norm_ffn = nn.GroupNorm(ng, channels, affine=False)
            self.film_ffn = FiLMModulation(channels, cond_dim, is_extra_scale=False)
            self.ffn = nn.Sequential(
                nn.Conv2d(channels, channels * ffn_factor, kernel_size=ffn_kernel_size,
                          padding=ffn_kernel_size // 2, bias=False),
                nn.SiLU(),
                nn.Conv2d(channels * ffn_factor, channels, kernel_size=ffn_kernel_size,
                          padding=ffn_kernel_size // 2, bias=False),
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        g1, b1, a1 = self.film(h, cond)
        h1 = (1 + g1) * h + b1
        out1 = a1 * self.cbam(h1) + x

        if self.if_ffn:
            h2 = self.norm_ffn(out1)
            g2, b2 = self.film_ffn(h2, cond)
            h3 = (1 + g2) * h2 + b2
            out2 = self.ffn(h3) + out1
            return out2
        else:
            return out1


class CrossCBAMDiTFusionV11(nn.Module):
    """
    concat(x1, x2) -> MultiHeadFusionBottleneck -> VLCBlock -> VLCBlock -> output
    """
    def __init__(
        self,
        mod1_channels: int,
        mod2_channels: int,
        out_channels: int,
        cond_dim: int,
        r: int = 2,
        num_groups: int = 8,
    ):
        super().__init__()
        self.mod1_channels = mod1_channels
        self.mod2_channels = mod2_channels
        self.out_channels = out_channels

        self.fusion_bottleneck = MultiHeadFusionBottleneck(
            mod1_channels + mod2_channels,
            out_channels,
        )
        self.vlc_fused = VLCBlock(out_channels, cond_dim, r=r, num_groups=num_groups,
                                  if_ffn=True, ffn_factor=1, ffn_kernel_size=3)
        self.vlc_fused2 = VLCBlock(out_channels, cond_dim, r=r, num_groups=num_groups,
                                   if_ffn=True, ffn_factor=1, ffn_kernel_size=3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1 = x1
        s2 = x2
        fused = self.fusion_bottleneck(torch.cat([s1, s2], dim=1))
        fused2 = self.vlc_fused(fused, cond)
        return self.vlc_fused2(fused2, cond)


# =============================================================================
# Main detector
# =============================================================================

@MODELS.register_module()
class VLCFusion(Base3DDetector):
    """
    CrossCBAM-DiT V11 variant of the LiDAR+RGB ensemble detector with
    VLC-block fusion conditioned on VLM environment descriptors.

    Replaces ConcatHeadWithCBAM with CrossCBAMDiTFusionV11:
      LiDAR feats (384ch) + RGB feats (256ch) ->
        MultiHeadFusionBottleneck -> VLCBlock -> VLCBlock -> backbone -> neck -> head
    """
    def __init__(self,
                 lidar_model_path: str,
                 lidar_model_cfg_path: str,
                 rgb_model_path: str,
                 train_jsonl_file: str = './vlm_conditions/night_day_training.jsonl',
                 val_jsonl_file: str = './vlm_conditions/night_day_validation.jsonl',
                 test_jsonl_file: str = './vlm_conditions/night_day_test.jsonl',
                 backbone: ConfigType = None,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 ) -> None:

        lidar_model = init_model(lidar_model_cfg_path, lidar_model_path)

        super().__init__(
            data_preprocessor=lidar_model.data_preprocessor, init_cfg=init_cfg,
        )

        self.train_conditions_dict = {}
        self.train_jsonl_file = train_jsonl_file
        self.val_conditions_dict = {}
        self.val_jsonl_file = val_jsonl_file
        self.test_conditions_dict = {}
        self.test_jsonl_file = test_jsonl_file

        self.n_conditions = 10
        self.load_conditions()

        IMAGE_SIZE = 480
        categories_to_tgttype = {0: 'pedestrian', 1: 'cyclist', 2: 'vehicle'}
        id2label = categories_to_tgttype
        label2id = {v: k for k, v in id2label.items()}
        config = AutoConfig.from_pretrained(
            rgb_model_path, label2id=label2id, id2label=id2label,
        )
        rgb_model = AutoModelForObjectDetection.from_pretrained(
            rgb_model_path, config=config, ignore_mismatched_sizes=True,
        )

        self.rgb_backbone = rgb_model.model.backbone
        self.rgb_input_projection = rgb_model.model.input_projection

        for module in [self.rgb_backbone, self.rgb_input_projection]:
            for param in module.parameters():
                param.requires_grad = False

        self._rgb_image_size = IMAGE_SIZE
        self.register_buffer('_rgb_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_rgb_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.lidar_data_preprocessor = lidar_model.data_preprocessor
        self.lidar_backbone = lidar_model.backbone
        self.lidar_voxel_encoder = lidar_model.voxel_encoder
        self.lidar_middle_encoder = lidar_model.middle_encoder

        for module in [
            self.lidar_data_preprocessor, self.lidar_backbone,
            self.lidar_voxel_encoder, self.lidar_middle_encoder,
        ]:
            for param in module.parameters():
                param.requires_grad = False

        cfg = lidar_model.cfg
        self.backbone = lidar_model.backbone
        self.neck = lidar_model.neck

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # RGB spatial upsampling (256ch DETR features -> 256ch at LiDAR BEV resolution)
        conv_transpose_layer = build_conv_layer(
            dict(type='ConvTranspose2d', bias=False),
            in_channels=256, out_channels=256,
            kernel_size=(6, 16), stride=(11, 16), padding=0,
        )
        self.rgb_transform_layer = nn.Sequential(
            conv_transpose_layer,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            build_conv_layer(dict(type='Conv2d'), 256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            build_conv_layer(dict(type='Conv2d'), 256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # CrossCBAM-DiT V11 fusion: LiDAR 384ch + RGB 256ch -> 384ch
        self.fusion_head = CrossCBAMDiTFusionV11(
            mod1_channels=384,
            mod2_channels=256,
            out_channels=384,
            cond_dim=self.n_conditions,
            r=2,
            num_groups=8,
        )

    # -----------------------------------------------------------------
    # Conditions
    # -----------------------------------------------------------------

    def load_conditions(self):
        if self.n_conditions == 10:
            indices_to_sample = np.array([31, 16, 9, 40, 27, 6, 3, 23, 10, 1])
        elif self.n_conditions == 3:
            indices_to_sample = np.array([1, 2, 3])
        elif self.n_conditions == 15:
            indices_to_sample = np.array([16, 31, 6, 40, 27, 23, 1, 38, 10, 3, 9, 20, 22, 37, 8])
        elif self.n_conditions == 20:
            indices_to_sample = np.array([16, 31, 6, 40, 27, 23, 1, 38, 10, 3, 9, 20, 22, 37, 8, 30, 15, 26, 25, 18])
        elif self.n_conditions == 30:
            indices_to_sample = np.array([16, 31, 6, 40, 27, 23, 1, 38, 10, 3, 9, 20, 22, 37, 8, 30, 15, 26, 25, 18, 21, 17, 35, 2, 32, 5, 19, 24, 14, 13])
        else:
            raise ValueError(f'Invalid number of conditions: {self.n_conditions}')

        for jsonl_path, target_dict in [
            (self.train_jsonl_file, self.train_conditions_dict),
            (self.val_jsonl_file,   self.val_conditions_dict),
            (self.test_jsonl_file,  self.test_conditions_dict),
        ]:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    image_path = os.path.basename(data['image_path'])
                    conditions = np.array(data['conditions'])[indices_to_sample - 1]
                    target_dict[image_path] = conditions

        print(f'Loaded {len(self.train_conditions_dict)} conditions.')
        print(f'Loaded {len(self.val_conditions_dict)} conditions.')
        print(f'Loaded {len(self.test_conditions_dict)} conditions.')

    def get_conditions(self, img_paths: str):
        conditions = []
        img_paths = img_paths[0]
        for img_path in img_paths:
            file_name = os.path.basename(img_path)
            if file_name[0] == '0':
                conditions.append(self.train_conditions_dict[file_name])
            elif file_name[0] == '1':
                conditions.append(self.val_conditions_dict[file_name])
            elif file_name[0] == '2':
                conditions.append(self.test_conditions_dict[file_name])
        return conditions

    # -----------------------------------------------------------------
    # Forward paths
    # -----------------------------------------------------------------

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples, results_list)
        return predictions

    def _forward(self, batch_inputs_dict: dict,
                 data_samples: OptSampleList = None,
                 **kwargs) -> Tuple[List[torch.Tensor]]:
        x = self.extract_feat(batch_inputs_dict, data_samples)
        results = self.bbox_head.forward(x)
        return results

    # -----------------------------------------------------------------
    # RGB preprocessing (GPU-side, matching DetrImageProcessorFast)
    # -----------------------------------------------------------------

    def _preprocess_rgb_gpu(self, imgs: torch.Tensor) -> dict:
        """Resize, rescale, normalize, and pad images entirely on GPU.

        Matches the DetrImageProcessorFast pipeline:
          1. BGR -> RGB
          2. Resize on 0-255 range (aspect-ratio-preserving, fit within SxS)
          3. Fused rescale (÷255) + ImageNet normalize
          4. Pad to SxS

        Input:  imgs [B, C, H, W] as loaded by mmcv (BGR, float, 0-255 range).
        Output: dict with 'pixel_values' [B, 3, S, S] and 'pixel_mask' [B, S, S].
        """
        S = self._rgb_image_size
        x = imgs[:, [2, 1, 0], :, :]

        B, C, H, W = x.shape
        scale = min(S / H, S / W)
        new_h, new_w = int(H * scale), int(W * scale)
        x = TF.resize(x, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR, antialias=False)

        x = x / 255.0
        x = (x - self._rgb_mean) / self._rgb_std

        pad_h, pad_w = S - new_h, S - new_w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

        mask = torch.ones(B, S, S, device=x.device, dtype=torch.int64)
        if pad_h > 0:
            mask[:, new_h:, :] = 0
        if pad_w > 0:
            mask[:, :, new_w:] = 0
        return {'pixel_values': x, 'pixel_mask': mask}

    # -----------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------

    def extract_feat(self, batch_inputs_dict: dict, batch_data_samples=None) -> Tuple[Tensor]:
        voxel_dict = batch_inputs_dict['voxels']

        with torch.no_grad():
            voxel_features = self.lidar_voxel_encoder(
                voxel_dict['voxels'], voxel_dict['num_points'], voxel_dict['coors'])
            batch_size = voxel_dict['coors'][-1, 0].item() + 1
            lidar_x = self.lidar_middle_encoder(
                voxel_features, voxel_dict['coors'], batch_size)

        batched_tensor_imgs = batch_inputs_dict['imgs']
        with torch.no_grad():
            rgb_inputs = self._preprocess_rgb_gpu(batched_tensor_imgs)
            features, _ = self.rgb_backbone(**rgb_inputs)
            feature_map, _ = features[-1]
            projected_feature_map = self.rgb_input_projection(feature_map)

        rgb_x = self.rgb_transform_layer(projected_feature_map)

        img_paths = [ds.metainfo['img_path'] for ds in batch_data_samples] if batch_data_samples else []
        batched_conditions = self.get_conditions([img_paths])
        cond_vec = torch.as_tensor(
            np.stack(batched_conditions, axis=0), dtype=torch.float, device=lidar_x.device)

        # CrossCBAM-DiT V11 fusion (replaces concat + ConcatHeadWithCBAM)
        x = self.fusion_head(lidar_x, rgb_x, cond_vec)

        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x
