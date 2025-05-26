import torch, torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

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

class FiLMModulation(nn.Module):
    def __init__(self, in_channels, cond_dim):
        super(FiLMModulation, self).__init__()
        self.linear = nn.Linear(cond_dim, in_channels * 2)

    def forward(self, x, cond):
        film_params = self.linear(cond)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)

        return (1 + gamma) * x + beta

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


class FusionSSDTransformLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionSSDTransformLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class FusionSSDTransformQueries(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionSSDTransformQueries, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class FusionSSDSelfAttentionTransformLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionSSDSelfAttentionTransformLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        
        self.self_attention = SelfAttention2D(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.self_attention(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class FusionSSDSelfAttentionTransformQueries(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionSSDSelfAttentionTransformQueries, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        
        self.self_attention = SelfAttention2D(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.self_attention(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class FusionSSDFiLMTransformLayer(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=14):
        super(FusionSSDFiLMTransformLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        
        self.film = FiLMModulation(out_channels, cond_dim)

    def forward(self, x, conditions):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.film(x, conditions)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class FusionSSDFiLMTransformQueries(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=14):
        super(FusionSSDFiLMTransformQueries, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        
        self.film = FiLMModulation(out_channels, cond_dim)

    def forward(self, x, conditions):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.film(x, conditions)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class CBAMTransformLayer(nn.Module):
    def __init__(self, in_channels, out_channels, r=2):
        super(CBAMTransformLayer, self).__init__()
        self.cbam = CBAM(channels=in_channels, r=r)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.cbam(x)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class CBAMTransformQueries(nn.Module):
    def __init__(self, in_channels, out_channels, r=2):
        super(CBAMTransformQueries, self).__init__()
        self.cbam = CBAM(channels=in_channels, r=r)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.cbam(x)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class CBAMFiLMTransformLayer(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=14, r=2):
        super(CBAMFiLMTransformLayer, self).__init__()
        self.cbam = CBAM(channels=in_channels, r=r)
        self.bn = nn.BatchNorm2d(in_channels)
        self.film = FiLMModulation(in_channels, cond_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
    
    def forward(self, x, conditions):
        x = self.cbam(x)
        x = self.bn(x)
        x = self.film(x, conditions)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class CBAMFiLMTransformQueries(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=14, r=2):
        super(CBAMFiLMTransformQueries, self).__init__()
        self.cbam = CBAM(channels=in_channels, r=r)
        self.bn = nn.BatchNorm2d(in_channels)
        self.film = FiLMModulation(in_channels, cond_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        
    def forward(self, x, conditions):
        x = self.cbam(x)
        x = self.bn(x)
        x = self.film(x, conditions)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class LearnableAlign(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LearnableAlign, self).__init__()
        
        self.modality_1_dim = in_channels // 2
        self.modality_2_dim = in_channels // 2
        self.emb_dim = in_channels // 2
        self.out_dim = out_channels
        self.q_embed = nn.Linear(self.modality_1_dim, self.emb_dim)
        self.k_embed = nn.Linear(self.modality_2_dim, self.emb_dim)
        self.v_embed = nn.Linear(self.modality_2_dim, self.emb_dim)
        self.out_fc = nn.Linear(self.emb_dim, self.out_dim)
        self.fusion_fc = nn.Linear(self.modality_1_dim + self.emb_dim, self.out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        modality_1 = x[:, :self.modality_1_dim, :, :]
        modality_2 = x[:, self.modality_1_dim:, :, :]
        B, C, H, W = modality_1.shape
        C_img = modality_2.shape[1]
        
        modality_1_flat = modality_1.permute(0, 2, 3, 1).reshape(B, -1, self.modality_1_dim)
        modality_2_flat = modality_2.permute(0, 2, 3, 1).reshape(B, -1, self.modality_2_dim)
        
        Q = self.q_embed(modality_1_flat)
        K = self.k_embed(modality_2_flat)
        V = self.v_embed(modality_2_flat)
        
        affinity = torch.sum(Q * K, dim=-1, keepdim=True) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32, device=Q.device))
        weights = F.softmax(affinity, dim=1)
        weights = self.dropout(weights)
        
        v_attn = weights * V 
        
        v_attn_proj = self.out_fc(v_attn)
        fused = torch.cat([modality_1_flat, v_attn_proj], dim=-1)
        fused = self.fusion_fc(fused)

        fused = fused.view(B, H, W, self.modality_1_dim).permute(0, 3, 1, 2)
        return fused

class LearnableAlignTransformLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LearnableAlignTransformLayer, self).__init__()
        self.la = LearnableAlign(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.la(x)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

class LearnableAlignTransformQueries(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LearnableAlignTransformQueries, self).__init__()
        self.la = LearnableAlign(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.la(x)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x