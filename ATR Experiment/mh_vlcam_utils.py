import torch, torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# Conditional Channel Attention Module
class MultiHeadCCAM(nn.Module):
    def __init__(self, channels, r, n_heads):
        super(MultiHeadCCAM, self).__init__()
        self.channels = channels
        self.r = r
        self.n_heads = n_heads
        reduced_dim = max(1, channels // r)
        # Grouped Conv1d acts as independent MLPs for each head
        self.mlp_1 = nn.Conv1d(channels, reduced_dim, kernel_size=1, groups=n_heads, bias=True)
        self.mlp_2 = nn.Conv1d(reduced_dim, channels, kernel_size=1, groups=n_heads, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_film):
        b, c, _, _ = x.size()
        # Global pooling -> [B, C, 1]
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c, 1)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c, 1)
        film_pool = F.adaptive_avg_pool2d(x_film, 1).view(b, c, 1)

        def run_mlp(feat):
            y = self.mlp_1(feat)
            y = self.relu(y)
            y = self.mlp_2(y)
            return y
        
        out = run_mlp(max_pool) + run_mlp(avg_pool) + run_mlp(film_pool)
        scale = F.sigmoid(out).view(b, c, 1, 1)
        return x * scale

# Conditional Spatial Attention Module
class MultiHeadCSAM(nn.Module):
    def __init__(self, n_heads, bias=False):
        super(MultiHeadCSAM, self).__init__()
        self.n_heads = n_heads
        # Input: 3 channels (max, avg, film) per head
        self.conv = nn.Conv2d(
            in_channels=3 * n_heads, 
            out_channels=n_heads, 
            kernel_size=7, padding=3, dilation=1, groups=n_heads, bias=bias
        )

    def forward(self, x, x_film):
        b, c, h, w = x.size()
        # Reshape to [B, Heads, C_per_head, H, W]
        x_reshaped = x.view(b, self.n_heads, c // self.n_heads, h, w)
        x_film_reshaped = x_film.view(b, self.n_heads, c // self.n_heads, h, w)
        
        # Calculate stats [B, Heads, H, W]
        max_pool = torch.max(x_reshaped, 2)[0]
        avg_pool = torch.mean(x_reshaped, 2)
        film_pool = torch.mean(x_film_reshaped, 2)
        
        # Stack for grouped conv: [B, H, 1, H, W] -> [B, H, 3, H, W] -> [B, H*3, H, W]
        combined = torch.cat([max_pool.unsqueeze(2), avg_pool.unsqueeze(2), film_pool.unsqueeze(2)], dim=2)
        combined = combined.view(b, self.n_heads * 3, h, w)
        
        # Spatial Map per head
        spatial_attn = self.conv(combined) # [B, Heads, H, W]
        spatial_attn = F.sigmoid(spatial_attn)
        
        # Expand back to channels
        spatial_attn = spatial_attn.unsqueeze(2) # [B, Heads, 1, H, W]
        output_scale = spatial_attn.expand(-1, -1, c // self.n_heads, -1, -1).reshape(b, c, h, w)
        return x * output_scale

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

class MHVLCAM(nn.Module):
    def __init__(self, channels, r=2, cond_dim=14, n_heads=2, num_groups=8):
        super(MHVLCAM, self).__init__()
        assert channels % n_heads == 0, f"Channels {channels} must be divisible by heads {n_heads}"
        
        self.ccam = MultiHeadCCAM(channels, r, n_heads)
        self.csam = MultiHeadCSAM(n_heads, bias=False)
        self.cam_film = FiLMModulation(channels, cond_dim=cond_dim)
        self.sam_film = FiLMModulation(channels, cond_dim=cond_dim)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def forward(self, x, conditions):
        # Pre-normalization
        out = self.gn1(x)
        
        # Channel Attention
        x_film_cam = self.cam_film(out, conditions)
        out = self.ccam(out, x_film_cam)
        
        # Spatial Attention
        x_film_sam = self.sam_film(out, conditions)
        out = self.csam(out, x_film_sam)
        
        # Peri-normalization
        out = self.gn2(out) 
        
        return out + x


class MHVLCAMTransformLayer(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=14, r=2, n_heads=1):
        super(MHVLCAMTransformLayer, self).__init__()
        # Initial Attention
        self.mhvlcam = MHVLCAM(channels=in_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.mhvlcam1 = MHVLCAM(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)
        self.silu1 = nn.SiLU()
        
        # Block 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.mhvlcam2 = MHVLCAM(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)
        self.silu2 = nn.SiLU()
        
        # Block 3
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.mhvlcam3 = MHVLCAM(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)
        self.silu3 = nn.SiLU()
    
    def forward(self, x, conditions):
        x = self.mhvlcam(x, conditions)
        
        x = self.conv1(x)
        x = self.mhvlcam1(x, conditions)
        x = self.silu1(x)
        
        x = self.conv2(x)
        x = self.mhvlcam2(x, conditions)
        x = self.silu2(x)
        
        x = self.conv3(x)
        x = self.mhvlcam3(x, conditions)
        x = self.silu3(x)
        return x

class MHVLCAMTransformQueries(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=14, r=2, n_heads=1):
        super(MHVLCAMTransformQueries, self).__init__()
        # Initial Attention
        self.mhvlcam = MHVLCAM(channels=in_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.mhvlcam1 = MHVLCAM(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)
        self.silu1 = nn.SiLU()
        
        # Block 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.mhvlcam2 = MHVLCAM(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)
        self.silu2 = nn.SiLU()
        
        # Block 3
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.mhvlcam3 = MHVLCAM(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)
        self.silu3 = nn.SiLU()
        
    def forward(self, x, conditions):
        x = self.mhvlcam(x, conditions)
        
        x = self.conv1(x)
        x = self.mhvlcam1(x, conditions)
        x = self.silu1(x)
        
        x = self.conv2(x)
        x = self.mhvlcam2(x, conditions)
        x = self.silu2(x)
        
        x = self.conv3(x)
        x = self.mhvlcam3(x, conditions)
        x = self.silu3(x)
        return x

# def __main__():
#     # Simple test to verify module functionality
#     b, c, h, w = 1, 200, 256, 224
#     cond_dim = 14
#     n_heads = 2
#     x = torch.randn(b, c, h, w)
#     cond = torch.randn(b, cond_dim)

#     mhvlcam = MHVLCAM(channels=c, r=2, cond_dim=cond_dim, n_heads=n_heads)
#     out = mhvlcam(x, cond)
#     print("MHVLCAM output shape:", out.shape)