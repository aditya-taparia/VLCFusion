import torch, torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


class AdaGN(nn.Module):
    """Adaptive GroupNorm modulation with zero-initialization.
    
    A single shared MLP that takes a condition vector and predicts all
    modulation parameters (scale, shift, gate) for the block.
    Zero-init ensures the block starts as identity.
    """
    def __init__(self, cond_dim, num_params):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, num_params)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond):
        # cond: [B, cond_dim] -> [B, num_params, 1, 1]
        return self.linear(self.silu(cond)).unsqueeze(-1).unsqueeze(-1)


class SwiGLUConv2d(nn.Module):
    """SwiGLU-style gated 1x1 convolution.
    Replaces Conv2d + SiLU with: SiLU(W1*x) * (W2*x).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.w1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.w2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)


class CAM(nn.Module):
    """Channel Attention via grouped SE-style squeeze-excitation."""
    def __init__(self, channels, r, n_heads=1):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.n_heads = n_heads
        reduced_dim = max(1, channels // r)
        self.mlp_1 = nn.Conv1d(channels, reduced_dim, kernel_size=1, groups=n_heads, bias=True)
        self.mlp_2 = nn.Conv1d(reduced_dim, channels, kernel_size=1, groups=n_heads, bias=True)
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c, 1)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c, 1)

        def run_mlp(feat):
            return self.mlp_2(self.silu(self.mlp_1(feat)))

        out = run_mlp(max_pool) + run_mlp(avg_pool)
        return x * F.sigmoid(out).view(b, c, 1, 1)

# Spatial Attention Module
class SAM(nn.Module):
    """Spatial Attention via per-head max/avg pooling and grouped conv."""
    def __init__(self, n_heads=1, bias=False):
        super(SAM, self).__init__()
        self.n_heads = n_heads
        self.conv = nn.Conv2d(
            in_channels=2 * n_heads,
            out_channels=n_heads,
            kernel_size=7, padding=3, dilation=1, groups=n_heads, bias=bias
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_reshaped = x.view(b, self.n_heads, c // self.n_heads, h, w)

        max_pool = torch.max(x_reshaped, 2)[0]   # [B, Heads, H, W]
        avg_pool = torch.mean(x_reshaped, 2)      # [B, Heads, H, W]

        combined = torch.cat([max_pool.unsqueeze(2), avg_pool.unsqueeze(2)], dim=2)
        combined = combined.view(b, self.n_heads * 2, h, w)

        spatial_attn = F.sigmoid(self.conv(combined))
        spatial_attn = spatial_attn.unsqueeze(2)
        output_scale = spatial_attn.expand(-1, -1, c // self.n_heads, -1, -1).reshape(b, c, h, w)
        return x * output_scale


class VLC(nn.Module):
    """Multi-Head Vision-Language Conditioned Attention Module.
    
    Uses AdaGN-Zero: a shared MLP predicts all modulation parameters
    (shift, scale, alpha) for both channel and spatial attention sub-blocks.
    At init, alpha=0 so the block is identity — safe for pretrained backbones.
    """
    def __init__(self, channels, r=2, cond_dim=14, n_heads=1, num_groups=8):
        super().__init__()
        assert channels % n_heads == 0, f"Channels {channels} must be divisible by heads {n_heads}"

        self.cam = CAM(channels, r, n_heads)
        self.sam = SAM(n_heads)

        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, affine=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, affine=False)

        # Shared MLP predicts 6 * channels params (shift, scale, alpha):
        #   shift₁, scale₁, alpha₁ (channel attn)  +  shift₂, scale₂, alpha₂ (spatial attn)
        self.adagn = AdaGN(cond_dim, channels * 6)

    def forward(self, x, conditions):
        params = self.adagn(conditions)
        shift_ca, scale_ca, alpha_ca, shift_sa, scale_sa, alpha_sa = params.chunk(6, dim=1)

        # Channel Attention sub-block
        h = self.gn1(x)
        h = h * (1 + scale_ca) + shift_ca
        h = self.cam(h)
        x = x + alpha_ca * h

        # Spatial Attention sub-block
        h = self.gn2(x)
        h = h * (1 + scale_sa) + shift_sa
        h = self.sam(h)
        x = x + alpha_sa * h

        return x


class VLCTransformLayer(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=14, r=2, n_heads=1):
        super().__init__()
        self.vlc = VLC(channels=in_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)
        
        # Block 1
        self.proj1 = SwiGLUConv2d(in_channels, out_channels)
        self.vlc1 = VLC(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)

        # Block 2
        self.proj2 = SwiGLUConv2d(out_channels, out_channels)
        self.vlc2 = VLC(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)

        # Block 3
        self.proj3 = SwiGLUConv2d(out_channels, out_channels)
        self.vlc3 = VLC(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)

    def forward(self, x, conditions):
        x = self.vlc(x, conditions)

        x = self.proj1(x)
        x = self.vlc1(x, conditions)

        x = self.proj2(x)
        x = self.vlc2(x, conditions)

        x = self.proj3(x)
        x = self.vlc3(x, conditions)
        return x


class VLCTransformQueries(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=14, r=2, n_heads=1):
        super().__init__()
        self.vlc = VLC(channels=in_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)

        # Block 1
        self.proj1 = SwiGLUConv2d(in_channels, out_channels)
        self.vlc1 = VLC(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)

        # Block 2
        self.proj2 = SwiGLUConv2d(out_channels, out_channels)
        self.vlc2 = VLC(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)

        # Block 3
        self.proj3 = SwiGLUConv2d(out_channels, out_channels)
        self.vlc3 = VLC(channels=out_channels, r=r, cond_dim=cond_dim, n_heads=n_heads)

    def forward(self, x, conditions):
        x = self.vlc(x, conditions)

        x = self.proj1(x)
        x = self.vlc1(x, conditions)

        x = self.proj2(x)
        x = self.vlc2(x, conditions)

        x = self.proj3(x)
        x = self.vlc3(x, conditions)
        return x

def __main__():
    # Simple test to verify module functionality
    b, c, h, w = 1, 200, 256, 224
    cond_dim = 14
    n_heads = 2
    x = torch.randn(b, c, h, w)
    cond = torch.randn(b, cond_dim)

    vlc = VLC(channels=c, r=2, cond_dim=cond_dim, n_heads=n_heads)
    out = vlc(x, cond)
    print("Output shape:", out.shape)
    
    x2 = torch.randn(b, c * 2, h, w)
    vlc_transform_layer = VLCTransformLayer(in_channels=c * 2, out_channels=c, cond_dim=cond_dim, r=2, n_heads=n_heads)
    out = vlc_transform_layer(x2, cond)
    print("Transform layer output shape:", out.shape)

    vlc_transform_queries = VLCTransformQueries(in_channels=c * 2, out_channels=c, cond_dim=cond_dim, r=2, n_heads=n_heads)
    out = vlc_transform_queries(x2, cond)
    print("Transform queries output shape:", out.shape)

if __name__ == "__main__":
    __main__()