"""
Cross-CBAM + FiLM-conditioned fusion block for multimodal feature fusion (V2).

Key difference from cross_cbam_dit_utils.py (V1):
  Uses per-stage FiLM modulation instead of a unified AdaLN-Zero MLP, and adds
  a point-wise FFN after the fuse projection.  Supports asymmetric modality
  channel dimensions (mod1_channels != mod2_channels).

Five modulated stages:
  0. Self-CBAM on modality 1           (mod1_channels)
  1. Self-CBAM on modality 2           (mod2_channels)
  2. Cross-CBAM: x1 guided by x2       (mod1_channels)
  3. Cross-CBAM: x2 guided by x1       (mod2_channels)
  4. FFN on fused features              (out_channels)

Every stage follows:
  x_out = FiLM(SubLayer(GN(x)), cond) + x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# =============================================================================
# Helpers
# =============================================================================

def _num_groups(channels: int, desired: int = 8) -> int:
    """Largest divisor of *channels* that is <= *desired*."""
    for g in range(min(desired, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# =============================================================================
# CBAM components (bare — no norm, no residual)
# =============================================================================

class ChannelAttention(nn.Module):
    """Avg + max pool -> shared MLP -> sigmoid gate."""
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
    """Channel pool (avg + max) -> conv -> sigmoid gate."""
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
    """Bare CBAM: channel attention then spatial attention (no norm, no residual)."""
    def __init__(self, channels: int, r: int = 2):
        super().__init__()
        self.cam = ChannelAttention(channels, r)
        self.sam = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sam(self.cam(x))


# =============================================================================
# Cross-CBAM: attention weights derived from a guide modality
# =============================================================================

class CrossChannelAttention(nn.Module):
    """Channel weights from *guide* applied to *target*."""
    def __init__(self, channels: int, r: int = 2):
        super().__init__()
        mid = max(1, channels // max(1, r))
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=True),
            nn.SiLU(),
            nn.Linear(mid, channels, bias=True),
        )

    def forward(self, target: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = guide.size()
        avg = F.adaptive_avg_pool2d(guide, 1).view(b, c)
        mx = F.adaptive_max_pool2d(guide, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return target * w


class CrossSpatialAttention(nn.Module):
    """Spatial mask from *guide* applied to *target*."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, stride=1, dilation=1, bias=False)

    def forward(self, target: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(guide, dim=1, keepdim=True)
        mx, _ = torch.max(guide, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return target * w


class CrossCBAM(nn.Module):
    """Cross-CBAM: guide produces channel then spatial weights for target."""
    def __init__(self, channels: int, r: int = 2):
        super().__init__()
        self.cross_cam = CrossChannelAttention(channels, r)
        self.cross_sam = CrossSpatialAttention()

    def forward(self, target: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        out = self.cross_cam(target, guide)
        return self.cross_sam(out, guide)


# =============================================================================
# FiLM modulation
# =============================================================================

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


# =============================================================================
# Fusion block
# =============================================================================

class CrossCBAMDiTFusionV2(nn.Module):
    """
    FiLM-style Cross-CBAM fusion block (V2).

    Per-stage FiLMModulation MLPs produce (gamma, beta) for each stage:
      0. self-CBAM on x1          (mod1_channels)
      1. self-CBAM on x2          (mod2_channels)
      2. cross-CBAM  x1 <- x2     (mod1_channels)
      3. cross-CBAM  x2 <- x1     (mod2_channels)
      4. FFN                       (out_channels)

    All stages:  x_out = FiLM(SubLayer(GN(x)), cond) + x
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
        ng_mod1 = _num_groups(mod1_channels, num_groups)
        ng_mod2 = _num_groups(mod2_channels, num_groups)
        ng_out = _num_groups(out_channels, num_groups)

        self.film_mod_self1 = FiLMModulation(mod1_channels, cond_dim)
        self.film_mod_self2 = FiLMModulation(mod2_channels, cond_dim)
        self.film_mod_cross1 = FiLMModulation(mod1_channels, cond_dim)
        self.film_mod_cross2 = FiLMModulation(mod2_channels, cond_dim)

        # Per-stage GroupNorms
        self.norm_self1  = nn.GroupNorm(ng_mod1, mod1_channels, eps=1e-6, affine=True)
        self.norm_self2  = nn.GroupNorm(ng_mod2, mod2_channels, eps=1e-6, affine=True)
        self.norm_cross1 = nn.GroupNorm(ng_mod1, mod1_channels, eps=1e-6, affine=True)
        self.norm_cross2 = nn.GroupNorm(ng_mod2, mod2_channels, eps=1e-6, affine=True)

        # Attention modules
        self.self_cbam1 = CBAM(mod1_channels, r=r)
        self.self_cbam2 = CBAM(mod2_channels, r=r)
        self.cross_cbam_1from2 = CrossCBAM(mod1_channels, r=r)
        self.cross_cbam_2from1 = CrossCBAM(mod2_channels, r=r)

        # Fusion projection: cat(u1, u2) [2*C] -> out_channels via 1x1 conv
        self.fuse_conv = nn.Conv2d(mod1_channels + mod2_channels, out_channels,
                                   kernel_size=1, bias=False)

        self.ffn = nn.Sequential(
            nn.GroupNorm(ng_out, out_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(ng_out, out_channels, eps=1e-6, affine=True),
        )
        self.film_mod_ffn = FiLMModulation(out_channels, cond_dim)
        self.ffn_activation = nn.SiLU()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x1:   [B, mod_channels, H, W]  (modality 1, e.g. IR)
            x2:   [B, mod_channels, H, W]  (modality 2, e.g. RGB)
            cond: [B, cond_dim]
        Returns:
            [B, out_channels, H, W]
        """
        # Modality 1 Self Attention
        identity_x1 = x1
        s1 = self.norm_self1(x1)
        s1 = self.self_cbam1(s1)
        s1 = self.film_mod_self1(s1,cond) + identity_x1
        
        # Modality 2 Self Attention
        identity_x2 = x2
        s2 = self.norm_self2(x2)
        s2 = self.self_cbam2(s2)
        s2 = self.film_mod_self2(s2,cond) + identity_x2
        
        # Cross Attention 1 from 2
        identity_x1 = s1
        u1 = self.norm_cross1(s1)
        u1 = self.cross_cbam_1from2(u1,s2)
        u1 = self.film_mod_cross1(u1,cond) + identity_x1
        
        # Cross Attention 2 from 1
        identity_x2 = s2
        u2 = self.norm_cross2(s2)
        u2 = self.cross_cbam_2from1(u2,s1)
        u2 = self.film_mod_cross2(u2,cond) + identity_x2
        
        # Fusion
        fused_base = self.fuse_conv(torch.cat([u1, u2], dim=1))
        
        fused_out = self.ffn(fused_base)
        fused_out = self.film_mod_ffn(fused_out, cond)
        fused_out = self.ffn_activation(fused_out) + fused_base
        
        return fused_out


# =============================================================================
# Transform wrappers (drop-in for MultimodalDetr: concat input -> single output)
# =============================================================================

class CrossCBAMDiTTransformLayerV2(nn.Module):
    """
    Fusion for concatenated feature maps [B, 2*C, H, W].
    Splits into two modalities, runs CrossCBAMDiTFusion, returns [B, out_channels, H, W].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 2,
        num_groups: int = 8,
    ):
        super().__init__()
        self.fusion = CrossCBAMDiTFusionV2(
            mod1_channels=in_channels // 2,
            mod2_channels=in_channels // 2,
            out_channels=out_channels,
            cond_dim=cond_dim,
            r=r,
            num_groups=num_groups,
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        c = x.size(1) // 2
        return self.fusion(x[:, :c], x[:, c:], conditions)


class CrossCBAMDiTTransformQueriesV2(nn.Module):
    """Same fusion for concatenated object queries [B, 2*C, 1, 1]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 2,
        num_groups: int = 8,
    ):
        super().__init__()
        self.fusion = CrossCBAMDiTFusionV2(
            mod1_channels=in_channels // 2,
            mod2_channels=in_channels // 2,
            out_channels=out_channels,
            cond_dim=cond_dim,
            r=r,
            num_groups=num_groups,
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        c = x.size(1) // 2
        return self.fusion(x[:, :c], x[:, c:], conditions)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    B, C, H, W = 2, 256, 15, 15
    cond_dim = 7

    # =========================================================================
    # Setting 1: Both modalities same shape [B, 100, 256, 224] (DETR features)
    # =========================================================================
    print("=" * 60)
    print("Setting 1: same-channel modalities [B, 100, 256, 224]")
    print("=" * 60)

    B, C, H, W = 2, 100, 256, 224
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)
    cond = torch.randn(B, cond_dim)

    # --- Shape tests ---
    fusion = CrossCBAMDiTFusionV2(
        mod1_channels=C, mod2_channels=C, out_channels=C, cond_dim=cond_dim, r=2,
    )
    out = fusion(x1, x2, cond)
    print(f"CrossCBAMDiTFusionV2: ({x1.shape}, {x2.shape}) + cond -> {out.shape}")
    assert out.shape == (B, C, H, W)

    in_ch, out_ch = 512, 256
    x_cat = torch.randn(B, in_ch, H, W)
    layer = CrossCBAMDiTTransformLayerV2(in_ch, out_ch, cond_dim=cond_dim, r=2)
    out_layer = layer(x_cat, cond)
    print(f"CrossCBAMDiTTransformLayerV2: {x_cat.shape} -> {out_layer.shape}")
    assert out_layer.shape == (B, out_ch, H, W)

    queries = CrossCBAMDiTTransformQueriesV2(in_ch, out_ch, cond_dim=cond_dim, r=2)
    x_q = torch.randn(B, in_ch, 1, 1)
    out_q = queries(x_q, cond)
    print(f"CrossCBAMDiTTransformQueriesV2: {x_q.shape} -> {out_q.shape}")
    assert out_q.shape == (B, out_ch, 1, 1)

    total = sum(p.numel() for p in fusion.parameters())
    print(f"Params (mod=out={C}, cond={cond_dim}): {total:,} total")
    print("All CrossCBAM-DiT V2 modules passed.")
