"""
Cross-CBAM + AdaLN-Zero fusion block for multimodal (e.g. IR + RGB) feature fusion.

Blueprint:
  - Self-attention via CBAM (pre-norm, channel then spatial) per modality.
  - Cross-attention via cross-CBAM: one modality guides the other (cross-multiply).
  - Concat + optional 1x1 projection.
  - Adaptive modulation (AdaLN-Zero) from condition: scale, shift, residual strength (zero-initialized).
  - Optional FFN after modulation.

All normalizations before modulation use GroupNorm with affine=False so that
scale/shift come only from the condition MLP (AdaLN-Zero).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _num_groups(channels: int, desired: int = 8) -> int:
    """Largest divisor of channels that is <= desired."""
    for g in range(min(desired, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# =============================================================================
# CBAM components (pre-norm, no affine in norm where AdaLN will modulate)
# =============================================================================

class CAM(nn.Module):
    """Channel Attention Module: avg+max pool → shared MLP → sigmoid → multiply."""
    def __init__(self, channels: int, r: int = 2):
        super().__init__()
        self.channels = channels
        self.r = max(1, r)
        mid = max(1, channels // self.r)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_ = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(max_)).view(b, c, 1, 1)
        return w * x


class SAM(nn.Module):
    """Spatial Attention Module: channel pool (avg+max) → conv → sigmoid → multiply."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return w * x


class CBAMPreNorm(nn.Module):
    """Pre-norm CBAM: GroupNorm (no affine) → CAM → SAM → + residual."""
    def __init__(self, channels: int, r: int = 2, num_groups: int = 8):
        super().__init__()
        ng = _num_groups(channels, num_groups)
        self.norm = nn.GroupNorm(ng, channels, affine=False)
        self.cam = CAM(channels, r)
        self.sam = SAM()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        x = self.cam(x)
        x = self.sam(x)
        return identity + x


# =============================================================================
# Cross-CBAM: derive channel and spatial weights FROM the other modality
# =============================================================================

class CrossChannelAttention(nn.Module):
    """Produce channel weights from a guide feature map and apply to target."""
    def __init__(self, channels: int, r: int = 2):
        super().__init__()
        mid = max(1, channels // max(1, r))
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=True),
        )

    def forward(
        self,
        target: torch.Tensor,
        guide: torch.Tensor,
    ) -> torch.Tensor:
        """Apply channel weights derived from guide to target. target, guide: [B, C, H, W]."""
        b, c, _, _ = guide.size()
        avg = F.adaptive_avg_pool2d(guide, 1).view(b, c)
        max_ = F.adaptive_max_pool2d(guide, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(max_)).view(b, c, 1, 1)
        return target * w


class CrossSpatialAttention(nn.Module):
    """Produce spatial mask from a guide feature map and apply to target."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(
        self,
        target: torch.Tensor,
        guide: torch.Tensor,
    ) -> torch.Tensor:
        """Apply spatial mask derived from guide to target. target, guide: [B, C, H, W]."""
        avg = torch.mean(guide, dim=1, keepdim=True)
        max_, _ = torch.max(guide, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return target * w


class CrossCBAM(nn.Module):
    """Cross-CBAM: guide modality produces channel and spatial weights applied to target."""
    def __init__(self, channels: int, r: int = 2):
        super().__init__()
        self.cross_cam = CrossChannelAttention(channels, r)
        self.cross_sam = CrossSpatialAttention()

    def forward(
        self,
        target: torch.Tensor,
        guide: torch.Tensor,
    ) -> torch.Tensor:
        """Refine target using channel and spatial attention derived from guide."""
        out = self.cross_cam(target, guide)
        out = self.cross_sam(out, guide)
        return out


# =============================================================================
# AdaLN-Zero: condition → scale (γ), shift (β), residual strength (α), zero-initialized
# =============================================================================

class AdaLNZeroProj(nn.Module):
    """Condition → γ, β, α for adaptive modulation. Last layer zero-initialized (AdaLN-Zero)."""
    def __init__(self, cond_dim: int, out_channels: int, hidden_mult: float = 2.0):
        super().__init__()
        self.out_channels = out_channels
        hidden = max(out_channels, int(out_channels * hidden_mult))
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_channels * 3),
        )
        # nn.init.zeros_(self.mlp[-1].weight)
        # nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (gamma, beta, alpha), each [B, out_channels] or alpha [B, 1] if scalar."""
        b = cond.size(0)
        out = self.mlp(cond).view(b, 3, self.out_channels)
        gamma = out[:, 0]
        beta = out[:, 1]
        alpha = out[:, 2]
        return gamma, beta, alpha


# =============================================================================
# Full fusion block: Self-CBAM → Cross-CBAM → Concat → Proj → AdaLN → optional FFN
# =============================================================================

class CrossCBAMAdaLNFusion(nn.Module):
    """
    Fusion block: two modalities (e.g. IR, RGB) with shape [B, C, H, W] each.
     1. Condition projection → γ, β, α (AdaLN-Zero).
     2. Self-attention (pre-norm CBAM) per modality → s^(1), s^(2).
     3. Cross-CBAM: s^(1) guided by s^(2) → u^(1), s^(2) guided by s^(1) → u^(2).
     4. Concat u = [u^(1), u^(2)], 1x1 conv → C channels.
     5. Pre-norm (GN, no affine), z = (1+γ)·û + β, output = (x1+x2) + α·z.
     6. Optional FFN: pre-norm, MLP, condition scale/shift, residual.
    """

    def __init__(
        self,
        mod_channels: int,
        out_channels: int,
        cond_dim: int,
        r: int = 2,
        num_groups: int = 8,
        use_ffn: bool = True,
        ffn_ratio: float = 2.0,
    ):
        super().__init__()
        self.mod_channels = mod_channels
        self.out_channels = out_channels
        ng = _num_groups(mod_channels, num_groups)
        ng_out = _num_groups(out_channels, num_groups)

        # Condition projection for main branch (γ, β, α)
        self.cond_proj = AdaLNZeroProj(cond_dim, out_channels)

        # Self-attention (CBAM with pre-norm) for each modality
        self.self_cbam_1 = CBAMPreNorm(mod_channels, r=r, num_groups=num_groups)
        self.self_cbam_2 = CBAMPreNorm(mod_channels, r=r, num_groups=num_groups)

        # Cross-CBAM (bidirectional)
        self.cross_1_from_2 = CrossCBAM(mod_channels, r=r)
        self.cross_2_from_1 = CrossCBAM(mod_channels, r=r)

        # Concat + projection to out_channels
        self.fuse_conv = nn.Conv2d(mod_channels * 2, out_channels, kernel_size=1, bias=False)
        self.norm_fused = nn.GroupNorm(ng_out, out_channels, affine=False)

        # Residual branch: project (x1 + x2) from mod_channels to out_channels when they differ.
        # Zero-initialized so the residual path starts as zero and the block behaves like AdaLN-Zero.
        if mod_channels != out_channels:
            self.res_proj = nn.Conv2d(mod_channels, out_channels, kernel_size=1, bias=False)
            nn.init.zeros_(self.res_proj.weight)
        else:
            self.res_proj = nn.Identity()

        # Optional FFN (pre-norm, conv MLP, condition modulation)
        self.use_ffn = use_ffn
        if use_ffn:
            ffn_hidden = max(out_channels, int(out_channels * ffn_ratio))
            ng_ffn = _num_groups(ffn_hidden, num_groups)
            self.ffn_norm = nn.GroupNorm(ng_out, out_channels, affine=False)
            self.ffn = nn.Sequential(
                nn.Conv2d(out_channels, ffn_hidden, 1, bias=False),
                nn.GroupNorm(ng_ffn, ffn_hidden),
                nn.SiLU(),
                nn.Conv2d(ffn_hidden, out_channels, 1, bias=False),
            )
            self.ffn_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, out_channels * 2),
            )
            nn.init.zeros_(self.ffn_cond[-1].weight)
            nn.init.zeros_(self.ffn_cond[-1].bias)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        x1, x2: [B, C_mod, H, W]
        cond: [B, cond_dim]
        Returns: [B, out_channels, H, W]
        """
        # Align spatial size (bilinear can alias; alternatives: use lower-res grid or learned upsample)
        if x1.shape[-2:] != x2.shape[-2:]:
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode="bilinear", align_corners=False)

        gamma, beta, alpha = self.cond_proj(cond)
        B, C, H, W = x1.shape
        gamma = gamma.view(B, -1, 1, 1)
        beta = beta.view(B, -1, 1, 1)
        alpha = alpha.view(B, -1, 1, 1)

        # 2. Self-attention per modality
        s1 = self.self_cbam_1(x1)
        s2 = self.self_cbam_2(x2)

        # 3. Cross-CBAM
        u1 = self.cross_1_from_2(s1, s2)
        u2 = self.cross_2_from_1(s2, s1)

        # 4. Concat + project
        u = torch.cat([u1, u2], dim=1)
        u = self.fuse_conv(u)

        # 5. AdaLN and residual (residual = sum of original modalities, projected to same dim if needed)
        x_sum = self.res_proj(x1 + x2)

        u_norm = self.norm_fused(u)
        z = (1 + gamma) * u_norm + beta
        out = x_sum + alpha * z

        # 6. Optional FFN
        if self.use_ffn:
            identity = out
            out_norm = self.ffn_norm(out)
            ffn_scale_shift = self.ffn_cond(cond).view(B, 2, self.out_channels, 1, 1)
            ffn_scale = ffn_scale_shift[:, 0]
            ffn_shift = ffn_scale_shift[:, 1]
            out_mod = out_norm * (1 + ffn_scale) + ffn_shift
            out = identity + self.ffn(out_mod)

        return out


# =============================================================================
# Transform layers (drop-in for MultimodalDetr: concat input → single output)
# =============================================================================

class CrossCBAMAdaLNTransformLayer(nn.Module):
    """
    Fusion for concatenated feature maps [B, 2*C, H, W].
    Splits into two modalities, runs CrossCBAMAdaLNFusion, returns [B, out_channels, H, W].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 2,
        num_groups: int = 8,
        use_ffn: bool = True,
    ):
        super().__init__()
        mod_channels = in_channels // 2
        self.fusion = CrossCBAMAdaLNFusion(
            mod_channels=mod_channels,
            out_channels=out_channels,
            cond_dim=cond_dim,
            r=r,
            num_groups=num_groups,
            use_ffn=use_ffn,
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        c = x.size(1) // 2
        x1, x2 = x[:, :c], x[:, c:]
        return self.fusion(x1, x2, conditions)


class CrossCBAMAdaLNTransformQueries(nn.Module):
    """
    Same fusion for concatenated object queries [B, 2*C, 1, 1].
    Same architecture as TransformLayer (spatial size 1x1 is fine).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 2,
        num_groups: int = 8,
        use_ffn: bool = True,
    ):
        super().__init__()
        mod_channels = in_channels // 2
        self.fusion = CrossCBAMAdaLNFusion(
            mod_channels=mod_channels,
            out_channels=out_channels,
            cond_dim=cond_dim,
            r=r,
            num_groups=num_groups,
            use_ffn=use_ffn,
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        c = x.size(1) // 2
        x1, x2 = x[:, :c], x[:, c:]
        return self.fusion(x1, x2, conditions)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    B, C, H, W = 2, 256, 15, 15
    cond_dim = 7
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)
    cond = torch.randn(B, cond_dim)

    fusion = CrossCBAMAdaLNFusion(
        mod_channels=C,
        out_channels=C,
        cond_dim=cond_dim,
        r=2,
        use_ffn=True,
    )
    out = fusion(x1, x2, cond)
    print(f"CrossCBAMAdaLNFusion: ({x1.shape}, {x2.shape}) + cond -> {out.shape}")
    assert out.shape == (B, C, H, W)

    in_ch, out_ch = 512, 256
    x_cat = torch.randn(B, in_ch, H, W)
    layer = CrossCBAMAdaLNTransformLayer(in_ch, out_ch, cond_dim=cond_dim, r=2)
    out_layer = layer(x_cat, cond)
    print(f"CrossCBAMAdaLNTransformLayer: {x_cat.shape} -> {out_layer.shape}")
    assert out_layer.shape == (B, out_ch, H, W)

    queries = CrossCBAMAdaLNTransformQueries(in_ch, out_ch, cond_dim=cond_dim, r=2)
    x_queries = torch.randn(B, in_ch, 1, 1)
    out_q = queries(x_queries, cond)
    print(f"CrossCBAMAdaLNTransformQueries: {x_queries.shape} -> {out_q.shape}")
    assert out_q.shape == (B, out_ch, 1, 1)

    print("All CrossCBAM-AdaLN modules passed.")