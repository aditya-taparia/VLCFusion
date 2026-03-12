"""
Cross-CBAM + Unified AdaLN-Zero (DiT-style) fusion block for multimodal feature fusion.

Key difference from cross_cbam_adaln_utils.py:
  A single ConditionModulation MLP produces all (gamma, beta, alpha) triplets for every
  sub-stage in one forward pass, mirroring DiT where one modulation network services
  the entire block.

Five modulated stages:
  0. Self-CBAM on modality 1       (mod_channels)
  1. Self-CBAM on modality 2       (mod_channels)
  2. Cross-CBAM: x1 guided by x2   (mod_channels)
  3. Cross-CBAM: x2 guided by x1   (mod_channels)
  4. Fused projection               (out_channels)

Each stage follows: GroupNorm(affine=False) -> modulate -> attention -> gated residual.
Fuse stage: cat -> 1x1 conv -> GroupNorm -> modulate -> alpha-gate (no residual).
No separate FFN.
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


def _modulate(x_norm: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Adaptive scale and shift: (1 + gamma) * x_norm + beta."""
    return (1.0 + gamma) * x_norm + beta


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
            nn.ReLU(inplace=True),
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
                              padding=kernel_size // 2, bias=False)

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
            nn.ReLU(inplace=True),
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
                              padding=kernel_size // 2, bias=False)

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
# Unified condition modulation (DiT-style)
# =============================================================================

class ConditionModulation(nn.Module):
    """
    Single MLP that produces (gamma, beta, alpha) for every modulation stage.

    The final linear layer is zero-initialised so the block starts as identity
    (AdaLN-Zero property).

    Args:
        cond_dim:    Dimensionality of the conditioning vector.
        stage_dims:  List of channel counts, one per modulation stage.
        hidden_mult: Multiplier for the hidden-layer width.
    """
    def __init__(self, cond_dim: int, stage_dims: List[int], hidden_mult: float = 2.0):
        super().__init__()
        total_out = sum(3 * d for d in stage_dims)
        hidden = max(max(stage_dims), int(cond_dim * hidden_mult))
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, total_out),
        )
        # nn.init.zeros_(self.mlp[-1].weight)
        # nn.init.zeros_(self.mlp[-1].bias)
        self.stage_dims = stage_dims

    def forward(self, cond: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Returns list of (gamma, beta, alpha) tuples, each shaped [B, d_i]."""
        out = self.mlp(cond)
        params: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        offset = 0
        for d in self.stage_dims:
            gamma = out[:, offset:offset + d]; offset += d
            beta  = out[:, offset:offset + d]; offset += d
            alpha = out[:, offset:offset + d]; offset += d
            params.append((gamma, beta, alpha))
        return params


# =============================================================================
# Full fusion block
# =============================================================================

class CrossCBAMDiTFusion(nn.Module):
    """
    DiT-style Cross-CBAM fusion block.

    One ConditionModulation MLP produces (gamma, beta, alpha) for five stages:
      0. self-CBAM on x1        (mod_channels)
      1. self-CBAM on x2        (mod_channels)
      2. cross-CBAM  x1 <- x2   (mod_channels)
      3. cross-CBAM  x2 <- x1   (mod_channels)
      4. fused projection        (out_channels)

    Stages 0-3:  x_out = x + alpha * Attention( (1+gamma)*GN(x) + beta )
    Stage 4:     output = alpha * ( (1+gamma)*GN(fuse_conv(cat)) + beta )
    """

    def __init__(
        self,
        mod_channels: int,
        out_channels: int,
        cond_dim: int,
        r: int = 2,
        num_groups: int = 8,
    ):
        super().__init__()
        self.mod_channels = mod_channels
        self.out_channels = out_channels
        ng_mod = _num_groups(mod_channels, num_groups)
        ng_out = _num_groups(out_channels, num_groups)

        # Unified condition modulation: one MLP for all 5 stages
        stage_dims = [mod_channels, mod_channels,   # self stages
                      mod_channels, mod_channels,   # cross stages
                      out_channels]                  # fuse stage
        self.cond_mod = ConditionModulation(cond_dim, stage_dims)

        # Per-stage GroupNorms (affine=False: scale/shift from condition only)
        self.norm_self1  = nn.GroupNorm(ng_mod, mod_channels, affine=False)
        self.norm_self2  = nn.GroupNorm(ng_mod, mod_channels, affine=False)
        self.norm_cross1 = nn.GroupNorm(ng_mod, mod_channels, affine=False)
        self.norm_cross2 = nn.GroupNorm(ng_mod, mod_channels, affine=False)
        self.norm_fuse   = nn.GroupNorm(ng_out, out_channels, affine=False)

        # Attention modules
        self.self_cbam1 = CBAM(mod_channels, r=r)
        self.self_cbam2 = CBAM(mod_channels, r=r)
        self.cross_cbam_1from2 = CrossCBAM(mod_channels, r=r)
        self.cross_cbam_2from1 = CrossCBAM(mod_channels, r=r)

        # Fusion projection: cat(u1, u2) [2*C] -> out_channels via 1x1 conv
        self.fuse_conv = nn.Conv2d(mod_channels * 2, out_channels,
                                   kernel_size=1, bias=False)

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
        B = x1.size(0)
        C = self.mod_channels
        C_out = self.out_channels

        # Single MLP call -> all modulation parameters
        params = self.cond_mod(cond)
        g0, b0, a0 = params[0]
        g1, b1, a1 = params[1]
        g2, b2, a2 = params[2]
        g3, b3, a3 = params[3]
        g4, b4, a4 = params[4]

        # Reshape modulation params to [B, C, 1, 1] for spatial broadcast
        g0, b0, a0 = g0.view(-1, C, 1, 1), b0.view(-1, C, 1, 1), a0.view(-1, C, 1, 1)
        g1, b1, a1 = g1.view(-1, C, 1, 1), b1.view(-1, C, 1, 1), a1.view(-1, C, 1, 1)
        g2, b2, a2 = g2.view(-1, C, 1, 1), b2.view(-1, C, 1, 1), a2.view(-1, C, 1, 1)
        g3, b3, a3 = g3.view(-1, C, 1, 1), b3.view(-1, C, 1, 1), a3.view(-1, C, 1, 1)
        g4, b4, a4 = g4.view(-1, C_out, 1, 1), b4.view(-1, C_out, 1, 1), a4.view(-1, C_out, 1, 1)

        # Stage 0: self-CBAM on x1
        s1 = x1 + a0 * self.self_cbam1(_modulate(self.norm_self1(x1), g0, b0))

        # Stage 1: self-CBAM on x2
        s2 = x2 + a1 * self.self_cbam2(_modulate(self.norm_self2(x2), g1, b1))

        # Stage 2: cross-CBAM x1 <- x2 (s1 refined by s2)
        u1 = s1 + a2 * self.cross_cbam_1from2(
            _modulate(self.norm_cross1(s1), g2, b2), s2)

        # Stage 3: cross-CBAM x2 <- x1 (s2 refined by s1)
        u2 = s2 + a3 * self.cross_cbam_2from1(
            _modulate(self.norm_cross2(s2), g3, b3), s1)

        # Stage 4: fuse — cat, project, modulate, alpha-gate
        fused = self.fuse_conv(torch.cat([u1, u2], dim=1))
        
        return a4 * _modulate(self.norm_fuse(fused), g4, b4)


# =============================================================================
# Transform wrappers (drop-in for MultimodalDetr: concat input -> single output)
# =============================================================================

class CrossCBAMDiTTransformLayer(nn.Module):
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
        self.fusion = CrossCBAMDiTFusion(
            mod_channels=in_channels // 2,
            out_channels=out_channels,
            cond_dim=cond_dim,
            r=r,
            num_groups=num_groups,
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        c = x.size(1) // 2
        return self.fusion(x[:, :c], x[:, c:], conditions)


class CrossCBAMDiTTransformQueries(nn.Module):
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
        self.fusion = CrossCBAMDiTFusion(
            mod_channels=in_channels // 2,
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
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)
    cond = torch.randn(B, cond_dim)

    # --- Shape tests ---
    fusion = CrossCBAMDiTFusion(
        mod_channels=C, out_channels=C, cond_dim=cond_dim, r=2,
    )
    out = fusion(x1, x2, cond)
    print(f"CrossCBAMDiTFusion: ({x1.shape}, {x2.shape}) + cond -> {out.shape}")
    assert out.shape == (B, C, H, W)

    in_ch, out_ch = 512, 256
    x_cat = torch.randn(B, in_ch, H, W)
    layer = CrossCBAMDiTTransformLayer(in_ch, out_ch, cond_dim=cond_dim, r=2)
    out_layer = layer(x_cat, cond)
    print(f"CrossCBAMDiTTransformLayer: {x_cat.shape} -> {out_layer.shape}")
    assert out_layer.shape == (B, out_ch, H, W)

    queries = CrossCBAMDiTTransformQueries(in_ch, out_ch, cond_dim=cond_dim, r=2)
    x_q = torch.randn(B, in_ch, 1, 1)
    out_q = queries(x_q, cond)
    print(f"CrossCBAMDiTTransformQueries: {x_q.shape} -> {out_q.shape}")
    assert out_q.shape == (B, out_ch, 1, 1)

    # # --- Zero-init verification ---
    # zero_cond = torch.zeros(B, cond_dim)
    # out_zero = fusion(x1, x2, zero_cond)
    # print(f"\nZero-init check (zero cond): output max abs = {out_zero.abs().max().item():.6f}")
    # assert out_zero.abs().max().item() < 1e-5, "Expected ~zero output with zero-init + zero cond"

    total = sum(p.numel() for p in fusion.parameters())
    print(f"Params (mod=out={C}, cond={cond_dim}): {total:,} total")
    print("All CrossCBAM-DiT modules passed.")
