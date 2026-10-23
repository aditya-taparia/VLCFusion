import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def _num_groups(channels: int, desired: int = 8) -> int:
    """Largest divisor of *channels* that is <= *desired*."""
    for g in range(min(desired, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


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
# FiLM modulation
# =============================================================================

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
        # Initialize the linear layer to zero
        # nn.init.constant_(self.linear.weight, 0)
        # nn.init.constant_(self.linear.bias, 0)

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
        
# =============================================================================
# Multi-head fusion bottleneck
# =============================================================================
class MultiHeadFusionBottleneck(nn.Module):
    """
    Concat-based multi-branch fusion bottleneck with softmax-weighted averaging.
    SKNet / DynamicConv style: multiple fusion heads, input-dependent weights.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_ratio: int = 4):
        super().__init__()

        # Three fusion heads
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
        # x: [B, C_in, H, W]
        branch_outs = [b(x) for b in self.branches]          # list of [B, C_out, H, W]
        y = torch.stack(branch_outs, dim=1)                  # [B, K, C_out, H, W]

        alpha = torch.softmax(self.router(x), dim=-1)        # [B, K]
        y = (y * alpha[:, :, None, None, None]).sum(dim=1)   # [B, C_out, H, W]
        return y


# =============================================================================
# VLC Block 
# =============================================================================

class VLCBlock(nn.Module):
    """
    GroupNorm -> FiLM(cond) -> CBAM -> residual,
    followed by a conditioned FFN block.

    Flow (CBAM arm):
        h = norm(x)
        g, b, a = film(h, cond)
        h = (1 + g) * h + b
        out = a * cbam(h) + x          # residual

    Flow (FFN arm):
        h = norm_ffn(out)
        g, b, a = film_ffn(h, cond)
        h = (1 + g) * h + b
        out = ffn(h) + out          # residual
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
                nn.Conv2d(channels, channels * ffn_factor, kernel_size=ffn_kernel_size, padding=ffn_kernel_size // 2, bias=False),
                nn.SiLU(),
                nn.Conv2d(channels * ffn_factor, channels, kernel_size=ffn_kernel_size, padding=ffn_kernel_size // 2, bias=False),
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


# =============================================================================
# Fusion block
# =============================================================================

class VLCFusion(nn.Module):
    """
    x1 -> VLCBlock -> s1
    x2 -> VLCBlock -> s2
    concat(s1, s2) -> fusion_bottleneck -> VLCBlock (with FFN) -> VLCBlock -> output
    """

    def __init__(
        self,
        mod1_channels: int,
        mod2_channels: int,
        out_channels: int,
        cond_dim: int,
        r: int = 2,
        num_groups: int = 8,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.mod1_channels = mod1_channels
        self.mod2_channels = mod2_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        self.fusion_bottleneck = MultiHeadFusionBottleneck(
            mod1_channels + mod2_channels,
            out_channels
        )

        # Stack of condition-modulated VLC blocks. Depth is configurable for the
        # num-blocks ablation (paper default = 2; ablation sweeps 2/4/6/8).
        self.vlc_blocks = nn.ModuleList([
            VLCBlock(out_channels, cond_dim, r=r, num_groups=num_groups,
                     if_ffn=True, ffn_factor=1, ffn_kernel_size=3)
            for _ in range(num_blocks)
        ])


    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x1:   [B, mod1_channels, H, W]  (modality 1, e.g. IR)
            x2:   [B, mod2_channels, H, W]  (modality 2, e.g. RGB)
            cond: [B, cond_dim]
        Returns:
            [B, out_channels, H, W]
        """
        fused = self.fusion_bottleneck(torch.cat([x1, x2], dim=1))
        for blk in self.vlc_blocks:
            fused = blk(fused, cond)
        return fused

    # V1 is with just fusion after concat with 2 vlc blocks.
    # V2 is with fusion after multi-head concat with 2 vlc blocks.
    # [Very bad] V3 is with fusion after multi-head concat with 2 vlc blocks and RGB detection head.

# =============================================================================
# Transform wrappers (drop-in for MultimodalDetr: concat input -> single output)
# =============================================================================

class VLCFusionTransformLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 2,
        num_groups: int = 8,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.fusion = VLCFusion(
            mod1_channels=in_channels // 2,
            mod2_channels=in_channels // 2,
            out_channels=out_channels,
            cond_dim=cond_dim,
            r=r,
            num_groups=num_groups,
            num_blocks=num_blocks,
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        c = x.size(1) // 2
        return self.fusion(x[:, :c], x[:, c:], conditions)


class VLCFusionTransformQueries(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 2,
        num_groups: int = 8,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.fusion = VLCFusion(
            mod1_channels=in_channels // 2,
            mod2_channels=in_channels // 2,
            out_channels=out_channels,
            cond_dim=cond_dim,
            r=r,
            num_groups=num_groups,
            num_blocks=num_blocks,
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
    fusion = VLCFusion(
        mod1_channels=C, mod2_channels=C, out_channels=C, cond_dim=cond_dim, r=2,
    )
    out = fusion(x1, x2, cond)
    print(f"VLCFusion: ({x1.shape}, {x2.shape}) + cond -> {out.shape}")
    assert out.shape == (B, C, H, W)

    in_ch, out_ch = 512, 256
    x_cat = torch.randn(B, in_ch, H, W)
    layer = VLCFusionTransformLayer(in_ch, out_ch, cond_dim=cond_dim, r=2)
    out_layer = layer(x_cat, cond)
    print(f"VLCFusionTransformLayer: {x_cat.shape} -> {out_layer.shape}")
    assert out_layer.shape == (B, out_ch, H, W)

    queries = VLCFusionTransformQueries(in_ch, out_ch, cond_dim=cond_dim, r=2)
    x_q = torch.randn(B, in_ch, 1, 1)
    out_q = queries(x_q, cond)
    print(f"VLCFusionTransformQueries: {x_q.shape} -> {out_q.shape}")
    assert out_q.shape == (B, out_ch, 1, 1)

    total = sum(p.numel() for p in fusion.parameters())
    print(f"Params (mod=out={C}, cond={cond_dim}): {total:,} total")
    print("All CrossCBAM-DiT V11 modules passed.")
