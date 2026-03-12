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
    def __init__(self, in_channels, cond_dim):
        super(FiLMModulation, self).__init__()
        self.linear = nn.Linear(cond_dim, in_channels * 3)
        # Initialize the linear layer to zero
        # nn.init.constant_(self.linear.weight, 0)
        # nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, cond):
        film_params = self.linear(cond)
        gamma, beta, alpha = film_params.chunk(3, dim=1)
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)
        alpha = alpha.view(-1, x.size(1), 1, 1)

        return gamma, beta, alpha


# =============================================================================
# Fusion block
# =============================================================================

class CrossCBAMDiTFusionV4(nn.Module):
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

        # Self-Attention modules
        self.film_mod_self1 = FiLMModulation(mod1_channels, cond_dim)
        self.film_mod_self2 = FiLMModulation(mod2_channels, cond_dim)
        
        # self.self_activation1 = nn.SiLU()
        # self.self_activation2 = nn.SiLU()

        self.norm_self1  = nn.GroupNorm(ng_mod1, mod1_channels, affine=False)
        self.norm_self2  = nn.GroupNorm(ng_mod2, mod2_channels, affine=False)

        self.self_cbam1 = CBAM(mod1_channels, r=r)
        self.self_cbam2 = CBAM(mod2_channels, r=r)

        # Fusion bottleneck: cat(u1, u2) [2*C] -> out_channels via 1x1 conv
        self.fusion_bottleneck = nn.Conv2d(mod1_channels + mod2_channels, out_channels,
                                   kernel_size=3, padding=1, bias=False)
        # self.fusion_bottleneck_activation = nn.SiLU()    
        self.film_mod_fusion_bottleneck = FiLMModulation(out_channels, cond_dim)
        self.norm_fusion_bottleneck = nn.GroupNorm(ng_out, out_channels, affine=False)
        self.self_cbam_fusion = CBAM(out_channels, r=r)

        # Fusion ffn module
        self.norm_fusion_ffn = nn.GroupNorm(ng_out, out_channels, affine=False)
        self.fusion_ffn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1, bias=False),
        )
        # Initialize the last layer to zero as we are using a residual connection
        # nn.init.constant_(self.fusion_ffn[-1].weight, 0)
        
        self.film_mod_fusion_ffn = FiLMModulation(out_channels, cond_dim)
        # self.fusion_out_activation = nn.SiLU()
        # self.fusion_out_cbam = CBAM(out_channels, r=r)

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
        s1 = self.norm_self1(x1)
        g1, b1, a1 = self.film_mod_self1(s1,cond)
        s1 = (1 + g1) * s1 + b1
        # s1 = self.self_activation1(s1)
        s1 = a1 * self.self_cbam1(s1) + x1
        
        # Modality 2 Self Attention
        s2 = self.norm_self2(x2)
        g2, b2, a2 = self.film_mod_self2(s2,cond)
        s2 = (1 + g2) * s2 + b2
        # s2 = self.self_activation2(s2)
        s2 = a2 * self.self_cbam2(s2) + x2
        
        # Fusion
        concat = torch.cat([s1, s2], dim=1)
        fused_base = self.fusion_bottleneck(concat)
        f = self.norm_fusion_bottleneck(fused_base)
        g3, b3, a3 = self.film_mod_fusion_bottleneck(f, cond)
        f = (1 + g3) * f + b3
        # f = self.fusion_bottleneck_activation(f)
        f = a3 * self.self_cbam_fusion(f) + fused_base
        
        # Fusion Feedforward Network
        f_ffn = self.norm_fusion_ffn(f)
        g4, b4, a4 = self.film_mod_fusion_ffn(f_ffn, cond)
        f_ffn = (1 + g4) * f_ffn + b4
        f_ffn = a4 * self.fusion_ffn(f_ffn) + f
        
        return f_ffn


# =============================================================================
# Transform wrappers (drop-in for MultimodalDetr: concat input -> single output)
# =============================================================================

class CrossCBAMDiTTransformLayerV4(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 2,
        num_groups: int = 8,
    ):
        super().__init__()
        self.fusion = CrossCBAMDiTFusionV4(
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


class CrossCBAMDiTTransformQueriesV4(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 2,
        num_groups: int = 8,
    ):
        super().__init__()
        self.fusion = CrossCBAMDiTFusionV4(
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
    fusion = CrossCBAMDiTFusionV4(
        mod1_channels=C, mod2_channels=C, out_channels=C, cond_dim=cond_dim, r=2,
    )
    out = fusion(x1, x2, cond)
    print(f"CrossCBAMDiTFusionV4: ({x1.shape}, {x2.shape}) + cond -> {out.shape}")
    assert out.shape == (B, C, H, W)

    in_ch, out_ch = 512, 256
    x_cat = torch.randn(B, in_ch, H, W)
    layer = CrossCBAMDiTTransformLayerV4(in_ch, out_ch, cond_dim=cond_dim, r=2)
    out_layer = layer(x_cat, cond)
    print(f"CrossCBAMDiTTransformLayerV4: {x_cat.shape} -> {out_layer.shape}")
    assert out_layer.shape == (B, out_ch, H, W)

    queries = CrossCBAMDiTTransformQueriesV4(in_ch, out_ch, cond_dim=cond_dim, r=2)
    x_q = torch.randn(B, in_ch, 1, 1)
    out_q = queries(x_q, cond)
    print(f"CrossCBAMDiTTransformQueriesV4: {x_q.shape} -> {out_q.shape}")
    assert out_q.shape == (B, out_ch, 1, 1)

    total = sum(p.numel() for p in fusion.parameters())
    print(f"Params (mod=out={C}, cond={cond_dim}): {total:,} total")
    print("All CrossCBAM-DiT V4 modules passed.")
