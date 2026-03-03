import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _num_groups(channels: int, desired: int = 8) -> int:
    """Largest divisor of *channels* that is <= *desired*."""
    for g in range(min(desired, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ====================================================================
# VLCA  –  Vision-Language Conditioned Attention  (self-attention)
# ====================================================================

class VLCA(nn.Module):
    """DiT-style adaptive normalization + CBAM-style channel/spatial attention.

    A single condition projection produces **7 modulation vectors** that gate
    the pooling, bias the attention maps, and control residual strength –
    making the attention computation itself condition-aware (unlike
    CBAM+FiLM which scales the output post-hoc).

    Modulation vectors:
        scale_ca, shift_ca, gate_ca   – channel-attention modulation
        scale_sa, shift_sa, gate_sa   – spatial-attention modulation
        residual_scale                – learnable residual strength
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        r: int = 4,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        bottleneck = max(1, channels // r)
        ng = _num_groups(channels, num_groups)
        ng_bn = _num_groups(bottleneck, num_groups)

        # Single projection for all modulation params (DiT-style adaLN)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, channels * 7),
        )
        # Small-normal init (avoids the zero-gradient stall seen with zero-init)
        nn.init.normal_(self.cond_proj[-1].weight, std=0.01)
        nn.init.zeros_(self.cond_proj[-1].bias)

        # Channel attention MLP (CBAM-style but condition-gated)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1, bias=False),
            nn.GroupNorm(ng_bn, bottleneck),
            nn.SiLU(),
            nn.Conv2d(bottleneck, channels, 1, bias=False),
        )

        # Spatial attention conv
        self.sa_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(1, 1),
        )

        # SwiGLU-style feature gating
        self.feature_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(ng, channels)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [B, C, H, W]
            conditions: [B, cond_dim]
        """
        B, C, H, W = x.shape
        identity = x

        # --- project conditions to 7 modulation vectors ---
        params = self.cond_proj(conditions).view(B, 7, C, 1, 1)
        scale_ca, shift_ca, gate_ca = params[:, 0], params[:, 1], params[:, 2]
        scale_sa, shift_sa, gate_sa = params[:, 3], params[:, 4], params[:, 5]
        residual_scale = params[:, 6]

        x_norm = self.norm(x)

        # --- conditioned channel attention ---
        avg_pool = F.adaptive_avg_pool2d(x_norm, 1)
        max_pool = F.adaptive_max_pool2d(x_norm, 1)
        pooled = (avg_pool + max_pool) * (1 + scale_ca) + shift_ca
        ca_weights = torch.sigmoid(self.ca_mlp(pooled)) * (1 + gate_ca)
        x_ca = self.dropout(x_norm * ca_weights)

        # --- conditioned spatial attention ---
        x_reweighted = x_ca * (1 + scale_sa) + shift_sa
        avg_out = torch.mean(x_reweighted, dim=1, keepdim=True)
        max_out, _ = torch.max(x_reweighted, dim=1, keepdim=True)
        sa_weights = torch.sigmoid(
            self.sa_conv(torch.cat([avg_out, max_out], dim=1))
        ) * (1 + gate_sa)
        x_sa = self.dropout(x_ca * sa_weights)

        # --- feature gating ---
        x_gated = x_sa * self.feature_gate(x_sa)

        # --- residual with learnable scale ---
        return identity + x_gated * (1 + residual_scale)


class VLCABlock(nn.Module):
    """VLCA + condition-modulated feed-forward network.

    Mirrors a Transformer block (attention → FFN with residual) but uses
    VLCA attention on spatial feature maps.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        r: int = 4,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        mlp_channels = int(channels * mlp_ratio)
        ng = _num_groups(channels, num_groups)
        ng_mlp = _num_groups(mlp_channels, num_groups)

        self.vlca = VLCA(channels, cond_dim, r, num_groups, dropout)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, mlp_channels, 1, bias=False),
            nn.GroupNorm(ng_mlp, mlp_channels),
            nn.SiLU(),
            nn.Conv2d(mlp_channels, channels, 1, bias=False),
            nn.GroupNorm(ng, channels),
        )

        # DiT-style: modulate BEFORE the FFN transform
        self.ffn_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 2),
        )
        nn.init.normal_(self.ffn_cond[-1].weight, std=0.01)
        nn.init.zeros_(self.ffn_cond[-1].bias)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        x = self.vlca(x, conditions)
        identity = x

        ffn_params = self.ffn_cond(conditions).view(x.shape[0], 2, x.shape[1], 1, 1)
        ffn_scale, ffn_shift = ffn_params[:, 0], ffn_params[:, 1]
        x_mod = x * (1 + ffn_scale) + ffn_shift

        return identity + self.ffn(x_mod)


# ====================================================================
# Cross-Attention Components
# ====================================================================

class ConditionalCrossChannelAttention(nn.Module):
    """Cross-channel attention conditioned on environment.

    Uses one modality's **global context** to weight the other modality's
    channels.  Efficiency: O(C) via global pooling (not O(HW) like
    transformer cross-attention).
    """

    def __init__(self, channels: int, cond_dim: int, r: int = 4):
        super().__init__()
        bottleneck = max(1, channels // r)

        self.cross_mlp = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, channels),
        )

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, channels),
        )
        nn.init.normal_(self.cond_proj[-1].weight, std=0.01)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(
        self,
        x_self: torch.Tensor,
        x_other: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_self:  [B, C, H, W]  modality to be re-weighted
        x_other: [B, C, H, W]  modality providing context
        """
        avg_pool = F.adaptive_avg_pool2d(x_other, 1).squeeze(-1).squeeze(-1)
        max_pool = F.adaptive_max_pool2d(x_other, 1).squeeze(-1).squeeze(-1)

        cross_weights = self.cross_mlp(avg_pool) + self.cross_mlp(max_pool)
        cond_bias = self.cond_proj(conditions)

        attn = torch.sigmoid(cross_weights + cond_bias).unsqueeze(-1).unsqueeze(-1)
        return x_self * attn


class ConditionalModalityGate(nn.Module):
    """Per-channel gating controlled by conditions.

    Can suppress an entire modality when conditions warrant it
    (e.g.  Night → suppress RGB, rely on IR).
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(conditions).view(x.shape[0], x.shape[1], 1, 1)
        return x * gate


class VLCACrossFusion(nn.Module):
    """Bidirectional cross-attention fusion with condition-based modality gating.

    Pipeline:
        1. Gate modalities based on conditions
        2. Bidirectional cross-channel attention (Mod1 ← Mod2 and Mod2 ← Mod1)
        3. Concatenate + project
        4. Refine with a VLCABlock (self-attention)
    """

    def __init__(
        self,
        mod_channels: int,
        out_channels: int,
        cond_dim: int,
        r: int = 4,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        ng_out = _num_groups(out_channels, num_groups)

        # Modality gating
        self.gate_mod1 = ConditionalModalityGate(mod_channels, cond_dim)
        self.gate_mod2 = ConditionalModalityGate(mod_channels, cond_dim)

        # Bidirectional cross-attention
        self.cross_attn_1 = ConditionalCrossChannelAttention(mod_channels, cond_dim, r)
        self.cross_attn_2 = ConditionalCrossChannelAttention(mod_channels, cond_dim, r)

        # Fuse projection
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(mod_channels * 2, out_channels, 1, bias=False),
            nn.GroupNorm(ng_out, out_channels),
            nn.SiLU(),
        )

        # Post-fusion refinement
        self.refine = VLCABlock(
            out_channels, cond_dim,
            mlp_ratio=2.0, r=r, num_groups=num_groups, dropout=dropout,
        )

    def forward(
        self,
        mod1: torch.Tensor,
        mod2: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        if mod1.shape[-2:] != mod2.shape[-2:]:
            mod2 = F.interpolate(
                mod2, size=mod1.shape[-2:], mode="bilinear", align_corners=False,
            )

        mod1_g = self.gate_mod1(mod1, conditions)
        mod2_g = self.gate_mod2(mod2, conditions)

        mod1_cross = self.cross_attn_1(mod1_g, mod2_g, conditions)
        mod2_cross = self.cross_attn_2(mod2_g, mod1_g, conditions)

        fused = self.fuse_conv(torch.cat([mod1_cross, mod2_cross], dim=1))
        return self.refine(fused, conditions)


# ====================================================================
# Transform Layers  (drop-in interface for MultimodalDetr)
# ====================================================================

class VLCATransformLayer(nn.Module):
    """Self-attention VLCA fusion for concatenated feature maps.

    Applies VLCA at the full concatenated channel dimension, then projects
    down to *out_channels*.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 4,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        ng_out = _num_groups(out_channels, num_groups)

        self.vlca = VLCA(in_channels, cond_dim, r, num_groups, dropout)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(ng_out, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(ng_out, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        return self.proj(self.vlca(x, conditions))


class VLCATransformQueries(nn.Module):
    """Self-attention VLCA fusion for concatenated object queries."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 4,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        ng_out = _num_groups(out_channels, num_groups)

        self.vlca = VLCA(in_channels, cond_dim, r, num_groups, dropout)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(ng_out, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(ng_out, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        return self.proj(self.vlca(x, conditions))


class VLCACrossTransformLayer(nn.Module):
    """Cross-attention VLCA fusion for concatenated feature maps.

    Splits the concatenated input into two equal-channel modalities, applies
    bidirectional cross-attention with condition-based gating, and projects
    to *out_channels*.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 4,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        mod_channels = in_channels // 2
        self.fusion = VLCACrossFusion(
            mod_channels, out_channels, cond_dim, r, num_groups, dropout,
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 2
        return self.fusion(x[:, :c], x[:, c:], conditions)


class VLCACrossTransformQueries(nn.Module):
    """Cross-attention VLCA fusion for concatenated object queries."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int = 14,
        r: int = 4,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        mod_channels = in_channels // 2
        self.fusion = VLCACrossFusion(
            mod_channels, out_channels, cond_dim, r, num_groups, dropout,
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 2
        return self.fusion(x[:, :c], x[:, c:], conditions)


# ====================================================================
# Smoke test
# ====================================================================

if __name__ == "__main__":
    B, C, H, W = 2, 256, 15, 15
    cond_dim = 7
    x = torch.randn(B, C, H, W)
    cond = torch.randn(B, cond_dim)

    vlca = VLCA(C, cond_dim)
    out = vlca(x, cond)
    print(f"VLCA:          {x.shape} -> {out.shape}")
    assert out.shape == x.shape

    block = VLCABlock(C, cond_dim)
    out = block(x, cond)
    print(f"VLCABlock:     {x.shape} -> {out.shape}")
    assert out.shape == x.shape

    in_ch, out_ch = 512, 256
    x_cat = torch.randn(B, in_ch, H, W)

    layer = VLCATransformLayer(in_ch, out_ch, cond_dim)
    out = layer(x_cat, cond)
    print(f"VLCATransformLayer:       {x_cat.shape} -> {out.shape}")
    assert out.shape == (B, out_ch, H, W)

    queries = VLCATransformQueries(in_ch, out_ch, cond_dim)
    out = queries(x_cat, cond)
    print(f"VLCATransformQueries:     {x_cat.shape} -> {out.shape}")
    assert out.shape == (B, out_ch, H, W)

    cross_layer = VLCACrossTransformLayer(in_ch, out_ch, cond_dim)
    out = cross_layer(x_cat, cond)
    print(f"VLCACrossTransformLayer:  {x_cat.shape} -> {out.shape}")
    assert out.shape == (B, out_ch, H, W)

    cross_queries = VLCACrossTransformQueries(in_ch, out_ch, cond_dim)
    out = cross_queries(x_cat, cond)
    print(f"VLCACrossTransformQueries:{x_cat.shape} -> {out.shape}")
    assert out.shape == (B, out_ch, H, W)

    total = sum(p.numel() for p in vlca.parameters())
    print(f"\nVLCA param count (C={C}): {total:,}")
    total_cross = sum(p.numel() for p in cross_layer.parameters())
    print(f"VLCACrossTransformLayer param count: {total_cross:,}")

    print("\nAll VLCA modules passed.")
