import torch, torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


class CGB(nn.Module):
    """Contextual Gated Bottleneck (CGB).

    Injects the condition vector directly INTO the attention computation
    rather than applying it as post-hoc modulation.

    How this differs from CBAM + FiLM:
      - CBAM + FiLM:  attention weights are condition-agnostic (CBAM runs
        on visual features alone), then FiLM applies an affine transform
        to the output.  The two mechanisms are independent.
      - CGB:  the condition generates a channel prior that gates the pooled
        features BEFORE the channel-attention MLP, producing attention
        weights that are themselves condition-dependent.  Similarly, the
        condition reweights channels before spatial pooling, controlling
        which features shape the spatial attention map.

    The condition is inside the attention loop, not outside it.
    """

    def __init__(self, channels, cond_dim, r=4):
        super().__init__()
        bottleneck = max(1, channels // r)

        # Condition -> channel-attention prior
        self.cond_channel = nn.Sequential(
            nn.Linear(cond_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, channels),
        )

        # Condition -> spatial-attention prior
        self.cond_spatial = nn.Sequential(
            nn.Linear(cond_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, channels),
        )

        # Channel attention MLP (SE-style bottleneck)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False),
        )

        # Spatial attention conv (7x7 like CBAM)
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x, cond):
        b, c, h, w = x.size()
        identity = x

        # --- Contextual Channel Attention ---
        # The condition prior gates the pooled features so the MLP sees a
        # condition-biased summary, producing different attention weights
        # under different environmental conditions.
        ca_prior = torch.sigmoid(self.cond_channel(cond)).view(b, c, 1, 1)
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        ca_weight = torch.sigmoid(self.ca_mlp((avg_pool + max_pool) * ca_prior))
        x = x * ca_weight

        # --- Contextual Spatial Attention ---
        # The condition reweights channels before the max/avg spatial pooling,
        # controlling which feature channels contribute to the spatial map.
        sa_prior = torch.sigmoid(self.cond_spatial(cond)).view(b, c, 1, 1)
        x_reweighted = x * sa_prior
        avg_out = torch.mean(x_reweighted, dim=1, keepdim=True)
        max_out, _ = torch.max(x_reweighted, dim=1, keepdim=True)
        sa_weight = torch.sigmoid(self.sa_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * sa_weight

        return self.bn(x + identity)


class CGBTransformLayer(nn.Module):
    """Feature-map fusion layer: CGB attention followed by 3x Conv-BN-ReLU.

    Deliberately mirrors CBAMFiLMTransformLayer's structure so the ONLY
    change is the attention mechanism itself (CGB vs CBAM+FiLM).
    """

    def __init__(self, in_channels, out_channels, cond_dim=14, r=4):
        super().__init__()
        self.cgb = CGB(in_channels, cond_dim, r)
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
        x = self.cgb(x, conditions)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x


class CGBTransformQueries(nn.Module):
    """Object-query fusion layer: same architecture as CGBTransformLayer."""

    def __init__(self, in_channels, out_channels, cond_dim=14, r=4):
        super().__init__()
        self.cgb = CGB(in_channels, cond_dim, r)
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
        x = self.cgb(x, conditions)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x


def __main__():
    b, c, h, w = 2, 256, 15, 15
    cond_dim = 7
    x = torch.randn(b, c, h, w)
    cond = torch.randn(b, cond_dim)

    cgb = CGB(channels=c, cond_dim=cond_dim, r=4)
    out = cgb(x, cond)
    print("CGB output shape:", out.shape)
    assert out.shape == x.shape

    in_ch, out_ch = 512, 256
    x2 = torch.randn(b, in_ch, h, w)

    layer = CGBTransformLayer(in_ch, out_ch, cond_dim=cond_dim, r=4)
    out2 = layer(x2, cond)
    print("CGBTransformLayer output shape:", out2.shape)
    assert out2.shape == (b, out_ch, h, w)

    queries = CGBTransformQueries(in_ch, out_ch, cond_dim=cond_dim, r=4)
    out3 = queries(x2, cond)
    print("CGBTransformQueries output shape:", out3.shape)
    assert out3.shape == (b, out_ch, h, w)

    print("All checks passed.")


if __name__ == "__main__":
    __main__()
