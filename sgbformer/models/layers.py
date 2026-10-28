"""Restormer building blocks used by the spectral backbone."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Layer normalization over channels at every spatial location."""

    def __init__(self, channels: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True) if self.bias is not None else 0.0
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + 1e-5)
        x = x * self.weight[None, :, None, None]
        if self.bias is not None:
            x = x + self.bias[None, :, None, None]
        return x


class GatedDconvFeedForward(nn.Module):
    """Gated depth-wise convolutional feed-forward network (GDFN)."""

    def __init__(
        self,
        channels: int,
        expansion_factor: float = 2.66,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(
            channels, hidden_channels * 2, kernel_size=1, bias=bias
        )
        self.depthwise = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(value) * gate)


class MultiDconvHeadTransposedAttention(nn.Module):
    """Restormer's channel-wise MDTA attention."""

    def __init__(self, channels: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError(
                f"channels ({channels}) must be divisible by heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_depthwise = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        head_channels = channels // self.num_heads
        q, k, v = self.qkv_depthwise(self.qkv(x)).chunk(3, dim=1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                batch, self.num_heads, head_channels, height * width
            )

        q, k, v = map(split_heads, (q, k, v))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        output = (attention @ v).reshape(batch, channels, height, width)
        return self.project_out(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        expansion_factor: float = 2.66,
        conv_bias: bool = False,
        norm_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels, bias=norm_bias)
        self.attention = MultiDconvHeadTransposedAttention(
            channels, num_heads, bias=conv_bias
        )
        self.norm2 = LayerNorm2d(channels, bias=norm_bias)
        self.feed_forward = GatedDconvFeedForward(
            channels, expansion_factor=expansion_factor, bias=conv_bias
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("Downsample input channels must be even")
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


def make_transformer_stage(
    channels: int,
    depth: int,
    heads: int,
    expansion_factor: float,
    conv_bias: bool,
    norm_bias: bool,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            TransformerBlock(
                channels,
                heads,
                expansion_factor=expansion_factor,
                conv_bias=conv_bias,
                norm_bias=norm_bias,
            )
            for _ in range(depth)
        ]
    )
