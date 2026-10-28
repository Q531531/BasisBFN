"""Condition-rich deterministic residual mean correction."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class MeanCorrectionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, feature: Tensor) -> Tensor:
        residual = self.norm(feature)
        residual = self.depthwise(F.silu(residual))
        residual = self.project(F.silu(residual))
        return feature + residual


class ConditionalMeanCorrector(nn.Module):
    """Predict E[x - b | y, b, y-b, F] without changing the fixed base."""

    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int = 32,
        blocks: int = 3,
        correction_scale: float = 0.1,
        detach_conditions: bool = True,
    ) -> None:
        super().__init__()
        if feature_channels < 1 or hidden_channels < 1 or blocks < 1:
            raise ValueError("Mean-corrector dimensions must be positive")
        if correction_scale <= 0:
            raise ValueError("correction_scale must be positive")
        self.image_projection = nn.Conv2d(9, hidden_channels, 3, padding=1)
        self.feature_projection = nn.Conv2d(
            feature_channels,
            hidden_channels,
            kernel_size=1,
        )
        self.fusion = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels,
            kernel_size=1,
        )
        self.blocks = nn.Sequential(
            *(MeanCorrectionBlock(hidden_channels) for _ in range(blocks))
        )
        self.output = nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.correction_scale = float(correction_scale)
        self.detach_conditions = bool(detach_conditions)
        self.condition_channels = hidden_channels

    def forward(
        self,
        degraded: Tensor,
        structural: Tensor,
        decoder_feature: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if decoder_feature.shape[-2:] != structural.shape[-2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=structural.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if self.detach_conditions:
            degraded = degraded.detach()
            structural = structural.detach()
            decoder_feature = decoder_feature.detach()
        image_condition = torch.cat(
            (degraded, structural, degraded - structural),
            dim=1,
        )
        image_feature = F.silu(self.image_projection(image_condition))
        backbone_feature = F.silu(
            self.feature_projection(decoder_feature)
        )
        condition = F.silu(
            self.fusion(torch.cat((image_feature, backbone_feature), dim=1))
        )
        condition = self.blocks(condition)
        correction = self.correction_scale * self.output(F.silu(condition))
        return correction, condition
