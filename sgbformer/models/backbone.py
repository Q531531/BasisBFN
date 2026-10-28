"""Lightweight Restormer U-Net with spectral and semantic conditioning."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .conditioning import DegradationAwareCrossAttention
from .layers import Downsample, Upsample, make_transformer_stage
from .spectral import SpectralGate


class SpectralGatedBackbone(nn.Module):
    """Deterministic structural estimator from SGBformer Sections 4.1-4.2."""

    required_multiple = 8
    has_paper_spectral_gate = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        conv_bias: bool = False,
        norm_bias: bool = True,
        semantic_dim: int = 512,
        semantic_tokens: int = 4,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("num_blocks and heads must each contain four values")
        self.dim = dim
        self.decoder_channels = dim * 2
        self.input_projection = nn.Conv2d(
            in_channels, dim, kernel_size=3, padding=1, bias=conv_bias
        )

        stage = lambda channels, depth, num_heads: make_transformer_stage(
            channels,
            depth,
            num_heads,
            ffn_expansion_factor,
            conv_bias,
            norm_bias,
        )

        self.encoder1 = stage(dim, num_blocks[0], heads[0])
        self.dac1 = DegradationAwareCrossAttention(
            dim, semantic_dim, heads[0], semantic_tokens
        )
        self.down1 = Downsample(dim)

        self.encoder2 = stage(dim * 2, num_blocks[1], heads[1])
        self.dac2 = DegradationAwareCrossAttention(
            dim * 2, semantic_dim, heads[1], semantic_tokens
        )
        self.down2 = Downsample(dim * 2)

        self.encoder3 = stage(dim * 4, num_blocks[2], heads[2])
        self.dac3 = DegradationAwareCrossAttention(
            dim * 4, semantic_dim, heads[2], semantic_tokens
        )
        self.down3 = Downsample(dim * 4)

        self.bottleneck = stage(dim * 8, num_blocks[3], heads[3])
        self.spectral_gate = SpectralGate(dim * 8)

        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=conv_bias)
        self.decoder3 = stage(dim * 4, num_blocks[2], heads[2])

        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=conv_bias)
        self.decoder2 = stage(dim * 2, num_blocks[1], heads[1])

        self.up1 = Upsample(dim * 2)
        self.decoder1 = stage(dim * 2, num_blocks[0], heads[0])
        self.refinement = stage(dim * 2, refinement_blocks, heads[0])
        self.output = nn.Conv2d(
            dim * 2, out_channels, kernel_size=3, padding=1, bias=conv_bias
        )

    def forward(
        self,
        image: Tensor,
        semantic: Tensor,
        return_spectral_gate: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        encoder1 = self.dac1(self.encoder1(self.input_projection(image)), semantic)
        encoder2 = self.dac2(self.encoder2(self.down1(encoder1)), semantic)
        encoder3 = self.dac3(self.encoder3(self.down2(encoder2)), semantic)

        bottleneck = self.bottleneck(self.down3(encoder3))
        if return_spectral_gate:
            bottleneck, gate = self.spectral_gate(
                bottleneck, return_gate=True
            )
        else:
            bottleneck = self.spectral_gate(bottleneck)

        decoder3 = self.up3(bottleneck)
        decoder3 = self.decoder3(
            self.reduce3(torch.cat((decoder3, encoder3), dim=1))
        )
        decoder2 = self.up2(decoder3)
        decoder2 = self.decoder2(
            self.reduce2(torch.cat((decoder2, encoder2), dim=1))
        )
        decoder1 = self.up1(decoder2)
        decoder1 = self.decoder1(torch.cat((decoder1, encoder1), dim=1))
        decoder_feature = self.refinement(decoder1)
        structural = image + self.output(decoder_feature)

        if return_spectral_gate:
            return structural, decoder_feature, gate
        return structural, decoder_feature
