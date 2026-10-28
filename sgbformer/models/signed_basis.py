"""Deterministic signed spatial-basis residual readout."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SignedSpatialBasisReadout(nn.Module):
    """Mix learned RGB basis maps with bounded signed spatial coefficients."""

    def __init__(
        self,
        channels: int,
        residual_channels: int,
        *,
        num_bases: int = 4,
        kernel_size: int = 3,
        basis_init: str = "dct_orthogonal",
        basis_init_gain: float = 1.0,
        coefficient_activation: str = "tanh",
        normalization: str = "inverse_sqrt_num_bases",
        zero_init_coefficients: bool = True,
    ) -> None:
        super().__init__()
        if num_bases < 2:
            raise ValueError("Signed basis readout requires at least two bases")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Signed basis kernel_size must be positive and odd")
        if basis_init != "dct_orthogonal":
            raise ValueError(
                f"Unsupported signed basis initialization: {basis_init!r}"
            )
        if not math.isfinite(basis_init_gain) or basis_init_gain <= 0:
            raise ValueError("Signed basis initialization gain must be positive")
        if coefficient_activation != "tanh":
            raise ValueError(
                "Signed basis coefficient_activation must be 'tanh'"
            )
        if normalization != "inverse_sqrt_num_bases":
            raise ValueError(
                "Signed basis normalization must be "
                "'inverse_sqrt_num_bases'"
            )
        if not zero_init_coefficients:
            raise ValueError(
                "Signed basis coefficients must use zero initialization"
            )

        self.channels = channels
        self.residual_channels = residual_channels
        self.num_bases = num_bases
        self.kernel_size = kernel_size
        self.basis_init = basis_init
        self.basis_init_gain = basis_init_gain
        self.coefficient_activation = coefficient_activation
        self.normalization = normalization
        self.zero_init_coefficients = zero_init_coefficients
        self.output_scale = 1.0 / math.sqrt(num_bases)

        self.norm = nn.GroupNorm(_group_count(channels), channels)
        padding = kernel_size // 2
        self.basis_projection = nn.Conv2d(
            channels,
            num_bases * residual_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.coefficient_projection = nn.Conv2d(
            channels,
            num_bases,
            kernel_size=kernel_size,
            padding=padding,
        )
        self._initialize_basis_projection()
        nn.init.zeros_(self.coefficient_projection.weight)
        nn.init.zeros_(self.coefficient_projection.bias)

        self.last_basis_maps: Tensor | None = None
        self.last_coefficients: Tensor | None = None
        self.last_correction: Tensor | None = None

    def _initialize_basis_projection(self) -> None:
        """Fill convolution rows with an RNG-independent orthonormal DCT basis."""

        weight = self.basis_projection.weight
        rows = weight.shape[0]
        columns = weight[0].numel()
        if rows > columns:
            raise ValueError(
                "DCT basis initialization requires output rows <= fan-in"
            )
        position = torch.arange(
            columns,
            device=weight.device,
            dtype=weight.dtype,
        ) + 0.5
        frequency = torch.arange(
            rows,
            device=weight.device,
            dtype=weight.dtype,
        )
        matrix = torch.cos(
            math.pi * frequency[:, None] * position[None, :] / columns
        )
        matrix[0].mul_(math.sqrt(1.0 / columns))
        if rows > 1:
            matrix[1:].mul_(math.sqrt(2.0 / columns))
        with torch.no_grad():
            weight.copy_(
                matrix.reshape_as(weight) * self.basis_init_gain
            )
            nn.init.zeros_(self.basis_projection.bias)

    @property
    def weight(self) -> Tensor:
        """Expose both projections for existing training diagnostics."""

        return torch.cat(
            (
                self.basis_projection.weight.flatten(),
                self.coefficient_projection.weight.flatten(),
            )
        )

    @property
    def bias(self) -> Tensor:
        return torch.cat(
            (
                self.basis_projection.bias.flatten(),
                self.coefficient_projection.bias.flatten(),
            )
        )

    def forward(self, feature: Tensor) -> Tensor:
        hidden = F.silu(self.norm(feature))
        batch, _, height, width = hidden.shape
        basis = self.basis_projection(hidden).reshape(
            batch,
            self.num_bases,
            self.residual_channels,
            height,
            width,
        )
        coefficients = self.coefficient_projection(hidden).tanh()
        correction = torch.einsum(
            "bkrhw,bkhw->brhw",
            basis,
            coefficients,
        ) * self.output_scale
        self.last_basis_maps = basis.detach()
        self.last_coefficients = coefficients.detach()
        self.last_correction = correction.detach()
        return correction
