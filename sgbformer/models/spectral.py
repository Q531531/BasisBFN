"""Frequency-domain soft gating from SGBformer Equation (5)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def spectral_scale_delta(feature: Tensor, scale_delta: Tensor) -> Tensor:
    """Apply only a real spectral-scale delta, preserving the base path."""

    spectrum = torch.fft.rfft2(feature.float(), norm="ortho")
    if scale_delta.shape != spectrum.shape:
        raise ValueError(
            "scale_delta must match the real FFT shape "
            f"{tuple(spectrum.shape)}, received {tuple(scale_delta.shape)}"
        )
    return torch.fft.irfft2(
        spectrum * scale_delta.float(),
        s=feature.shape[-2:],
        norm="ortho",
    ).to(feature.dtype)


class SpectralGate(nn.Module):
    """Modulate FFT amplitudes with a learned real-valued gate.

    Multiplying a complex spectrum by a positive real gate changes amplitude
    while preserving phase. ``rfft2`` is used because the input is real.
    """

    def __init__(
        self,
        channels: int,
        bias: bool = True,
        initial_gate: float | None = None,
    ) -> None:
        super().__init__()
        if initial_gate is not None and not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie in (0, 1)")
        if initial_gate is not None and not bias:
            raise ValueError("initial_gate requires a learnable bias")
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=bias)
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            gate_bias = (
                2.0
                if initial_gate is None
                else math.log(initial_gate / (1.0 - initial_gate))
            )
            nn.init.constant_(self.gate.bias, gate_bias)

    def forward(
        self, x: Tensor, return_gate: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        height, width = x.shape[-2:]
        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        real_imag = torch.cat((spectrum.real, spectrum.imag), dim=1)
        gate = torch.sigmoid(self.gate(real_imag))
        filtered = torch.fft.irfft2(
            spectrum * gate, s=(height, width), norm="ortho"
        ).to(dtype=x.dtype)
        if return_gate:
            return filtered, gate
        return filtered
