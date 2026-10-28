"""Deterministic feature corrections with separately calibrated image risk."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spectral import SpectralGate, spectral_scale_delta


class PaperSpectralGateAdapter(nn.Module):
    """Equation (5) spectral gate for an external foundation bottleneck."""

    is_paper_spectral_gate = True

    def __init__(
        self,
        channels: int,
        initial_gate: float = 0.999,
    ) -> None:
        super().__init__()
        self.spectral_gate = SpectralGate(
            channels,
            initial_gate=initial_gate,
        )
        self.last_regularization: Tensor | None = None

    def forward(self, feature: Tensor) -> Tensor:
        filtered = self.spectral_gate(feature)
        self.last_regularization = (
            filtered.float() - feature.float()
        ).square().mean()
        return filtered


class StateConditionedSpectralAdapter(nn.Module):
    """Zero-initialized decoder adapter conditioned on a BFN belief state.

    The adapter is attached to one frozen SSR decoder module.  Without a
    runtime state it returns the feature exactly, which makes the base
    restoration available without a second model copy.
    """

    requires_runtime_state = True

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if channels < 1 or hidden_channels < 1:
            raise ValueError("Adapter channel counts must be positive")
        groups = next(
            group for group in (32, 16, 8, 4, 2, 1) if channels % group == 0
        )
        self.norm = nn.GroupNorm(groups, channels)
        self.feature_projection = nn.Conv2d(channels, hidden_channels, 1)
        self.observation_projection = nn.Conv2d(9, hidden_channels, 1)
        self.belief_projection = nn.Conv2d(3, hidden_channels, 1)
        self.precision_projection = nn.Conv2d(1, hidden_channels, 1)
        self.fusion = nn.Conv2d(hidden_channels * 4, hidden_channels, 1)
        self.local = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            3,
            padding=1,
            groups=hidden_channels,
        )
        self.spectral_gate = nn.Conv2d(
            hidden_channels * 2, hidden_channels, 1
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 2),
        )
        self.output = nn.Conv2d(hidden_channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_regularization: Tensor | None = None
        self.last_state_delta: Tensor | None = None

    @staticmethod
    def _resize(value: Tensor, size: tuple[int, int]) -> Tensor:
        if value.shape[-2:] == size:
            return value
        return F.interpolate(
            value,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        feature: Tensor,
        state: Mapping[str, Tensor | str] | None = None,
    ) -> Tensor:
        if state is None:
            self.last_regularization = feature.new_zeros(())
            self.last_state_delta = feature.new_zeros(())
            return feature

        required = ("degraded", "baseline", "belief_mean", "precision", "time")
        missing = [name for name in required if name not in state]
        if missing:
            raise ValueError(
                "State-conditioned adapter is missing: " + ", ".join(missing)
            )
        degraded = state["degraded"]
        baseline = state["baseline"]
        belief_mean = state["belief_mean"]
        precision = state["precision"]
        time = state["time"]
        if not all(
            isinstance(value, Tensor)
            for value in (
                degraded,
                baseline,
                belief_mean,
                precision,
                time,
            )
        ):
            raise TypeError("State-conditioned adapter expects tensor state values")
        if state.get("mode", "live") == "neutral":
            belief_mean = torch.zeros_like(belief_mean)
            precision = torch.ones_like(precision)
            time = torch.zeros_like(time)

        size = feature.shape[-2:]
        observation = torch.cat(
            (
                self._resize(degraded, size),
                self._resize(baseline, size),
                self._resize(degraded - baseline, size),
            ),
            dim=1,
        )
        belief = self._resize(belief_mean, size)
        log_precision = self._resize(precision.clamp_min(1e-6).log(), size)
        hidden = torch.cat(
            (
                F.silu(self.feature_projection(self.norm(feature))),
                F.silu(self.observation_projection(observation)),
                F.silu(self.belief_projection(belief)),
                F.silu(self.precision_projection(log_precision)),
            ),
            dim=1,
        )
        hidden = F.silu(self.fusion(hidden))
        scale, shift = self.time_embedding(time.float().view(-1, 1)).chunk(
            2, dim=1
        )
        hidden = hidden * (scale[:, :, None, None] + 1.0) + shift[
            :, :, None, None
        ]
        hidden = F.silu(self.local(hidden))
        spectrum = torch.fft.rfft2(hidden.float(), norm="ortho")
        descriptor = torch.cat((spectrum.real, spectrum.imag), dim=1)
        gate = torch.tanh(self.spectral_gate(descriptor.to(hidden.dtype))).float()
        evidence = torch.fft.irfft2(
            spectrum * gate,
            s=hidden.shape[-2:],
            norm="ortho",
        ).to(hidden.dtype)
        delta = self.output(F.silu(hidden + evidence))
        self.last_regularization = delta.float().square().mean()
        self.last_state_delta = delta.float().square().mean().sqrt()
        return feature + delta


class LatentConditionedFiLMSpectralAdapter(nn.Module):
    """Identity-initialized FiLM and spectral modulation from a latent belief."""

    requires_runtime_state = True
    requires_latent_state = True

    def __init__(
        self,
        channels: int,
        global_dim: int = 32,
        spatial_dim: int = 8,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if min(channels, global_dim, spatial_dim, hidden_channels) < 1:
            raise ValueError("Adapter dimensions must be positive")
        self.channels = int(channels)
        self.global_dim = int(global_dim)
        self.spatial_dim = int(spatial_dim)
        self.global_projection = nn.Sequential(
            nn.Linear(global_dim * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(spatial_dim * 2, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
            ),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1),
            nn.SiLU(),
        )
        self.gamma_projection = nn.Conv2d(hidden_channels, channels, 1)
        self.beta_projection = nn.Conv2d(hidden_channels, channels, 1)
        self.spectral_projection = nn.Conv2d(hidden_channels, channels, 1)
        self.fusion_gain = nn.Parameter(torch.ones(()))
        for projection in (
            self.gamma_projection,
            self.beta_projection,
            self.spectral_projection,
        ):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.last_regularization: Tensor | None = None
        self.last_gamma_delta: Tensor | None = None
        self.last_beta: Tensor | None = None
        self.last_spectral_delta: Tensor | None = None

    @staticmethod
    def _require_tensor(
        state: Mapping[str, Tensor | str],
        name: str,
    ) -> Tensor:
        value = state.get(name)
        if not isinstance(value, Tensor):
            raise ValueError(f"Latent adapter requires tensor state '{name}'")
        return value

    def forward(
        self,
        feature: Tensor,
        state: Mapping[str, Tensor | str] | None = None,
    ) -> Tensor:
        if state is None:
            zero = feature.new_zeros(())
            self.last_regularization = zero
            self.last_gamma_delta = zero
            self.last_beta = zero
            self.last_spectral_delta = zero
            return feature

        global_mean = self._require_tensor(state, "global_mean")
        global_precision = self._require_tensor(state, "global_precision")
        spatial_mean = self._require_tensor(state, "spatial_mean")
        spatial_precision = self._require_tensor(state, "spatial_precision")
        if global_mean.shape[1] != self.global_dim:
            raise ValueError("Latent global dimension does not match the adapter")
        if spatial_mean.shape[1] != self.spatial_dim:
            raise ValueError("Latent spatial dimension does not match the adapter")

        global_hidden = self.global_projection(
            torch.cat(
                (
                    global_mean,
                    global_precision.clamp_min(1e-6).log(),
                ),
                dim=1,
            )
        )
        spatial_hidden = self.spatial_projection(
            torch.cat(
                (
                    spatial_mean,
                    spatial_precision.clamp_min(1e-6).log(),
                ),
                dim=1,
            )
        )
        spatial_hidden = F.interpolate(
            spatial_hidden,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        global_hidden = global_hidden[:, :, None, None].expand(
            -1,
            -1,
            *feature.shape[-2:],
        )
        hidden = self.fusion(torch.cat((global_hidden, spatial_hidden), dim=1))

        gamma_delta = self.gamma_projection(hidden)
        beta = self.beta_projection(hidden)
        spectral_delta_scale = self.spectral_projection(hidden)
        spectral_delta_scale = F.adaptive_avg_pool2d(
            spectral_delta_scale.float(),
            (
                feature.shape[-2],
                feature.shape[-1] // 2 + 1,
            ),
        )
        spectral_delta = spectral_scale_delta(
            feature,
            spectral_delta_scale,
        )
        correction = (
            gamma_delta * feature
            + beta
            + spectral_delta
        )
        output = feature + self.fusion_gain.to(correction.dtype) * correction
        self.last_gamma_delta = gamma_delta.float().abs().mean()
        self.last_beta = beta.float().abs().mean()
        self.last_spectral_delta = spectral_delta.float().abs().mean()
        self.last_regularization = (
            gamma_delta.float().square().mean()
            + beta.float().square().mean()
            + spectral_delta.float().square().mean()
        )
        return output


class CalibratedSpectralAdapter(nn.Module):
    """Neutral spectral/spatial adapter; variance estimates RGB image MSE.

    This is not a BFN or a posterior over feature corrections. The risk head
    is trained only by its calibration score, not by reconstruction gradients.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        residual_scale: float = 0.1,
        use_spectral: bool = True,
        uncertainty_mode: str = "gated",
        reference_error: float = 0.05,
        confidence_strength: float = 0.5,
        risk_log_ratio_mode: str = "clamp",
        spectral_reference_size: int | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_channels if hidden_channels is not None else max(channels // 4, 16)
        if channels < 1 or hidden < 1:
            raise ValueError("Adapter channel counts must be positive")
        if not 0.0 < residual_scale < 1.0:
            raise ValueError("residual_scale must lie in (0, 1)")
        if uncertainty_mode not in {"none", "auxiliary", "gated", "uniform"}:
            raise ValueError("Unsupported uncertainty_mode")
        if reference_error <= 0 or not math.isfinite(reference_error):
            raise ValueError("reference_error must be finite and positive")
        if not 0.0 <= confidence_strength < 1.0:
            raise ValueError("confidence_strength must lie in [0, 1)")
        if risk_log_ratio_mode not in {"clamp", "smooth"}:
            raise ValueError("risk_log_ratio_mode must be clamp or smooth")
        if spectral_reference_size is not None and (
            isinstance(spectral_reference_size, bool)
            or not isinstance(spectral_reference_size, int)
            or spectral_reference_size < 1
        ):
            raise ValueError("spectral_reference_size must be a positive integer")
        groups = next(g for g in (32, 16, 8, 4, 2, 1) if channels % g == 0)
        self.norm = nn.GroupNorm(groups, channels)
        self.project_in = nn.Conv2d(channels, hidden, 1)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        # This layer stays active in the spatial control, matching capacity.
        self.evidence_gate = nn.Conv2d(hidden, hidden, 1)
        self.mean_head = nn.Conv2d(hidden, channels, 1)
        self.risk_head = nn.Conv2d(hidden, 1, 1)
        self.gain_logit = nn.Parameter(
            torch.tensor(math.log(residual_scale / (1.0 - residual_scale)))
        )
        self.use_spectral = bool(use_spectral)
        self.uncertainty_mode = uncertainty_mode
        self.reference_variance = float(reference_error) ** 2
        self.confidence_strength = float(confidence_strength)
        self.risk_log_ratio_mode = risk_log_ratio_mode
        self.spectral_reference_size = spectral_reference_size
        for head in (self.evidence_gate, self.mean_head, self.risk_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.risk_head.requires_grad_(uncertainty_mode != "none")
        self.last_uncertainty: Tensor | None = None
        self.last_regularization: Tensor | None = None

    def spectral_evidence(self, hidden: Tensor) -> Tensor:
        spectrum = torch.fft.rfft2(hidden.float(), norm="ortho")
        amplitude = spectrum.abs()
        if self.spectral_reference_size is not None:
            # Normalize only the descriptor, keeping the reconstruction FFT intact.
            height, width = hidden.shape[-2:]
            amplitude = amplitude * (
                self.spectral_reference_size / math.sqrt(height * width)
            )
        gate = torch.tanh(
            self.evidence_gate(amplitude.log1p().to(hidden.dtype))
        ).float()
        return torch.fft.irfft2(
            spectrum * gate, s=hidden.shape[-2:], norm="ortho"
        ).to(hidden.dtype)

    def forward(self, feature: Tensor) -> Tensor:
        hidden = F.silu(self.local(self.project_in(self.norm(feature))))
        if self.use_spectral:
            evidence = self.spectral_evidence(hidden)
        else:
            evidence = hidden * torch.tanh(self.evidence_gate(hidden))
        fused = hidden + evidence
        mean = self.mean_head(F.silu(fused))
        self.last_uncertainty = None
        confidence: Tensor | float = 1.0
        if self.uncertainty_mode != "none":
            raw_ratio = self.risk_head(fused.detach()).float()
            log_ratio = (
                6.0 * torch.tanh(raw_ratio / 6.0)
                if self.risk_log_ratio_mode == "smooth"
                else raw_ratio.clamp(-6.0, 6.0)
            )
            variance = self.reference_variance * log_ratio.exp()
            self.last_uncertainty = variance
            if self.uncertainty_mode in {"gated", "uniform"}:
                gate_ratio = log_ratio.detach()
                if self.uncertainty_mode == "uniform":
                    gate_ratio = (
                        variance.detach().mean(dim=(-2, -1), keepdim=True)
                        / self.reference_variance
                    ).log()
                # Bounded, initially one, and never able to erase the update.
                confidence = (
                    1.0 - self.confidence_strength * torch.tanh(gate_ratio / 2.0)
                ).to(mean.dtype)
        correction = self.gain_logit.sigmoid().to(mean.dtype) * confidence * mean
        # Regularize the actual correction, not a fictitious latent Gaussian KL.
        self.last_regularization = correction.float().square().mean()
        return feature + correction

    def calibration_loss(
        self,
        squared_error: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        variance = self.last_uncertainty
        if variance is None:
            return squared_error.new_zeros(())
        error = squared_error.detach().float()
        valid = torch.ones_like(error) if valid_mask is None else valid_mask.float()
        size = variance.shape[-2:]
        mass = F.interpolate(valid, size=size, mode="area")
        target = F.interpolate(error * valid, size=size, mode="area")
        target = target / mass.clamp_min(1e-8)
        # Gaussian log score (constant omitted), minimized at predicted MSE.
        score = 0.5 * (target / variance + variance.log())
        return (score * mass).sum() / mass.sum().clamp_min(1.0)
