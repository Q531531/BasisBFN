"""Time-conditioned residual receiver used by the Bayesian flow."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .coefficient_state import CoefficientStateBasisReadout
from .layers import MultiDconvHeadTransposedAttention
from .signed_basis import SignedSpatialBasisReadout


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4:
            raise ValueError("Time embedding dimension must be at least four")
        self.embedding_dim = embedding_dim

    def forward(self, time: Tensor) -> Tensor:
        half = self.embedding_dim // 2
        frequency = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(half - 1, 1)
        )
        phase = time[:, None] * frequency[None] * 1_000.0
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if embedding.shape[-1] < self.embedding_dim:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class AdaptiveGroupNorm(nn.Module):
    """AdaGN(x, t) = gamma(t) * Norm(x) + beta(t)."""

    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.modulation = nn.Linear(time_dim, channels * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        scale, shift = self.modulation(time_embedding).chunk(2, dim=-1)
        scale = scale[:, :, None, None] + 1.0
        shift = shift[:, :, None, None]
        return scale * self.norm(x) + shift


class TimeResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = AdaptiveGroupNorm(in_channels, time_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaptiveGroupNorm(out_channels, time_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        residual = self.conv1(F.silu(self.norm1(x, time_embedding)))
        residual = self.conv2(
            self.dropout(F.silu(self.norm2(residual, time_embedding)))
        )
        return self.skip(x) + residual


class TimeAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int, time_dim: int) -> None:
        super().__init__()
        self.norm = AdaptiveGroupNorm(channels, time_dim)
        self.attention = MultiDconvHeadTransposedAttention(
            channels, num_heads=heads, bias=True
        )

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        return x + self.attention(self.norm(x, time_embedding))


class TimeStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        for index in range(depth):
            blocks.append(
                TimeResidualBlock(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    time_dim,
                    dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, time_embedding)
        return x


class RoutedResidualHead(nn.Module):
    """One zero-initialized private residual projection."""

    def __init__(
        self,
        channels: int,
        residual_channels: int,
        *,
        symmetry_offset: float,
        symmetry_break_epsilon: float,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.output = nn.Conv2d(
            channels,
            residual_channels,
            kernel_size=3,
            padding=1,
        )
        with torch.no_grad():
            pattern = torch.linspace(
                -1.0,
                1.0,
                channels,
                dtype=self.norm.weight.dtype,
                device=self.norm.weight.device,
            )
            pattern -= pattern.mean()
            self.norm.weight.add_(
                symmetry_offset * symmetry_break_epsilon * pattern
            )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature: Tensor) -> Tensor:
        return self.output(F.silu(self.norm(feature)))


class PrivateBottleneckRoutedResidualHead(nn.Module):
    """An independently initialized private path with a zero RGB projection."""

    def __init__(
        self,
        channels: int,
        residual_channels: int,
        *,
        symmetry_offset: float,
        symmetry_break_epsilon: float,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.bottleneck = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.hidden_norm = nn.GroupNorm(_group_count(channels), channels)
        self.output = nn.Conv2d(
            channels,
            residual_channels,
            kernel_size=3,
            padding=1,
        )
        with torch.no_grad():
            pattern = torch.linspace(
                -1.0,
                1.0,
                channels,
                dtype=self.norm.weight.dtype,
                device=self.norm.weight.device,
            )
            pattern -= pattern.mean()
            self.norm.weight.add_(
                symmetry_offset * symmetry_break_epsilon * pattern
            )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature: Tensor) -> Tensor:
        hidden = F.silu(self.bottleneck(self.norm(feature)))
        hidden = F.silu(self.hidden_norm(hidden))
        return self.output(hidden)


class CategoryRoutedReadout(nn.Module):
    """Soft mixture of private residual heads with an identity-safe start."""

    def __init__(
        self,
        channels: int,
        residual_channels: int,
        categories: Sequence[str],
        *,
        head_type: str = "groupnorm_silu_conv3x3",
        symmetry_break_epsilon: float = 1e-3,
        acceptance_gate: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        category_names = tuple(str(name) for name in categories)
        if len(category_names) != 3 or len(set(category_names)) != 3:
            raise ValueError(
                "Category-routed readout requires three unique categories"
            )
        if symmetry_break_epsilon <= 0:
            raise ValueError("Head symmetry-break epsilon must be positive")
        head_types = {
            "groupnorm_silu_conv3x3": RoutedResidualHead,
            "groupnorm_conv3x3_silu_groupnorm_silu_conv3x3": (
                PrivateBottleneckRoutedResidualHead
            ),
        }
        if head_type not in head_types:
            raise ValueError(
                f"Unsupported routed readout head_type: {head_type!r}"
            )
        self.categories = category_names
        self.head_type = head_type
        head_class = head_types[head_type]
        self.heads = nn.ModuleDict(
            {
                name: head_class(
                    channels,
                    residual_channels,
                    symmetry_offset=float(index - 1),
                    symmetry_break_epsilon=symmetry_break_epsilon,
                )
                for index, name in enumerate(category_names)
            }
        )
        self.router_pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Linear(channels, len(category_names))
        nn.init.zeros_(self.router.weight)
        nn.init.constant_(
            self.router.bias,
            math.log(1.0 / len(category_names)),
        )
        self.last_router_entropy: Tensor | None = None
        self.last_router_probabilities: Tensor | None = None
        gate_config = dict(acceptance_gate or {})
        self.acceptance_gate_enabled = bool(
            gate_config.pop("enabled", False)
        )
        self.freeze_base_for_acceptance_gate = bool(
            gate_config.pop("freeze_base_receiver", False)
        )
        if self.acceptance_gate_enabled:
            expected = {
                "type": "global_average_linear_hard_clamp",
                "range": [0.0, 1.0],
                "zero_init": True,
            }
            for key, expected_value in expected.items():
                value = gate_config.pop(key, expected_value)
                if value != expected_value:
                    raise ValueError(
                        f"acceptance_gate.{key} must be "
                        f"{expected_value!r}, got {value!r}"
                    )
            if not self.freeze_base_for_acceptance_gate:
                raise ValueError(
                    "Acceptance gate requires freeze_base_receiver=true"
                )
            self.acceptance_gate: nn.Linear | None = nn.Linear(channels, 1)
            nn.init.zeros_(self.acceptance_gate.weight)
            nn.init.zeros_(self.acceptance_gate.bias)
        else:
            if gate_config or self.freeze_base_for_acceptance_gate:
                raise ValueError(
                    "Acceptance gate fields require enabled=true"
                )
            self.acceptance_gate = None
        self.last_acceptance: Tensor | None = None
        self.last_ungated_mixture: Tensor | None = None

    @property
    def weight(self) -> Tensor:
        """Expose routed output weights for existing training diagnostics."""

        return torch.cat(
            [head.output.weight.flatten() for head in self.heads.values()]
        )

    @property
    def bias(self) -> Tensor:
        return torch.cat(
            [head.output.bias.flatten() for head in self.heads.values()]
        )

    def forward(self, feature: Tensor) -> Tensor:
        pooled = self.router_pool(feature).flatten(1)
        logits = self.router(pooled)
        probabilities = F.softmax(logits.float(), dim=1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=1).mean()
        self.last_router_entropy = entropy.detach()
        self.last_router_probabilities = probabilities.detach()
        mixture = None
        for index, head in enumerate(self.heads.values()):
            correction = head(feature)
            weight = probabilities[:, index].to(correction.dtype)
            weighted = weight[:, None, None, None] * correction
            mixture = weighted if mixture is None else mixture + weighted
        if mixture is None:
            raise RuntimeError("Category-routed readout has no heads")
        self.last_ungated_mixture = mixture.detach()
        if self.acceptance_gate is not None:
            raw_acceptance = self.acceptance_gate(pooled)
            # Keep the historical unit gradient at both boundaries explicitly:
            # a zero-initialized gate must be able to leave the identity state.
            acceptance = torch.where(
                raw_acceptance < 0.0,
                torch.zeros_like(raw_acceptance),
                torch.where(
                    raw_acceptance > 1.0,
                    torch.ones_like(raw_acceptance),
                    raw_acceptance,
                ),
            )
            self.last_acceptance = acceptance.detach()
            mixture = acceptance[:, :, None, None].to(mixture.dtype) * mixture
        else:
            self.last_acceptance = None
        return mixture


class ResidualReceiver(nn.Module):
    """A compact conditional U-Net that predicts the clean residual."""

    def __init__(
        self,
        residual_channels: int = 3,
        condition_channels: int = 3,
        feature_channels: int = 64,
        base_channels: int = 24,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        blocks_per_level: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.0,
        observation_channels: int = 0,
        routed_readout: dict[str, Any] | None = None,
        signed_basis_readout: dict[str, Any] | None = None,
        coefficient_state_readout: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not channel_multipliers:
            raise ValueError("channel_multipliers cannot be empty")
        time_dim = base_channels * 4
        if observation_channels not in {0, 6}:
            raise ValueError("observation_channels must be zero or six")
        self.observation_channels = observation_channels
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        input_channels = (
            residual_channels + 1 + condition_channels + feature_channels
            + observation_channels
        )
        self.input_projection = nn.Conv2d(
            input_channels, base_channels, kernel_size=3, padding=1
        )

        level_channels = [base_channels * value for value in channel_multipliers]
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = base_channels
        for index, channels in enumerate(level_channels):
            self.encoder_stages.append(
                TimeStage(
                    current_channels,
                    channels,
                    time_dim,
                    blocks_per_level,
                    dropout,
                )
            )
            current_channels = channels
            if index < len(level_channels) - 1:
                next_channels = level_channels[index + 1]
                self.downsamples.append(
                    nn.Conv2d(current_channels, next_channels, 3, stride=2, padding=1)
                )
                current_channels = next_channels

        self.middle1 = TimeResidualBlock(
            current_channels, current_channels, time_dim, dropout
        )
        self.middle_attention = TimeAttentionBlock(
            current_channels, attention_heads, time_dim
        )
        self.middle2 = TimeResidualBlock(
            current_channels, current_channels, time_dim, dropout
        )

        self.upsamples = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for channels in reversed(level_channels[:-1]):
            self.upsamples.append(
                nn.Conv2d(current_channels, channels, kernel_size=3, padding=1)
            )
            self.decoder_stages.append(
                TimeStage(
                    channels * 2,
                    channels,
                    time_dim,
                    blocks_per_level,
                    dropout,
                )
            )
            current_channels = channels

        routed_config = dict(routed_readout or {})
        self.routed_readout_enabled = bool(
            routed_config.pop("enabled", False)
        )
        signed_basis_config = dict(signed_basis_readout or {})
        self.signed_basis_readout_enabled = bool(
            signed_basis_config.pop("enabled", False)
        )
        coefficient_state_config = dict(coefficient_state_readout or {})
        self.coefficient_state_readout_enabled = bool(
            coefficient_state_config.pop("enabled", False)
        )
        enabled_readouts = sum(
            (
                self.routed_readout_enabled,
                self.signed_basis_readout_enabled,
                self.coefficient_state_readout_enabled,
            )
        )
        if enabled_readouts > 1:
            raise ValueError(
                "routed, signed-basis, and coefficient-state readouts are "
                "mutually exclusive"
            )
        if self.routed_readout_enabled:
            categories = routed_config.pop(
                "categories",
                ("snow", "outdoor_rain", "raindrop"),
            )
            symmetry_break_epsilon = float(
                routed_config.pop("head_symmetry_break_epsilon", 1e-3)
            )
            head_type = str(
                routed_config.pop(
                    "head_type",
                    "groupnorm_silu_conv3x3",
                )
            )
            acceptance_gate = routed_config.pop("acceptance_gate", None)
            expected_metadata = {
                "router_type": "global_average_linear",
                "router_supervision": "none",
                "router_bias_init": "uniform_log_probability",
                "zero_init_output": True,
            }
            for metadata_key, expected in expected_metadata.items():
                value = routed_config.pop(metadata_key, expected)
                if value != expected:
                    raise ValueError(
                        f"routed_readout.{metadata_key} must be "
                        f"{expected!r}, got {value!r}"
                    )
            if routed_config:
                raise ValueError(
                    "Unknown routed_readout fields: "
                    + ", ".join(sorted(routed_config))
                )
            self.output_norm = nn.Identity()
            self.output = CategoryRoutedReadout(
                current_channels,
                residual_channels,
                categories,
                head_type=head_type,
                symmetry_break_epsilon=symmetry_break_epsilon,
                acceptance_gate=acceptance_gate,
            )
        elif self.signed_basis_readout_enabled:
            readout_type = signed_basis_config.pop(
                "type",
                "spatial_rgb_basis",
            )
            if readout_type != "spatial_rgb_basis":
                raise ValueError(
                    "signed_basis_readout.type must be 'spatial_rgb_basis'"
                )
            coefficient_range = signed_basis_config.pop(
                "coefficient_range",
                [-1.0, 1.0],
            )
            if coefficient_range != [-1.0, 1.0]:
                raise ValueError(
                    "signed_basis_readout.coefficient_range must be [-1, 1]"
                )
            allowed_fields = {
                "num_bases",
                "kernel_size",
                "basis_init",
                "basis_init_gain",
                "coefficient_activation",
                "normalization",
                "zero_init_coefficients",
            }
            unknown_fields = set(signed_basis_config) - allowed_fields
            if unknown_fields:
                raise ValueError(
                    "Unknown signed_basis_readout fields: "
                    + ", ".join(sorted(unknown_fields))
                )
            self.output_norm = nn.Identity()
            self.output = SignedSpatialBasisReadout(
                current_channels,
                residual_channels,
                **signed_basis_config,
            )
        elif self.coefficient_state_readout_enabled:
            readout_type = coefficient_state_config.pop(
                "type",
                "spatial_coefficient_flow",
            )
            if readout_type != "spatial_coefficient_flow":
                raise ValueError(
                    "coefficient_state_readout.type must be "
                    "'spatial_coefficient_flow'"
                )
            coefficient_range = coefficient_state_config.pop(
                "coefficient_range",
                [-1.0, 1.0],
            )
            if coefficient_range != [-1.0, 1.0]:
                raise ValueError(
                    "coefficient_state_readout.coefficient_range must be "
                    "[-1, 1]"
                )
            allowed_fields = {
                "num_bases",
                "steps",
                "hidden_channels",
                "kernel_size",
                "basis_init",
                "basis_init_gain",
                "observation_activation",
                "coefficient_activation",
                "normalization",
                "zero_init_observer",
                "update_mode",
                "min_variance",
                "accuracy_schedule",
                "min_accuracy_scale",
                "target_ridge",
                "acceptance_gate",
            }
            unknown_fields = set(coefficient_state_config) - allowed_fields
            if unknown_fields:
                raise ValueError(
                    "Unknown coefficient_state_readout fields: "
                    + ", ".join(sorted(unknown_fields))
                )
            self.output_norm = nn.Identity()
            self.output = CoefficientStateBasisReadout(
                current_channels,
                residual_channels,
                **coefficient_state_config,
            )
        else:
            if routed_config:
                raise ValueError(
                    "routed_readout fields require enabled=true"
                )
            if signed_basis_config:
                raise ValueError(
                    "signed_basis_readout fields require enabled=true"
                )
            if coefficient_state_config:
                raise ValueError(
                    "coefficient_state_readout fields require enabled=true"
                )
            self.output_norm = nn.GroupNorm(
                _group_count(current_channels), current_channels
            )
            self.output = nn.Conv2d(
                current_channels, residual_channels, kernel_size=3, padding=1
            )
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        belief_mean: Tensor,
        uncertainty: Tensor,
        time: Tensor,
        structural: Tensor,
        decoder_feature: Tensor,
        degraded: Tensor | None = None,
        deterministic: bool = True,
    ) -> Tensor:
        time_embedding = self.time_embedding(time.float())
        inputs = [belief_mean, uncertainty, structural, decoder_feature]
        if self.observation_channels:
            if degraded is None:
                raise ValueError(
                    "Observation-rich receiver requires the degraded image"
                )
            inputs.extend((degraded, degraded - structural))
        x = torch.cat(inputs, dim=1)
        x = self.input_projection(x)

        skips = []
        for index, stage in enumerate(self.encoder_stages):
            x = stage(x, time_embedding)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        x = self.middle1(x, time_embedding)
        x = self.middle_attention(x, time_embedding)
        x = self.middle2(x, time_embedding)

        for index, (upsample, stage) in enumerate(
            zip(self.upsamples, self.decoder_stages)
        ):
            x = F.interpolate(
                x, scale_factor=2.0, mode="bilinear", align_corners=False
            )
            x = upsample(x)
            skip = skips[-index - 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = stage(torch.cat((x, skip), dim=1), time_embedding)

        if (
            self.routed_readout_enabled
            or self.signed_basis_readout_enabled
            or self.coefficient_state_readout_enabled
        ):
            if self.coefficient_state_readout_enabled:
                return self.output(x, deterministic=deterministic)
            return self.output(x)
        return self.output(F.silu(self.output_norm(x)))

    def router_entropy(self, reference: Tensor) -> Tensor:
        if not self.routed_readout_enabled:
            return reference.new_zeros(())
        entropy = self.output.last_router_entropy
        if entropy is None:
            return reference.new_zeros(())
        return entropy.to(device=reference.device, dtype=reference.dtype)

    def adapter_diagnostic(self, reference: Tensor) -> Tensor:
        if self.coefficient_state_readout_enabled:
            acceptance = self.output.last_acceptance
            if (
                self.output.acceptance_gate_enabled
                and acceptance is not None
            ):
                return acceptance.mean().to(
                    device=reference.device,
                    dtype=reference.dtype,
                )
            return reference.new_zeros(())
        if not self.routed_readout_enabled:
            return reference.new_zeros(())
        acceptance = self.output.last_acceptance
        if self.output.acceptance_gate_enabled and acceptance is not None:
            return acceptance.mean().to(
                device=reference.device,
                dtype=reference.dtype,
            )
        return self.router_entropy(reference)
