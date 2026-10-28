"""Continuous Gaussian Bayesian flow utilities for residual refinement."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@torch.no_grad()
def local_entropy_map(
    image: Tensor,
    kernel_size: int = 9,
    bins: int = 16,
) -> Tensor:
    """Compute a normalized local Shannon entropy map in [0, 1]."""

    if image.ndim != 4:
        raise ValueError("Expected image in BCHW format")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    if bins < 2:
        raise ValueError("bins must be at least two")

    if image.shape[1] == 3:
        weights = image.new_tensor((0.299, 0.587, 0.114))[None, :, None, None]
        gray = (image.clamp(0, 1) * weights).sum(dim=1)
    else:
        gray = image.clamp(0, 1).mean(dim=1)

    indices = torch.round(gray * (bins - 1)).long()
    one_hot = F.one_hot(indices, num_classes=bins).permute(0, 3, 1, 2).float()
    padding = kernel_size // 2
    pad_mode = (
        "reflect"
        if min(image.shape[-2:]) > padding
        else "replicate"
    )
    one_hot = F.pad(one_hot, (padding, padding, padding, padding), mode=pad_mode)
    probabilities = F.avg_pool2d(one_hot, kernel_size, stride=1)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=1, keepdim=True)
    return entropy / math.log(bins)


class ResidualBayesianFlow(nn.Module):
    """Gaussian BFN schedule and closed-form Bayesian parameter updates."""

    def __init__(
        self,
        min_variance: float = 1e-4,
        accuracy_schedule: str = "exponential",
        entropy_kernel: int = 9,
        entropy_bins: int = 16,
        entropy_threshold: float = 0.25,
        min_accuracy_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0.0 < min_variance < 1.0:
            raise ValueError("min_variance must lie in (0, 1)")
        if not 0.0 <= entropy_threshold <= 1.0:
            raise ValueError("entropy_threshold must lie in [0, 1]")
        if accuracy_schedule not in {
            "exponential",
            "linear_beta",
            "equal_alpha",
        }:
            raise ValueError(
                "accuracy_schedule must be exponential, linear_beta, "
                "or equal_alpha"
            )
        self.min_variance = float(min_variance)
        self.accuracy_schedule = accuracy_schedule
        self.entropy_kernel = int(entropy_kernel)
        self.entropy_bins = int(entropy_bins)
        self.entropy_threshold = float(entropy_threshold)
        self.min_accuracy_scale = float(min_accuracy_scale)

    def cumulative_accuracy(self, time: Tensor) -> Tensor:
        """Return cumulative accuracy for the configured schedule."""

        min_variance = torch.as_tensor(
            self.min_variance, device=time.device, dtype=time.dtype
        )
        if self.accuracy_schedule in {"linear_beta", "equal_alpha"}:
            return time * (min_variance.reciprocal() - 1.0)
        return min_variance.pow(-time) - 1.0

    def step_accuracy(
        self,
        step: int | Tensor,
        num_steps: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Discrete accuracy alpha_i from the official continuous-data BFN."""

        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        step_tensor = torch.as_tensor(step, device=device, dtype=dtype or torch.float32)
        if self.accuracy_schedule in {"linear_beta", "equal_alpha"}:
            min_variance = torch.as_tensor(
                self.min_variance,
                device=step_tensor.device,
                dtype=step_tensor.dtype,
            )
            return (min_variance.reciprocal() - 1.0) / num_steps
        current = self.cumulative_accuracy(step_tensor / num_steps)
        previous = self.cumulative_accuracy(
            (step_tensor - 1.0) / num_steps
        )
        return current - previous

    def spatial_schedule(self, structural: Tensor) -> tuple[Tensor, Tensor]:
        """Return the binary active mask and adaptive accuracy scale."""

        entropy = local_entropy_map(
            structural.detach().clamp(0, 1),
            kernel_size=self.entropy_kernel,
            bins=self.entropy_bins,
        )
        mask = (entropy > self.entropy_threshold).to(structural.dtype)
        scale = mask * (
            self.min_accuracy_scale
            + (1.0 - self.min_accuracy_scale) * entropy.to(structural.dtype)
        )
        return mask, scale

    def sender_state(
        self,
        residual: Tensor,
        time: Tensor,
        accuracy_scale: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample (mu_t, rho_t) directly from the accumulated sender flow."""

        if time.ndim != 1 or time.shape[0] != residual.shape[0]:
            raise ValueError("time must have shape [batch]")
        beta = self.cumulative_accuracy(time).view(-1, 1, 1, 1)
        beta = beta * accuracy_scale
        precision = 1.0 + beta
        if noise is None:
            noise = torch.randn_like(residual)
        mean = (beta * residual + beta.clamp_min(0).sqrt() * noise) / precision
        return mean, precision

    @staticmethod
    def bayesian_update(
        mean: Tensor,
        precision: Tensor,
        observation: Tensor,
        accuracy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        new_precision = precision + accuracy
        new_mean = (
            precision * mean + accuracy * observation
        ) / new_precision.clamp_min(1e-8)
        return new_mean, new_precision

    @staticmethod
    def uncertainty(precision: Tensor) -> Tensor:
        return precision.reciprocal()

    def discrete_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        step: Tensor,
        num_steps: int,
        accuracy_scale: Tensor,
    ) -> Tensor:
        """Equation (11), estimated with one random discrete step per item."""

        prediction = prediction.float()
        target = target.float()
        accuracy_scale = accuracy_scale.float()
        alpha = self.step_accuracy(
            step,
            num_steps,
            device=prediction.device,
            dtype=torch.float32,
        ).view(-1, 1, 1, 1)
        alpha = alpha * accuracy_scale
        weighted_error = 0.5 * num_steps * alpha * (target - prediction).square()
        active_values = (accuracy_scale > 0).sum() * prediction.shape[1]
        return weighted_error.sum() / active_values.clamp_min(1)


def degradation_latent_targets(
    degraded: Tensor,
    target: Tensor,
    *,
    global_dim: int,
    spatial_dim: int,
    spatial_size: tuple[int, int],
    category: Tensor | None,
    num_categories: int,
    residual_scale: float = 0.2,
) -> tuple[Tensor, Tensor]:
    """Build fixed global and coarse spatial degradation descriptors."""

    if global_dim < num_categories + 2:
        raise ValueError("global_dim is too small for category and severity targets")
    if spatial_dim < 1 or residual_scale <= 0:
        raise ValueError("spatial_dim and residual_scale must be positive")
    residual = (degraded.float() - target.float()).detach()
    absolute = residual.abs()
    gray = residual.mean(dim=1, keepdim=True)
    gray_absolute = gray.abs()

    global_target = residual.new_zeros((residual.shape[0], global_dim))
    if category is not None:
        one_hot = F.one_hot(
            category.long(),
            num_classes=num_categories,
        ).to(global_target.dtype)
        global_target[:, :num_categories] = one_hot
    severity = residual.square().mean(dim=(1, 2, 3)).sqrt()
    global_target[:, num_categories] = (
        severity / residual_scale
    ).clamp(0, 1)

    spectral_slots = global_dim - num_categories - 1
    spectrum = torch.fft.rfft2(gray, norm="ortho").abs().log1p()
    spectral = F.adaptive_avg_pool2d(
        spectrum,
        (1, spectral_slots),
    ).flatten(1)
    spectral = spectral / spectral.amax(dim=1, keepdim=True).clamp_min(1e-6)
    global_target[:, num_categories + 1 :] = spectral

    gradient_x = F.pad(
        (gray[..., 1:] - gray[..., :-1]).abs(),
        (0, 1, 0, 0),
    )
    gradient_y = F.pad(
        (gray[..., 1:, :] - gray[..., :-1, :]).abs(),
        (0, 0, 0, 1),
    )
    lowpass = F.avg_pool2d(gray_absolute, 5, stride=1, padding=2)
    highpass = (gray_absolute - lowpass).abs()
    spatial_basis = torch.cat(
        (
            absolute,
            gray_absolute,
            gradient_x,
            gradient_y,
            lowpass,
            highpass,
        ),
        dim=1,
    )
    if spatial_basis.shape[1] < spatial_dim:
        repeats = math.ceil(spatial_dim / spatial_basis.shape[1])
        spatial_basis = spatial_basis.repeat(1, repeats, 1, 1)
    spatial_target = F.adaptive_avg_pool2d(
        spatial_basis[:, :spatial_dim],
        spatial_size,
    )
    spatial_target = (spatial_target / residual_scale).clamp(0, 1)
    return global_target.detach(), spatial_target.detach()


class DegradationEvidenceEncoder(nn.Module):
    """Encode one cached SSR prefix feature for lightweight latent inference."""

    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.SiLU(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        return self.layers(feature)


class DegradationLatentBFN(nn.Module):
    """Matched iterative flow over global and coarse spatial degradation."""

    def __init__(
        self,
        feature_channels: int,
        global_dim: int = 32,
        spatial_dim: int = 8,
        hidden_channels: int = 64,
        num_categories: int = 3,
        min_variance: float = 0.01,
        min_accuracy_scale: float = 0.25,
        spatial_downsample: int = 4,
        accuracy_schedule: str = "exponential",
        loss_mode: str = "all_observations",
        update_mode: str = "bayesian",
        loss_weighting: str = "none",
    ) -> None:
        super().__init__()
        if min(feature_channels, global_dim, spatial_dim, hidden_channels) < 1:
            raise ValueError("Latent BFN dimensions must be positive")
        if num_categories < 2 or global_dim <= num_categories:
            raise ValueError("global_dim must exceed num_categories")
        if not 0.0 < min_accuracy_scale <= 1.0:
            raise ValueError("min_accuracy_scale must lie in (0, 1]")
        if spatial_downsample < 1:
            raise ValueError("spatial_downsample must be positive")
        if loss_mode not in {"all_observations", "final_mean"}:
            raise ValueError(
                "loss_mode must be all_observations or final_mean"
            )
        if update_mode not in {"bayesian", "deterministic"}:
            raise ValueError(
                "update_mode must be bayesian or deterministic"
            )
        if loss_weighting not in {"none", "adaptive_accuracy"}:
            raise ValueError(
                "loss_weighting must be none or adaptive_accuracy"
            )
        self.global_dim = int(global_dim)
        self.spatial_dim = int(spatial_dim)
        self.num_categories = int(num_categories)
        self.min_accuracy_scale = float(min_accuracy_scale)
        self.spatial_downsample = int(spatial_downsample)
        self.loss_mode = loss_mode
        self.update_mode = update_mode
        self.loss_weighting = loss_weighting
        self.flow = ResidualBayesianFlow(
            min_variance=min_variance,
            accuracy_schedule=accuracy_schedule,
        )

        self.degradation_encoder = DegradationEvidenceEncoder(
            feature_channels,
            hidden_channels,
        )
        self.global_context = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.spatial_context = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
        )
        self.global_observer = nn.Sequential(
            nn.Linear(
                hidden_channels + global_dim * 2 + 1,
                hidden_channels,
            ),
            nn.SiLU(),
            nn.Linear(hidden_channels, global_dim * 2),
        )
        self.spatial_observer = nn.Sequential(
            nn.Conv2d(
                hidden_channels + spatial_dim * 2 + 1,
                hidden_channels,
                1,
            ),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, spatial_dim * 2, 1),
        )
        self.category_head = nn.Linear(global_dim, num_categories)

    def _accuracy_scale(self, logits: Tensor) -> Tensor:
        return self.min_accuracy_scale + (
            1.0 - self.min_accuracy_scale
        ) * logits.sigmoid()

    def adaptive_accuracy(
        self,
        base_accuracy: Tensor,
        logits: Tensor,
    ) -> Tensor:
        """Construct the single adaptive alpha used by state and loss."""

        return base_accuracy * self._accuracy_scale(logits)

    def update_state(
        self,
        mean: Tensor,
        precision: Tensor,
        observation: Tensor,
        accuracy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the frozen arm-specific update without adding parameters."""

        if self.update_mode == "bayesian":
            return self.flow.bayesian_update(
                mean,
                precision,
                observation,
                accuracy,
            )
        new_precision = precision + accuracy
        # D interprets each head output as a deterministic residual increment.
        # The same adaptive alpha and precision state gate that increment.
        new_mean = mean + accuracy / new_precision.clamp_min(1e-8) * observation
        return new_mean, new_precision

    def forward(
        self,
        feature: Tensor,
        *,
        steps: int,
        return_trajectory: bool = False,
    ) -> dict[str, Tensor | list[Tensor]]:
        if steps < 1:
            raise ValueError("Latent BFN steps must be positive")
        hidden = self.degradation_encoder(feature)
        global_context = self.global_context(hidden.mean(dim=(-2, -1)))
        spatial_size = (
            max(1, feature.shape[-2] // self.spatial_downsample),
            max(1, feature.shape[-1] // self.spatial_downsample),
        )
        spatial_context = self.spatial_context(
            F.adaptive_avg_pool2d(hidden, spatial_size)
        )

        global_mean = feature.new_zeros((feature.shape[0], self.global_dim))
        global_precision = feature.new_ones(global_mean.shape)
        spatial_mean = feature.new_zeros(
            (feature.shape[0], self.spatial_dim, *spatial_size)
        )
        spatial_precision = feature.new_ones(spatial_mean.shape)
        global_observations: list[Tensor] = []
        spatial_observations: list[Tensor] = []
        global_accuracies: list[Tensor] = []
        spatial_accuracies: list[Tensor] = []
        trajectory: list[Tensor] = []

        for index in range(1, steps + 1):
            time = feature.new_full(
                (feature.shape[0], 1),
                (index - 1) / steps,
            )
            global_output = self.global_observer(
                torch.cat(
                    (
                        global_context,
                        global_mean,
                        global_precision.clamp_min(1e-6).log(),
                        time,
                    ),
                    dim=1,
                )
            )
            global_observation, global_accuracy_logits = (
                global_output.chunk(2, dim=1)
            )
            global_observation = global_observation.tanh()

            time_map = time[:, :, None, None].expand(
                -1,
                -1,
                *spatial_size,
            )
            spatial_output = self.spatial_observer(
                torch.cat(
                    (
                        spatial_context,
                        spatial_mean,
                        spatial_precision.clamp_min(1e-6).log(),
                        time_map,
                    ),
                    dim=1,
                )
            )
            spatial_observation, spatial_accuracy_logits = (
                spatial_output.chunk(2, dim=1)
            )
            spatial_observation = spatial_observation.tanh()

            base_accuracy = self.flow.step_accuracy(
                index,
                steps,
                device=feature.device,
                dtype=feature.dtype,
            )
            global_accuracy = self.adaptive_accuracy(
                base_accuracy,
                global_accuracy_logits
            )
            spatial_accuracy = self.adaptive_accuracy(
                base_accuracy,
                spatial_accuracy_logits
            )
            global_mean, global_precision = self.update_state(
                global_mean,
                global_precision,
                global_observation,
                global_accuracy,
            )
            spatial_mean, spatial_precision = self.update_state(
                spatial_mean,
                spatial_precision,
                spatial_observation,
                spatial_accuracy,
            )
            global_observations.append(global_observation)
            spatial_observations.append(spatial_observation)
            global_accuracies.append(global_accuracy)
            spatial_accuracies.append(spatial_accuracy)
            if return_trajectory:
                trajectory.append(global_mean.clone())

        return {
            "global_mean": global_mean,
            "global_precision": global_precision,
            "spatial_mean": spatial_mean,
            "spatial_precision": spatial_precision,
            "category_logits": self.category_head(global_mean),
            "global_observations": global_observations,
            "spatial_observations": spatial_observations,
            "global_accuracies": global_accuracies,
            "spatial_accuracies": spatial_accuracies,
            "trajectory": trajectory,
        }

    @staticmethod
    def _weighted_mse(
        prediction: Tensor,
        target: Tensor,
        weight: Tensor | None,
    ) -> Tensor:
        error = (prediction.float() - target.float()).square()
        if weight is None:
            return error.mean()
        normalized = weight.detach().float()
        normalized = normalized / normalized.mean().clamp_min(1e-8)
        return (error * normalized).mean()

    def latent_loss_weights(
        self,
        outputs: dict[str, Tensor | list[Tensor]],
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.loss_weighting == "none":
            return None, None
        global_precision = outputs.get("global_precision")
        spatial_precision = outputs.get("spatial_precision")
        if not isinstance(global_precision, Tensor) or not isinstance(
            spatial_precision,
            Tensor,
        ):
            raise TypeError("Adaptive latent loss requires precision tensors")
        return (
            (global_precision - 1.0).detach(),
            (spatial_precision - 1.0).detach(),
        )

    def latent_loss(
        self,
        outputs: dict[str, Tensor | list[Tensor]],
        global_target: Tensor,
        spatial_target: Tensor,
    ) -> Tensor:
        global_mean = outputs["global_mean"]
        spatial_mean = outputs["spatial_mean"]
        if not isinstance(global_mean, Tensor) or not isinstance(
            spatial_mean,
            Tensor,
        ):
            raise TypeError("Latent means must be tensors")
        global_weight, spatial_weight = self.latent_loss_weights(outputs)
        losses = [
            self._weighted_mse(global_mean, global_target, global_weight),
            self._weighted_mse(spatial_mean, spatial_target, spatial_weight),
        ]
        if self.loss_mode == "all_observations":
            global_accuracies = outputs.get("global_accuracies")
            spatial_accuracies = outputs.get("spatial_accuracies")
            if self.loss_weighting == "adaptive_accuracy" and (
                not isinstance(global_accuracies, list)
                or not isinstance(spatial_accuracies, list)
            ):
                raise TypeError(
                    "Adaptive observation loss requires per-step accuracies"
                )
            for index, observation in enumerate(
                outputs["global_observations"]
            ):
                losses.append(
                    self._weighted_mse(
                        observation,
                        global_target,
                        (
                            global_accuracies[index]
                            if isinstance(global_accuracies, list)
                            else None
                        ),
                    )
                )
            for index, observation in enumerate(
                outputs["spatial_observations"]
            ):
                losses.append(
                    self._weighted_mse(
                        observation,
                        spatial_target,
                        (
                            spatial_accuracies[index]
                            if isinstance(spatial_accuracies, list)
                            else None
                        ),
                    )
                )
        return torch.stack(losses).mean()
