"""Iterative coefficient-state readout over a shared RGB residual basis."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bfn import ResidualBayesianFlow


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class BasisBFNReadout(nn.Module):
    """Build one basis and iteratively infer its signed spatial coefficients."""

    def __init__(
        self,
        channels: int,
        residual_channels: int,
        *,
        num_bases: int = 4,
        steps: int = 4,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        basis_init: str = "dct_orthogonal",
        basis_init_gain: float = 1.0,
        observation_activation: str = "tanh",
        coefficient_activation: str = "tanh",
        normalization: str = "inverse_sqrt_num_bases",
        zero_init_observer: bool = True,
        update_mode: str = "bayesian",
        min_variance: float = 0.01,
        accuracy_schedule: str = "linear_beta",
        min_accuracy_scale: float = 0.25,
        target_ridge: float = 1e-3,
        acceptance_gate: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if num_bases < 2:
            raise ValueError("Coefficient-state readout requires at least two bases")
        if steps < 1:
            raise ValueError("Coefficient-state readout steps must be positive")
        if hidden_channels < 1:
            raise ValueError("Coefficient-state hidden_channels must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Coefficient-state kernel_size must be positive and odd")
        if basis_init != "dct_orthogonal":
            raise ValueError(
                f"Unsupported coefficient-state basis initialization: {basis_init!r}"
            )
        if not math.isfinite(basis_init_gain) or basis_init_gain <= 0:
            raise ValueError("Coefficient-state basis_init_gain must be positive")
        if observation_activation != "tanh":
            raise ValueError("Coefficient observations must use tanh")
        if coefficient_activation != "tanh":
            raise ValueError("Coefficient readout must use tanh")
        if normalization != "inverse_sqrt_num_bases":
            raise ValueError(
                "Coefficient-state normalization must be inverse_sqrt_num_bases"
            )
        if not zero_init_observer:
            raise ValueError("Coefficient observer must use zero initialization")
        if update_mode not in {"deterministic", "bayesian"}:
            raise ValueError(
                "Coefficient-state update_mode must be deterministic or bayesian"
            )
        if not 0.0 < min_accuracy_scale <= 1.0:
            raise ValueError("min_accuracy_scale must lie in (0, 1]")
        if not math.isfinite(target_ridge) or target_ridge <= 0:
            raise ValueError("target_ridge must be positive")

        gate_config = dict(acceptance_gate or {})
        self.acceptance_gate_enabled = bool(gate_config.pop("enabled", False))
        expected_gate = {
            "type": "posterior_spatial_sigmoid",
            "condition": ["context", "posterior_mean", "log_precision"],
            "range": [0.0, 1.0],
            "initial_probability": 0.5,
        }
        for key, expected in expected_gate.items():
            actual = gate_config.pop(key, expected)
            if actual != expected:
                raise ValueError(
                    f"coefficient acceptance_gate.{key} must be "
                    f"{expected!r}, got {actual!r}"
                )
        if gate_config:
            raise ValueError(
                "Unknown coefficient acceptance_gate fields: "
                + ", ".join(sorted(gate_config))
            )

        self.channels = int(channels)
        self.residual_channels = int(residual_channels)
        self.num_bases = int(num_bases)
        self.steps = int(steps)
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.basis_init = basis_init
        self.basis_init_gain = float(basis_init_gain)
        self.observation_activation = observation_activation
        self.coefficient_activation = coefficient_activation
        self.normalization = normalization
        self.zero_init_observer = zero_init_observer
        self.update_mode = update_mode
        self.min_accuracy_scale = float(min_accuracy_scale)
        self.target_ridge = float(target_ridge)
        self.output_scale = 1.0 / math.sqrt(num_bases)
        self.flow = ResidualBayesianFlow(
            min_variance=min_variance,
            accuracy_schedule=accuracy_schedule,
        )

        padding = kernel_size // 2
        self.basis_norm = nn.GroupNorm(_group_count(channels), channels)
        self.basis_projection = nn.Conv2d(
            channels,
            num_bases * residual_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        observer_input_channels = channels + num_bases * 2 + 1
        self.observer_input = nn.Conv2d(
            observer_input_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.observer_norm = nn.GroupNorm(
            _group_count(hidden_channels),
            hidden_channels,
        )
        self.observer_output = nn.Conv2d(
            hidden_channels,
            num_bases * 2,
            kernel_size=kernel_size,
            padding=padding,
        )

        gate_input_channels = channels + num_bases * 2
        self.gate_input = nn.Conv2d(
            gate_input_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.gate_norm = nn.GroupNorm(
            _group_count(hidden_channels),
            hidden_channels,
        )
        self.gate_output = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=kernel_size,
            padding=padding,
        )

        self._initialize_basis_projection()
        nn.init.zeros_(self.observer_output.weight)
        nn.init.zeros_(self.observer_output.bias)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

        self.last_basis_maps: Tensor | None = None
        self.last_coefficients: Tensor | None = None
        self.last_posterior_mean: Tensor | None = None
        self.last_posterior_precision: Tensor | None = None
        self.last_acceptance: Tensor | None = None
        self.last_correction: Tensor | None = None
        self.last_trajectory: list[Tensor] = []
        self.last_bfn_target: Tensor | None = None
        self.last_bfn_prediction: Tensor | None = None
        self.last_bfn_time: Tensor | None = None
        self.last_bfn_sender_mean: Tensor | None = None
        self.last_bfn_sender_precision: Tensor | None = None
        self.last_basis_evaluations = 0
        self.last_observer_calls = 0
        self._last_context: Tensor | None = None
        self._last_basis_maps: Tensor | None = None

    def _initialize_basis_projection(self) -> None:
        """Use an explicit RNG-independent orthonormal DCT row basis."""

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
            weight.copy_(matrix.reshape_as(weight) * self.basis_init_gain)
            nn.init.zeros_(self.basis_projection.bias)

    @property
    def weight(self) -> Tensor:
        """Expose readout projections for gradient and parameter inspection."""

        return torch.cat(
            (
                self.basis_projection.weight.flatten(),
                self.observer_output.weight.flatten(),
                self.gate_output.weight.flatten(),
            )
        )

    @property
    def bias(self) -> Tensor:
        return torch.cat(
            (
                self.basis_projection.bias.flatten(),
                self.observer_output.bias.flatten(),
                self.gate_output.bias.flatten(),
            )
        )

    def update_state(
        self,
        mean: Tensor,
        precision: Tensor,
        observation: Tensor,
        accuracy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the sole behavioral difference between matched D/B arms."""

        if self.update_mode == "bayesian":
            return self.flow.bayesian_update(
                mean,
                precision,
                observation,
                accuracy,
            )
        new_precision = precision + accuracy
        new_mean = mean + (
            accuracy / new_precision.clamp_min(1e-8)
        ) * observation
        return new_mean, new_precision

    def predict_distribution(
        self,
        context: Tensor,
        mean: Tensor,
        precision: Tensor,
        time: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict coefficient mean and adaptive precision scale."""

        batch, _, height, width = context.shape
        if time.ndim == 1:
            time = time[:, None]
        time_map = time[:, :, None, None].expand(
            batch,
            1,
            height,
            width,
        )
        observer_hidden = self.observer_input(
            torch.cat(
                (
                    context,
                    mean,
                    precision.clamp_min(1e-6).log(),
                    time_map,
                ),
                dim=1,
            )
        )
        observer_hidden = F.silu(self.observer_norm(observer_hidden))
        observer_output = self.observer_output(observer_hidden)
        prediction, accuracy_logits = observer_output.chunk(2, dim=1)
        return prediction.tanh(), accuracy_logits

    def coefficient_target(
        self,
        normalized_residual: Tensor,
    ) -> Tensor:
        """Project an RGB residual onto the current basis with ridge stability."""

        if self._last_basis_maps is None:
            raise RuntimeError("Coefficient target requires a preceding readout")
        with torch.autocast(
            device_type=normalized_residual.device.type,
            enabled=False,
        ):
            basis = self._last_basis_maps.detach().float() * self.output_scale
            residual = normalized_residual.detach().float()
            if residual.shape[1] != self.residual_channels:
                raise ValueError(
                    "Coefficient target residual has wrong channels"
                )
            if residual.shape[-2:] != basis.shape[-2:]:
                raise ValueError("Coefficient target and basis sizes differ")
            design = basis.permute(0, 3, 4, 2, 1)
            gram = design @ design.transpose(-1, -2)
            identity = torch.eye(
                self.residual_channels,
                device=gram.device,
                dtype=torch.float32,
            )
            projected = torch.linalg.solve(
                gram + self.target_ridge * identity,
                residual.permute(0, 2, 3, 1).unsqueeze(-1),
            )
            coefficients = design.transpose(-1, -2) @ projected
        return coefficients.squeeze(-1).permute(0, 3, 1, 2).clamp(
            -1.0,
            1.0,
        ).to(normalized_residual.dtype).detach()

    def bfn_training_loss(
        self,
        normalized_residual: Tensor,
        *,
        step: Tensor | None = None,
    ) -> Tensor:
        """Apply the continuous-data BFN sender and discrete-time objective."""

        if self._last_context is None:
            raise RuntimeError("BFN loss requires a preceding readout")
        target = self.coefficient_target(normalized_residual)
        batch, _, height, width = target.shape
        if step is None:
            step = torch.randint(
                1,
                self.steps + 1,
                (batch,),
                device=target.device,
            )
        elif step.shape != (batch,):
            raise ValueError("BFN training step must have shape [batch]")
        step = step.to(device=target.device, dtype=torch.long)
        if torch.any((step < 1) | (step > self.steps)):
            raise ValueError("BFN training step lies outside [1, steps]")
        time = (step.to(target.dtype) - 1.0) / self.steps
        accuracy_scale = target.new_ones((batch, 1, height, width))
        mean, precision = self.flow.sender_state(
            target,
            time,
            accuracy_scale,
        )
        precision = precision.expand_as(target)
        prediction, _ = self.predict_distribution(
            self._last_context,
            mean,
            precision,
            time,
        )
        self.last_observer_calls += 1
        loss = self.flow.discrete_loss(
            prediction,
            target,
            step,
            self.steps,
            accuracy_scale,
        ) / self.steps
        self.last_bfn_target = target
        self.last_bfn_prediction = prediction.detach()
        self.last_bfn_time = time.detach()
        self.last_bfn_sender_mean = mean.detach()
        self.last_bfn_sender_precision = precision.detach()
        return loss

    def forward(
        self,
        feature: Tensor,
        *,
        deterministic: bool = True,
    ) -> Tensor:
        context = F.silu(self.basis_norm(feature))
        batch, _, height, width = context.shape
        basis = self.basis_projection(context).reshape(
            batch,
            self.num_bases,
            self.residual_channels,
            height,
            width,
        )
        self._last_context = context
        self._last_basis_maps = basis
        self.last_basis_evaluations = 1
        self.last_bfn_target = None
        self.last_bfn_prediction = None
        self.last_bfn_time = None
        self.last_bfn_sender_mean = None
        self.last_bfn_sender_precision = None
        mean = context.new_zeros((batch, self.num_bases, height, width))
        precision = torch.ones_like(mean)
        trajectory: list[Tensor] = []
        self.last_observer_calls = 0
        for index in range(1, self.steps + 1):
            time = context.new_full(
                (batch,),
                (index - 1) / self.steps,
            )
            observation, accuracy_logits = self.predict_distribution(
                context,
                mean,
                precision,
                time,
            )
            base_accuracy = self.flow.step_accuracy(
                index,
                self.steps,
                device=feature.device,
                dtype=feature.dtype,
            )
            accuracy_scale = self.min_accuracy_scale + (
                1.0 - self.min_accuracy_scale
            ) * accuracy_logits.sigmoid()
            accuracy = base_accuracy.view(1, 1, 1, 1) * accuracy_scale
            if self.update_mode == "bayesian" and not deterministic:
                observation = observation + (
                    torch.randn_like(observation)
                    / accuracy.clamp_min(1e-8).sqrt()
                )
            mean, precision = self.update_state(
                mean,
                precision,
                observation,
                accuracy,
            )
            trajectory.append(mean.detach())
            self.last_observer_calls += 1

        coefficients = mean.tanh()
        gate_hidden = self.gate_input(
            torch.cat(
                (
                    context,
                    mean,
                    precision.clamp_min(1e-6).log(),
                ),
                dim=1,
            )
        )
        acceptance = self.gate_output(
            F.silu(self.gate_norm(gate_hidden))
        ).sigmoid()
        applied_acceptance = (
            acceptance if self.acceptance_gate_enabled else torch.ones_like(acceptance)
        )
        correction = torch.einsum(
            "bkrhw,bkhw->brhw",
            basis,
            coefficients,
        )
        correction = correction * applied_acceptance * self.output_scale

        self.last_basis_maps = basis.detach()
        self.last_coefficients = coefficients.detach()
        self.last_posterior_mean = mean.detach()
        self.last_posterior_precision = precision.detach()
        self.last_acceptance = acceptance.detach()
        self.last_correction = correction.detach()
        self.last_trajectory = trajectory
        return correction


CoefficientStateBasisReadout = BasisBFNReadout
