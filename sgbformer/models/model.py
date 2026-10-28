"""End-to-end SGBformer model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import SpectralGatedBackbone
from .bfn import (
    DegradationLatentBFN,
    ResidualBayesianFlow,
    degradation_latent_targets,
)
from .conditioning import build_semantic_encoder
from .foundation import build_foundation
from .mean import ConditionalMeanCorrector
from .receiver import ResidualReceiver


def _pad_to_multiple(image: Tensor, multiple: int) -> tuple[Tensor, tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return image, (height, width)
    mode = (
        "reflect"
        if height > pad_height and width > pad_width
        else "replicate"
    )
    return F.pad(image, (0, pad_width, 0, pad_height), mode=mode), (height, width)


def _crop(image: Tensor, size: tuple[int, int]) -> Tensor:
    return image[..., : size[0], : size[1]]


def _validate_state_ablation(value: str) -> str:
    normalized = value.lower()
    if normalized not in {"normal", "reset-belief", "shuffle-belief"}:
        raise ValueError(
            "state_ablation must be normal, reset-belief, or shuffle-belief"
        )
    return normalized


class BasisBFN(nn.Module):
    """Bayesian residual refinement with image-dependent bases."""

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
        semantic: dict[str, Any] | None = None,
        bfn: dict[str, Any] | None = None,
        latent_bfn: dict[str, Any] | None = None,
        foundation: dict[str, Any] | None = None,
        mean_correction: dict[str, Any] | None = None,
        core: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        semantic = dict(semantic or {})
        bfn = dict(bfn or {})
        latent_config = dict(latent_bfn or {})
        core = dict(core or {})
        semantic_dim = int(semantic.get("embedding_dim", 512))
        semantic_tokens = int(semantic.get("context_tokens", 4))
        self.semantic_encoder = build_semantic_encoder(semantic)

        if foundation:
            self.backbone = build_foundation(dict(foundation))
        else:
            self.backbone = SpectralGatedBackbone(
                in_channels=in_channels,
                out_channels=out_channels,
                dim=dim,
                num_blocks=num_blocks,
                refinement_blocks=refinement_blocks,
                heads=heads,
                ffn_expansion_factor=ffn_expansion_factor,
                conv_bias=conv_bias,
                norm_bias=norm_bias,
                semantic_dim=semantic_dim,
                semantic_tokens=semantic_tokens,
            )
        mean_config = dict(mean_correction or {})
        mean_enabled = bool(mean_config.pop("enabled", False))
        self.mean_corrector = (
            ConditionalMeanCorrector(
                feature_channels=self.backbone.decoder_channels,
                **mean_config,
            )
            if mean_enabled
            else None
        )
        self.flow = ResidualBayesianFlow(
            min_variance=float(bfn.get("min_variance", 1e-4)),
            accuracy_schedule=str(
                bfn.get("accuracy_schedule", "exponential")
            ),
            entropy_kernel=int(bfn.get("entropy_kernel", 9)),
            entropy_bins=int(bfn.get("entropy_bins", 16)),
            entropy_threshold=float(bfn.get("entropy_threshold", 0.25)),
            min_accuracy_scale=float(bfn.get("min_accuracy_scale", 0.25)),
        )
        self.residual_scale = float(bfn.get("residual_scale", 1.0))
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("BFN residual_scale must lie in (0, 1]")
        self.training_final_state = str(
            bfn.get("training_final_state", "sampled")
        ).lower()
        if self.training_final_state not in {"sampled", "terminal", "rollout"}:
            raise ValueError(
                "BFN training_final_state must be sampled, terminal, or rollout"
            )
        self.rollout_batch_fraction = float(
            bfn.get("rollout_batch_fraction", 1.0)
        )
        if not 0.0 < self.rollout_batch_fraction <= 1.0:
            raise ValueError("BFN rollout_batch_fraction must lie in (0, 1]")
        self.receiver_output_mode = str(
            bfn.get("receiver_output_mode", "identity")
        ).lower()
        if self.receiver_output_mode not in {"identity", "tanh"}:
            raise ValueError(
                "BFN receiver_output_mode must be identity or tanh"
            )
        self.residual_update_mode = str(
            bfn.get("update_mode", "bayesian")
        ).lower()
        if self.residual_update_mode not in {"deterministic", "bayesian"}:
            raise ValueError(
                "BFN update_mode must be deterministic or bayesian"
            )
        self.residual_readout_mode = str(
            bfn.get("readout_mode", "receiver")
        ).lower()
        if self.residual_readout_mode not in {"receiver", "posterior_mean"}:
            raise ValueError(
                "BFN readout_mode must be receiver or posterior_mean"
            )
        self.residual_base_clamp = bool(
            bfn.get("residual_base_clamp", False)
        )
        self.refinement_mode = str(
            bfn.get("inference_mode", "iterative")
        ).lower()
        if self.refinement_mode not in {"iterative", "direct", "none"}:
            raise ValueError("BFN inference_mode must be iterative, direct, or none")
        self.receiver_type = str(bfn.get("receiver_type", "unet")).lower()
        if self.receiver_type not in {"unet", "decoder_state"}:
            raise ValueError("BFN receiver_type must be unet or decoder_state")
        self.decoder_state_mode = str(
            bfn.get("state_condition_mode", "live")
        ).lower()
        if self.decoder_state_mode not in {"live", "neutral"}:
            raise ValueError(
                "BFN state_condition_mode must be live or neutral"
            )
        if self.receiver_type == "decoder_state" and not getattr(
            self.backbone, "has_state_conditioned_adapter", False
        ):
            raise ValueError(
                "decoder_state BFN requires a state_spectral foundation adapter"
            )
        if self.receiver_type == "decoder_state" and self.mean_corrector is not None:
            raise ValueError(
                "decoder_state BFN requires the fixed foundation output as baseline"
            )
        self.decoder_state_use_cache = bool(
            bfn.get("decoder_state_cache", False)
        )
        if self.decoder_state_use_cache and not getattr(
            self.backbone, "supports_decoder_state_cache", False
        ):
            raise ValueError(
                "decoder_state_cache requires the standard split-capable SSR "
                "decoder layout"
            )
        self.receiver_condition_mode = str(
            bfn.get("condition_mode", "structural_feature")
        ).lower()
        if self.receiver_condition_mode not in {
            "structural_feature",
            "observation_rich",
        }:
            raise ValueError(
                "BFN condition_mode must be structural_feature or "
                "observation_rich"
            )
        self.receiver = (
            None
            if self.refinement_mode == "none"
            or self.receiver_type == "decoder_state"
            else ResidualReceiver(
                residual_channels=out_channels,
                condition_channels=out_channels,
                feature_channels=self.backbone.decoder_channels,
                base_channels=int(bfn.get("base_channels", 24)),
                channel_multipliers=tuple(
                    bfn.get("channel_multipliers", (1, 2, 4))
                ),
                blocks_per_level=int(bfn.get("blocks_per_level", 2)),
                attention_heads=int(bfn.get("attention_heads", 4)),
                dropout=float(bfn.get("dropout", 0.0)),
                observation_channels=(
                    6 if self.receiver_condition_mode == "observation_rich"
                    else 0
                ),
                routed_readout=bfn.get("routed_readout"),
                signed_basis_readout=bfn.get("signed_basis_readout"),
                coefficient_state_readout=bfn.get(
                    "coefficient_state_readout"
                ),
            )
        )
        if self.receiver is not None:
            self.receiver.requires_grad_(
                bool(bfn.get("receiver_trainable", True))
            )
            output = self.receiver.output
            if getattr(
                output,
                "freeze_base_for_acceptance_gate",
                False,
            ):
                self.receiver.requires_grad_(False)
                acceptance_gate = getattr(output, "acceptance_gate", None)
                if acceptance_gate is None:
                    raise ValueError(
                        "Frozen acceptance-gate receiver has no gate"
                    )
                acceptance_gate.requires_grad_(True)
        self.bfn_steps = int(bfn.get("steps", 10))
        if self.bfn_steps < 1:
            raise ValueError("BFN steps must be positive")
        self.coefficient_state_readout_enabled = bool(
            self.receiver is not None
            and getattr(
                self.receiver,
                "coefficient_state_readout_enabled",
                False,
            )
        )
        if self.coefficient_state_readout_enabled:
            if self.refinement_mode != "direct":
                raise ValueError(
                    "Coefficient-state readout requires direct receiver mode"
                )
            if self.receiver.output.steps != self.bfn_steps:
                raise ValueError(
                    "Coefficient-state readout steps must match model.bfn.steps"
                )
        self.training_rollout_steps = int(
            bfn.get("training_rollout_steps", self.bfn_steps)
        )
        if self.training_rollout_steps < 1:
            raise ValueError("BFN training_rollout_steps must be positive")
        self.rollout_backprop_steps = int(
            bfn.get("rollout_backprop_steps", 0)
        )
        if not 0 <= self.rollout_backprop_steps <= self.training_rollout_steps:
            raise ValueError(
                "BFN rollout_backprop_steps must lie between zero and "
                "training_rollout_steps"
            )
        if (
            self.residual_readout_mode == "posterior_mean"
            and self.training_final_state != "rollout"
        ):
            raise ValueError(
                "posterior_mean readout requires training_final_state=rollout"
            )
        self.mask_mode = str(bfn.get("mask_mode", "entropy")).lower()
        if self.mask_mode not in {"entropy", "none"}:
            raise ValueError("BFN mask_mode must be entropy or none")

        self.oracle_z_enabled = bool(latent_config.get("oracle_z", False))
        self.latent_bfn_enabled = bool(
            latent_config.get("enabled", latent_bfn is not None)
            or self.oracle_z_enabled
        )
        self.latent_flow: DegradationLatentBFN | None = None
        self.latent_steps = 0
        self.latent_global_dim = 0
        self.latent_spatial_dim = 0
        self.latent_spatial_downsample = 0
        self.latent_num_categories = 0
        self.latent_target_residual_scale = 0.2
        self.latent_update_mode = "bayesian"
        self.latent_loss_weighting = "none"
        self.latent_adapter_precision_mode = "belief"
        if self.latent_bfn_enabled:
            if not foundation:
                raise ValueError("latent_bfn requires an external SSR foundation")
            if self.mean_corrector is not None:
                raise ValueError("latent_bfn does not use an RGB mean correction")
            if not getattr(self.backbone, "supports_latent_cache", False):
                raise ValueError(
                    "latent_bfn requires decoder_level3 and decoder_level2 "
                    "latent_film_spectral adapters on a split-capable SSR"
                )
            configured_steps = int(latent_config.get("steps", 4))
            self.latent_steps = (
                0 if self.oracle_z_enabled else configured_steps
            )
            self.latent_global_dim = int(
                latent_config.get("global_dim", 32)
            )
            self.latent_spatial_dim = int(
                latent_config.get("spatial_dim", 8)
            )
            self.latent_num_categories = int(
                latent_config.get("num_categories", 3)
            )
            self.latent_spatial_downsample = int(
                latent_config.get("spatial_downsample", 4)
            )
            self.latent_target_residual_scale = float(
                latent_config.get("target_residual_scale", 0.2)
            )
            self.latent_update_mode = str(
                latent_config.get("update_mode", "bayesian")
            ).lower()
            self.latent_loss_weighting = str(
                latent_config.get("loss_weighting", "none")
            ).lower()
            self.latent_adapter_precision_mode = str(
                latent_config.get("adapter_precision_mode", "belief")
            ).lower()
            if self.latent_adapter_precision_mode not in {"belief", "unit"}:
                raise ValueError(
                    "latent_bfn.adapter_precision_mode must be belief or unit"
                )
            if not self.oracle_z_enabled and self.latent_steps < 1:
                raise ValueError("latent_bfn.steps must be positive")
            if self.latent_spatial_downsample < 1:
                raise ValueError(
                    "latent_bfn.spatial_downsample must be positive"
                )
            if self.latent_target_residual_scale <= 0:
                raise ValueError(
                    "latent_bfn.target_residual_scale must be positive"
                )
            feature_channels = int(
                latent_config.get(
                    "feature_channels",
                    getattr(self.backbone, "latent_feature_channels", 0),
                )
            )
            if feature_channels < 1:
                raise ValueError(
                    "latent_bfn.feature_channels must match decoder_level3"
                )
            if self.oracle_z_enabled:
                self.bfn_steps = 0
                self.refinement_mode = "none"
                self.receiver_type = "latent_oracle"
            else:
                self.latent_flow = DegradationLatentBFN(
                    feature_channels=feature_channels,
                    global_dim=self.latent_global_dim,
                    spatial_dim=self.latent_spatial_dim,
                    hidden_channels=int(
                        latent_config.get("hidden_channels", 64)
                    ),
                    num_categories=self.latent_num_categories,
                    min_variance=float(
                        latent_config.get("min_variance", 0.01)
                    ),
                    min_accuracy_scale=float(
                        latent_config.get("min_accuracy_scale", 0.25)
                    ),
                    spatial_downsample=self.latent_spatial_downsample,
                    accuracy_schedule=str(
                        latent_config.get(
                            "accuracy_schedule",
                            "exponential",
                        )
                    ),
                    loss_mode=str(
                        latent_config.get(
                            "loss_mode",
                            "all_observations",
                        )
                    ),
                    update_mode=self.latent_update_mode,
                    loss_weighting=self.latent_loss_weighting,
                )
                self.latent_flow.requires_grad_(
                    bool(latent_config.get("trainable", True))
                )
                self.bfn_steps = self.latent_steps
                self.refinement_mode = "iterative"
                self.receiver_type = "latent"
        self._validate_core_requirements(core)

    def _validate_core_requirements(self, core: dict[str, Any]) -> None:
        if core.get("require_spectral_gate", False) and not getattr(
            self.backbone, "has_paper_spectral_gate", False
        ):
            raise ValueError(
                "Paper core requires Equation (5) spectral gating in the backbone"
            )
        if (
            core.get("require_iterative_bfn", False)
            and self.refinement_mode != "iterative"
        ):
            raise ValueError(
                "Paper core requires iterative Bayesian residual refinement"
            )
        required_steps = core.get("bfn_steps")
        if required_steps is not None and self.bfn_steps != int(required_steps):
            raise ValueError(
                f"Paper core requires {int(required_steps)} BFN steps, "
                f"received {self.bfn_steps}"
            )
        if core.get("require_entropy_mask", False) and self.mask_mode != "entropy":
            raise ValueError(
                "Paper core requires entropy-guided spatial sampling"
            )

    def receiver_prediction(
        self,
        belief_mean: Tensor,
        uncertainty: Tensor,
        time: Tensor,
        structural: Tensor,
        decoder_feature: Tensor,
        degraded: Tensor | None = None,
        precision: Tensor | None = None,
        decoder_cache: Any | None = None,
        deterministic: bool = True,
    ) -> Tensor:
        if self.receiver_type == "decoder_state":
            if degraded is None:
                raise ValueError(
                    "decoder_state BFN requires the degraded image"
                )
            if precision is None:
                precision = uncertainty.reciprocal()
            state = {
                "degraded": degraded,
                "baseline": structural,
                "belief_mean": belief_mean,
                "precision": precision,
                "time": time,
                "mode": self.decoder_state_mode,
            }
            if decoder_cache is None:
                decoded, _ = self.backbone.forward_with_state(
                    degraded,
                    state,
                )
            else:
                decoded, _ = self.backbone.decode_from_cached_decoder3(
                    decoder_cache,
                    state,
                )
            prediction = (decoded - structural) / self.residual_scale
        else:
            receiver_inputs = (
                belief_mean,
                uncertainty,
                time,
                structural,
                decoder_feature,
            )
            if self.coefficient_state_readout_enabled:
                prediction = self.receiver(
                    *receiver_inputs,
                    *(
                        (degraded,)
                        if self.receiver_condition_mode == "observation_rich"
                        else ()
                    ),
                    deterministic=deterministic,
                )
            else:
                prediction = (
                    self.receiver(*receiver_inputs, degraded)
                    if self.receiver_condition_mode == "observation_rich"
                    else self.receiver(*receiver_inputs)
                )
        if self.receiver_output_mode == "tanh":
            prediction = prediction.tanh()
        return prediction

    def rollout_belief(
        self,
        structural: Tensor,
        decoder_feature: Tensor,
        accuracy_scale: Tensor,
        steps: int,
        degraded: Tensor | None = None,
        gradient_steps: int = 0,
        decoder_cache: Any | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not 0 <= gradient_steps <= steps:
            raise ValueError("gradient_steps must lie between zero and steps")
        belief_mean = torch.zeros_like(structural)
        precision = torch.ones_like(accuracy_scale)
        for index in range(1, steps + 1):
            tracks_gradient = index > steps - gradient_steps
            context = nullcontext() if tracks_gradient else torch.no_grad()
            with context:
                time = structural.new_full(
                    (structural.shape[0],),
                    (index - 1) / steps,
                )
                prediction = self.receiver_prediction(
                    belief_mean,
                    self.flow.uncertainty(precision),
                    time,
                    structural,
                    decoder_feature,
                    degraded,
                    precision,
                    decoder_cache,
                )
                alpha = self.flow.step_accuracy(
                    index,
                    steps,
                    device=structural.device,
                    dtype=structural.dtype,
                )
                alpha = alpha.view(1, 1, 1, 1) * accuracy_scale
                belief_mean, precision = self.update_residual_state(
                    belief_mean,
                    precision,
                    prediction,
                    alpha,
                )
        return belief_mean, precision

    def update_residual_state(
        self,
        mean: Tensor,
        precision: Tensor,
        observation: Tensor,
        accuracy: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the registered parameter-free residual-state update."""

        if self.residual_update_mode == "bayesian":
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

    def spatial_schedule(self, structural: Tensor) -> tuple[Tensor, Tensor]:
        if self.mask_mode == "none":
            mask = structural.new_ones(
                (structural.shape[0], 1, *structural.shape[-2:])
            )
            return mask, mask
        return self.flow.spatial_schedule(structural)

    def adapter_regularization(self, reference: Tensor) -> Tensor:
        if self.receiver is not None:
            uses_receiver_diagnostic = getattr(
                self.receiver,
                "routed_readout_enabled",
                False,
            ) or getattr(
                self.receiver,
                "coefficient_state_readout_enabled",
                False,
            )
            if uses_receiver_diagnostic:
                return self.receiver.adapter_diagnostic(reference)
        regularizer = getattr(
            self.backbone, "adapter_regularization", None
        )
        if regularizer is None:
            return reference.new_zeros(())
        return regularizer(reference)

    def coefficient_state_diagnostics(
        self,
        reference: Tensor,
        original_size: tuple[int, int] | None = None,
    ) -> dict[str, Tensor | int | str | bool]:
        if not self.coefficient_state_readout_enabled or self.receiver is None:
            return {}
        readout = self.receiver.output
        precision = readout.last_posterior_precision
        mean = readout.last_posterior_mean
        coefficients = readout.last_coefficients
        acceptance = readout.last_acceptance
        if any(
            value is None
            for value in (precision, mean, coefficients, acceptance)
        ):
            raise RuntimeError(
                "Coefficient-state readout did not record its posterior"
            )
        assert isinstance(precision, Tensor)
        assert isinstance(mean, Tensor)
        assert isinstance(coefficients, Tensor)
        assert isinstance(acceptance, Tensor)
        size = original_size or reference.shape[-2:]
        return {
            "coefficient_mean": _crop(mean.to(reference), size),
            "coefficient_precision": _crop(precision.to(reference), size),
            "coefficient_uncertainty": _crop(
                precision.reciprocal().mean(
                    dim=1,
                    keepdim=True,
                ).to(reference),
                size,
            ),
            "coefficient_maps": _crop(coefficients.to(reference), size),
            "coefficient_acceptance": _crop(
                acceptance.to(reference),
                size,
            ),
            "coefficient_gate_enabled": bool(
                readout.acceptance_gate_enabled
            ),
            "coefficient_update_mode": str(readout.update_mode),
            "coefficient_state_updates": int(readout.last_observer_calls),
            "basis_evaluations": int(readout.last_basis_evaluations),
        }

    def uncertainty_calibration(
        self,
        residual_target: Tensor,
        reference: Tensor,
        original_size: tuple[int, int],
    ) -> Tensor:
        calibration = getattr(
            self.backbone, "uncertainty_calibration", None
        )
        if calibration is None:
            return reference.new_zeros(())
        return calibration(residual_target, reference, original_size)

    def encode_semantic(self, image: Tensor) -> Tensor:
        return self.semantic_encoder(image)

    def degradation_targets_for_input(
        self,
        degraded: Tensor,
        target: Tensor,
        *,
        spatial_size: tuple[int, int],
        category: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Construct the configured GT-derived latent targets."""

        padded_degraded, degraded_size = _pad_to_multiple(
            degraded,
            self.backbone.required_multiple,
        )
        padded_target, target_size = _pad_to_multiple(
            target,
            self.backbone.required_multiple,
        )
        if degraded_size != target_size:
            raise ValueError("Degraded and target image sizes differ")
        return degradation_latent_targets(
            padded_degraded,
            padded_target,
            global_dim=self.latent_global_dim,
            spatial_dim=self.latent_spatial_dim,
            spatial_size=spatial_size,
            category=category,
            num_categories=self.latent_num_categories,
            residual_scale=self.latent_target_residual_scale,
        )

    def latent_restoration_forward(
        self,
        image: Tensor,
        semantic: Tensor | None = None,
        num_steps: int | None = None,
        return_trajectory: bool = False,
        adapter_precision_mode: str | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        dict[str, Tensor | list[Tensor]],
        Tensor,
        tuple[int, int],
    ]:
        """Run one SSR prefix, a lightweight latent flow, and one decoder."""

        (
            cache,
            semantic,
            latent_outputs,
            padded,
            original_size,
        ) = self.predict_latent_state(
            image,
            semantic=semantic,
            num_steps=num_steps,
            return_trajectory=return_trajectory,
        )
        precision_mode = (
            self.latent_adapter_precision_mode
            if adapter_precision_mode is None
            else adapter_precision_mode
        )
        state, _ = self.adapter_state_from_latent_outputs(
            latent_outputs,
            precision_mode=precision_mode,
        )
        restored, decoder_feature = (
            self.backbone.decode_from_cached_decoder3(cache, state)
        )
        return (
            restored,
            decoder_feature,
            semantic,
            latent_outputs,
            padded,
            original_size,
        )

    def predict_latent_state(
        self,
        image: Tensor,
        *,
        semantic: Tensor | None = None,
        num_steps: int | None = None,
        return_trajectory: bool = False,
    ) -> tuple[
        Any,
        Tensor,
        dict[str, Tensor | list[Tensor]],
        Tensor,
        tuple[int, int],
    ]:
        """Run the cached SSR prefix and latent predictor without decoding."""

        if self.oracle_z_enabled:
            raise RuntimeError(
                "Oracle-z bypasses the latent flow and requires a target"
            )
        if not self.latent_bfn_enabled or self.latent_flow is None:
            raise RuntimeError("Latent-BFN is not enabled")
        steps = int(num_steps or self.latent_steps)
        if steps < 1:
            raise ValueError("Latent-BFN steps must be positive")
        padded, original_size = _pad_to_multiple(
            image,
            self.backbone.required_multiple,
        )
        if semantic is None:
            semantic = self.encode_semantic(padded)
        cache = self.backbone.cache_decoder_prefix(padded)
        latent_outputs = self.latent_flow(
            cache.decoder_level3,
            steps=steps,
            return_trajectory=return_trajectory,
        )
        return (
            cache,
            semantic,
            latent_outputs,
            padded,
            original_size,
        )

    @staticmethod
    def adapter_state_from_latent_outputs(
        latent_outputs: dict[str, Tensor | list[Tensor]],
        *,
        precision_mode: str,
    ) -> tuple[dict[str, Tensor], dict[str, float | str]]:
        """Separate Bayesian update precision from the Adapter input regime."""

        if precision_mode not in {"belief", "unit"}:
            raise ValueError("precision_mode must be belief or unit")
        global_mean = latent_outputs.get("global_mean")
        global_precision = latent_outputs.get("global_precision")
        spatial_mean = latent_outputs.get("spatial_mean")
        spatial_precision = latent_outputs.get("spatial_precision")
        if not all(
            isinstance(value, Tensor)
            for value in (
                global_mean,
                global_precision,
                spatial_mean,
                spatial_precision,
            )
        ):
            raise TypeError("Latent-BFN returned an incomplete state")
        if precision_mode == "unit":
            adapter_global_precision = torch.ones_like(global_precision)
            adapter_spatial_precision = torch.ones_like(spatial_precision)
        else:
            adapter_global_precision = global_precision
            adapter_spatial_precision = spatial_precision
        global_log = adapter_global_precision.clamp_min(1e-6).log()
        spatial_log = adapter_spatial_precision.clamp_min(1e-6).log()
        if precision_mode == "unit" and (
            torch.count_nonzero(global_log).item() != 0
            or torch.count_nonzero(spatial_log).item() != 0
        ):
            raise RuntimeError(
                "Unit Adapter precision must produce exactly zero log-precision"
            )
        state = {
            "global_mean": global_mean,
            "global_precision": adapter_global_precision,
            "spatial_mean": spatial_mean,
            "spatial_precision": adapter_spatial_precision,
        }
        guard: dict[str, float | str] = {
            "adapter_input_regime": (
                "unit_precision"
                if precision_mode == "unit"
                else "belief_precision"
            ),
            "global_mean_abs_log_precision": float(
                global_log.detach().float().abs().mean()
            ),
            "spatial_mean_abs_log_precision": float(
                spatial_log.detach().float().abs().mean()
            ),
        }
        return state, guard

    def decode_cached_latent_state(
        self,
        cache: Any,
        state: dict[str, Tensor],
        *,
        original_size: tuple[int, int],
    ) -> Tensor:
        """Decode one cached prefix from one externally transformed state."""

        required = {
            "global_mean",
            "global_precision",
            "spatial_mean",
            "spatial_precision",
        }
        if set(state) != required:
            raise ValueError("Latent state keys are incomplete")
        restored, _ = self.backbone.decode_from_cached_decoder3(
            cache,
            state,
        )
        return _crop(restored.clamp(0, 1), original_size)

    def oracle_z_restoration_forward(
        self,
        degraded: Tensor,
        target: Tensor,
        semantic: Tensor | None = None,
        category: Tensor | None = None,
        ablation: str = "full",
        global_replacement: Tensor | None = None,
        spatial_replacement: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        dict[str, dict[str, Tensor]],
        tuple[int, int],
    ]:
        """Inject GT-derived degradation targets without a latent predictor."""

        if not self.oracle_z_enabled:
            raise RuntimeError("Oracle-z mode is not enabled")
        if self.latent_flow is not None:
            raise RuntimeError("Oracle-z must not instantiate a latent flow")
        if category is None:
            raise ValueError("Oracle-z requires GT degradation categories")
        if ablation not in {"full", "global", "spatial"}:
            raise ValueError(
                "Oracle-z ablation must be full, global, or spatial"
            )
        padded_degraded, original_size = _pad_to_multiple(
            degraded,
            self.backbone.required_multiple,
        )
        padded_target, target_size = _pad_to_multiple(
            target,
            self.backbone.required_multiple,
        )
        if target_size != original_size:
            raise ValueError("Oracle-z degraded and target sizes differ")
        if semantic is None:
            semantic = self.encode_semantic(padded_degraded)
        cache = self.backbone.cache_decoder_prefix(padded_degraded)
        spatial_size = (
            max(
                1,
                cache.decoder_level3.shape[-2]
                // self.latent_spatial_downsample,
            ),
            max(
                1,
                cache.decoder_level3.shape[-1]
                // self.latent_spatial_downsample,
            ),
        )
        global_target, spatial_target = degradation_latent_targets(
            padded_degraded,
            padded_target,
            global_dim=self.latent_global_dim,
            spatial_dim=self.latent_spatial_dim,
            spatial_size=spatial_size,
            category=category,
            num_categories=self.latent_num_categories,
            residual_scale=self.latent_target_residual_scale,
        )
        if ablation == "spatial":
            if global_replacement is None:
                injected_global = torch.zeros_like(global_target)
            else:
                injected_global = global_replacement.to(
                    device=global_target.device,
                    dtype=global_target.dtype,
                )
                if injected_global.ndim == 1:
                    injected_global = injected_global[None]
                injected_global = injected_global.expand_as(global_target)
        else:
            injected_global = global_target
        if ablation == "global":
            if spatial_replacement is None:
                injected_spatial = torch.zeros_like(spatial_target)
            else:
                injected_spatial = spatial_replacement.to(
                    device=spatial_target.device,
                    dtype=spatial_target.dtype,
                )
                if injected_spatial.ndim == 3:
                    injected_spatial = injected_spatial[None]
                injected_spatial = F.interpolate(
                    injected_spatial,
                    size=spatial_target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).expand_as(spatial_target)
        else:
            injected_spatial = spatial_target
        state = {
            "global_mean": injected_global,
            "global_precision": torch.ones_like(global_target),
            "spatial_mean": injected_spatial,
            "spatial_precision": torch.ones_like(spatial_target),
        }
        restored, decoder_feature = (
            self.backbone.decode_from_cached_decoder3(cache, state)
        )
        diagnostics_fn = getattr(
            self.backbone,
            "latent_modulation_diagnostics",
            None,
        )
        if diagnostics_fn is None:
            raise RuntimeError(
                "Oracle-z foundation cannot report modulation diagnostics"
            )
        diagnostics = diagnostics_fn()
        return (
            restored,
            decoder_feature,
            semantic,
            global_target,
            spatial_target,
            injected_global,
            injected_spatial,
            diagnostics,
            original_size,
        )

    def oracle_z_training_outputs(
        self,
        degraded: Tensor,
        target: Tensor,
        semantic: Tensor | None = None,
        category: Tensor | None = None,
    ) -> dict[str, Tensor]:
        (
            restored,
            _,
            semantic,
            global_target,
            spatial_target,
            _,
            _,
            _,
            original_size,
        ) = self.oracle_z_restoration_forward(
            degraded,
            target,
            semantic,
            category,
        )
        cropped = _crop(restored, original_size)
        return {
            "restored": cropped,
            "structural": cropped,
            "foundation": cropped,
            "mean_correction": torch.zeros_like(cropped),
            "residual_prediction": torch.zeros_like(cropped),
            "residual_target": target - cropped,
            "uncertainty": cropped.new_ones(
                (cropped.shape[0], 1, *cropped.shape[-2:])
            ),
            "sampling_mask": cropped.new_ones(
                (cropped.shape[0], 1, *cropped.shape[-2:])
            ),
            "bfn_loss": cropped.new_zeros(()),
            "adapter_regularization": self.adapter_regularization(restored),
            "uncertainty_calibration": cropped.new_zeros(()),
            "time": cropped.new_zeros((cropped.shape[0],)),
            "semantic": semantic,
            "latent_global_target": global_target,
            "latent_spatial_target": spatial_target,
            "latent_bfn_steps": cropped.new_tensor(0),
            "decoder_calls": cropped.new_tensor(1),
            "oracle_z": cropped.new_tensor(True),
        }

    @torch.no_grad()
    def restore_oracle_z(
        self,
        degraded: Tensor,
        target: Tensor,
        *,
        category: Tensor,
        semantic: Tensor | None = None,
        ablation: str = "full",
        global_replacement: Tensor | None = None,
        spatial_replacement: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | dict[str, Any]:
        (
            restored,
            _,
            semantic,
            global_target,
            spatial_target,
            injected_global,
            injected_spatial,
            diagnostics,
            original_size,
        ) = self.oracle_z_restoration_forward(
            degraded,
            target,
            semantic,
            category,
            ablation,
            global_replacement,
            spatial_replacement,
        )
        restored = _crop(restored.clamp(0, 1), original_size)
        if not return_details:
            return restored
        return {
            "restored": restored,
            "structural": restored,
            "foundation": restored,
            "mean_correction": torch.zeros_like(restored),
            "residual": torch.zeros_like(restored),
            "uncertainty": restored.new_ones(
                (restored.shape[0], 1, *restored.shape[-2:])
            ),
            "sampling_mask": restored.new_ones(
                (restored.shape[0], 1, *restored.shape[-2:])
            ),
            "semantic": semantic,
            "latent_global_target": global_target,
            "latent_spatial_target": spatial_target,
            "injected_global_mean": injected_global,
            "injected_spatial_mean": injected_spatial,
            "modulation_diagnostics": diagnostics,
            "latent_bfn_steps": 0,
            "decoder_calls": 1,
            "oracle_z": True,
            "oracle_z_ablation": ablation,
            "deployable": False,
        }

    @torch.no_grad()
    def restore_from_latent_state(
        self,
        degraded: Tensor,
        *,
        global_mean: Tensor,
        global_precision: Tensor,
        spatial_mean: Tensor,
        spatial_precision: Tensor,
        semantic: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | dict[str, Any]:
        """Decode one externally supplied predicted latent state."""

        if self.oracle_z_enabled or self.latent_flow is None:
            raise RuntimeError(
                "External predicted states require a deployable latent model"
            )
        padded, original_size = _pad_to_multiple(
            degraded,
            self.backbone.required_multiple,
        )
        if semantic is None:
            semantic = self.encode_semantic(padded)
        cache = self.backbone.cache_decoder_prefix(padded)
        state = {
            "global_mean": global_mean,
            "global_precision": global_precision,
            "spatial_mean": spatial_mean,
            "spatial_precision": spatial_precision,
        }
        restored, _ = self.backbone.decode_from_cached_decoder3(
            cache,
            state,
        )
        restored = _crop(restored.clamp(0, 1), original_size)
        if not return_details:
            return restored
        diagnostics_fn = getattr(
            self.backbone,
            "latent_modulation_diagnostics",
            None,
        )
        if diagnostics_fn is None:
            raise RuntimeError(
                "Latent foundation cannot report modulation diagnostics"
            )
        return {
            "restored": restored,
            "structural": restored,
            "foundation": restored,
            "latent_global_mean": global_mean,
            "latent_global_precision": global_precision,
            "latent_spatial_mean": spatial_mean,
            "latent_spatial_precision": spatial_precision,
            "modulation_diagnostics": diagnostics_fn(),
            "decoder_calls": 1,
            "deployable": True,
        }

    def latent_training_outputs(
        self,
        degraded: Tensor,
        target: Tensor,
        semantic: Tensor | None = None,
        num_steps: int | None = None,
        category: Tensor | None = None,
    ) -> dict[str, Tensor]:
        (
            restored,
            _,
            semantic,
            latent_outputs,
            padded_degraded,
            original_size,
        ) = self.latent_restoration_forward(
            degraded,
            semantic,
            num_steps,
        )
        padded_target, _ = _pad_to_multiple(
            target,
            self.backbone.required_multiple,
        )
        spatial_mean = latent_outputs["spatial_mean"]
        category_logits = latent_outputs["category_logits"]
        if not isinstance(spatial_mean, Tensor) or not isinstance(
            category_logits,
            Tensor,
        ):
            raise TypeError("Latent-BFN training outputs are incomplete")
        global_target, spatial_target = degradation_latent_targets(
            padded_degraded,
            padded_target,
            global_dim=self.latent_global_dim,
            spatial_dim=self.latent_spatial_dim,
            spatial_size=spatial_mean.shape[-2:],
            category=category,
            num_categories=self.latent_num_categories,
            residual_scale=self.latent_target_residual_scale,
        )
        latent_loss = self.latent_flow.latent_loss(
            latent_outputs,
            global_target,
            spatial_target,
        )
        cropped = _crop(restored, original_size)
        residual_target = _crop(padded_target - restored, original_size)
        spatial_precision = latent_outputs["spatial_precision"]
        if not isinstance(spatial_precision, Tensor):
            raise TypeError("Latent spatial precision must be a tensor")
        uncertainty = F.interpolate(
            spatial_precision.reciprocal().mean(dim=1, keepdim=True),
            size=restored.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        outputs = {
            "restored": cropped,
            "structural": cropped,
            # Computing a separate unmodulated image would execute the decoder
            # twice. Arm C supplies that comparison in the paired protocol.
            "foundation": cropped,
            "mean_correction": torch.zeros_like(cropped),
            "residual_prediction": torch.zeros_like(cropped),
            "residual_target": residual_target,
            "uncertainty": _crop(uncertainty, original_size),
            "sampling_mask": cropped.new_ones(
                (cropped.shape[0], 1, *cropped.shape[-2:])
            ),
            "bfn_loss": latent_loss,
            "adapter_regularization": self.adapter_regularization(restored),
            "uncertainty_calibration": restored.new_zeros(()),
            "time": restored.new_full(
                (restored.shape[0],),
                1.0,
            ),
            "semantic": semantic,
            "latent_category_logits": category_logits,
            "latent_global_target": global_target,
            "latent_spatial_target": spatial_target,
            "latent_bfn_steps": restored.new_tensor(
                int(num_steps or self.latent_steps)
            ),
            "decoder_calls": restored.new_tensor(1),
        }
        if category is not None:
            outputs["latent_category_target"] = category
        return outputs

    def restoration_forward(
        self,
        image: Tensor,
        semantic: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor,
        Tensor,
        tuple[int, int],
    ]:
        padded, original_size = _pad_to_multiple(
            image, self.backbone.required_multiple
        )
        if semantic is None:
            semantic = self.encode_semantic(padded)
        foundation, decoder_feature = self.backbone(padded, semantic)
        if self.mean_corrector is None:
            mean_correction = torch.zeros_like(foundation)
            condition = None
        else:
            mean_correction, condition = self.mean_corrector(
                padded,
                foundation,
                decoder_feature,
            )
        structural = foundation + mean_correction
        return (
            structural,
            foundation,
            mean_correction,
            condition,
            decoder_feature,
            semantic,
            original_size,
        )

    @torch.no_grad()
    def decoder_state_restoration(
        self,
        image: Tensor,
        semantic: Tensor | None = None,
    ) -> tuple[
        tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor | None,
            Tensor,
            Tensor,
            tuple[int, int],
        ],
        Any,
    ]:
        """Return the fixed base restoration and one reusable SSR decoder cache."""

        if self.receiver_type != "decoder_state" or not self.decoder_state_use_cache:
            raise RuntimeError("Decoder-state cache is not enabled")
        padded, original_size = _pad_to_multiple(
            image, self.backbone.required_multiple
        )
        if semantic is None:
            semantic = self.encode_semantic(padded)
        cache = self.backbone.cache_decoder_prefix(padded)
        foundation, decoder_feature = self.backbone.decode_from_cached_decoder3(
            cache
        )
        mean_correction = torch.zeros_like(foundation)
        return (
            (
                foundation,
                foundation,
                mean_correction,
                None,
                decoder_feature,
                semantic,
                original_size,
            ),
            cache,
        )

    def structural_forward(
        self,
        image: Tensor,
        semantic: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[int, int]]:
        structural, _, _, _, decoder_feature, semantic, original_size = (
            self.restoration_forward(image, semantic)
        )
        return structural, decoder_feature, semantic, original_size

    @torch.no_grad()
    def restore_structural(
        self,
        degraded: Tensor,
        semantic: Tensor | None = None,
    ) -> Tensor:
        """Return the deployed frozen-foundation output without BFN refinement."""

        if self.oracle_z_enabled:
            raise RuntimeError(
                "Oracle-z is non-deployable and requires the clean target"
            )
        if self.latent_bfn_enabled:
            restored = self.restore(
                degraded,
                semantic=semantic,
                deterministic=True,
            )
            if not isinstance(restored, Tensor):
                raise TypeError("Latent restoration did not return a tensor")
            return restored
        if self.receiver_type == "decoder_state" and self.decoder_state_use_cache:
            restoration, _ = self.decoder_state_restoration(
                degraded,
                semantic,
            )
        else:
            restoration = self.restoration_forward(degraded, semantic)
        structural, _, _, _, _, _, original_size = restoration
        return _crop(structural.clamp(0, 1), original_size)

    def training_outputs(
        self,
        degraded: Tensor,
        target: Tensor,
        semantic: Tensor | None = None,
        num_steps: int | None = None,
        training_stage: str = "joint",
        category: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if training_stage not in {
            "structure",
            "mean",
            "residual",
            "bfn",
            "joint",
        }:
            raise ValueError(f"Unsupported training stage: {training_stage}")
        if self.oracle_z_enabled:
            if training_stage != "joint":
                raise ValueError("Oracle-z uses the joint training stage")
            return self.oracle_z_training_outputs(
                degraded,
                target,
                semantic,
                category,
            )
        if self.latent_bfn_enabled:
            if training_stage != "joint":
                raise ValueError(
                    "Latent-BFN uses the joint training stage"
                )
            return self.latent_training_outputs(
                degraded,
                target,
                semantic,
                num_steps,
                category,
            )
        if (
            self.receiver is None
            and self.receiver_type == "unet"
            and training_stage not in {"structure", "mean"}
        ):
            raise ValueError(
                "inference_mode=none requires training_stage=structure or mean"
            )
        steps = int(num_steps or self.bfn_steps)
        if (
            self.coefficient_state_readout_enabled
            and steps != self.receiver.output.steps
        ):
            raise ValueError(
                "Coefficient-state steps are fixed by the readout config"
            )
        decoder_cache = None
        if self.receiver_type == "decoder_state" and self.decoder_state_use_cache:
            restoration, decoder_cache = self.decoder_state_restoration(
                degraded,
                semantic,
            )
        elif (
            training_stage in {"residual", "bfn"}
            or self.receiver_type == "decoder_state"
        ):
            with torch.no_grad():
                restoration = self.restoration_forward(
                    degraded,
                    semantic,
                )
        else:
            restoration = self.restoration_forward(
                degraded,
                semantic,
            )
        (
            structural,
            foundation,
            mean_correction,
            condition,
            decoder_feature,
            semantic,
            original_size,
        ) = restoration
        if self.residual_base_clamp:
            structural = structural.clamp(0, 1)
        degraded_condition, _ = _pad_to_multiple(
            degraded, self.backbone.required_multiple
        )
        target, _ = _pad_to_multiple(target, self.backbone.required_multiple)
        residual_target = target - structural
        normalized_target = residual_target / self.residual_scale
        mask, accuracy_scale = self.spatial_schedule(structural)
        adapter_regularization = self.adapter_regularization(structural)
        uncertainty_calibration = self.uncertainty_calibration(
            residual_target,
            structural,
            original_size,
        )

        if training_stage in {"structure", "mean"}:
            zero_residual = torch.zeros_like(structural)
            return {
                "restored": _crop(structural, original_size),
                "structural": _crop(structural, original_size),
                "foundation": _crop(foundation, original_size),
                "mean_correction": _crop(
                    mean_correction,
                    original_size,
                ),
                "residual_prediction": _crop(zero_residual, original_size),
                "residual_target": _crop(residual_target, original_size),
                "uncertainty": _crop(torch.ones_like(mask), original_size),
                "sampling_mask": _crop(mask, original_size),
                "bfn_loss": structural.new_zeros(()),
                "adapter_regularization": adapter_regularization,
                "uncertainty_calibration": uncertainty_calibration,
                "time": structural.new_zeros((structural.shape[0],)),
                "semantic": semantic,
            }

        if training_stage == "residual":
            uncertainty = torch.ones_like(mask)
            time = structural.new_ones((structural.shape[0],))
            residual_prediction = self.receiver_prediction(
                torch.zeros_like(structural),
                uncertainty,
                time,
                structural,
                decoder_feature,
                degraded_condition,
                torch.ones_like(uncertainty),
                decoder_cache,
            )
            residual_prediction = residual_prediction * self.residual_scale
            restored = structural + mask * residual_prediction
            coefficient_diagnostics = self.coefficient_state_diagnostics(
                structural,
                original_size,
            )
            bfn_loss = structural.new_zeros(())
            if coefficient_diagnostics:
                uncertainty = coefficient_diagnostics[
                    "coefficient_uncertainty"
                ]
                assert isinstance(uncertainty, Tensor)
                bfn_loss = self.receiver.output.bfn_training_loss(
                    normalized_target
                )
            outputs: dict[str, Tensor | int | str | bool] = {
                "restored": _crop(restored, original_size),
                "structural": _crop(structural, original_size),
                "foundation": _crop(foundation, original_size),
                "mean_correction": _crop(
                    mean_correction,
                    original_size,
                ),
                "residual_prediction": _crop(
                    residual_prediction, original_size
                ),
                "residual_target": _crop(residual_target, original_size),
                "uncertainty": _crop(uncertainty, original_size),
                "sampling_mask": _crop(mask, original_size),
                "bfn_loss": bfn_loss,
                "adapter_regularization": self.adapter_regularization(
                    structural
                ),
                "uncertainty_calibration": uncertainty_calibration,
                "time": time,
                "semantic": semantic,
            }
            outputs.update(coefficient_diagnostics)
            return outputs

        step = torch.randint(
            1, steps + 1, (degraded.shape[0],), device=degraded.device
        )
        time = (step.to(degraded.dtype) - 1.0) / steps
        belief_mean, precision = self.flow.sender_state(
            normalized_target, time, accuracy_scale
        )
        normalized_uncertainty = self.flow.uncertainty(precision)
        normalized_prediction = self.receiver_prediction(
            belief_mean,
            normalized_uncertainty,
            time,
            structural,
            decoder_feature,
            degraded_condition,
            precision,
            decoder_cache,
        )
        bfn_loss = self.flow.discrete_loss(
            normalized_prediction,
            normalized_target,
            step,
            steps,
            accuracy_scale,
        ) * self.residual_scale**2
        residual_prediction = normalized_prediction * self.residual_scale

        final_prediction = residual_prediction
        final_loss_mask = structural.new_ones(
            (structural.shape[0], 1, 1, 1)
        )
        if training_stage == "joint" and self.training_final_state == "terminal":
            final_time = structural.new_ones((structural.shape[0],))
            final_mean, final_precision = self.flow.sender_state(
                normalized_target,
                final_time,
                accuracy_scale,
            )
            final_prediction = self.receiver_prediction(
                final_mean,
                self.flow.uncertainty(final_precision),
                final_time,
                structural,
                decoder_feature,
                degraded_condition,
                final_precision,
                decoder_cache,
            ) * self.residual_scale
        elif training_stage == "joint" and self.training_final_state == "rollout":
            rollout_count = max(
                1,
                math.ceil(
                    structural.shape[0] * self.rollout_batch_fraction
                ),
            )
            indices = torch.randperm(
                structural.shape[0],
                device=structural.device,
            )[:rollout_count]
            rollout_structural = structural.index_select(0, indices)
            rollout_feature = decoder_feature.index_select(0, indices)
            rollout_scale = accuracy_scale.index_select(0, indices)
            rollout_degraded = degraded_condition.index_select(0, indices)
            rollout_cache = (
                decoder_cache.index_select(indices)
                if decoder_cache is not None
                else None
            )
            rollout_mean, rollout_precision = self.rollout_belief(
                rollout_structural,
                rollout_feature,
                rollout_scale,
                self.training_rollout_steps,
                degraded=rollout_degraded,
                gradient_steps=self.rollout_backprop_steps,
                decoder_cache=rollout_cache,
            )
            if self.residual_readout_mode == "posterior_mean":
                rollout_prediction = rollout_mean * self.residual_scale
            else:
                final_time = structural.new_ones((rollout_count,))
                rollout_prediction = self.receiver_prediction(
                    rollout_mean,
                    self.flow.uncertainty(rollout_precision),
                    final_time,
                    rollout_structural,
                    rollout_feature,
                    rollout_degraded,
                    rollout_precision,
                    rollout_cache,
                ) * self.residual_scale
            final_prediction = torch.zeros_like(
                residual_prediction
            ).index_copy(0, indices, rollout_prediction)
            final_loss_mask = structural.new_zeros(
                (structural.shape[0], 1, 1, 1)
            ).index_fill(0, indices, 1.0)
        restored = structural + mask * final_prediction

        return {
            "restored": _crop(restored, original_size),
            "structural": _crop(structural, original_size),
            "foundation": _crop(foundation, original_size),
            "mean_correction": _crop(mean_correction, original_size),
            "residual_prediction": _crop(residual_prediction, original_size),
            "residual_target": _crop(residual_target, original_size),
            "uncertainty": _crop(
                normalized_uncertainty * self.residual_scale**2,
                original_size,
            ),
            "sampling_mask": _crop(mask, original_size),
            "final_loss_mask": final_loss_mask,
            "bfn_loss": bfn_loss,
            "adapter_regularization": self.adapter_regularization(structural),
            "uncertainty_calibration": uncertainty_calibration,
            "time": time,
            "semantic": semantic,
        }

    @torch.no_grad()
    def restore(
        self,
        degraded: Tensor,
        semantic: Tensor | None = None,
        num_steps: int | None = None,
        deterministic: bool = False,
        residual_gain: float = 1.0,
        state_ablation: str = "normal",
        return_details: bool = False,
        return_trajectory: bool = False,
        adapter_precision_mode: str | None = None,
    ) -> Tensor | dict[str, Any]:
        if not 0.0 <= residual_gain <= 1.0:
            raise ValueError("residual_gain must lie in [0, 1]")
        state_ablation = _validate_state_ablation(state_ablation)
        if self.oracle_z_enabled:
            raise RuntimeError(
                "Oracle-z is non-deployable; use restore_oracle_z with a target"
            )
        if self.latent_bfn_enabled:
            precision_mode = (
                self.latent_adapter_precision_mode
                if adapter_precision_mode is None
                else adapter_precision_mode
            )
            if residual_gain != 1.0:
                raise ValueError(
                    "residual_gain is not defined for Latent-BFN modulation"
                )
            if state_ablation != "normal":
                raise ValueError(
                    "state_ablation is not defined for Latent-BFN"
                )
            (
                restored,
                _,
                semantic,
                latent_outputs,
                _,
                original_size,
            ) = self.latent_restoration_forward(
                degraded,
                semantic,
                num_steps,
                return_trajectory=return_trajectory,
                adapter_precision_mode=precision_mode,
            )
            restored = _crop(restored.clamp(0, 1), original_size)
            if not return_details:
                return restored
            spatial_precision = latent_outputs["spatial_precision"]
            if not isinstance(spatial_precision, Tensor):
                raise TypeError("Latent spatial precision must be a tensor")
            uncertainty = F.interpolate(
                spatial_precision.reciprocal().mean(
                    dim=1,
                    keepdim=True,
                ),
                size=restored.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            details: dict[str, Any] = {
                "restored": restored,
                "structural": restored,
                "structural_raw": restored,
                "foundation": restored,
                "mean_correction": torch.zeros_like(restored),
                "residual": torch.zeros_like(restored),
                "uncertainty": uncertainty,
                "sampling_mask": torch.ones_like(uncertainty),
                "semantic": semantic,
                "latent_global_mean": latent_outputs["global_mean"],
                "latent_global_precision": latent_outputs[
                    "global_precision"
                ],
                "latent_spatial_mean": latent_outputs["spatial_mean"],
                "latent_spatial_precision": spatial_precision,
                "latent_category_logits": latent_outputs[
                    "category_logits"
                ],
                "latent_bfn_steps": int(num_steps or self.latent_steps),
                "decoder_calls": 1,
            }
            _, precision_guard = self.adapter_state_from_latent_outputs(
                latent_outputs,
                precision_mode=precision_mode,
            )
            details["adapter_precision_guard"] = precision_guard
            details["latent_update_mode"] = self.latent_update_mode
            diagnostics_fn = getattr(
                self.backbone,
                "latent_modulation_diagnostics",
                None,
            )
            if diagnostics_fn is not None:
                details["modulation_diagnostics"] = diagnostics_fn()
            if return_trajectory:
                details["trajectory"] = latent_outputs["trajectory"]
            return details
        steps = int(num_steps or self.bfn_steps)
        if (
            self.coefficient_state_readout_enabled
            and steps != self.receiver.output.steps
        ):
            raise ValueError(
                "Coefficient-state steps are fixed by the readout config"
            )
        decoder_state_calls = 0
        receiver_calls = 0
        state_update_schedule: list[dict[str, float | int]] = []

        def predict_residual(*inputs: Any) -> Tensor:
            nonlocal decoder_state_calls, receiver_calls
            receiver_calls += 1
            if self.receiver_type == "decoder_state":
                decoder_state_calls += 1
            return self.receiver_prediction(*inputs)

        decoder_cache = None
        if self.receiver_type == "decoder_state" and self.decoder_state_use_cache:
            restoration, decoder_cache = self.decoder_state_restoration(
                degraded,
                semantic,
            )
        else:
            restoration = self.restoration_forward(degraded, semantic)
        (
            structural,
            foundation,
            mean_correction,
            _,
            decoder_feature,
            semantic,
            original_size,
        ) = restoration
        if self.residual_base_clamp:
            structural = structural.clamp(0, 1)
        degraded_condition, _ = _pad_to_multiple(
            degraded, self.backbone.required_multiple
        )
        if self.refinement_mode == "none":
            if state_ablation != "normal":
                raise ValueError(
                    "state_ablation requires iterative decoder-state BFN inference"
                )
            restored = _crop(structural.clamp(0, 1), original_size)
            if not return_details:
                return restored
            details = {
                "restored": restored,
                "structural": restored,
                "foundation": _crop(
                    foundation.clamp(0, 1),
                    original_size,
                ),
                "mean_correction": _crop(
                    mean_correction,
                    original_size,
                ),
                "residual": torch.zeros_like(restored),
                "semantic": semantic,
            }
            uncertainty_fn = getattr(self.backbone, "adapter_uncertainty_map", None)
            variance = uncertainty_fn(structural) if uncertainty_fn else None
            if variance is not None:
                details["adapter_uncertainty"] = _crop(variance, original_size)
            if return_trajectory:
                details["trajectory"] = []
            return details
        mask, accuracy_scale = self.spatial_schedule(structural)
        if self.refinement_mode == "direct":
            if state_ablation != "normal":
                raise ValueError(
                    "state_ablation requires iterative decoder-state BFN inference"
                )
            uncertainty = torch.ones_like(mask)
            final_time = structural.new_ones((structural.shape[0],))
            raw_residual = predict_residual(
                torch.zeros_like(structural),
                uncertainty,
                final_time,
                structural,
                decoder_feature,
                degraded_condition,
                torch.ones_like(uncertainty),
                decoder_cache,
                deterministic,
            )
            raw_residual = raw_residual * self.residual_scale
            residual = raw_residual * residual_gain
            coefficient_diagnostics = self.coefficient_state_diagnostics(
                structural,
                original_size,
            )
            if coefficient_diagnostics:
                uncertainty = coefficient_diagnostics[
                    "coefficient_uncertainty"
                ]
                assert isinstance(uncertainty, Tensor)
            restored = _crop(
                (structural + mask * residual).clamp(0, 1),
                original_size,
            )
            if not return_details:
                return restored
            details: dict[str, Tensor | list[Tensor]] = {
                "restored": restored,
                "foundation": _crop(
                    foundation.clamp(0, 1),
                    original_size,
                ),
                "mean_correction": _crop(
                    mean_correction,
                    original_size,
                ),
                "structural_raw": _crop(structural, original_size),
                "structural": _crop(
                    structural.clamp(0, 1), original_size
                ),
                "residual": _crop(residual, original_size),
                "residual_ungained": _crop(raw_residual, original_size),
                "uncertainty": _crop(uncertainty, original_size),
                "sampling_mask": _crop(mask, original_size),
                "semantic": semantic,
                "decoder_state_calls": decoder_state_calls,
                "receiver_calls": receiver_calls,
                "decoder_state_updates": 0,
                "state_update_schedule": state_update_schedule,
                "terminal_readout_time": float(final_time[0]),
            }
            details.update(coefficient_diagnostics)
            if return_trajectory and self.coefficient_state_readout_enabled:
                details["trajectory"] = [
                    _crop(
                        state.mean(dim=1, keepdim=True),
                        original_size,
                    )
                    for state in self.receiver.output.last_trajectory
                ]
            if return_trajectory:
                details.setdefault(
                    "trajectory",
                    [_crop(residual.clone(), original_size)],
                )
            return details

        if state_ablation != "normal" and self.receiver_type != "decoder_state":
            raise ValueError(
                "state_ablation requires receiver_type=decoder_state"
            )
        if state_ablation == "shuffle-belief" and structural.shape[0] < 2:
            raise ValueError(
                "shuffle-belief requires inference batch size of at least two"
            )
        belief_mean = torch.zeros_like(structural)
        precision = torch.ones_like(mask)
        trajectory: list[Tensor] = []

        for index in range(1, steps + 1):
            time = structural.new_full(
                (structural.shape[0],), (index - 1) / steps
            )
            receiver_belief = belief_mean
            receiver_precision = precision
            if state_ablation == "reset-belief":
                receiver_belief = torch.zeros_like(belief_mean)
                receiver_precision = torch.ones_like(precision)
            elif state_ablation == "shuffle-belief":
                shift = index % structural.shape[0] or 1
                receiver_belief = belief_mean.roll(shift, dims=0)
                receiver_precision = precision.roll(shift, dims=0)
            uncertainty = self.flow.uncertainty(receiver_precision)
            residual_prediction = predict_residual(
                receiver_belief,
                uncertainty,
                time,
                structural,
                decoder_feature,
                degraded_condition,
                receiver_precision,
                decoder_cache,
            )
            alpha_scalar = self.flow.step_accuracy(
                index,
                steps,
                device=structural.device,
                dtype=structural.dtype,
            )
            state_update_schedule.append({
                "step": index,
                "time": float(time[0]),
                "alpha": float(alpha_scalar),
            })
            alpha = alpha_scalar.view(1, 1, 1, 1) * accuracy_scale
            if deterministic:
                observation = residual_prediction
            else:
                noise = torch.randn(
                    residual_prediction.shape,
                    device=residual_prediction.device,
                    dtype=residual_prediction.dtype,
                )
                observation = residual_prediction + noise / alpha.clamp_min(
                    1e-8
                ).sqrt()
            belief_mean, precision = self.update_residual_state(
                belief_mean, precision, observation, alpha
            )
            if return_trajectory:
                trajectory.append(
                    _crop(
                        belief_mean.clone() * self.residual_scale,
                        original_size,
                    )
                )

        final_time = structural.new_ones((structural.shape[0],))
        receiver_belief = belief_mean
        receiver_precision = precision
        if state_ablation == "reset-belief":
            receiver_belief = torch.zeros_like(belief_mean)
            receiver_precision = torch.ones_like(precision)
        elif state_ablation == "shuffle-belief":
            shift = (steps + 1) % structural.shape[0] or 1
            receiver_belief = belief_mean.roll(shift, dims=0)
            receiver_precision = precision.roll(shift, dims=0)
        uncertainty = self.flow.uncertainty(receiver_precision)
        if self.residual_readout_mode == "posterior_mean":
            raw_residual = receiver_belief * self.residual_scale
            terminal_readout_time: float | None = None
        else:
            raw_residual = predict_residual(
                receiver_belief,
                uncertainty,
                final_time,
                structural,
                decoder_feature,
                degraded_condition,
                receiver_precision,
                decoder_cache,
            ) * self.residual_scale
            terminal_readout_time = float(final_time[0])
        residual = raw_residual * residual_gain
        restored = (structural + mask * residual).clamp(0, 1)
        restored = _crop(restored, original_size)
        if not return_details:
            return restored
        details: dict[str, Tensor | list[Tensor]] = {
            "restored": restored,
            "foundation": _crop(
                foundation.clamp(0, 1),
                original_size,
            ),
            "mean_correction": _crop(
                mean_correction,
                original_size,
            ),
            "structural_raw": _crop(structural, original_size),
            "structural": _crop(structural.clamp(0, 1), original_size),
            "residual": _crop(residual, original_size),
            "residual_ungained": _crop(raw_residual, original_size),
            "uncertainty": _crop(
                uncertainty * self.residual_scale**2,
                original_size,
            ),
            "sampling_mask": _crop(mask, original_size),
            "semantic": semantic,
            "state_ablation": state_ablation,
            "decoder_state_calls": decoder_state_calls,
            "receiver_calls": receiver_calls,
            "decoder_state_updates": steps,
            "state_update_schedule": state_update_schedule,
            "terminal_readout_time": terminal_readout_time,
            "residual_update_mode": self.residual_update_mode,
            "residual_readout_mode": self.residual_readout_mode,
        }
        if return_trajectory:
            details["trajectory"] = trajectory
        return details

    def forward(
        self,
        degraded: Tensor,
        target: Tensor | None = None,
        semantic: Tensor | None = None,
        num_steps: int | None = None,
        training_stage: str = "joint",
        category: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        if target is None:
            return self.restore(
                degraded, semantic=semantic, num_steps=num_steps
            )
        return self.training_outputs(
            degraded,
            target,
            semantic=semantic,
            num_steps=num_steps,
            training_stage=training_stage,
            category=category,
        )


SGBFormer = BasisBFN


def build_model(config: dict[str, Any]) -> BasisBFN:
    model_config = dict(config.get("model", config))
    return BasisBFN(**model_config)
