"""Adapters for externally maintained restoration foundations."""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import (
    CalibratedSpectralAdapter,
    LatentConditionedFiLMSpectralAdapter,
    PaperSpectralGateAdapter,
    StateConditionedSpectralAdapter,
)


@dataclass(frozen=True)
class SSRDecoderCache:
    """Frozen SSR prefix features required after decoder_level3."""

    image: Tensor
    encoder_level1: Tensor
    encoder_level2: Tensor
    decoder_level3: Tensor

    def index_select(self, indices: Tensor) -> "SSRDecoderCache":
        return SSRDecoderCache(
            image=self.image.index_select(0, indices),
            encoder_level1=self.encoder_level1.index_select(0, indices),
            encoder_level2=self.encoder_level2.index_select(0, indices),
            decoder_level3=self.decoder_level3.index_select(0, indices),
        )


def _load_module(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"External architecture was not found: {path}")
    safe_path = re.sub(r"\W", "_", str(path.resolve()))
    module_name = f"_sgbformer_external_ssr_{safe_path}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import external architecture: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _extract_state_dict(
    checkpoint: object,
    param_key: str | None,
) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Foundation checkpoint must contain a state dictionary")
    keys = [param_key] if param_key else []
    keys.extend(("params_ema", "params", "state_dict", "model"))
    state: object = checkpoint
    for key in keys:
        if key and key in checkpoint and isinstance(checkpoint[key], dict):
            state = checkpoint[key]
            break
    if not isinstance(state, dict) or not all(
        isinstance(value, Tensor) for value in state.values()
    ):
        raise TypeError(
            "Could not locate a tensor state dictionary in the foundation "
            "checkpoint"
        )
    cleaned = {}
    for name, value in state.items():
        for prefix in ("module.", "_orig_mod.", "net_g."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


class CheckpointCompatibleEDFFN(nn.Module):
    """Frequency FFN required by the released SSR checkpoint.

    The checkpoint contains the EDFFN parameters used by EVSSM, while the
    current SSR repository defines a plain FFN under the same class name.
    """

    def __init__(
        self,
        dim: int,
        ffn_expansion_factor: float,
        bias: bool,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = int(patch_size)
        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.fft = nn.Parameter(
            torch.ones(
                (
                    dim,
                    1,
                    1,
                    self.patch_size,
                    self.patch_size // 2 + 1,
                )
            )
        )
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        batch, channels, height, width = x.shape
        patch = self.patch_size
        pad_height = (patch - height % patch) % patch
        pad_width = (patch - width % patch) % patch
        if pad_height or pad_width:
            mode = (
                "reflect"
                if height > pad_height and width > pad_width
                else "replicate"
            )
            x = F.pad(x, (0, pad_width, 0, pad_height), mode=mode)
        padded_height, padded_width = x.shape[-2:]
        patches = x.reshape(
            batch,
            channels,
            padded_height // patch,
            patch,
            padded_width // patch,
            patch,
        ).permute(0, 1, 2, 4, 3, 5)
        spectrum = torch.fft.rfft2(patches.float()) * self.fft
        patches = torch.fft.irfft2(spectrum, s=(patch, patch))
        x = patches.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, channels, padded_height, padded_width
        )
        return x[..., :height, :width]


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SpectralUncertaintyAdapter(nn.Module):
    """Zero-initialized frequency adapter with uncertainty modulation."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        residual_scale: float = 0.1,
        use_spectral: bool = True,
        use_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels or max(channels // 4, 16))
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.project_in = nn.Conv2d(channels, hidden, 1)
        self.local = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden
        )
        self.spectral_gate = nn.Conv2d(hidden, hidden, 1)
        self.mean_head = nn.Conv2d(hidden, channels, 1)
        self.log_variance_head = nn.Conv2d(hidden, channels, 1)
        if not 0.0 < residual_scale < 1.0:
            raise ValueError("residual_scale must lie in (0, 1)")
        self.gain_logit = nn.Parameter(
            torch.tensor(math.log(residual_scale / (1.0 - residual_scale)))
        )
        self.use_spectral = bool(use_spectral)
        self.use_uncertainty = bool(use_uncertainty)
        nn.init.zeros_(self.spectral_gate.weight)
        nn.init.zeros_(self.spectral_gate.bias)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)
        self.spectral_gate.requires_grad_(self.use_spectral)
        self.log_variance_head.requires_grad_(self.use_uncertainty)
        self.last_regularization: Tensor | None = None
        self.last_uncertainty: Tensor | None = None

    def forward(self, feature: Tensor) -> Tensor:
        hidden = F.silu(self.local(self.project_in(self.norm(feature))))
        if self.use_spectral:
            spectrum = torch.fft.rfft2(hidden.float(), norm="ortho")
            amplitude = spectrum.abs().add(1e-6).log1p()
            gate = torch.sigmoid(
                self.spectral_gate(amplitude.to(hidden.dtype))
            ).float()
            spectral = torch.fft.irfft2(
                spectrum * gate,
                s=hidden.shape[-2:],
                norm="ortho",
            ).to(hidden.dtype)
            hidden = hidden + spectral
        fused = hidden
        mean = self.mean_head(F.silu(fused))
        if self.use_uncertainty:
            log_variance = self.log_variance_head(fused).clamp(-6.0, 4.0)
            confidence = torch.sigmoid(-log_variance)
            residual = confidence * mean
            self.last_uncertainty = log_variance.exp().mean(
                dim=1, keepdim=True
            )
            self.last_regularization = 0.5 * (
                mean.float().square()
                + log_variance.float().exp()
                - log_variance.float()
                - 1.0
            ).mean()
        else:
            residual = mean
            self.last_uncertainty = None
            self.last_regularization = 0.5 * mean.float().square().mean()
        gain = torch.sigmoid(self.gain_logit).to(residual.dtype)
        return feature + gain * residual


class ExternalSSRFoundation(nn.Module):
    """Wrap the official SSR implementation without vendoring its source.

    The wrapper captures SSR's final decoder feature through a forward hook,
    providing the ``(restored, feature)`` contract expected by the residual
    BFN. The official repository and checkpoint remain external dependencies.
    """

    def __init__(
        self,
        repo_path: str,
        checkpoint: str,
        architecture_file: str = "basicsr/models/archs/SSR_arch.py",
        model_class: str = "SSR",
        feature_module: str = "decoder_level1",
        param_key: str | None = "params",
        strict_load: bool = True,
        freeze: bool = True,
        trainable_modules: list[str] | tuple[str, ...] | None = None,
        feature_adapters: dict[str, dict[str, Any]] | None = None,
        frozen_modules_eval: bool = False,
        required_multiple: int = 8,
        checkpoint_compatible_edffn: bool = True,
        architecture: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        repo = Path(repo_path).expanduser().resolve()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        module = _load_module(repo / architecture_file)
        if checkpoint_compatible_edffn:
            module.FFN = CheckpointCompatibleEDFFN
        model_type = getattr(module, model_class, None)
        if model_type is None:
            raise AttributeError(
                f"{model_class!r} is not defined in {repo / architecture_file}"
            )
        kwargs = dict(architecture or {})
        self.model = model_type(**kwargs)
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = _extract_state_dict(raw, param_key)
        result = self.model.load_state_dict(state, strict=False)
        if strict_load and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError(
                "SSR checkpoint mismatch. "
                f"Missing: {result.missing_keys}; "
                f"unexpected: {result.unexpected_keys}"
            )

        self.required_multiple = int(required_multiple)
        if self.required_multiple < 1:
            raise ValueError("required_multiple must be positive")
        self.decoder_channels = int(kwargs.get("dim", 48))
        self.latent_feature_channels = self.decoder_channels * 4
        self._frozen_modules_eval = bool(frozen_modules_eval and freeze)
        self._trainable_modules = tuple(trainable_modules or ())
        self.model.requires_grad_(not freeze)
        for module_name in self._trainable_modules:
            self.model.get_submodule(module_name).requires_grad_(True)
        self.foundation_trainable = any(
            parameter.requires_grad for parameter in self.model.parameters()
        )
        if not self.foundation_trainable:
            self.model.eval()

        self.adapters = nn.ModuleDict()
        self._adapter_paths: dict[str, str] = {}
        self._runtime_adapter_state: dict[str, Tensor | str] | None = None
        for module_name, adapter_config in (feature_adapters or {}).items():
            key = module_name.replace(".", "__")
            if key in self.adapters:
                raise ValueError(f"Duplicate adapter key: {module_name}")
            adapter_kwargs = dict(adapter_config)
            adapter_type = adapter_kwargs.pop("type", "legacy")
            adapter_trainable = bool(adapter_kwargs.pop("trainable", True))
            if adapter_type not in {
                "legacy",
                "calibrated",
                "latent_film_spectral",
                "paper_spectral_gate",
                "state_spectral",
            }:
                raise ValueError(f"Unsupported adapter type: {adapter_type}")
            channels = int(adapter_kwargs.pop("channels"))
            adapter_classes = {
                "legacy": SpectralUncertaintyAdapter,
                "calibrated": CalibratedSpectralAdapter,
                "latent_film_spectral": LatentConditionedFiLMSpectralAdapter,
                "paper_spectral_gate": PaperSpectralGateAdapter,
                "state_spectral": StateConditionedSpectralAdapter,
            }
            adapter_class = adapter_classes[adapter_type]
            adapter = adapter_class(
                channels=channels,
                **adapter_kwargs,
            )
            adapter.requires_grad_(adapter_trainable)
            self.adapters[key] = adapter
            self._adapter_paths[module_name] = key
            layer = self.model.get_submodule(module_name)
            layer.register_forward_hook(self._make_adapter_hook(key))

        self.has_paper_spectral_gate = any(
            getattr(adapter, "is_paper_spectral_gate", False)
            for adapter in self.adapters.values()
        )
        self.has_state_conditioned_adapter = any(
            isinstance(adapter, StateConditionedSpectralAdapter)
            for adapter in self.adapters.values()
        )
        self.has_latent_conditioned_adapter = any(
            isinstance(adapter, LatentConditionedFiLMSpectralAdapter)
            for adapter in self.adapters.values()
        )
        self._state_adapter_module_name = next(
            (
                module_name
                for module_name, key in self._adapter_paths.items()
                if isinstance(
                    self.adapters[key],
                    StateConditionedSpectralAdapter,
                )
            ),
            None,
        )
        self._latent_adapter_module_names = tuple(
            module_name
            for module_name, key in self._adapter_paths.items()
            if isinstance(
                self.adapters[key],
                LatentConditionedFiLMSpectralAdapter,
            )
        )
        self.has_trainable_path = (
            self.foundation_trainable
            or any(
                parameter.requires_grad
                for adapter in self.adapters.values()
                for parameter in adapter.parameters()
            )
        )
        self._decoder_feature: Tensor | None = None
        feature_layer = self.model.get_submodule(feature_module)
        feature_layer.register_forward_hook(self._capture_decoder_feature)
        self.train(self.training)

    @property
    def state_conditioned_adapter(
        self,
    ) -> StateConditionedSpectralAdapter | None:
        adapters = [
            adapter
            for adapter in self.adapters.values()
            if isinstance(adapter, StateConditionedSpectralAdapter)
        ]
        if len(adapters) == 1:
            return adapters[0]
        return None

    def _conditioned_adapter(
        self,
        module_name: str,
    ) -> StateConditionedSpectralAdapter | LatentConditionedFiLMSpectralAdapter | None:
        key = self._adapter_paths.get(module_name)
        if key is None:
            return None
        adapter = self.adapters[key]
        if isinstance(
            adapter,
            (
                StateConditionedSpectralAdapter,
                LatentConditionedFiLMSpectralAdapter,
            ),
        ):
            return adapter
        return None

    @staticmethod
    def _standard_decoder_modules() -> tuple[str, ...]:
        return (
            "patch_embed",
            "encoder_level1",
            "down1_2",
            "encoder_level2",
            "down2_3",
            "encoder_level3",
            "decoder_level3",
            "up3_2",
            "decoder_level2",
            "up2_1",
            "decoder_level1",
            "output",
        )

    @property
    def supports_decoder_state_cache(self) -> bool:
        return (
            self._state_adapter_module_name == "decoder_level3"
            and self.state_conditioned_adapter is not None
            and all(
                hasattr(self.model, name)
                for name in self._standard_decoder_modules()
            )
        )

    @property
    def supports_latent_cache(self) -> bool:
        return (
            self.has_latent_conditioned_adapter
            and "decoder_level3" in self._latent_adapter_module_names
            and set(self._latent_adapter_module_names).issubset(
                {"decoder_level3", "decoder_level2"}
            )
            and all(
                hasattr(self.model, name)
                for name in self._standard_decoder_modules()
            )
        )

    def cache_decoder_prefix(self, image: Tensor) -> SSRDecoderCache:
        """Run the SSR encoder and decoder-level-3 prefix exactly once."""

        if not (
            self.supports_decoder_state_cache
            or self.supports_latent_cache
        ):
            raise RuntimeError(
                "SSR decoder cache requires a decoder_level3 conditioned adapter "
                "and the standard SSR encoder/decoder module layout"
            )
        context = nullcontext() if self.foundation_trainable else torch.no_grad()
        with context:
            inp_enc_level1 = self.model.patch_embed(image)
            out_enc_level1 = self.model.encoder_level1(inp_enc_level1)
            inp_enc_level2 = self.model.down1_2(out_enc_level1)
            out_enc_level2 = self.model.encoder_level2(inp_enc_level2)
            inp_enc_level3 = self.model.down2_3(out_enc_level2)
            out_enc_level3 = self.model.encoder_level3(inp_enc_level3)
            out_dec_level3 = self.model.decoder_level3(out_enc_level3)
        return SSRDecoderCache(
            image=image,
            encoder_level1=out_enc_level1,
            encoder_level2=out_enc_level2,
            decoder_level3=out_dec_level3,
        )

    def decode_from_cached_decoder3(
        self,
        cache: SSRDecoderCache,
        state: dict[str, Tensor | str] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply conditioned adapters and execute the decoder suffix once."""

        if not (
            self.supports_decoder_state_cache
            or self.supports_latent_cache
        ):
            raise RuntimeError("SSR decoder cache is unavailable")
        decoder_level3 = cache.decoder_level3
        if state is not None:
            adapter = self._conditioned_adapter("decoder_level3")
            if adapter is None:
                raise RuntimeError(
                    "decoder_level3 conditioned adapter is unavailable"
                )
            decoder_level3 = adapter(decoder_level3, state)
        context = (
            self._state_adapter_context(state)
            if state is not None
            else nullcontext()
        )
        with context:
            inp_dec_level2 = self.model.up3_2(decoder_level3)
            inp_dec_level2 = inp_dec_level2 + cache.encoder_level2
            out_dec_level2 = self.model.decoder_level2(inp_dec_level2)
            inp_dec_level1 = self.model.up2_1(out_dec_level2)
            inp_dec_level1 = inp_dec_level1 + cache.encoder_level1
            out_dec_level1 = self.model.decoder_level1(inp_dec_level1)
            restored = self.model.output(out_dec_level1) + cache.image
        return restored, out_dec_level1

    def _make_adapter_hook(self, key: str):
        def apply_adapter(
            module: nn.Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> Tensor:
            del module, inputs
            if not isinstance(output, Tensor):
                raise TypeError("SSR adapter target must return a tensor")
            adapter = self.adapters[key]
            if getattr(adapter, "requires_runtime_state", False):
                return adapter(output, self._runtime_adapter_state)
            return adapter(output)

        return apply_adapter

    @contextmanager
    def _state_adapter_context(
        self,
        state: dict[str, Tensor | str],
    ):
        previous = self._runtime_adapter_state
        self._runtime_adapter_state = state
        try:
            yield
        finally:
            self._runtime_adapter_state = previous

    def _capture_decoder_feature(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs
        if not isinstance(output, Tensor):
            raise TypeError("SSR feature module must return a tensor")
        self._decoder_feature = output

    def train(self, mode: bool = True) -> "ExternalSSRFoundation":
        super().train(mode)
        if self._frozen_modules_eval:
            self.model.eval()
            for module_name in self._trainable_modules:
                self.model.get_submodule(module_name).train(mode)
        elif not self.foundation_trainable:
            self.model.eval()
        self.adapters.train(mode)
        return self

    def adapter_regularization(self, reference: Tensor) -> Tensor:
        values = [
            adapter.last_regularization
            for adapter in self.adapters.values()
            if adapter.last_regularization is not None
        ]
        if not values:
            return reference.new_zeros(())
        return torch.stack(values).mean()

    def latent_modulation_diagnostics(
        self,
    ) -> dict[str, dict[str, Tensor]]:
        diagnostics = {}
        for module_name, key in self._adapter_paths.items():
            adapter = self.adapters[key]
            if not isinstance(
                adapter,
                LatentConditionedFiLMSpectralAdapter,
            ):
                continue
            values = {
                "mean_abs_gamma_delta": adapter.last_gamma_delta,
                "mean_abs_beta": adapter.last_beta,
                "mean_abs_spectral_delta": adapter.last_spectral_delta,
            }
            if any(value is None for value in values.values()):
                raise RuntimeError(
                    f"Latent adapter {module_name} has not executed"
                )
            diagnostics[module_name] = {
                name: value
                for name, value in values.items()
                if isinstance(value, Tensor)
            }
        return diagnostics

    def uncertainty_calibration(
        self,
        residual_target: Tensor,
        reference: Tensor,
        original_size: tuple[int, int] | None = None,
    ) -> Tensor:
        residual = residual_target.detach().float()
        error_map = residual.abs().mean(dim=1, keepdim=True)
        # Match deployed [0, 1] clipping without backpropagating through the target.
        restored = reference.detach().float()
        clipped_residual = residual + (restored - restored.clamp(0, 1))
        squared_error = clipped_residual.square().mean(dim=1, keepdim=True)
        valid = torch.ones_like(squared_error)
        if original_size is not None:
            valid[..., original_size[0] :, :] = 0
            valid[..., :, original_size[1] :] = 0
        losses = []
        for adapter in self.adapters.values():
            if isinstance(adapter, CalibratedSpectralAdapter):
                if adapter.last_uncertainty is not None:
                    losses.append(adapter.calibration_loss(squared_error, valid))
                continue
            uncertainty = getattr(adapter, "last_uncertainty", None)
            if uncertainty is None:
                continue
            target = F.interpolate(
                error_map.detach().float(),
                size=uncertainty.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            predicted = uncertainty.float()
            predicted = predicted / predicted.mean(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-6)
            target = target / target.mean(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-6)
            losses.append(
                F.smooth_l1_loss(
                    predicted.log1p(),
                    target.log1p(),
                )
            )
        if not losses:
            return reference.new_zeros(())
        return torch.stack(losses).mean()

    def adapter_uncertainty_map(self, reference: Tensor) -> Tensor | None:
        maps = [
            adapter.last_uncertainty
            for adapter in self.adapters.values()
            if isinstance(adapter, CalibratedSpectralAdapter)
            and adapter.last_uncertainty is not None
        ]
        if not maps:
            return None
        finest = max(maps, key=lambda value: value.shape[-2] * value.shape[-1])
        return F.interpolate(
            finest, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self,
        image: Tensor,
        semantic: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del semantic
        self._decoder_feature = None
        context = (
            nullcontext()
            if self.has_trainable_path
            else torch.no_grad()
        )
        with context:
            restored = self.model(image)
        feature = self._decoder_feature
        self._decoder_feature = None
        if not isinstance(restored, Tensor) or feature is None:
            raise RuntimeError("SSR did not produce the expected image and feature")
        if restored.shape[-2:] != feature.shape[-2:]:
            raise RuntimeError(
                "SSR decoder feature and restored image have different sizes"
            )
        return restored, feature

    def forward_with_state(
        self,
        image: Tensor,
        state: dict[str, Tensor | str],
        semantic: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not (
            self.has_state_conditioned_adapter
            or self.has_latent_conditioned_adapter
        ):
            raise RuntimeError(
                "Foundation has no runtime-conditioned decoder adapter"
            )
        with self._state_adapter_context(state):
            return self.forward(image, semantic)


def build_foundation(config: dict[str, Any]) -> nn.Module:
    foundation_type = str(config.get("type", "")).lower()
    if foundation_type in {"external_ssr", "ssr"}:
        kwargs = dict(config)
        kwargs.pop("type", None)
        return ExternalSSRFoundation(**kwargs)
    raise ValueError(f"Unsupported restoration foundation: {foundation_type}")
