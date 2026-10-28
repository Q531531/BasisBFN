"""Exponential moving average for trainable model parameters."""

from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import contextmanager
from collections.abc import Iterator, Mapping

import torch
from torch import Tensor, nn


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class ExponentialMovingAverage:
    """Track and temporarily apply an EMA without copying frozen CLIP weights."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_updates: int = 2_000,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        self.decay = float(decay)
        self.warmup_updates = int(warmup_updates)
        self.updates = 0
        self.shadow = OrderedDict(
            (name, parameter.detach().clone())
            for name, parameter in _unwrap(model).named_parameters()
            if parameter.requires_grad
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay
        if self.warmup_updates > 0:
            decay *= 1.0 - math.exp(-self.updates / self.warmup_updates)
        parameters = dict(_unwrap(model).named_parameters())
        for name, average in self.shadow.items():
            average.lerp_(parameters[name].detach(), 1.0 - decay)

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "warmup_updates": self.warmup_updates,
            "updates": self.updates,
            "shadow": OrderedDict(
                (name, value.detach().clone())
                for name, value in self.shadow.items()
            ),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping):
            raise ValueError("EMA state is missing the shadow parameter mapping")
        missing = sorted(set(self.shadow) - set(shadow))
        unexpected = sorted(set(shadow) - set(self.shadow))
        if missing or unexpected:
            raise RuntimeError(
                f"EMA checkpoint mismatch. Missing: {missing}; "
                f"unexpected: {unexpected}"
            )
        self.decay = float(state.get("decay", self.decay))
        self.warmup_updates = int(
            state.get("warmup_updates", self.warmup_updates)
        )
        self.updates = int(state.get("updates", 0))
        for name, value in shadow.items():
            self.shadow[name].copy_(value)

    @contextmanager
    @torch.no_grad()
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        parameters = dict(_unwrap(model).named_parameters())
        original = OrderedDict(
            (name, parameters[name].detach().clone()) for name in self.shadow
        )
        try:
            for name, average in self.shadow.items():
                parameters[name].copy_(average)
            yield
        finally:
            for name, value in original.items():
                parameters[name].copy_(value)
