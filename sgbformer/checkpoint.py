"""Checkpoint serialization."""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def capture_rng_state(
    device: torch.device,
    data_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "device_type": device.type,
    }
    if device.type == "cuda":
        state["device"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["device"] = torch.mps.get_rng_state()
    if data_generator is not None:
        state["data_generator"] = data_generator.get_state()
    return state


def restore_rng_state(
    state: dict[str, Any],
    device: torch.device,
    data_generator: torch.Generator | None = None,
) -> None:
    if state["device_type"] != device.type:
        raise ValueError("Exact RNG resume requires the same device type")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["name"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["device"].cpu(), device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["device"].cpu())
    if data_generator is not None and "data_generator" in state:
        data_generator.set_state(state["data_generator"].cpu())


def _unwrapped(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _checkpoint_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    module = _unwrapped(model)
    state = module.state_dict()
    semantic_encoder = getattr(module, "semantic_encoder", None)
    omit_semantic = semantic_encoder is not None and not any(
        parameter.requires_grad for parameter in semantic_encoder.parameters()
    )
    return {
        name: value
        for name, value in state.items()
        if not (omit_semantic and name.startswith("semantic_encoder."))
    }


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Replace a checkpoint only after serialization succeeds; clean failed writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    ema: Any = None,
    scaler: Any = None,
    rng_states: list[dict[str, Any]] | None = None,
    best_psnr: float | None = None,
    best_lpips: float | None = None,
) -> None:
    state = _checkpoint_state_dict(model)
    omitted = sorted(set(_unwrapped(model).state_dict()) - set(state))
    payload: dict[str, Any] = {
        "format_version": 2,
        "model": state,
        "omitted_state_keys": omitted,
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "best_lpips": best_lpips,
        "frozen_parameters_excluded": bool(omitted),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
        payload["ema_parameter_names"] = sorted(payload["ema"]["shadow"])
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if rng_states is not None:
        payload["rng_states"] = rng_states
    atomic_torch_save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    use_ema: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if use_ema and "ema" in checkpoint:
        if strict and checkpoint.get("format_version", 0) >= 2:
            if set(checkpoint["ema"]["shadow"]) != set(checkpoint["ema_parameter_names"]):
                raise RuntimeError("EMA checkpoint has missing or unexpected shadow keys")
        # EMA only tracks trainable parameters; restore frozen weights and buffers too.
        state_dict = dict(checkpoint.get("model", {}))
        state_dict.update(checkpoint["ema"]["shadow"])
    else:
        state_dict = checkpoint.get("model", checkpoint)
    result = _unwrapped(model).load_state_dict(state_dict, strict=False)
    if strict:
        trainable_names = {
            name
            for name, parameter in _unwrapped(model).named_parameters()
            if parameter.requires_grad
        }
        if checkpoint.get("format_version", 0) >= 2:
            allowed_missing = {
                name for name in checkpoint.get("omitted_state_keys", [])
                if name.startswith("semantic_encoder.") and name not in trainable_names
            }
            missing = sorted(set(result.missing_keys) - allowed_missing)
        else:
            missing = sorted(set(result.missing_keys) & trainable_names)
            missing_frozen = [
                name for name in result.missing_keys
                if name not in trainable_names and not name.startswith("semantic_encoder.")
            ]
            if missing_frozen:
                warnings.warn(
                    f"Legacy checkpoint omits frozen state: {missing_frozen}",
                    RuntimeWarning,
                )
        if missing or result.unexpected_keys:
            raise RuntimeError(
                "Checkpoint mismatch. "
                f"Missing required keys: {missing}; "
                f"unexpected keys: {result.unexpected_keys}"
            )
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
