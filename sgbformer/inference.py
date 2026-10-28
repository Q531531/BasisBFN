"""Checkpoint loading and optional tiled inference."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor

from .checkpoint import load_checkpoint
from .config import load_config
from .models import BasisBFN, build_model


def validate_inference_config(saved: dict, requested: dict) -> None:
    models = [deepcopy(config.get("model", config)) for config in (saved, requested)]
    for model in models:
        foundation = model.get("foundation")
        if isinstance(foundation, dict):
            for key in ("repo_path", "checkpoint"):
                foundation.pop(key, None)
    if models[0] != models[1]:
        raise ValueError(
            "Inference config changes the checkpoint model. Only foundation path "
            "relocation is allowed; changed model semantics require a separate experiment."
        )


def load_model_for_inference(
    checkpoint_path: str | Path,
    device: torch.device,
    config_path: str | Path | None = None,
) -> tuple[BasisBFN, dict]:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = load_config(config_path) if config_path else raw.get("config")
    if config is None:
        raise ValueError(
            "Checkpoint has no embedded config; provide --config explicitly"
        )
    if config_path is not None and raw.get("config") is not None:
        validate_inference_config(raw["config"], config)
    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device, use_ema=True)
    model.eval()
    return model, config


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile < 1 or overlap < 0 or overlap >= tile:
        raise ValueError("tile must be positive and overlap must lie in [0, tile)")
    if tile >= length:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def restoration_tile_count(
    height: int,
    width: int,
    tile: int | None,
    overlap: int,
) -> int:
    """Return the number of model restoration trajectories for one image."""

    if tile is None or max(height, width) <= tile:
        return 1
    return len(_tile_starts(height, tile, overlap)) * len(
        _tile_starts(width, tile, overlap)
    )


@torch.no_grad()
def restore_tensor(
    model: BasisBFN,
    image: Tensor,
    *,
    steps: int | None = None,
    deterministic: bool = False,
    residual_gain: float = 1.0,
    state_ablation: str = "normal",
    tile: int | None = None,
    overlap: int = 32,
) -> Tensor:
    if tile is not None and (tile < 1 or overlap < 0 or overlap >= tile):
        raise ValueError("tile must be positive and overlap must lie in [0, tile)")
    if tile is None or max(image.shape[-2:]) <= tile:
        return model.restore(
            image,
            num_steps=steps,
            deterministic=deterministic,
            residual_gain=residual_gain,
            state_ablation=state_ablation,
        )
    if image.shape[0] != 1:
        raise ValueError("Tiled inference currently expects batch size one")

    semantic = model.encode_semantic(image)
    height, width = image.shape[-2:]
    output = torch.zeros_like(image)
    normalization = torch.zeros_like(image[:, :1])
    y_starts = _tile_starts(height, tile, overlap)
    x_starts = _tile_starts(width, tile, overlap)

    for top in y_starts:
        for left in x_starts:
            crop = image[..., top : top + tile, left : left + tile]
            restored = model.restore(
                crop,
                semantic=semantic,
                num_steps=steps,
                deterministic=deterministic,
                residual_gain=residual_gain,
                state_ablation=state_ablation,
            )
            crop_height, crop_width = crop.shape[-2:]
            weight_y = torch.hann_window(
                crop_height,
                periodic=False,
                device=image.device,
                dtype=image.dtype,
            ).clamp_min(0.1)
            weight_x = torch.hann_window(
                crop_width,
                periodic=False,
                device=image.device,
                dtype=image.dtype,
            ).clamp_min(0.1)
            weight = (weight_y[:, None] * weight_x[None, :])[None, None]
            output[..., top : top + crop_height, left : left + crop_width] += (
                restored * weight
            )
            normalization[
                ..., top : top + crop_height, left : left + crop_width
            ] += weight
    return output / normalization.clamp_min(1e-8)


@torch.no_grad()
def restore_structural_tensor(
    model: BasisBFN,
    image: Tensor,
    *,
    tile: int | None = None,
    overlap: int = 32,
) -> Tensor:
    """Restore with the frozen structural path using the requested tile protocol."""

    if tile is not None and (tile < 1 or overlap < 0 or overlap >= tile):
        raise ValueError("tile must be positive and overlap must lie in [0, tile)")
    if tile is None or max(image.shape[-2:]) <= tile:
        return model.restore_structural(image)
    if image.shape[0] != 1:
        raise ValueError("Tiled inference currently expects batch size one")

    semantic = model.encode_semantic(image)
    height, width = image.shape[-2:]
    output = torch.zeros_like(image)
    normalization = torch.zeros_like(image[:, :1])
    y_starts = _tile_starts(height, tile, overlap)
    x_starts = _tile_starts(width, tile, overlap)

    for top in y_starts:
        for left in x_starts:
            crop = image[..., top : top + tile, left : left + tile]
            restored = model.restore_structural(crop, semantic=semantic)
            crop_height, crop_width = crop.shape[-2:]
            weight_y = torch.hann_window(
                crop_height,
                periodic=False,
                device=image.device,
                dtype=image.dtype,
            ).clamp_min(0.1)
            weight_x = torch.hann_window(
                crop_width,
                periodic=False,
                device=image.device,
                dtype=image.dtype,
            ).clamp_min(0.1)
            weight = (weight_y[:, None] * weight_x[None, :])[None, None]
            output[..., top : top + crop_height, left : left + crop_width] += (
                restored * weight
            )
            normalization[
                ..., top : top + crop_height, left : left + crop_width
            ] += weight
    return output / normalization.clamp_min(1e-8)
