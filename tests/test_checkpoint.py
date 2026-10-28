import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from sgbformer.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from sgbformer.ema import ExponentialMovingAverage


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.semantic_encoder = nn.Linear(2, 2)
        self.semantic_encoder.requires_grad_(False)
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)


def test_checkpoint_keeps_stage_frozen_weights_and_ema(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = ToyModel()
    ema = ExponentialMovingAverage(model, decay=0.5, warmup_updates=0)
    initial_head = model.head.weight.detach().clone()
    with torch.no_grad():
        model.head.weight.add_(1.0)
    ema.update(model)
    model.backbone.requires_grad_(False)

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        config={"model": {}},
        epoch=1,
        global_step=2,
        ema=ema,
    )
    raw = torch.load(checkpoint_path, weights_only=False)
    assert "head.weight" in raw["model"]
    assert "backbone.weight" in raw["model"]
    assert "semantic_encoder.weight" not in raw["model"]
    assert "ema" in raw

    restored = ToyModel()
    load_checkpoint(checkpoint_path, restored)
    torch.testing.assert_close(restored.head.weight, model.head.weight)

    averaged = ToyModel()
    load_checkpoint(checkpoint_path, averaged, use_ema=True)
    torch.testing.assert_close(averaged.head.weight, initial_head + 0.5)


def test_ema_loading_keeps_frozen_weights_and_buffers(tmp_path: Path) -> None:
    model = ToyModel()
    model.backbone.requires_grad_(False)
    model.register_buffer("running_value", torch.tensor(5.0))
    ema = ExponentialMovingAverage(model, decay=0.5, warmup_updates=0)
    checkpoint_path = tmp_path / "frozen.pt"
    save_checkpoint(
        checkpoint_path, model, config={}, epoch=0, global_step=0, ema=ema
    )
    restored = ToyModel()
    restored.backbone.requires_grad_(False)
    restored.register_buffer("running_value", torch.tensor(0.0))
    load_checkpoint(checkpoint_path, restored, use_ema=True)
    torch.testing.assert_close(restored.backbone.weight, model.backbone.weight)
    torch.testing.assert_close(restored.running_value, model.running_value)


def test_rng_and_scaler_roundtrip_is_weights_only_safe(tmp_path: Path) -> None:
    model = ToyModel()
    generator = torch.Generator().manual_seed(19)
    scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler.scale(model.head(torch.ones(1, 2)).sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    state = capture_rng_state(torch.device("cpu"), generator)
    checkpoint = tmp_path / "random.pt"
    save_checkpoint(
        checkpoint, model, config={}, epoch=0, global_step=1,
        optimizer=optimizer, scaler=scaler, rng_states=[state],
    )
    expected = (
        random.random(), np.random.random(), torch.rand(3),
        torch.rand(3, generator=generator),
    )
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    generator.manual_seed(99)
    restored_scaler = torch.amp.GradScaler("cpu")
    raw = load_checkpoint(checkpoint, model, scaler=restored_scaler)
    restore_rng_state(raw["rng_states"][0], torch.device("cpu"), generator)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], atol=0, rtol=0)
    torch.testing.assert_close(
        torch.rand(3, generator=generator), expected[3], atol=0, rtol=0
    )
    assert restored_scaler.state_dict() == scaler.state_dict()


@pytest.mark.parametrize("missing_key", ["backbone.weight", "running_value"])
def test_new_checkpoint_rejects_missing_frozen_state(tmp_path, missing_key) -> None:
    model = ToyModel()
    model.backbone.requires_grad_(False)
    model.register_buffer("running_value", torch.tensor(5.0))
    path = tmp_path / "incomplete.pt"
    save_checkpoint(path, model, config={}, epoch=0, global_step=0)
    raw = torch.load(path, weights_only=True)
    del raw["model"][missing_key]
    torch.save(raw, path)
    with pytest.raises(RuntimeError, match="Missing required keys"):
        load_checkpoint(path, model)


def test_ema_inference_rejects_incomplete_shadow(tmp_path) -> None:
    model = ToyModel()
    ema = ExponentialMovingAverage(model)
    path = tmp_path / "ema.pt"
    save_checkpoint(path, model, config={}, epoch=0, global_step=0, ema=ema)
    raw = torch.load(path, weights_only=True)
    del raw["ema"]["shadow"]["head.weight"]
    torch.save(raw, path)
    with pytest.raises(RuntimeError, match="shadow keys"):
        load_checkpoint(path, model, use_ema=True)


def test_legacy_checkpoint_still_loads_without_new_metadata(tmp_path) -> None:
    model = ToyModel()
    model.backbone.requires_grad_(False)
    state = {
        key: value for key, value in model.state_dict().items()
        if not key.startswith("semantic_encoder.")
    }
    path = tmp_path / "legacy.pt"
    torch.save({"model": state}, path)
    restored = ToyModel()
    restored.backbone.requires_grad_(False)
    load_checkpoint(path, restored)
    torch.testing.assert_close(restored.backbone.weight, model.backbone.weight)
    del state["backbone.weight"]
    torch.save({"model": state}, path)
    with pytest.warns(RuntimeWarning, match="Legacy checkpoint omits frozen"):
        load_checkpoint(path, restored)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_mps_rng_restore() -> None:
    device = torch.device("mps")
    state = capture_rng_state(device)
    expected = torch.rand(10, device=device)
    restore_rng_state(state, device)
    torch.testing.assert_close(torch.rand(10, device=device), expected, atol=0, rtol=0)
