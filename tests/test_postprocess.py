import sys
from pathlib import Path

import pytest
import torch

from evaluate import (
    SELF_ENSEMBLE_TRANSFORMS,
    _geometric_view,
    _inverse_geometric_view,
    self_ensemble_restore,
)
from scripts.average_checkpoints import (
    average_states,
    main as average_checkpoints_main,
    weight_view,
)


@pytest.mark.parametrize("index", range(SELF_ENSEMBLE_TRANSFORMS))
def test_self_ensemble_geometric_views_are_exactly_invertible(index: int) -> None:
    image = torch.arange(45).reshape(1, 3, 3, 5)
    restored = _inverse_geometric_view(_geometric_view(image, index), index)
    torch.testing.assert_close(restored, image, rtol=0, atol=0)


def test_self_ensemble_runs_all_views_and_restores_original_geometry() -> None:
    image = torch.rand(2, 3, 5, 7)
    seen_shapes = []

    def restore(view: torch.Tensor) -> torch.Tensor:
        seen_shapes.append(tuple(view.shape))
        return view + 0.25

    restored = self_ensemble_restore(restore, image)
    assert len(seen_shapes) == SELF_ENSEMBLE_TRANSFORMS
    assert set(seen_shapes) == {(2, 3, 5, 7), (2, 3, 7, 5)}
    torch.testing.assert_close(restored, image + 0.25, rtol=0, atol=2e-7)


def test_checkpoint_average_uses_ema_view_and_preserves_equal_buffers() -> None:
    first = {
        "model": {
            "weight": torch.tensor([1.0, 3.0]),
            "counter": torch.tensor(7, dtype=torch.int64),
        },
        "ema": {"shadow": {"weight": torch.tensor([2.0, 4.0])}},
    }
    second = {
        "model": {
            "weight": torch.tensor([5.0, 7.0]),
            "counter": torch.tensor(7, dtype=torch.int64),
        },
        "ema": {"shadow": {"weight": torch.tensor([6.0, 8.0])}},
    }
    paths = [Path("first.pt"), Path("second.pt")]
    states = [
        weight_view(checkpoint, path, "ema")
        for checkpoint, path in zip((first, second), paths)
    ]
    averaged = average_states(states, paths)
    torch.testing.assert_close(
        averaged["weight"], torch.tensor([4.0, 6.0]), rtol=0, atol=0,
    )
    torch.testing.assert_close(
        averaged["counter"], torch.tensor(7, dtype=torch.int64), rtol=0, atol=0,
    )


def test_checkpoint_average_cli_writes_non_resumable_eval_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = []
    for index, value in enumerate((2.0, 4.0), start=1):
        path = tmp_path / f"epoch_{index:04d}.pt"
        torch.save(
            {
                "format_version": 2,
                "model": {"weight": torch.tensor([value - 1.0])},
                "ema": {
                    "shadow": {"weight": torch.tensor([value])},
                    "decay": 0.999,
                    "warmup_updates": 2_000,
                    "updates": index,
                },
                "ema_parameter_names": ["weight"],
                "config": {"model": {"kind": "test"}},
                "epoch": index - 1,
                "global_step": index * 10,
                "optimizer": {"state": {}},
                "scheduler": {"last_epoch": index},
                "rng_states": [],
            },
            path,
        )
        inputs.append(path)
    output = tmp_path / "averaged.pt"
    monkeypatch.setattr(
        sys, "argv",
        [
            "average_checkpoints.py", "--inputs",
            *(str(path) for path in inputs), "--weights", "ema",
            "--output", str(output),
        ],
    )
    average_checkpoints_main()
    payload = torch.load(output, map_location="cpu", weights_only=True)
    torch.testing.assert_close(
        payload["model"]["weight"], torch.tensor([3.0]), rtol=0, atol=0,
    )
    assert payload["averaged_weights"] == "ema"
    assert payload["averaged_checkpoints"] == 2
    assert payload["averaged_epochs"] == [0, 1]
    assert payload["resumable"] is False
    for key in ("optimizer", "scheduler", "rng_states", "ema"):
        assert key not in payload
