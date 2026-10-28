from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import torch

from sgbformer.checkpoint import save_checkpoint
from sgbformer.models import build_model
from sgbformer.utils import save_image


def test_evaluate_json_records_runtime_cost(tmp_path: Path, monkeypatch) -> None:
    config = {
        "model": {
            "dim": 8,
            "num_blocks": [1, 1, 1, 1],
            "refinement_blocks": 1,
            "heads": [1, 2, 4, 8],
            "semantic": {"encoder": "none", "embedding_dim": 16},
            "bfn": {"inference_mode": "none", "mask_mode": "none"},
        }
    }
    model = build_model(config)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        config=config,
        epoch=0,
        global_step=0,
    )
    input_dir = tmp_path / "input"
    target_dir = tmp_path / "target"
    image = torch.rand(3, 16, 16)
    save_image(image, input_dir / "sample.png")
    save_image(image, target_dir / "sample.png")
    output = tmp_path / "evaluation.json"

    script = Path(__file__).resolve().parents[1] / "evaluate.py"
    spec = spec_from_file_location("evaluate_cli_runtime_test", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--input-dir",
            str(input_dir),
            "--target-dir",
            str(target_dir),
            "--device",
            "cpu",
            "--quiet",
            "--json",
            str(output),
        ],
    )
    module.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    runtime = result["runtime"]
    assert runtime["images"] == 1
    assert runtime["restoration_trajectories"] == 1
    assert runtime["average_restoration_trajectories_per_image"] == 1
    assert runtime["average_decoder_state_calls_per_image"] == 0
    assert runtime["average_inference_seconds_per_image"] > 0
    assert runtime["peak_memory_allocated_mb"] is None
    assert runtime["peak_incremental_memory_mb"] is None
    assert result["protocol"]["foundation_signature"] is None
    assert result["protocol"]["checkpoint_path"] == str(checkpoint.resolve())
    assert result["protocol"]["checkpoint_epoch"] == 0
    assert result["protocol"]["checkpoint_global_step"] == 0
    assert result["protocol"]["checkpoint_weights"] == "model"

    tiled_output = tmp_path / "evaluation_tiled.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--input-dir",
            str(input_dir),
            "--target-dir",
            str(target_dir),
            "--device",
            "cpu",
            "--tile",
            "8",
            "--overlap",
            "4",
            "--compare-structural",
            "--quiet",
            "--json",
            str(tiled_output),
        ],
    )
    module.main()
    tiled = json.loads(tiled_output.read_text(encoding="utf-8"))
    assert tiled["runtime"]["restoration_trajectories"] == 9
    assert tiled["runtime"]["average_decoder_state_calls_per_image"] == 0
    assert tiled["protocol"]["tile"] == 8
