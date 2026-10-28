import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import train
import evaluate
from sgbformer.inference import load_model_for_inference
from test_foundation import _write_dummy_ssr


@pytest.mark.parametrize(
    "stochastic,workers,experimental",
    [(False, 0, False), (True, 0, False), (True, 1, False), (True, 0, True)],
)
def test_adapter_train_resume_and_ema_inference(
    tmp_path: Path, monkeypatch, stochastic: bool, workers: int, experimental: bool
) -> None:
    repo, checkpoint = _write_dummy_ssr(tmp_path)
    if stochastic:
        build_model = train.build_model

        def build_with_dropout(config):
            model = build_model(config)
            model.backbone.model.decoder_level1.register_forward_hook(
                lambda layer, inputs, output: torch.nn.functional.dropout(
                    output, p=0.25, training=layer.training
                )
            )
            return model

        monkeypatch.setattr(train, "build_model", build_with_dropout)
    input_dir, target_dir = tmp_path / "input", tmp_path / "gt"
    input_dir.mkdir()
    target_dir.mkdir()
    generator = np.random.default_rng(42)
    for index in range(3):
        target = generator.integers(20, 230, (16, 16, 3), dtype=np.uint8)
        Image.fromarray(target).save(target_dir / f"{index}.png")
        Image.fromarray(target - 10).save(input_dir / f"{index}.png")
    output = tmp_path / "run"
    config = {
        "seed": 42,
        "model": {
            "semantic": {"encoder": "none", "embedding_dim": 1},
            "foundation": {
                "type": "external_ssr", "repo_path": str(repo),
                "checkpoint": str(checkpoint), "architecture": {"dim": 4},
                "trainable_modules": ["decoder_level1", "output"],
                "feature_adapters": {
                    "decoder_level1": {
                        "type": "calibrated", "channels": 4, "hidden_channels": 4,
                    },
                },
            },
            "bfn": {"inference_mode": "none", "mask_mode": "none"},
        },
        "data": {
            "workers": workers,
            "train": {
                "input_dir": str(input_dir), "target_dir": str(target_dir),
                "patch_size": 16,
            },
            "val": None,
        },
        # Must not construct/download LPIPS when no validation loader exists.
        "validation": {"lpips": True},
        "training": {
            "output_dir": str(output), "epochs": 2, "batch_size": 2,
            "micro_batch_size": 1, "amp": False, "log_interval": 1,
            "save_interval": 1, "reproducible_data": True,
            "schedule_minimum_mode": "per_group", "learning_rate": 1e-3,
            "parameter_lrs": {"backbone.model": 1e-4},
            "stages": [{
                "name": "structure", "epochs": 2,
                "loss": {"bfn_weight": 0, "uncertainty_weight": 0.01},
            }],
        },
    }
    if experimental:
        config["model"]["foundation"]["feature_adapters"]["decoder_level1"].update(
            spectral_reference_size=16, risk_log_ratio_mode="smooth",
        )
        config["training"]["stages"][0]["loss"]["structural_loss_type"] = "mse"
        config["training"]["sampling"] = {
            "mode": "category_balanced",
            "category_patterns": {"toy": r"\d+\.png"},
            "expected_counts": {"toy": 3},
            "target_fractions": {"toy": 1.0},
        }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path), "--device", "cpu"])
    train.main()
    latest = torch.load(output / "latest.pt", weights_only=True)
    first = torch.load(output / "epoch_0001.pt", weights_only=True)
    assert latest["global_step"] == 4  # Last partial accumulation group is retained.
    model, _ = load_model_for_inference(output / "latest.pt", torch.device("cpu"))
    assert model.receiver is None
    restored = model.restore(torch.rand(1, 3, 17, 19), return_details=True)
    assert restored["adapter_uncertainty"].shape == (1, 1, 17, 19)
    assert torch.isfinite(restored["restored"]).all()
    with pytest.raises(FileExistsError, match="already contains"):
        train.main()
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--config", str(config_path), "--device", "cpu",
        "--resume", str(output / "epoch_0001.pt"),
    ])
    history_before = (output / "metrics.jsonl").read_bytes()
    with pytest.raises(ValueError, match="existing history"):
        train.main()
    assert (output / "metrics.jsonl").read_bytes() == history_before

    config["training"]["output_dir"] = str(tmp_path / "resumed")
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--config", str(config_path), "--device", "cpu",
        "--resume", str(output / "epoch_0001.pt"),
    ])
    train.main()
    resumed = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=True)
    assert first["global_step"] == 2
    for key, value in latest["model"].items():
        torch.testing.assert_close(resumed["model"][key], value, rtol=0, atol=0)
    for key, value in latest["ema"]["shadow"].items():
        torch.testing.assert_close(resumed["ema"]["shadow"][key], value, rtol=0, atol=0)
    history = json.loads((tmp_path / "resumed" / "metrics.jsonl").read_text().strip())
    assert history["epoch"] == 2
    assert len(history["next_learning_rates"]) == 2
    metrics_path = tmp_path / "evaluation.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--checkpoint", str(output / "latest.pt"),
        "--input-dir", str(input_dir), "--target-dir", str(target_dir),
        "--device", "cpu", "--test-y-channel", "--uncertainty",
        "--compare-structural", "--json", str(metrics_path),
    ])
    evaluate.main()
    evaluation = json.loads(metrics_path.read_text())
    assert len(evaluation["images"]) == 3
    assert evaluation["average"]["delta_psnr"] == 0
    assert "risk_spatial_nll_delta" in evaluation["average"]
    assert evaluation["protocol"]["refinement_mode"] == "none"
    assert evaluation["protocol"]["steps"] == 0
    assert evaluation["protocol"]["overlap"] is None
    assert evaluation["protocol"]["output_quantization"] == "float32_clamp_0_1"
    assert evaluation["protocol"]["checkpoint_epoch"] == 1
    assert evaluation["protocol"]["checkpoint_weights"] == "ema"
