from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re

import pytest
import torch

from sgbformer.checkpoint import save_checkpoint
from sgbformer.ema import ExponentialMovingAverage
from sgbformer.inference import load_model_for_inference
from sgbformer.models import build_model
from sgbformer.utils import save_image


SPLIT_SSR = """
from torch import nn


class SSR(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=4,
        num_blocks=None,
        ffn_expansion_factor=2,
        bias=True,
    ):
        super().__init__()
        del num_blocks, ffn_expansion_factor
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, padding=1, bias=bias)
        self.encoder_level1 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=bias))
        self.down1_2 = nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1, bias=bias)
        self.encoder_level2 = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 3, padding=1, bias=bias))
        self.down2_3 = nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1, bias=bias)
        self.encoder_level3 = nn.Sequential(nn.Conv2d(dim * 4, dim * 4, 3, padding=1, bias=bias))
        self.decoder_level3 = nn.Sequential(nn.Conv2d(dim * 4, dim * 4, 3, padding=1, bias=bias))
        self.up3_2 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2, bias=bias)
        self.decoder_level2 = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 3, padding=1, bias=bias))
        self.up2_1 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2, bias=bias)
        self.decoder_level1 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=bias))
        self.output = nn.Conv2d(dim, out_channels, 3, padding=1, bias=bias)

    def forward(self, inp_img, return_degrade=False):
        del return_degrade
        level1 = self.encoder_level1(self.patch_embed(inp_img))
        level2 = self.encoder_level2(self.down1_2(level1))
        level3 = self.encoder_level3(self.down2_3(level2))
        decoded3 = self.decoder_level3(level3)
        decoded2 = self.decoder_level2(self.up3_2(decoded3) + level2)
        decoded1 = self.decoder_level1(self.up2_1(decoded2) + level1)
        return self.output(decoded1) + inp_img
"""


def _load_probe_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "probe_latent_leverage.py"
    spec = spec_from_file_location("probe_latent_leverage_test", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_ssr(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "SSR"
    architecture = repo / "basicsr" / "models" / "archs" / "SSR_arch.py"
    architecture.parent.mkdir(parents=True)
    architecture.write_text(SPLIT_SSR, encoding="utf-8")
    spec = spec_from_file_location("_probe_ssr", architecture)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "net_g_best.pth"
    torch.save({"params": module.SSR(dim=4).state_dict()}, checkpoint)
    return repo, checkpoint


def _config(repo: Path, foundation_checkpoint: Path) -> dict:
    return {
        "seed": 42,
        "model": {
            "semantic": {"encoder": "none", "embedding_dim": 1},
            "foundation": {
                "type": "external_ssr",
                "repo_path": str(repo),
                "checkpoint": str(foundation_checkpoint),
                "architecture": {"dim": 4},
                "freeze": False,
                "trainable_modules": [],
                "frozen_modules_eval": False,
                "feature_adapters": {
                    "encoder_level3": {
                        "type": "paper_spectral_gate",
                        "channels": 16,
                        "initial_gate": 0.999,
                    },
                    "decoder_level3": {
                        "type": "latent_film_spectral",
                        "channels": 16,
                        "global_dim": 8,
                        "spatial_dim": 2,
                        "hidden_channels": 4,
                    },
                    "decoder_level2": {
                        "type": "latent_film_spectral",
                        "channels": 8,
                        "global_dim": 8,
                        "spatial_dim": 2,
                        "hidden_channels": 4,
                    },
                },
            },
            "bfn": {"inference_mode": "none", "mask_mode": "none"},
            "latent_bfn": {
                "enabled": True,
                "steps": 2,
                "global_dim": 8,
                "spatial_dim": 2,
                "hidden_channels": 4,
                "feature_channels": 16,
                "num_categories": 3,
                "spatial_downsample": 2,
                "category_patterns": {
                    "snow": r".*\.jpg",
                    "outdoor_rain": r"im_\d+_s\d+_a0[456]\.png",
                    "raindrop": r"\d+_rain\.png",
                },
            },
        },
        "training": {
            "sampling": {
                "target_fractions": {
                    "snow": 0.5,
                    "outdoor_rain": 0.25,
                    "raindrop": 0.25,
                }
            }
        },
    }


def _checkpoint(tmp_path: Path) -> tuple[Path, dict]:
    repo, foundation_checkpoint = _write_ssr(tmp_path)
    config = _config(repo, foundation_checkpoint)
    model = build_model(config).eval()
    with torch.no_grad():
        model.backbone.adapters["decoder_level3"].beta_projection.bias.fill_(0.03)
        model.backbone.adapters["decoder_level2"].beta_projection.bias.fill_(0.02)
        model.backbone.adapters["decoder_level3"].fusion_gain.fill_(0.7)
        model.backbone.adapters["decoder_level2"].fusion_gain.fill_(0.6)
        model.backbone.adapters["encoder_level3"].spectral_gate.gate.bias.zero_()
    ema = ExponentialMovingAverage(model, decay=0.0, warmup_updates=0)
    checkpoint = tmp_path / "v26d.pt"
    save_checkpoint(
        checkpoint,
        model,
        config=config,
        epoch=2,
        global_step=1,
        ema=ema,
    )
    return checkpoint, config


def _write_three_categories(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "input"
    target_dir = tmp_path / "target"
    torch.manual_seed(7)
    for name in (
        "snow.jpg",
        "im_0001_s80_a04.png",
        "123_rain.png",
    ):
        target = torch.rand(3, 16, 16)
        degraded = (target + 0.08).clamp(0, 1)
        save_image(degraded, input_dir / name)
        save_image(target, target_dir / name)
    return input_dir, target_dir


def test_probe_fusion_context_builds_raw_f0_and_restores_gains(
    tmp_path: Path,
) -> None:
    probe = _load_probe_module()
    checkpoint, config = _checkpoint(tmp_path)
    model, _ = load_model_for_inference(checkpoint, torch.device("cpu"))
    original = probe.current_fusion_gains(model)

    with probe.temporary_fusion_gains(
        model,
        {"decoder_level3": 0.0, "decoder_level2": 0.0},
    ):
        assert probe.current_fusion_gains(model) == {
            "decoder_level3": 0.0,
            "decoder_level2": 0.0,
        }
    assert probe.current_fusion_gains(model) == original

    encoder_gate = probe.encoder_gate_adapter(model)
    feature = torch.rand(1, 16, 4, 4)
    gated = encoder_gate(feature)
    assert "forward" not in encoder_gate.__dict__
    with probe.temporary_encoder_gate_bypass(model, enabled=True):
        assert "forward" in encoder_gate.__dict__
        torch.testing.assert_close(encoder_gate(feature), feature, rtol=0, atol=0)
    assert "forward" not in encoder_gate.__dict__
    torch.testing.assert_close(encoder_gate(feature), gated, rtol=0, atol=0)
    assert not torch.equal(gated, feature)

    raw = probe.build_raw_foundation(config, torch.device("cpu"))
    assert not raw.adapters
    assert not any(parameter.requires_grad for parameter in raw.model.parameters())


def test_evaluate_variant_disables_autograd(tmp_path: Path) -> None:
    probe = _load_probe_module()
    checkpoint, config = _checkpoint(tmp_path)
    observed_grad_modes = []

    class SingleImageDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            assert index == 0
            image = torch.rand(3, 16, 16)
            return {
                "degraded": image,
                "target": image,
                "input_path": Path("snow.jpg"),
                "name": "snow",
            }

    def restore(degraded, target, category):
        del target, category
        observed_grad_modes.append(torch.is_grad_enabled())
        return degraded, {}

    report = probe.evaluate_variant(
        dataset=SingleImageDataset(),
        category_patterns={"snow": re.compile(r".*\.jpg")},
        category_indices=None,
        variant="no_grad_guard",
        config=config,
        crop_border=0,
        test_y_channel=True,
        checkpoint_path=checkpoint,
        checkpoint_sha256="test",
        device=torch.device("cpu"),
        restore=restore,
        gains=None,
        quiet=True,
    )

    assert observed_grad_modes == [False]
    assert report["images"][0]["psnr"] == pytest.approx(120.0)


def test_probe_cli_outputs_three_way_sweep_and_unit_precision_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    checkpoint, _ = _checkpoint(tmp_path)
    input_dir, target_dir = _write_three_categories(tmp_path)
    output_dir = tmp_path / "probe"
    probe.main([
        "--checkpoint",
        str(checkpoint),
        "--input-dir",
        str(input_dir),
        "--target-dir",
        str(target_dir),
        "--device",
        "cpu",
        "--output-dir",
        str(output_dir),
        "--rgb-metrics",
        "--fusion-sweep",
        "--fusion-alpha",
        "0",
        "--fusion-alpha",
        "1",
        "--single-injection-sweep",
        "--oracle",
        "--oracle-precision-scale",
        "0.5",
        "--bootstrap-samples",
        "100",
        "--quiet",
    ])

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["images"] == 3
    assert summary["guardrails"]["fusion_gains_restored"] is True
    assert set(summary["comparisons"]) == {
        "bfn_latent_F2_minus_F1",
        "ema_ssr_weight_drift_Fback_minus_F0",
        "encoder_spectral_gate_F1_minus_Fback",
        "total_F2_minus_F0",
        "oracle_mean_predicted_precision_minus_F1",
        "oracle_mean_unit_precision_condition_minus_F1",
        "oracle_mean_predicted_precision_scale_0p5_minus_F1",
    }
    for report in summary["comparisons"].values():
        assert report["bootstrap_samples"] == 100
        assert report["macro_weighted"]["weights"] == {
            "snow": 0.5,
            "outdoor_rain": 0.25,
            "raindrop": 0.25,
        }

    f0 = json.loads((output_dir / "f0_foundation.json").read_text(encoding="utf-8"))
    fback = json.loads(
        (output_dir / "fback_ema_ssr_no_adapters.json").read_text(encoding="utf-8")
    )
    f1 = json.loads((output_dir / "f1_fusion_off.json").read_text(encoding="utf-8"))
    f2 = json.loads((output_dir / "f2_normal.json").read_text(encoding="utf-8"))
    assert (
        len(f0["images"])
        == len(fback["images"])
        == len(f1["images"])
        == len(f2["images"])
        == 3
    )
    assert f0["probe"]["fusion_gains"] is None
    assert fback["probe"]["encoder_gate_bypassed"] is True
    assert fback["probe"]["fusion_gains"] == {
        "decoder_level3": 0.0,
        "decoder_level2": 0.0,
    }
    assert f1["probe"]["fusion_gains"] == {
        "decoder_level3": 0.0,
        "decoder_level2": 0.0,
    }
    assert f2["probe"]["fusion_gains"] == {
        "decoder_level3": pytest.approx(0.7),
        "decoder_level2": pytest.approx(0.6),
    }
    assert any(
        abs(left["psnr"] - right["psnr"]) > 1e-8
        for left, right in zip(f1["images"], f2["images"])
    )
    assert any(
        abs(left["psnr"] - right["psnr"]) > 1e-8
        for left, right in zip(fback["images"], f1["images"])
    )

    sweep = json.loads(
        (output_dir / "fusion_sweep" / "index.json").read_text(encoding="utf-8")
    )
    assert {
        "decoder_level3_off",
        "decoder_level2_off",
        "both_relative_alpha_0",
        "both_relative_alpha_1",
        "decoder_level3_only_relative_alpha_0",
        "decoder_level3_only_relative_alpha_1",
        "decoder_level2_only_relative_alpha_0",
        "decoder_level2_only_relative_alpha_1",
    } == set(sweep["variants"])
    assert sweep["alpha_mode"] == "relative"
    assert sweep["variants"]["both_relative_alpha_1"]["fusion_gains"] == {
        "decoder_level3": pytest.approx(0.7),
        "decoder_level2": pytest.approx(0.6),
    }

    predicted_oracle = json.loads(
        (
            output_dir / "oracle_mean_predicted_precision.json"
        ).read_text(encoding="utf-8")
    )
    unit_oracle = json.loads(
        (
            output_dir / "oracle_mean_unit_precision.json"
        ).read_text(encoding="utf-8")
    )
    scaled_oracle = json.loads(
        (
            output_dir / "oracle_mean_predicted_precision_scale_0p5.json"
        ).read_text(encoding="utf-8")
    )
    assert predicted_oracle["protocol"]["oracle_precision"] == "predicted"
    assert unit_oracle["protocol"]["oracle_precision"] == "unit_condition"
    assert scaled_oracle["protocol"]["oracle_precision"] == "predicted_scaled_0.5"

    model, _ = load_model_for_inference(checkpoint, torch.device("cpu"))
    degraded = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(degraded)
    captured: dict[str, torch.Tensor] = {}
    original = model.restore_from_latent_state

    def wrapped(image, **kwargs):
        captured["global_precision"] = kwargs["global_precision"]
        captured["spatial_precision"] = kwargs["spatial_precision"]
        return original(image, **kwargs)

    monkeypatch.setattr(model, "restore_from_latent_state", wrapped)
    monkeypatch.setattr(
        model,
        "restore_oracle_z",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("probe must not call restore_oracle_z")
        ),
    )
    prediction, _ = probe.oracle_restore(
        model,
        degraded,
        target,
        category=torch.tensor([0]),
        precision_source="unit",
    )
    assert prediction.shape == degraded.shape
    assert torch.equal(
        captured["global_precision"],
        torch.ones_like(captured["global_precision"]),
    )
    assert torch.equal(
        captured["spatial_precision"],
        torch.ones_like(captured["spatial_precision"]),
    )
    (
        _,
        _,
        predicted_global_precision,
        predicted_spatial_precision,
    ) = probe.oracle_latent_state(
        model,
        degraded,
        target,
        category=torch.tensor([0]),
    )
    prediction, _ = probe.oracle_restore(
        model,
        degraded,
        target,
        category=torch.tensor([0]),
        precision_source="predicted",
        precision_scale=0.5,
    )
    assert prediction.shape == degraded.shape
    torch.testing.assert_close(
        captured["global_precision"],
        predicted_global_precision.mul(0.5).clamp_min(1e-6),
    )
    torch.testing.assert_close(
        captured["spatial_precision"],
        predicted_spatial_precision.mul(0.5).clamp_min(1e-6),
    )
