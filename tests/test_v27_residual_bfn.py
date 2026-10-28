from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sgbformer.config import load_config
from sgbformer.data import CategoryBalancedSampler
from sgbformer.models import build_model
from sgbformer.utils import parameter_count, seed_everything
from sgbformer.v27 import (
    V27_EXPECTED_TRAINABLE_PARAMETERS,
    V27_C0_E1_TARGET_COUNTS,
    audit_v27_c0_e1_configs,
    normalized_v27_config,
    resolved_config_sha256,
    trainable_parameter_signature,
    trainable_state_sha256,
    validate_v27_config,
    validate_v27_c0_e1_config,
    validate_v27_launch_artifacts,
)
from sgbformer.protocol import implementation_fingerprint, sha256_file
from train import build_optimizer


SPLIT_SSR = """
from torch import nn


class SSR(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
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


def write_ssr(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "SSR"
    architecture = repo / "basicsr" / "models" / "archs" / "SSR_arch.py"
    architecture.parent.mkdir(parents=True)
    architecture.write_text(SPLIT_SSR, encoding="utf-8")
    spec = spec_from_file_location("_v27_ssr", architecture)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "net_g_best.pth"
    torch.save({"params": module.SSR(dim=48, bias=False).state_dict()}, checkpoint)
    return repo, checkpoint


def load_v27_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict]:
    repo, checkpoint = write_ssr(tmp_path)
    monkeypatch.setenv("SSR_REPO", str(repo))
    monkeypatch.setenv("SSR_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SGB_DATA_ROOT", str(tmp_path / "data"))
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    return {
        arm: load_config(root / f"ssr_residual_v27_{arm.lower()}_proxy.yaml")
        for arm in ("C0", "D1", "D4", "B4")
    }


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = spec_from_file_location(f"_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_c0_e1_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict]:
    configs = load_v27_configs(tmp_path, monkeypatch)
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    exploratory = load_config(
        root / "ssr_residual_v27_c0_e1_rainsafe.yaml"
    )
    return configs["C0"], exploratory


def test_v27_configs_freeze_the_registered_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = load_v27_configs(tmp_path, monkeypatch)
    for arm, config in configs.items():
        assert validate_v27_config(config, expected_arm=arm) == arm
        assert config["model"]["foundation"]["feature_adapters"] == {}
        assert config["model"]["latent_bfn"]["enabled"] is False
        assert config["training"]["sampling"] == {"mode": "shuffle"}
        assert config["loss"]["latent_category_weight"] == 0

    assert normalized_v27_config(configs["D4"]) == normalized_v27_config(
        configs["B4"]
    )


def test_v27_arms_have_exactly_matched_receiver_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = load_v27_configs(tmp_path, monkeypatch)
    signatures = {}
    initial_states = {}
    for arm, config in configs.items():
        seed_everything(42)
        model = build_model(config)
        assert parameter_count(model) == V27_EXPECTED_TRAINABLE_PARAMETERS
        assert model.receiver is not None
        assert not model.backbone.adapters
        assert not model.latent_bfn_enabled
        assert all(
            not parameter.requires_grad
            for parameter in model.backbone.parameters()
        )
        assert all(
            name.startswith("receiver.")
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        signatures[arm] = trainable_parameter_signature(model)
        initial_states[arm] = trainable_state_sha256(model)

    assert len({str(value) for value in signatures.values()}) == 1
    assert len(set(initial_states.values())) == 1


@pytest.mark.parametrize("arm", ("C0", "D1", "D4", "B4"))
def test_v27_initial_output_is_exactly_the_frozen_foundation(
    arm: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_v27_configs(tmp_path, monkeypatch)[arm]
    model = build_model(config).eval()
    image = torch.rand(1, 3, 17, 19)
    baseline = model.restore_structural(image)
    details = model.restore(image, deterministic=True, return_details=True)

    torch.testing.assert_close(details["restored"], baseline, atol=0, rtol=0)
    torch.testing.assert_close(
        details["residual"],
        torch.zeros_like(details["residual"]),
        atol=0,
        rtol=0,
    )
    assert details["receiver_calls"] == (1 if arm in {"C0", "D1"} else 4)


@pytest.mark.parametrize("arm", ("C0", "B4"))
def test_v27_zero_projection_opens_gradient_path_after_one_step(
    arm: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_v27_configs(tmp_path, monkeypatch)[arm]
    seed_everything(42)
    model = build_model(config).train()
    optimizer = build_optimizer(model, config["training"])
    image = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(image)
    stage = "residual" if arm == "C0" else "joint"
    output_ids = {id(parameter) for parameter in model.receiver.output.parameters()}

    optimizer.zero_grad(set_to_none=True)
    outputs = model(image, target, training_stage=stage)
    F.mse_loss(outputs["restored"], target).backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.receiver.output.parameters()
    )
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.receiver.parameters()
        if id(parameter) not in output_ids
    )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(image, target, training_stage=stage)
    F.mse_loss(outputs["restored"], target).backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.receiver.parameters()
        if id(parameter) not in output_ids
    )
    assert all(
        parameter.grad is None for parameter in model.backbone.parameters()
    )


def test_v27_validator_rejects_a_category_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_v27_configs(tmp_path, monkeypatch)["B4"]
    config["loss"]["latent_category_weight"] = 0.01
    with pytest.raises(ValueError, match="latent_category_weight"):
        validate_v27_config(config)


def test_v27_launch_requires_matching_preflight_and_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_v27_configs(tmp_path, monkeypatch)["C0"]
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0_proxy.yaml"
    )
    preflight_path = tmp_path / "preflight.json"
    profile_path = tmp_path / "profile.json"
    config["training"]["v27_preflight_report"] = str(preflight_path)
    config["training"]["v27_profile_report"] = str(profile_path)
    preflight = {
        "protocol": "residual_bfn_v27",
        "status": "passed",
        "configs": {
            "C0": {
                "source_sha256": sha256_file(config_path),
                "resolved_sha256": resolved_config_sha256(config),
            }
        },
        "source": {"core": implementation_fingerprint()},
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    profile = {
        "protocol": "residual_bfn_v27",
        "arm": "C0",
        "source_config_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_config_sha256(config),
        "optimizer_steps": 200,
        "trainable_parameters": V27_EXPECTED_TRAINABLE_PARAMETERS,
        "preflight": {"sha256": sha256_file(preflight_path)},
        "implementation": implementation_fingerprint(),
        "device": "test-device",
        "seconds_per_optimizer_step": 1.0,
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    result = validate_v27_launch_artifacts(config_path, config)
    assert result["arm"] == "C0"
    assert result["profile_device"] == "test-device"

    profile["optimizer_steps"] = 199
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(RuntimeError, match="200 optimizer steps"):
        validate_v27_launch_artifacts(config_path, config)


def test_v27_capacity_gate_uses_fixed_epoch_and_category_safety() -> None:
    module = load_script("analyze_v27_proxy.py")

    summary = {
        "macro_delta": 0.01,
        "macro_ci95": [-0.01, 0.03],
        "categories": {
            "snow": {"delta": 0.02, "ci95": [0.0, 0.04]},
            "outdoor_rain": {"delta": 0.0, "ci95": [-0.01, 0.01]},
        },
    }
    assert module.capacity_decision(summary)["status"] == "CONDITIONAL_GO"

    summary["categories"]["outdoor_rain"] = {
        "delta": -0.03,
        "ci95": [-0.04, -0.02],
    }
    decision = module.capacity_decision(summary)
    assert decision["status"] == "STOP"
    assert "outdoor_rain" in decision["reason"]


def test_v27_preflight_checks_structure_identity_and_gradient_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("preflight_v27_residual_bfn.py")
    config = load_v27_configs(tmp_path, monkeypatch)["B4"]
    model = build_model(config)
    structure = module.assert_model_structure(model)
    identity = module.check_identity(model, torch.device("cpu"))
    sample = {
        "degraded": torch.rand(3, 16, 16),
        "target": torch.rand(3, 16, 16),
    }
    gradients = module.check_gradient_path(
        model,
        config,
        sample,
        torch.device("cpu"),
    )

    assert structure["trainable_parameters"] == 614_455
    assert all(case["output_equal"] for case in identity["cases"])
    assert gradients["foundation_state_unchanged"] is True
    assert gradients["first_image_backward"]["output_projection"] > 0
    assert gradients["second_image_backward"]["upstream"] > 0


def test_v27_analysis_writes_a_fixed_epoch_capacity_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("analyze_v27_proxy.py")
    protocol = {
        "test_y_channel": True,
        "crop_border": 0,
        "lpips_network": None,
        "tile": None,
        "stochastic": False,
        "output_quantization": "float32_clamp_0_1",
        "foundation_signature": "same",
    }
    rows = [
        ("snow", "snow", 30.0),
        ("im_0001_s80_a04", "outdoor_rain", 30.0),
        ("123_rain", "raindrop", 30.0),
    ]
    baseline = {
        "average": {"psnr": 30.0, "ssim": 0.9},
        "images": [
            {"name": name, "category": category, "psnr": value, "ssim": 0.9}
            for name, category, value in rows
        ],
        "protocol": protocol,
    }
    baseline_path = tmp_path / "foundation.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    pairs = tmp_path / "val_pairs.tsv"
    pairs.write_text(
        "snow.jpg\tsnow.jpg\n"
        "im_0001_s80_a04.png\tim_0001_s80_a04.png\n"
        "123_rain.png\t123_rain.png\n",
        encoding="utf-8",
    )
    arm_dir = tmp_path / "C0"
    arm_dir.mkdir()
    for epoch in range(1, 7):
        candidate = {
            "average": {"psnr": 30.1, "ssim": 0.9},
            "images": [
                {
                    "name": name,
                    "category": category,
                    "psnr": value + 0.1,
                    "ssim": 0.9,
                }
                for name, category, value in rows
            ],
            "protocol": {
                **protocol,
                "experiment_protocol": "residual_bfn_v27",
                "experiment_arm": "C0",
                "resolved_config_sha256": "config",
            },
        }
        (arm_dir / f"epoch_{epoch:04d}.json").write_text(
            json.dumps(candidate),
            encoding="utf-8",
        )
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_v27_proxy.py",
            "--baseline",
            str(baseline_path),
            "--arm",
            f"C0={arm_dir}",
            "--pairs-file",
            str(pairs),
            "--output-dir",
            str(output_dir),
        ],
    )
    module.main()

    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["primary_checkpoint"] == "epoch_0006"
    assert summary["stage_a0_decision"]["status"] == "GO"
    assert summary["arm_resolved_config_sha256"] == {"C0": "config"}


def test_c0_e1_changes_only_the_registered_sampling_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c0, exploratory = load_c0_e1_config(tmp_path, monkeypatch)
    assert validate_v27_c0_e1_config(exploratory, c0) == "C0-E1"
    audit = audit_v27_c0_e1_configs(c0, exploratory)

    assert audit["behavioral_differences"] == ["training.sampling"]
    assert exploratory["training"]["epochs"] == c0["training"]["epochs"] == 6
    assert exploratory["training"]["stages"] == c0["training"]["stages"]
    for key in (
        "learning_rate",
        "minimum_learning_rate",
        "minimum_learning_rate_ratio",
        "schedule_minimum_mode",
        "warmup_epochs",
        "batch_size",
        "micro_batch_size",
    ):
        assert exploratory["training"][key] == c0["training"][key]
    assert exploratory["model"] == c0["model"]
    assert exploratory["loss"] == c0["loss"]
    assert exploratory["data"] == c0["data"]
    assert exploratory["seed"] == c0["seed"] == 42


def test_c0_e1_sampler_targets_keep_the_epoch_size() -> None:
    labels = (
        ["snow"] * 8_551
        + ["outdoor_rain"] * 7_838
        + ["raindrop"] * 777
    )
    sampler = CategoryBalancedSampler(
        labels,
        {
            "snow": 0.30,
            "outdoor_rain": 0.45,
            "raindrop": 0.25,
        },
        num_samples=17_166,
        seed=42,
    )
    assert len(sampler) == 17_166
    assert dict(sampler.target_counts) == V27_C0_E1_TARGET_COUNTS


def test_c0_e1_preserves_parameters_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c0_config, exploratory_config = load_c0_e1_config(
        tmp_path,
        monkeypatch,
    )
    models = {}
    for name, config in (
        ("C0", c0_config),
        ("C0-E1", exploratory_config),
    ):
        seed_everything(42)
        models[name] = build_model(config)
    assert parameter_count(models["C0"]) == parameter_count(
        models["C0-E1"]
    ) == V27_EXPECTED_TRAINABLE_PARAMETERS
    assert trainable_parameter_signature(
        models["C0"]
    ) == trainable_parameter_signature(models["C0-E1"])
    assert trainable_state_sha256(models["C0"]) == trainable_state_sha256(
        models["C0-E1"]
    )
    image = torch.rand(1, 3, 17, 19)
    for model in models.values():
        baseline = model.restore_structural(image)
        restored = model.restore(image, deterministic=True)
        torch.testing.assert_close(restored, baseline, atol=0, rtol=0)


def test_c0_e1_fixed_decision_requires_rain_safety_and_improvement() -> None:
    module = load_script("analyze_v27_c0_e1.py")

    def report(outdoor, raindrop, macro=0.01):
        return {
            "macro_weighted": {"psnr": {"delta": macro}},
            "categories": {
                "outdoor_rain": {
                    "n": 411,
                    "psnr": {
                        "delta": outdoor[0],
                        "paired_image_bootstrap_ci95": outdoor[1],
                        "improved": 200,
                    },
                },
                "raindrop": {
                    "n": 55,
                    "psnr": {
                        "delta": raindrop[0],
                        "paired_image_bootstrap_ci95": raindrop[1],
                        "improved": 30,
                    },
                },
            },
        }

    vs_f0 = report(
        (-0.005, [-0.02, 0.005]),
        (-0.009, [-0.03, 0.002]),
    )
    vs_c0 = report(
        (0.007, [-0.001, 0.015]),
        (0.005, [-0.01, 0.02]),
    )
    assert module.decision(vs_f0, vs_c0)["status"] == "PASS"

    vs_f0["categories"]["raindrop"]["psnr"][
        "paired_image_bootstrap_ci95"
    ] = [-0.03, -0.001]
    result = module.decision(vs_f0, vs_c0)
    assert result["status"] == "STOP"
    assert "raindrop_vs_f0_ci_upper_nonnegative" in result["failed_checks"]


def test_c0_e1_profile_accepts_only_matching_exploratory_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("profile_v27_residual_bfn.py")
    c0, exploratory = load_c0_e1_config(tmp_path, monkeypatch)
    del c0
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0_e1_rainsafe.yaml"
    )
    preflight_path = tmp_path / "preflight.json"
    report = {
        "protocol": "residual_bfn_v27_exploratory",
        "status": "passed",
        "arm": "C0-E1",
        "configs": {
            "C0-E1": {
                "source_sha256": sha256_file(config_path),
                "resolved_sha256": resolved_config_sha256(exploratory),
            }
        },
        "implementation": implementation_fingerprint(),
    }
    preflight_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = module.load_verified_preflight(
        preflight_path,
        arm="C0-E1",
        config_path=config_path,
        config=exploratory,
    )
    assert loaded["status"] == "passed"

    report["protocol"] = "residual_bfn_v27"
    preflight_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a v27 artifact"):
        module.load_verified_preflight(
            preflight_path,
            arm="C0-E1",
            config_path=config_path,
            config=exploratory,
        )
