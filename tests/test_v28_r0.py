from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sgbformer.config import load_config
from sgbformer.models import SignedSpatialBasisReadout, build_model
from sgbformer.protocol import implementation_fingerprint, sha256_file
from sgbformer.utils import parameter_count, seed_everything
from sgbformer.v27 import resolved_config_sha256
from sgbformer.v28 import (
    V28_PEAK_MEMORY_GATE_MB,
    V28_PROFILE_STEPS,
    V28_PROTOCOL,
    V28_R0_ARM,
    V28_R0_EXPECTED_TRAINABLE_PARAMETERS,
    audit_v28_r0_config,
    validate_v28_launch_artifacts,
    validate_v28_r0_config,
)


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
    spec = spec_from_file_location("_v28_r0_ssr", architecture)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "net_g_best.pth"
    torch.save({"params": module.SSR(dim=48, bias=False).state_dict()}, checkpoint)
    return repo, checkpoint


def load_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict, Path]:
    repo, checkpoint = write_ssr(tmp_path)
    monkeypatch.setenv("SSR_REPO", str(repo))
    monkeypatch.setenv("SSR_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SGB_DATA_ROOT", str(tmp_path / "data"))
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    candidate_path = root / "ssr_residual_v28_r0_deterministic_basis.yaml"
    return (
        load_config(root / "ssr_residual_v27_c0_proxy.yaml"),
        load_config(candidate_path),
        candidate_path,
    )


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = spec_from_file_location(f"_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v28_r0_changes_only_the_signed_basis_readout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, candidate, _ = load_configs(tmp_path, monkeypatch)
    assert validate_v28_r0_config(candidate, reference) == V28_R0_ARM
    audit = audit_v28_r0_config(candidate, reference)
    assert audit["behavioral_differences"] == [
        "model.bfn.signed_basis_readout"
    ]
    assert candidate["training"]["sampling"] == {"mode": "shuffle"}
    assert candidate["loss"] == reference["loss"]
    assert candidate["data"] == reference["data"]
    assert candidate["seed"] == reference["seed"] == 42

    candidate["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="outside protocol metadata"):
        validate_v28_r0_config(candidate, reference)


def test_signed_basis_initialization_is_deterministic_and_identity_safe() -> None:
    torch.manual_seed(1)
    first = SignedSpatialBasisReadout(16, 3)
    torch.manual_seed(999)
    second = SignedSpatialBasisReadout(16, 3)
    torch.testing.assert_close(
        first.basis_projection.weight,
        second.basis_projection.weight,
        atol=0,
        rtol=0,
    )
    assert torch.count_nonzero(first.basis_projection.weight) > 0
    assert torch.count_nonzero(first.coefficient_projection.weight) == 0
    assert torch.count_nonzero(first.coefficient_projection.bias) == 0

    feature = torch.rand(2, 16, 8, 8, requires_grad=True)
    target = torch.rand(2, 3, 8, 8)
    output = first(feature)
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
    assert first.last_basis_maps is not None
    assert torch.count_nonzero(first.last_basis_maps) > 0
    assert first.last_coefficients is not None
    assert torch.count_nonzero(first.last_coefficients) == 0

    optimizer = torch.optim.SGD(first.parameters(), lr=0.1)
    F.mse_loss(output, target).backward()
    assert first.coefficient_projection.weight.grad is not None
    assert torch.count_nonzero(first.coefficient_projection.weight.grad) > 0
    assert first.basis_projection.weight.grad is not None
    assert torch.count_nonzero(first.basis_projection.weight.grad) == 0
    assert feature.grad is not None
    assert torch.count_nonzero(feature.grad) == 0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    feature.grad = None
    F.mse_loss(first(feature), target).backward()
    assert first.basis_projection.weight.grad is not None
    assert torch.count_nonzero(first.basis_projection.weight.grad) > 0
    assert feature.grad is not None
    assert torch.count_nonzero(feature.grad) > 0


def test_v28_r0_model_structure_identity_and_gradient_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config, _ = load_configs(tmp_path, monkeypatch)
    module = load_script("preflight_v28_r0.py")
    seed_everything(42)
    model = build_model(config)
    assert parameter_count(model) == V28_R0_EXPECTED_TRAINABLE_PARAMETERS
    samples = [
        {
            "degraded": torch.rand(3, 16 + index % 2, 18 + index % 3),
            "target": torch.rand(3, 16 + index % 2, 18 + index % 3),
            "input_path": f"dev-{index}",
        }
        for index in range(8)
    ]
    structure = module.assert_r0_structure(model)
    identity = module.check_r0_identity(
        model,
        samples,
        torch.device("cpu"),
    )
    gradients = module.check_r0_gradient_path(
        model,
        config,
        samples[0],
        torch.device("cpu"),
    )
    assert structure["trainable_parameters"] == 616_340
    assert structure["dct_gram_max_abs_error"] <= 1e-5
    assert identity["maximum_absolute_difference"] <= 1e-6
    assert identity["receiver_calls_per_image"] == 1
    assert gradients["first_image_backward"]["coefficient_projection"] > 0
    assert gradients["first_image_backward"]["basis_projection"] == 0
    assert gradients["first_image_backward"]["receiver_upstream"] == 0
    assert gradients["second_image_backward"]["basis_projection"] > 0
    assert gradients["second_image_backward"]["receiver_upstream"] > 0
    assert gradients["foundation_state_unchanged"] is True


def test_v28_r0_launch_requires_matching_preflight_and_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config, config_path = load_configs(tmp_path, monkeypatch)
    preflight_path = tmp_path / "preflight_r0.json"
    profile_path = tmp_path / "R0.json"
    config["training"]["v28_preflight_report"] = str(preflight_path)
    config["training"]["v28_profile_report"] = str(profile_path)
    preflight = {
        "protocol": V28_PROTOCOL,
        "arm": V28_R0_ARM,
        "status": "passed",
        "config": {
            "source_sha256": sha256_file(config_path),
            "resolved_sha256": resolved_config_sha256(config),
        },
        "implementation": implementation_fingerprint(),
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    profile = {
        "protocol": V28_PROTOCOL,
        "arm": V28_R0_ARM,
        "status": "passed",
        "source_config_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_config_sha256(config),
        "optimizer_steps": V28_PROFILE_STEPS,
        "trainable_parameters": V28_R0_EXPECTED_TRAINABLE_PARAMETERS,
        "receiver_calls_per_image": 1,
        "quality_metrics_observed": False,
        "preflight": {"sha256": sha256_file(preflight_path)},
        "implementation": implementation_fingerprint(),
        "device": "test-device",
        "seconds_per_optimizer_step": 1.0,
        "peak_memory_allocated_mb": V28_PEAK_MEMORY_GATE_MB,
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    launch = validate_v28_launch_artifacts(config_path, config)
    assert launch["arm"] == V28_R0_ARM
    assert launch["profile_device"] == "test-device"

    profile["receiver_calls_per_image"] = 2
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(RuntimeError, match="one receiver call"):
        validate_v28_launch_artifacts(config_path, config)


def test_v28_r0_profile_accepts_only_matching_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config, config_path = load_configs(tmp_path, monkeypatch)
    module = load_script("profile_v28_r0.py")
    preflight_path = tmp_path / "preflight_r0.json"
    report = {
        "protocol": V28_PROTOCOL,
        "arm": V28_R0_ARM,
        "status": "passed",
        "config": {
            "source_sha256": sha256_file(config_path),
            "resolved_sha256": resolved_config_sha256(config),
        },
        "implementation": implementation_fingerprint(),
    }
    preflight_path.write_text(json.dumps(report), encoding="utf-8")
    loaded = module.load_verified_preflight(
        preflight_path,
        config_path=config_path,
        config=config,
    )
    assert loaded["arm"] == V28_R0_ARM

    report["config"]["resolved_sha256"] = "changed"
    preflight_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resolved config fingerprint"):
        module.load_verified_preflight(
            preflight_path,
            config_path=config_path,
            config=config,
        )


def test_v28_r0_decision_requires_capacity_and_rain_safety() -> None:
    module = load_script("analyze_v28_r0.py")

    def report(
        macro: tuple[float, list[float]] = (0.05, [0.001, 0.10]),
        outdoor: tuple[float, list[float]] = (-0.003, [-0.01, 0.001]),
        raindrop: tuple[float, list[float]] = (-0.005, [-0.02, 0.001]),
    ) -> dict:
        return {
            "macro_weighted": {
                "psnr": {
                    "delta": macro[0],
                    "paired_image_bootstrap_ci95": macro[1],
                }
            },
            "categories": {
                "outdoor_rain": {
                    "n": 400,
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
                        "improved": 28,
                    },
                },
            },
        }

    assert module.continuation_decision(report())["status"] == "GO"
    result = module.continuation_decision(
        report(macro=(0.0499, [0.001, 0.10]))
    )
    assert result["status"] == "STOP"
    assert "macro_delta_at_least_0p05" in result["failed_checks"]

    result = module.continuation_decision(
        report(outdoor=(-0.004, [-0.01, 0.001]))
    )
    assert result["status"] == "STOP"
    assert (
        "outdoor_rain_point_at_least_neg_0p003"
        in result["failed_checks"]
    )


def test_v28_r0_analyzer_binds_evaluation_to_fixed_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config, _ = load_configs(tmp_path, monkeypatch)
    module = load_script("analyze_v28_r0.py")
    checkpoint_path = tmp_path / "epoch_0006.pt"
    torch.save(
        {
            "epoch": 5,
            "global_step": 123,
            "ema": {"shadow": {}},
            "config": config,
        },
        checkpoint_path,
    )
    evaluation_path = tmp_path / "evaluation.json"
    evaluation = {
        "protocol": {
            "experiment_protocol": V28_PROTOCOL,
            "experiment_arm": V28_R0_ARM,
            "resolved_config_sha256": resolved_config_sha256(config),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": 5,
            "checkpoint_weights": "ema",
            "test_y_channel": True,
            "crop_border": 0,
            "stochastic": False,
            "tile": None,
            "self_ensemble": False,
            "refinement_mode": "direct",
            "receiver_calls_per_trajectory": 1,
        }
    }
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    endpoint = module.validate_fixed_endpoint(
        checkpoint_path,
        evaluation_path,
    )
    assert endpoint["checkpoint_epoch"] == 5

    evaluation["protocol"]["checkpoint_sha256"] = "changed"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint hash"):
        module.validate_fixed_endpoint(checkpoint_path, evaluation_path)
