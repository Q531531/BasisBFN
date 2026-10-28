from importlib.util import module_from_spec, spec_from_file_location
import copy
import json
import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sgbformer.config import load_config
from sgbformer.models import CoefficientStateBasisReadout, build_model
from sgbformer.protocol import implementation_fingerprint, sha256_file
from sgbformer.utils import parameter_count
from sgbformer.v27 import resolved_config_sha256
from sgbformer.basisbfn import (
    BASISBFN_ARMS,
    BASISBFN_EXPECTED_TRAINABLE_PARAMETERS,
    BASISBFN_PEAK_MEMORY_GATE_MB,
    BASISBFN_PROFILE_STEPS,
    BASISBFN_PROTOCOL,
    audit_basisbfn_config,
    audit_basisbfn_matrix,
    canonical_v27_c0_config,
    validate_basisbfn_config,
    validate_basisbfn_launch_artifacts,
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

CONFIG_NAMES = {
    "D4": "basisbfn_d4.yaml",
    "D4-G": "basisbfn_d4g.yaml",
    "B4": "basisbfn_b4.yaml",
    "B4-G": "basisbfn_b4g.yaml",
}


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = spec_from_file_location(f"_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_configs(tmp_path: Path) -> tuple[dict[str, dict], dict[str, Path]]:
    repo = tmp_path / "SSR"
    architecture = repo / "basicsr" / "models" / "archs" / "SSR_arch.py"
    architecture.parent.mkdir(parents=True)
    architecture.write_text(SPLIT_SSR, encoding="utf-8")
    spec = spec_from_file_location("_basisbfn_ssr", architecture)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "net_g_best.pth"
    torch.save({"params": module.SSR(dim=48, bias=False).state_dict()}, checkpoint)
    os.environ["SSR_REPO"] = str(repo)
    os.environ["SSR_CHECKPOINT"] = str(checkpoint)
    os.environ["SGB_DATA_ROOT"] = str(tmp_path / "data")
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    paths = {arm: root / name for arm, name in CONFIG_NAMES.items()}
    return {arm: load_config(path) for arm, path in paths.items()}, paths


def test_coefficient_readout_is_identity_safe_and_uses_one_basis() -> None:
    readout = CoefficientStateBasisReadout(
        16,
        3,
        update_mode="bayesian",
        acceptance_gate={"enabled": True},
    )
    basis_calls = 0
    observer_calls = 0
    gate_input: torch.Tensor | None = None

    def count_basis(_module, _inputs, _output) -> None:
        nonlocal basis_calls
        basis_calls += 1

    def count_observer(_module, _inputs, _output) -> None:
        nonlocal observer_calls
        observer_calls += 1

    def capture_gate(_module, inputs) -> None:
        nonlocal gate_input
        gate_input = inputs[0].detach()

    hooks = [
        readout.basis_projection.register_forward_hook(count_basis),
        readout.observer_output.register_forward_hook(count_observer),
        readout.gate_input.register_forward_pre_hook(capture_gate),
    ]
    feature = torch.randn(2, 16, 9, 11, requires_grad=True)
    target = torch.randn(2, 3, 9, 11)
    output = readout(feature)
    for hook in hooks:
        hook.remove()

    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
    assert basis_calls == readout.last_basis_evaluations == 1
    assert observer_calls == readout.last_observer_calls == 4
    assert readout.last_posterior_precision is not None
    assert torch.all(readout.last_posterior_precision > 1)
    assert readout.last_coefficients is not None
    assert torch.count_nonzero(readout.last_coefficients) == 0
    assert readout.last_acceptance is not None
    torch.testing.assert_close(
        readout.last_acceptance,
        torch.full_like(readout.last_acceptance, 0.5),
        atol=0,
        rtol=0,
    )
    assert gate_input is not None
    posterior_start = readout.channels
    torch.testing.assert_close(
        gate_input[:, posterior_start : posterior_start + readout.num_bases],
        readout.last_posterior_mean,
    )
    torch.testing.assert_close(
        gate_input[:, posterior_start + readout.num_bases :],
        readout.last_posterior_precision.log(),
    )

    F.mse_loss(output, target).backward()
    assert readout.observer_output.weight.grad is not None
    observation_gradient = readout.observer_output.weight.grad[
        : readout.num_bases
    ]
    assert torch.count_nonzero(observation_gradient) > 0
    assert readout.basis_projection.weight.grad is not None
    assert torch.count_nonzero(readout.basis_projection.weight.grad) == 0
    assert feature.grad is not None
    assert torch.count_nonzero(feature.grad) == 0


def test_deterministic_and_bayesian_arms_are_parameter_matched_but_distinct() -> None:
    torch.manual_seed(42)
    deterministic = CoefficientStateBasisReadout(
        16,
        3,
        update_mode="deterministic",
    )
    torch.manual_seed(42)
    bayesian = CoefficientStateBasisReadout(
        16,
        3,
        update_mode="bayesian",
    )
    assert list(deterministic.state_dict()) == list(bayesian.state_dict())
    for name, value in deterministic.state_dict().items():
        torch.testing.assert_close(value, bayesian.state_dict()[name])
    assert sum(p.numel() for p in deterministic.parameters()) == sum(
        p.numel() for p in bayesian.parameters()
    )
    with torch.no_grad():
        deterministic.observer_output.bias[:4].fill_(0.2)
        bayesian.observer_output.bias[:4].fill_(0.2)
    feature = torch.randn(1, 16, 8, 8)
    deterministic(feature)
    bayesian(feature)
    assert deterministic.last_posterior_mean is not None
    assert bayesian.last_posterior_mean is not None
    assert not torch.allclose(
        deterministic.last_posterior_mean,
        bayesian.last_posterior_mean,
    )
    torch.testing.assert_close(
        deterministic.last_posterior_precision,
        bayesian.last_posterior_precision,
    )


def test_bayesian_arm_uses_sender_noise_and_coefficient_bfn_loss() -> None:
    readout = CoefficientStateBasisReadout(
        16,
        3,
        update_mode="bayesian",
    )
    feature = torch.randn(2, 16, 8, 8)
    normalized_residual = torch.randn(2, 3, 8, 8)
    readout(feature)
    step = torch.full((2,), 4, dtype=torch.long)
    torch.manual_seed(11)
    loss = readout.bfn_training_loss(
        normalized_residual,
        step=step,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0
    assert readout.last_bfn_target is not None
    assert readout.last_bfn_target.shape == (2, 4, 8, 8)
    assert readout.last_bfn_prediction is not None
    expected = (
        0.5
        * readout.flow.step_accuracy(
            4,
            readout.steps,
            device=loss.device,
            dtype=torch.float32,
        )
        * (
            readout.last_bfn_target.float()
            - readout.last_bfn_prediction.float()
        ).square().mean()
    )
    torch.testing.assert_close(loss, expected)
    assert readout.last_bfn_time is not None
    assert readout.last_bfn_sender_mean is not None
    assert readout.last_observer_calls == 5
    first_sender_mean = readout.last_bfn_sender_mean.clone()
    torch.manual_seed(12)
    readout.bfn_training_loss(normalized_residual, step=step)
    assert readout.last_bfn_sender_mean is not None
    assert not torch.allclose(
        first_sender_mean,
        readout.last_bfn_sender_mean,
    )
    assert readout.last_bfn_sender_precision is not None
    assert torch.all(readout.last_bfn_sender_precision > 1)
    assert readout.last_observer_calls == 6

    deterministic = CoefficientStateBasisReadout(
        16,
        3,
        update_mode="deterministic",
    )
    deterministic(feature)
    deterministic_loss = deterministic.bfn_training_loss(
        normalized_residual,
        step=step,
    )
    assert float(deterministic_loss.detach()) > 0
    assert deterministic.last_observer_calls == 5


def test_coefficient_target_solve_stays_fp32_under_bfloat16_autocast() -> None:
    readout = CoefficientStateBasisReadout(
        16,
        3,
        update_mode="bayesian",
    )
    feature = torch.randn(2, 16, 8, 8)
    normalized_residual = torch.randn(2, 3, 8, 8)
    with torch.autocast(
        device_type="cpu",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        readout(feature)
        loss = readout.bfn_training_loss(normalized_residual)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0
    assert readout.last_bfn_target is not None
    assert readout.last_bfn_target.dtype == torch.float32


def test_bayesian_inference_can_sample_coefficient_messages() -> None:
    readout = CoefficientStateBasisReadout(
        16,
        3,
        update_mode="bayesian",
    )
    with torch.no_grad():
        readout.observer_output.bias[:4].fill_(0.2)
    feature = torch.randn(1, 16, 8, 8)
    deterministic = readout(feature, deterministic=True)
    torch.manual_seed(1)
    stochastic_a = readout(feature, deterministic=False)
    torch.manual_seed(2)
    stochastic_b = readout(feature, deterministic=False)
    assert not torch.allclose(deterministic, stochastic_a)
    assert not torch.allclose(stochastic_a, stochastic_b)


def test_basisbfn_matrix_and_full_model_contract(tmp_path: Path) -> None:
    configs, _ = load_configs(tmp_path)
    matrix = audit_basisbfn_matrix(configs)
    assert matrix["arms"] == BASISBFN_ARMS
    assert matrix["parameter_matched"] is True
    audit = audit_basisbfn_config(configs["B4"], canonical_v27_c0_config())
    assert audit["behavioral_differences"] == [
        "loss.bfn_weight",
        "model.bfn.steps",
        "model.bfn.coefficient_state_readout",
    ]

    parameter_names = None
    for arm, config in configs.items():
        assert validate_basisbfn_config(config) == arm
        model = build_model(config)
        assert parameter_count(model) == (
            BASISBFN_EXPECTED_TRAINABLE_PARAMETERS
        )
        assert model.coefficient_state_readout_enabled is True
        assert model.receiver is not None
        assert isinstance(model.receiver.output, CoefficientStateBasisReadout)
        names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if parameter_names is None:
            parameter_names = names
        else:
            assert names == parameter_names
        assert all(name.startswith("receiver.") for name in names)
        assert not any(
            parameter.requires_grad
            for parameter in model.backbone.parameters()
        )

    model = build_model(configs["B4-G"]).eval()
    image = torch.rand(1, 3, 17, 19)
    baseline = model.restore_structural(image)
    details = model.restore(
        image,
        deterministic=True,
        return_details=True,
        return_trajectory=True,
    )
    torch.testing.assert_close(details["restored"], baseline, atol=1e-6, rtol=0)
    assert details["receiver_calls"] == 1
    assert details["coefficient_state_updates"] == 4
    assert details["basis_evaluations"] == 1
    assert details["coefficient_update_mode"] == "bayesian"
    assert details["coefficient_gate_enabled"] is True
    assert len(details["trajectory"]) == 4
    assert details["coefficient_maps"].shape[-2:] == image.shape[-2:]
    assert details["coefficient_uncertainty"].shape[-2:] == image.shape[-2:]
    assert torch.all((details["coefficient_acceptance"] >= 0))
    assert torch.all((details["coefficient_acceptance"] <= 1))
    with pytest.raises(ValueError, match="steps are fixed"):
        model.restore(image, num_steps=3)

    target = torch.rand_like(image)
    model.train()
    bayesian_outputs = model(
        image,
        target,
        training_stage="residual",
    )
    assert float(bayesian_outputs["bfn_loss"].detach()) > 0
    assert model.receiver.output.last_bfn_target is not None
    assert model.receiver.output.last_observer_calls == 5

    deterministic_model = build_model(configs["D4"]).train()
    deterministic_outputs = deterministic_model(
        image,
        target,
        training_stage="residual",
    )
    assert float(deterministic_outputs["bfn_loss"].detach()) > 0
    assert deterministic_model.receiver.output.last_observer_calls == 5


def test_basisbfn_accepts_all_four_paper_seeds(tmp_path: Path) -> None:
    configs, _ = load_configs(tmp_path)
    for seed in (42, 43, 44, 45):
        config = copy.deepcopy(configs["B4-G"])
        config["seed"] = seed
        config["training"]["output_dir"] = (
            f"runs/basisbfn/b4-g/seed_{seed}"
        )
        config["training"]["basisbfn_preflight_report"] = (
            f"runs/basisbfn/artifacts/b4-g/seed_{seed}/"
            "preflight/preflight_b4-g.json"
        )
        config["training"]["basisbfn_profile_report"] = (
            f"runs/basisbfn/artifacts/b4-g/seed_{seed}/profile.json"
        )
        assert validate_basisbfn_config(config) == "B4-G"

    config["seed"] = 46
    with pytest.raises(ValueError, match="seed must be one of"):
        validate_basisbfn_config(config)


def test_basisbfn_launch_requires_bound_artifacts(tmp_path: Path) -> None:
    configs, paths = load_configs(tmp_path)
    config = configs["B4-G"]
    config_path = paths["B4-G"]
    preflight_path = tmp_path / "preflight_b4-g.json"
    profile_path = tmp_path / "B4-G.json"
    config["training"]["basisbfn_preflight_report"] = str(
        preflight_path
    )
    config["training"]["basisbfn_profile_report"] = str(profile_path)
    preflight = {
        "protocol": BASISBFN_PROTOCOL,
        "arm": "B4-G",
        "status": "passed",
        "config": {
            "source_sha256": sha256_file(config_path),
            "resolved_sha256": resolved_config_sha256(config),
        },
        "implementation": implementation_fingerprint(),
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    profile = {
        "protocol": BASISBFN_PROTOCOL,
        "arm": "B4-G",
        "status": "passed",
        "source_config_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_config_sha256(config),
        "optimizer_steps": BASISBFN_PROFILE_STEPS,
        "trainable_parameters": (
            BASISBFN_EXPECTED_TRAINABLE_PARAMETERS
        ),
        "receiver_calls_per_image": 1,
        "coefficient_state_updates_per_image": 4,
        "bfn_training_observer_calls_per_image": 1,
        "basis_evaluations_per_image": 1,
        "quality_metrics_observed": False,
        "preflight": {"sha256": sha256_file(preflight_path)},
        "implementation": implementation_fingerprint(),
        "device": "test-device",
        "seconds_per_optimizer_step": 1.0,
        "peak_memory_allocated_mb": BASISBFN_PEAK_MEMORY_GATE_MB,
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    launch = validate_basisbfn_launch_artifacts(config_path, config)
    assert launch["arm"] == "B4-G"

    profile["basis_evaluations_per_image"] = 2
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(RuntimeError, match="basis once"):
        validate_basisbfn_launch_artifacts(config_path, config)


def test_basisbfn_analyzer_binds_fixed_endpoint(tmp_path: Path) -> None:
    configs, _ = load_configs(tmp_path)
    config = configs["B4-G"]
    module = load_script("analyze_basisbfn.py")
    checkpoint_dir = tmp_path / "b4-g"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "epoch_0006.pt"
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
            "experiment_protocol": BASISBFN_PROTOCOL,
            "experiment_arm": "B4-G",
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
            "steps": 4,
            "receiver_calls_per_trajectory": 1,
            "coefficient_state_updates_per_trajectory": 4,
            "basis_evaluations_per_trajectory": 1,
            "coefficient_update_mode": "bayesian",
            "coefficient_gate_enabled": True,
        }
    }
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    endpoint = module.validate_fixed_endpoint(
        "B4-G",
        checkpoint_path,
        evaluation_path,
    )
    assert endpoint["checkpoint_epoch"] == 5

    evaluation["protocol"]["coefficient_state_updates_per_trajectory"] = 3
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="must be 4"):
        module.validate_fixed_endpoint(
            "B4-G",
            checkpoint_path,
            evaluation_path,
        )
