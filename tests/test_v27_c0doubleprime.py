from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from sgbformer.checkpoint import save_checkpoint
from sgbformer.comparison import foundation_signature
from sgbformer.config import load_config
from sgbformer.data import PairedImageDataset
from sgbformer.ema import ExponentialMovingAverage
from sgbformer.metrics import psnr, ssim
from sgbformer.models import (
    CategoryRoutedReadout,
    PrivateBottleneckRoutedResidualHead,
    RoutedResidualHead,
    build_model,
)
from sgbformer.protocol import implementation_fingerprint, sha256_file
from sgbformer.utils import parameter_count, save_image, seed_everything
from sgbformer.v27 import (
    V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS,
    V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS,
    audit_v27_c0accept_configs,
    audit_v27_c0doubleprime_configs,
    resolved_config_sha256,
    validate_v27_c0accept_config,
    validate_v27_c0doubleprime_config,
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
    spec = spec_from_file_location("_v27_c0doubleprime_ssr", architecture)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "net_g_best.pth"
    torch.save({"params": module.SSR(dim=48, bias=False).state_dict()}, checkpoint)
    return repo, checkpoint


def load_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict, dict]:
    repo, checkpoint = write_ssr(tmp_path)
    monkeypatch.setenv("SSR_REPO", str(repo))
    monkeypatch.setenv("SSR_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SGB_DATA_ROOT", str(tmp_path / "data"))
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    return (
        load_config(root / "ssr_residual_v27_c0_proxy.yaml"),
        load_config(root / "ssr_residual_v27_c0prime_routed.yaml"),
        load_config(
            root / "ssr_residual_v27_c0doubleprime_bottleneck_routed.yaml"
        ),
    )


def load_accept_config() -> dict:
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    return load_config(root / "ssr_residual_v27_c0accept_gate.yaml")


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = spec_from_file_location(f"_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pairwise_relative_distances(
    tensors: list[torch.Tensor],
) -> list[float]:
    distances = []
    for index, left in enumerate(tensors):
        for right in tensors[index + 1 :]:
            denominator = max(
                0.5 * float(left.norm() + right.norm()),
                1e-12,
            )
            distances.append(float((left - right).norm()) / denominator)
    return distances


def test_c0doubleprime_changes_only_private_head_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c0, c0prime, candidate = load_configs(tmp_path, monkeypatch)
    assert (
        validate_v27_c0doubleprime_config(candidate, c0, c0prime)
        == "C0-double-prime"
    )
    audit = audit_v27_c0doubleprime_configs(c0, c0prime, candidate)
    assert audit["behavioral_differences"] == [
        "model.bfn.routed_readout.head_type"
    ]
    assert candidate["training"]["sampling"] == {"mode": "shuffle"}
    assert candidate["loss"] == c0prime["loss"]
    assert candidate["data"] == c0prime["data"]
    assert candidate["seed"] == c0prime["seed"] == 42

    candidate["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="outside protocol metadata"):
        validate_v27_c0doubleprime_config(candidate, c0, c0prime)


def test_c0accept_changes_only_the_acceptance_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c0, c0prime, c0doubleprime = load_configs(tmp_path, monkeypatch)
    candidate = load_accept_config()
    assert (
        validate_v27_c0accept_config(
            candidate,
            c0,
            c0prime,
            c0doubleprime,
        )
        == "C0-accept"
    )
    audit = audit_v27_c0accept_configs(
        c0,
        c0prime,
        c0doubleprime,
        candidate,
    )
    assert audit["behavioral_differences"] == [
        "model.bfn.routed_readout.acceptance_gate"
    ]
    assert candidate["loss"] == c0doubleprime["loss"]
    assert candidate["data"] == c0doubleprime["data"]
    assert candidate["training"]["sampling"] == {"mode": "shuffle"}
    assert candidate["seed"] == 42

    candidate["loss"]["final_weight"] = 0.5
    with pytest.raises(ValueError, match="outside protocol metadata"):
        validate_v27_c0accept_config(
            candidate,
            c0,
            c0prime,
            c0doubleprime,
        )


def test_c0accept_gate_is_identity_safe_and_only_gate_trains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, c0doubleprime = load_configs(tmp_path, monkeypatch)
    candidate_config = load_accept_config()
    source = build_model(c0doubleprime).eval()
    with torch.no_grad():
        for index, head in enumerate(
            source.receiver.output.heads.values(),
            start=1,
        ):
            head.output.weight.normal_(mean=0.0, std=0.01 * index)
    candidate = build_model(candidate_config).eval()
    preflight = load_script("preflight_v27_c0accept.py")
    transplant = preflight.transplant_c0doubleprime_ema(candidate, source)

    assert transplant["source_and_candidate_shared_state_equal"] is True
    assert (
        parameter_count(candidate)
        == V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS
        == 17
    )
    assert {
        name
        for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    } == preflight.GATE_PARAMETER_NAMES
    image = torch.rand(2, 3, 16, 16)
    target = torch.rand_like(image)
    baseline = candidate.restore_structural(image)
    details = candidate.restore(
        image,
        deterministic=True,
        return_details=True,
    )
    torch.testing.assert_close(details["restored"], baseline, atol=0, rtol=0)
    readout = candidate.receiver.output
    assert torch.count_nonzero(readout.last_acceptance) == 0
    assert torch.count_nonzero(readout.last_ungated_mixture) > 0

    candidate.train()
    outputs = candidate(image, target, training_stage="residual")
    F.mse_loss(outputs["restored"], target).backward()
    assert readout.acceptance_gate.weight.grad is not None
    assert float(readout.acceptance_gate.weight.grad.norm()) > 1e-12
    assert outputs["adapter_regularization"].requires_grad is False
    assert float(outputs["adapter_regularization"]) == 0.0
    assert all(
        parameter.grad is None
        for name, parameter in candidate.named_parameters()
        if name not in preflight.GATE_PARAMETER_NAMES
    )


def test_c0doubleprime_parameter_identity_and_c0prime_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, c0prime, candidate = load_configs(tmp_path, monkeypatch)
    seed_everything(42)
    old_model = build_model(c0prime).eval()
    seed_everything(42)
    model = build_model(candidate).eval()

    assert parameter_count(old_model) == 615_440
    assert (
        parameter_count(model)
        == V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS
    )
    assert parameter_count(model) < 2 * 614_455
    assert model.receiver is not None
    assert old_model.receiver is not None
    old_readout = old_model.receiver.output
    readout = model.receiver.output
    assert isinstance(old_readout, CategoryRoutedReadout)
    assert isinstance(readout, CategoryRoutedReadout)
    assert all(
        isinstance(head, RoutedResidualHead)
        for head in old_readout.heads.values()
    )
    assert not any(
        ".bottleneck." in name for name in old_readout.state_dict()
    )
    assert all(
        isinstance(head, PrivateBottleneckRoutedResidualHead)
        for head in readout.heads.values()
    )
    bottlenecks = [
        head.bottleneck.weight.detach()
        for head in readout.heads.values()
    ]
    assert min(pairwise_relative_distances(bottlenecks)) > 1.0
    assert torch.count_nonzero(readout.router.weight) == 0
    for head in readout.heads.values():
        assert torch.count_nonzero(head.output.weight) == 0
        assert torch.count_nonzero(head.output.bias) == 0

    images = torch.rand(4, 3, 17, 19)
    baseline = model.restore_structural(images)
    details = model.restore(images, deterministic=True, return_details=True)
    torch.testing.assert_close(details["restored"], baseline, atol=1e-6, rtol=0)
    assert torch.count_nonzero(details["residual"]) == 0
    assert readout.last_router_probabilities is not None
    torch.testing.assert_close(
        readout.last_router_probabilities,
        torch.full((4, 3), 1.0 / 3.0),
        atol=1e-7,
        rtol=0,
    )


def test_c0doubleprime_materially_separates_first_step_head_gradients() -> None:
    torch.manual_seed(42)
    readout = CategoryRoutedReadout(
        16,
        3,
        ("snow", "outdoor_rain", "raindrop"),
        head_type="groupnorm_conv3x3_silu_groupnorm_silu_conv3x3",
        symmetry_break_epsilon=1e-3,
    )
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)
    feature = torch.rand(2, 16, 8, 8)
    target = torch.rand(2, 3, 8, 8)

    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(readout(feature), target).backward()
    first_gradients = [
        head.output.weight.grad.detach().clone()
        for head in readout.heads.values()
    ]
    assert min(pairwise_relative_distances(first_gradients)) > 0.01
    assert readout.router.weight.grad is not None
    assert torch.count_nonzero(readout.router.weight.grad) == 0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(readout(feature), target).backward()
    assert readout.router.weight.grad is not None
    assert float(readout.router.weight.grad.norm()) > 1e-12
    assert readout.last_router_entropy is not None
    assert float(readout.last_router_entropy) == pytest.approx(
        math.log(3.0),
        abs=1e-3,
    )


def test_c0doubleprime_preflight_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("preflight_v27_c0doubleprime.py")
    _, _, config = load_configs(tmp_path, monkeypatch)
    seed_everything(42)
    model = build_model(config)
    samples = [
        {
            "degraded": torch.rand(3, 16 + index % 2, 18 + index % 3),
            "target": torch.rand(3, 16 + index % 2, 18 + index % 3),
            "input_path": f"dev-{index}",
        }
        for index in range(8)
    ]

    structure = module.assert_c0doubleprime_structure(model)
    identity = module.check_c0doubleprime_identity(
        model,
        samples,
        torch.device("cpu"),
    )
    gradients = module.check_c0doubleprime_gradient_path(
        model,
        config,
        samples[0],
        torch.device("cpu"),
    )

    assert structure["trainable_parameters"] == 622_496
    assert structure["bottlenecks_pairwise_distinct"] is True
    assert identity["maximum_absolute_difference"] <= 1e-6
    assert identity["zero_input_correction_nonzero"] == 0
    assert len(identity["random_dev_images"]) == 8
    assert gradients["initial_loss_relative_difference"] <= 0.01
    assert (
        gradients["minimum_first_head_gradient_relative_distance"]
        >= 0.01
    )
    assert gradients["first_router_gradient_norm"] == 0
    assert gradients["second_router_gradient_norm"] > 1e-12
    assert gradients["router_entropy_detached"] is True
    assert gradients["foundation_state_unchanged"] is True


def test_c0doubleprime_profile_accepts_only_matching_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("profile_v27_residual_bfn.py")
    _, _, config = load_configs(tmp_path, monkeypatch)
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0doubleprime_bottleneck_routed.yaml"
    )
    preflight_path = tmp_path / "preflight_c0doubleprime.json"
    report = {
        "protocol": "residual_bfn_v27_exploratory",
        "status": "passed",
        "arm": "C0-double-prime",
        "configs": {
            "C0-double-prime": {
                "source_sha256": sha256_file(config_path),
                "resolved_sha256": resolved_config_sha256(config),
            }
        },
        "implementation": implementation_fingerprint(),
    }
    preflight_path.write_text(json.dumps(report), encoding="utf-8")
    loaded = module.load_verified_preflight(
        preflight_path,
        arm="C0-double-prime",
        config_path=config_path,
        config=config,
    )
    assert loaded["arm"] == "C0-double-prime"

    report["configs"]["C0-double-prime"]["resolved_sha256"] = "changed"
    preflight_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="config fingerprint changed"):
        module.load_verified_preflight(
            preflight_path,
            arm="C0-double-prime",
            config_path=config_path,
            config=config,
        )


def test_c0accept_profile_binds_the_initialization_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_script("profile_v27_residual_bfn.py")
    preflight = load_script("preflight_v27_c0accept.py")
    _, _, source_config = load_configs(tmp_path, monkeypatch)
    config = load_accept_config()
    source = build_model(source_config)
    with torch.no_grad():
        for head in source.receiver.output.heads.values():
            head.output.weight.normal_(mean=0.0, std=0.01)
    candidate = build_model(config)
    preflight.transplant_c0doubleprime_ema(candidate, source)
    initialization_path = tmp_path / "c0accept_init.pt"
    initialization = preflight.save_initialization_checkpoint(
        initialization_path,
        candidate,
        config,
    )
    assert initialization["epoch"] == -1
    assert initialization["global_step"] == 0
    assert initialization["optimizer_present"] is True
    assert initialization["scheduler_present"] is True
    assert initialization["ema_present"] is True

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0accept_gate.yaml"
    )
    preflight_path = tmp_path / "preflight_c0accept.json"
    report = {
        "protocol": "residual_bfn_v27_exploratory",
        "status": "passed",
        "arm": "C0-accept",
        "configs": {
            "C0-accept": {
                "source_sha256": sha256_file(config_path),
                "resolved_sha256": resolved_config_sha256(config),
            }
        },
        "implementation": implementation_fingerprint(),
        "initialization_checkpoint": initialization,
    }
    preflight_path.write_text(json.dumps(report), encoding="utf-8")
    loaded = profile.load_verified_preflight(
        preflight_path,
        arm="C0-accept",
        config_path=config_path,
        config=config,
        initialization_checkpoint=initialization_path,
    )
    assert loaded["arm"] == "C0-accept"

    initialization_path.write_bytes(
        initialization_path.read_bytes() + b"changed"
    )
    with pytest.raises(
        RuntimeError,
        match="initialization checkpoint changed",
    ):
        profile.load_verified_preflight(
            preflight_path,
            arm="C0-accept",
            config_path=config_path,
            config=config,
            initialization_checkpoint=initialization_path,
        )


def test_c0doubleprime_analyzer_rejects_the_wrong_arm(
    tmp_path: Path,
) -> None:
    module = load_script("analyze_v27_c0doubleprime.py")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "protocol": {
                    "experiment_protocol": "residual_bfn_v27_exploratory",
                    "experiment_arm": "C0-prime",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="C0-double-prime"):
        module.validate_candidate(
            evaluation,
            protocol="residual_bfn_v27_exploratory",
            arm="C0-double-prime",
        )


def test_c0accept_analyzer_uses_fixed_source_and_candidate_arms(
    tmp_path: Path,
) -> None:
    module = load_script("analyze_v27_c0accept.py")
    source = tmp_path / "source.json"
    candidate = tmp_path / "candidate.json"
    source.write_text(
        json.dumps(
            {
                "protocol": {
                    "experiment_protocol": "residual_bfn_v27_exploratory",
                    "experiment_arm": "C0-double-prime",
                }
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "protocol": {
                    "experiment_protocol": "residual_bfn_v27_exploratory",
                    "experiment_arm": "C0-accept",
                }
            }
        ),
        encoding="utf-8",
    )

    assert module.validate_candidate(
        source,
        protocol="residual_bfn_v27_exploratory",
        arm="C0-double-prime",
    )["protocol"]["experiment_arm"] == "C0-double-prime"
    assert module.validate_candidate(
        candidate,
        protocol="residual_bfn_v27_exploratory",
        arm="C0-accept",
    )["protocol"]["experiment_arm"] == "C0-accept"
    with pytest.raises(ValueError, match="C0-accept"):
        module.validate_candidate(
            source,
            protocol="residual_bfn_v27_exploratory",
            arm="C0-accept",
        )


def test_c0doubleprime_probe_runs_read_only_on_fixed_epoch_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("probe_v27_c0prime_routing.py")
    _, _, config = load_configs(tmp_path, monkeypatch)
    input_dir = tmp_path / "data" / "input"
    target_dir = tmp_path / "data" / "gt"
    input_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    filenames = (
        "snow.jpg",
        "im_0001_s80_a04.png",
        "123_rain.png",
    )
    pairs_file = tmp_path / "data" / "val_pairs.tsv"
    pairs_file.write_text(
        "".join(f"{name}\t{name}\n" for name in filenames),
        encoding="utf-8",
    )
    (tmp_path / "data" / "train_pairs.tsv").write_text(
        pairs_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    generator = torch.Generator().manual_seed(42)
    for filename in filenames:
        image = torch.rand(3, 16, 16, generator=generator)
        save_image(image, input_dir / filename)
        save_image(image, target_dir / filename)

    model = build_model(config).eval()
    ema = ExponentialMovingAverage(model, decay=0.999, warmup_updates=2000)
    checkpoint = tmp_path / "epoch_0003.pt"
    save_checkpoint(
        checkpoint,
        model,
        config=config,
        epoch=2,
        global_step=3,
        ema=ema,
    )
    dataset = PairedImageDataset(
        input_dir,
        target_dir,
        pairs_file=pairs_file,
    )
    protocol = {
        "test_y_channel": True,
        "crop_border": 0,
        "lpips_network": None,
        "tile": None,
        "stochastic": False,
        "output_quantization": "float32_clamp_0_1",
        "foundation_signature": foundation_signature(config),
    }
    baseline_rows = []
    registered_rows = []
    with torch.inference_mode():
        for sample in dataset:
            degraded = sample["degraded"].unsqueeze(0)
            target = sample["target"].unsqueeze(0)
            structural = model.restore_structural(degraded)
            details = model.restore(
                degraded,
                deterministic=True,
                return_details=True,
            )

            def row(prediction):
                return {
                    "name": sample["name"],
                    "psnr": float(
                        psnr(
                            prediction,
                            target,
                            test_y_channel=True,
                        )[0]
                    ),
                    "ssim": float(
                        ssim(
                            prediction,
                            target,
                            test_y_channel=True,
                        )[0]
                    ),
                }

            baseline_rows.append(row(structural))
            registered_rows.append(row(details["restored"]))
    baseline_path = tmp_path / "foundation.json"
    registered_path = tmp_path / "registered.json"
    baseline_path.write_text(
        json.dumps({"images": baseline_rows, "protocol": protocol}),
        encoding="utf-8",
    )
    registered_path.write_text(
        json.dumps(
            {
                "images": registered_rows,
                "protocol": {
                    **protocol,
                    "experiment_protocol": "residual_bfn_v27_exploratory",
                    "experiment_arm": "C0-double-prime",
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"
    report_path = tmp_path / "route_probe.md"
    monkeypatch.setattr(module, "V27_EXPECTED_DEV_PAIRS", 3)
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe_v27_c0prime_routing.py",
            "--checkpoint",
            str(checkpoint),
            "--baseline",
            str(baseline_path),
            "--registered-evaluation",
            str(registered_path),
            "--input-dir",
            str(input_dir),
            "--target-dir",
            str(target_dir),
            "--pairs-file",
            str(pairs_file),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
            "--device",
            "cpu",
            "--bootstrap-samples",
            "10000",
            "--seed",
            "42",
        ],
    )
    checkpoint_before = sha256_file(checkpoint)
    module.main()
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "completed"
    assert summary["arm"] == "C0-double-prime"
    assert summary["protocol"] == "v27_c0doubleprime_epoch3_route_probe"
    assert summary["checkpoint"]["epoch"] == 3
    assert summary["dataset"]["images"] == 3
    assert (
        summary["implementation"]["candidate_trainable_parameters"]
        == 622_496
    )
    assert summary["parameter_diagnostics"]["head_parameters"] == {
        "snow": 2_819,
        "outdoor_rain": 2_819,
        "raindrop": 2_819,
    }
    assert summary["integrity"]["checkpoint_sha256_unchanged"] is True
    assert summary["integrity"]["model_state_unchanged"] is True
    assert summary["integrity"]["maximum_soft_reconstruction_error"] <= 1e-6
    assert sha256_file(checkpoint) == checkpoint_before
    assert len(
        (output_dir / "router_per_image.jsonl").read_text().splitlines()
    ) == 3
    assert "# v27 C0-double-prime" in report_path.read_text()


def test_c0doubleprime_residual_geometry_audit_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("audit_v27_residual_geometry.py")
    _, _, config = load_configs(tmp_path, monkeypatch)
    input_dir = tmp_path / "data" / "input"
    target_dir = tmp_path / "data" / "gt"
    input_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    filenames = (
        "snow.jpg",
        "im_0001_s80_a04.png",
        "123_rain.png",
    )
    pairs_file = tmp_path / "data" / "val_pairs.tsv"
    pairs_file.write_text(
        "".join(f"{name}\t{name}\n" for name in filenames),
        encoding="utf-8",
    )
    (tmp_path / "data" / "train_pairs.tsv").write_text(
        pairs_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    generator = torch.Generator().manual_seed(42)
    for filename in filenames:
        degraded = torch.rand(3, 16, 16, generator=generator)
        target = torch.rand(3, 16, 16, generator=generator)
        save_image(degraded, input_dir / filename)
        save_image(target, target_dir / filename)

    model = build_model(config).eval()
    ema = ExponentialMovingAverage(model, decay=0.999, warmup_updates=2000)
    checkpoint = tmp_path / "epoch_0003.pt"
    save_checkpoint(
        checkpoint,
        model,
        config=config,
        epoch=2,
        global_step=3,
        ema=ema,
    )
    dataset = PairedImageDataset(
        input_dir,
        target_dir,
        pairs_file=pairs_file,
    )
    protocol = {
        "test_y_channel": True,
        "crop_border": 0,
        "lpips_network": None,
        "tile": None,
        "stochastic": False,
        "output_quantization": "float32_clamp_0_1",
        "foundation_signature": foundation_signature(config),
        "experiment_protocol": "residual_bfn_v27_exploratory",
        "experiment_arm": "C0-double-prime",
        "resolved_config_sha256": resolved_config_sha256(config),
    }
    registered_rows = []
    with torch.inference_mode():
        for sample in dataset:
            degraded = sample["degraded"].unsqueeze(0)
            target = sample["target"].unsqueeze(0)
            prediction = model.restore(
                degraded,
                deterministic=True,
            )
            registered_rows.append(
                {
                    "name": sample["name"],
                    "psnr": float(
                        psnr(
                            prediction,
                            target,
                            test_y_channel=True,
                        )[0]
                    ),
                    "ssim": float(
                        ssim(
                            prediction,
                            target,
                            test_y_channel=True,
                        )[0]
                    ),
                }
            )
    registered_path = tmp_path / "registered.json"
    registered_path.write_text(
        json.dumps(
            {
                "images": registered_rows,
                "protocol": protocol,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "geometry"
    report_path = tmp_path / "geometry.md"
    monkeypatch.setattr(module, "V27_EXPECTED_DEV_PAIRS", 3)
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_v27_residual_geometry.py",
            "--checkpoint",
            str(checkpoint),
            "--registered-evaluation",
            str(registered_path),
            "--input-dir",
            str(input_dir),
            "--target-dir",
            str(target_dir),
            "--pairs-file",
            str(pairs_file),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
            "--device",
            "cpu",
            "--bootstrap-samples",
            "10000",
            "--seed",
            "42",
            "--mixture-chunk-size",
            "21",
        ],
    )
    checkpoint_before = sha256_file(checkpoint)
    module.main()
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "completed"
    assert summary["checkpoint"]["epoch"] == 3
    assert summary["dataset"]["images"] == 3
    assert summary["geometry"]["identity_passed"] is True
    assert summary["endpoint_reproduction"]["alpha_zero_passed"] is True
    assert summary["endpoint_reproduction"]["alpha_one_passed"] is True
    assert summary["continuous_mixture"]["grid_size"] == 231
    assert summary["integrity"]["receiver_forward_passes_per_image"] == 1
    assert summary["integrity"]["checkpoint_sha256_unchanged"] is True
    assert summary["integrity"]["model_state_unchanged"] is True
    assert summary["integrity"]["model_parameter_gradients_absent"] is True
    assert sha256_file(checkpoint) == checkpoint_before
    assert len(
        (output_dir / "per_image.jsonl").read_text().splitlines()
    ) == 3
    assert report_path.is_file()


def test_c0accept_gate_audit_statistics_and_diagnosis() -> None:
    module = load_script("audit_v27_c0accept_gate.py")
    records = [
        {
            "category": "snow",
            "acceptance": 0.99,
            "source_rgb_mse_improvement": 0.2,
            "source_psnr_gain_vs_f0": 0.2,
            "candidate_psnr_gain_vs_f0": 0.2,
            "candidate_psnr_gain_vs_source": 0.0,
        },
        {
            "category": "outdoor_rain",
            "acceptance": 0.98,
            "source_rgb_mse_improvement": -0.1,
            "source_psnr_gain_vs_f0": -0.1,
            "candidate_psnr_gain_vs_f0": -0.1,
            "candidate_psnr_gain_vs_source": 0.0,
        },
        {
            "category": "raindrop",
            "acceptance": 1.0,
            "source_rgb_mse_improvement": -0.2,
            "source_psnr_gain_vs_f0": -0.2,
            "candidate_psnr_gain_vs_f0": -0.2,
            "candidate_psnr_gain_vs_source": 0.0,
        },
    ]
    summary = module.summarize_records(records)
    diagnosis = module.diagnose_gate(summary, formal_status="STOP")

    assert module.binary_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert summary["overall"]["acceptance"]["near_one_fraction"] == 1.0
    assert diagnosis["code"] == "G1"
    assert diagnosis["flags"]["formal_decision_is_stop"] is True


def test_c0accept_gate_audit_runs_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("audit_v27_c0accept_gate.py")
    _, _, source_config = load_configs(tmp_path, monkeypatch)
    candidate_config = load_accept_config()
    input_dir = tmp_path / "data" / "input"
    target_dir = tmp_path / "data" / "gt"
    input_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    filenames = (
        "snow.jpg",
        "im_0001_s80_a04.png",
        "123_rain.png",
    )
    pairs_file = tmp_path / "data" / "val_pairs.tsv"
    pairs_file.write_text(
        "".join(f"{name}\t{name}\n" for name in filenames),
        encoding="utf-8",
    )
    (tmp_path / "data" / "train_pairs.tsv").write_text(
        pairs_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    generator = torch.Generator().manual_seed(42)
    for filename in filenames:
        degraded = torch.rand(3, 16, 16, generator=generator)
        target = torch.rand(3, 16, 16, generator=generator)
        save_image(degraded, input_dir / filename)
        save_image(target, target_dir / filename)

    source = build_model(source_config).eval()
    with torch.no_grad():
        for index, head in enumerate(
            source.receiver.output.heads.values(),
            start=1,
        ):
            head.output.weight.normal_(mean=0.0, std=0.01 * index)
    candidate = build_model(candidate_config).eval()
    preflight = load_script("preflight_v27_c0accept.py")
    preflight.transplant_c0doubleprime_ema(candidate, source)
    with torch.no_grad():
        candidate.receiver.output.acceptance_gate.weight.normal_(
            mean=0.0,
            std=0.01,
        )
        candidate.receiver.output.acceptance_gate.bias.fill_(0.5)
    ema = ExponentialMovingAverage(
        candidate,
        decay=0.999,
        warmup_updates=2000,
    )
    checkpoint = tmp_path / "epoch_0003.pt"
    save_checkpoint(
        checkpoint,
        candidate,
        config=candidate_config,
        epoch=2,
        global_step=3,
        ema=ema,
    )

    dataset = PairedImageDataset(
        input_dir,
        target_dir,
        pairs_file=pairs_file,
    )
    protocol = {
        "test_y_channel": True,
        "crop_border": 0,
        "lpips_network": None,
        "tile": None,
        "stochastic": False,
        "output_quantization": "float32_clamp_0_1",
        "foundation_signature": foundation_signature(candidate_config),
        "experiment_protocol": "residual_bfn_v27_exploratory",
    }
    baseline_rows = []
    source_rows = []
    candidate_rows = []
    with torch.inference_mode():
        for sample in dataset:
            degraded = sample["degraded"].unsqueeze(0)
            target = sample["target"].unsqueeze(0)
            details = candidate.restore(
                degraded,
                deterministic=True,
                return_details=True,
            )
            height, width = target.shape[-2:]
            ungated = (
                candidate.receiver.output.last_ungated_mixture[
                    ..., :height, :width
                ]
                * candidate.residual_scale
            )
            source_prediction = (
                details["structural"] + ungated
            ).clamp(0, 1)

            def row(prediction: torch.Tensor) -> dict:
                return {
                    "name": sample["name"],
                    "psnr": float(
                        psnr(
                            prediction,
                            target,
                            test_y_channel=True,
                        )[0]
                    ),
                    "ssim": float(
                        ssim(
                            prediction,
                            target,
                            test_y_channel=True,
                        )[0]
                    ),
                }

            baseline_rows.append(row(details["structural"]))
            source_rows.append(row(source_prediction))
            candidate_rows.append(row(details["restored"]))

    baseline_path = tmp_path / "baseline.json"
    source_path = tmp_path / "source.json"
    candidate_path = tmp_path / "candidate.json"
    decision_path = tmp_path / "decision.json"
    baseline_path.write_text(
        json.dumps({"images": baseline_rows, "protocol": protocol}),
        encoding="utf-8",
    )
    source_path.write_text(
        json.dumps(
            {
                "images": source_rows,
                "protocol": {
                    **protocol,
                    "experiment_arm": "C0-double-prime",
                },
            }
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "images": candidate_rows,
                "protocol": {
                    **protocol,
                    "experiment_arm": "C0-accept",
                },
            }
        ),
        encoding="utf-8",
    )
    decision_path.write_text(
        json.dumps(
            {
                "protocol": "residual_bfn_v27_exploratory",
                "arm": "C0-accept",
                "primary_endpoint": "fixed_epoch_3",
                "decision": {
                    "status": "STOP",
                    "failed_checks": ["rain_safety"],
                },
                "vs_f0": {
                    "macro_weighted": {
                        "psnr": {"delta": 0.0}
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "gate_audit"
    report_path = tmp_path / "gate_audit.md"
    monkeypatch.setattr(module, "V27_EXPECTED_DEV_PAIRS", 3)
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_v27_c0accept_gate.py",
            "--checkpoint",
            str(checkpoint),
            "--baseline",
            str(baseline_path),
            "--source-evaluation",
            str(source_path),
            "--candidate-evaluation",
            str(candidate_path),
            "--decision-summary",
            str(decision_path),
            "--input-dir",
            str(input_dir),
            "--target-dir",
            str(target_dir),
            "--pairs-file",
            str(pairs_file),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
            "--device",
            "cpu",
            "--seed",
            "42",
        ],
    )
    checkpoint_before = sha256_file(checkpoint)
    module.main()
    summary = json.loads(
        (output_dir / "gate_summary.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "completed"
    assert summary["checkpoint"]["epoch"] == 3
    assert summary["formal_decision"]["status"] == "STOP"
    assert summary["dataset"]["images"] == 3
    assert summary["gate_parameters"]["trainable_parameters"] == 17
    assert summary["integrity"]["checkpoint_sha256_unchanged"] is True
    assert summary["integrity"]["model_state_unchanged"] is True
    assert summary["integrity"]["parameter_gradients_absent"] is True
    assert summary["integrity"]["maximum_gate_reconstruction_error"] <= 1e-6
    assert summary["integrity"]["maximum_source_metric_error"] <= 1e-4
    assert summary["integrity"]["maximum_candidate_metric_error"] <= 1e-4
    assert sha256_file(checkpoint) == checkpoint_before
    assert len(
        (output_dir / "gate_per_image.jsonl").read_text().splitlines()
    ) == 3
    assert report_path.is_file()


def test_signed_coefficient_oracle_runs_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("audit_v27_signed_coefficient_oracle.py")
    _, _, config = load_configs(tmp_path, monkeypatch)
    input_dir = tmp_path / "data" / "input"
    target_dir = tmp_path / "data" / "gt"
    input_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    filenames = (
        "snow.jpg",
        "im_0001_s80_a04.png",
        "123_rain.png",
    )
    pairs_file = tmp_path / "data" / "val_pairs.tsv"
    pairs_file.write_text(
        "".join(f"{name}\t{name}\n" for name in filenames),
        encoding="utf-8",
    )
    (tmp_path / "data" / "train_pairs.tsv").write_text(
        pairs_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    generator = torch.Generator().manual_seed(42)
    for filename in filenames:
        degraded = torch.rand(3, 16, 16, generator=generator)
        target = torch.rand(3, 16, 16, generator=generator)
        save_image(degraded, input_dir / filename)
        save_image(target, target_dir / filename)

    model = build_model(config).eval()
    with torch.no_grad():
        for index, head in enumerate(
            model.receiver.output.heads.values(),
            start=1,
        ):
            head.output.weight.normal_(mean=0.0, std=0.01 * index)
    ema = ExponentialMovingAverage(model, decay=0.999, warmup_updates=2000)
    checkpoint = tmp_path / "epoch_0003.pt"
    save_checkpoint(
        checkpoint,
        model,
        config=config,
        epoch=2,
        global_step=3,
        ema=ema,
    )

    dataset = PairedImageDataset(
        input_dir,
        target_dir,
        pairs_file=pairs_file,
    )
    protocol = {
        "test_y_channel": True,
        "crop_border": 0,
        "lpips_network": None,
        "tile": None,
        "stochastic": False,
        "output_quantization": "float32_clamp_0_1",
        "foundation_signature": foundation_signature(config),
    }
    baseline_rows = []
    registered_rows = []
    with torch.inference_mode():
        for sample in dataset:
            degraded = sample["degraded"].unsqueeze(0)
            target = sample["target"].unsqueeze(0)
            details = model.restore(
                degraded,
                deterministic=True,
                return_details=True,
            )

            def row(prediction: torch.Tensor) -> dict:
                return {
                    "name": sample["name"],
                    "psnr": float(
                        psnr(
                            prediction,
                            target,
                            test_y_channel=True,
                        )[0]
                    ),
                    "ssim": float(
                        ssim(
                            prediction,
                            target,
                            test_y_channel=True,
                        )[0]
                    ),
                }

            baseline_rows.append(row(details["structural"]))
            registered_rows.append(row(details["restored"]))

    baseline_path = tmp_path / "baseline.json"
    registered_path = tmp_path / "registered.json"
    baseline_path.write_text(
        json.dumps({"images": baseline_rows, "protocol": protocol}),
        encoding="utf-8",
    )
    registered_path.write_text(
        json.dumps(
            {
                "images": registered_rows,
                "protocol": {
                    **protocol,
                    "experiment_protocol": "residual_bfn_v27_exploratory",
                    "experiment_arm": "C0-double-prime",
                },
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "signed_oracle"
    report_path = tmp_path / "signed_oracle.md"
    monkeypatch.setattr(module, "V27_EXPECTED_DEV_PAIRS", 3)
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_v27_signed_coefficient_oracle.py",
            "--checkpoint",
            str(checkpoint),
            "--baseline",
            str(baseline_path),
            "--registered-evaluation",
            str(registered_path),
            "--input-dir",
            str(input_dir),
            "--target-dir",
            str(target_dir),
            "--pairs-file",
            str(pairs_file),
            "--output-dir",
            str(output_dir),
            "--report",
            str(report_path),
            "--device",
            "cpu",
            "--bootstrap-samples",
            "10000",
            "--seed",
            "42",
            "--mixture-chunk-size",
            "7",
        ],
    )
    checkpoint_before = sha256_file(checkpoint)
    module.main()
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "completed"
    assert summary["checkpoint"]["epoch"] == 3
    assert summary["dataset"]["images"] == 3
    assert summary["grid"]["size"] == 129
    assert summary["grid"]["contains_zero"] is True
    assert summary["grid"]["contains_negative_coefficients"] is True
    assert summary["integrity"]["checkpoint_sha256_unchanged"] is True
    assert summary["integrity"]["model_state_unchanged"] is True
    assert summary["integrity"]["parameter_gradients_absent"] is True
    assert summary["integrity"]["maximum_baseline_metric_error"] <= 1e-4
    assert summary["integrity"]["maximum_registered_metric_error"] <= 1e-4
    assert sha256_file(checkpoint) == checkpoint_before
    assert len(
        (output_dir / "signed_oracle_per_image.jsonl").read_text().splitlines()
    ) == 3
    assert report_path.is_file()
