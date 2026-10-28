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
from sgbformer.models import CategoryRoutedReadout, build_model
from sgbformer.protocol import implementation_fingerprint, sha256_file
from sgbformer.utils import parameter_count, save_image, seed_everything
from sgbformer.v27 import (
    V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS,
    audit_v27_c0prime_configs,
    resolved_config_sha256,
    validate_v27_c0prime_config,
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
    spec = spec_from_file_location("_v27_c0prime_ssr", architecture)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "net_g_best.pth"
    torch.save({"params": module.SSR(dim=48, bias=False).state_dict()}, checkpoint)
    return repo, checkpoint


def load_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict]:
    repo, checkpoint = write_ssr(tmp_path)
    monkeypatch.setenv("SSR_REPO", str(repo))
    monkeypatch.setenv("SSR_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SGB_DATA_ROOT", str(tmp_path / "data"))
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    return (
        load_config(root / "ssr_residual_v27_c0_proxy.yaml"),
        load_config(root / "ssr_residual_v27_c0prime_routed.yaml"),
    )


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = spec_from_file_location(f"_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c0prime_changes_only_the_routed_readout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c0, c0prime = load_configs(tmp_path, monkeypatch)
    assert validate_v27_c0prime_config(c0prime, c0) == "C0-prime"
    audit = audit_v27_c0prime_configs(c0, c0prime)

    assert audit["behavioral_differences"] == ["model.bfn.routed_readout"]
    assert c0prime["training"]["sampling"] == c0["training"]["sampling"] == {
        "mode": "shuffle"
    }
    for key in (
        "epochs",
        "stages",
        "learning_rate",
        "minimum_learning_rate",
        "minimum_learning_rate_ratio",
        "schedule_minimum_mode",
        "warmup_epochs",
        "batch_size",
        "micro_batch_size",
    ):
        assert c0prime["training"][key] == c0["training"][key]
    assert c0prime["loss"] == c0["loss"]
    assert c0prime["data"] == c0["data"]
    assert c0prime["seed"] == c0["seed"] == 42

    c0prime["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="outside protocol metadata"):
        validate_v27_c0prime_config(c0prime, c0)


def test_c0prime_parameter_freeze_and_identity_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config = load_configs(tmp_path, monkeypatch)
    seed_everything(42)
    model = build_model(config).eval()
    assert parameter_count(model) == V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS
    assert parameter_count(model) < 2 * 614_455
    assert all(
        not parameter.requires_grad for parameter in model.backbone.parameters()
    )
    assert all(
        name.startswith("receiver.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert model.receiver is not None
    readout = model.receiver.output
    assert isinstance(readout, CategoryRoutedReadout)
    assert tuple(readout.heads) == ("snow", "outdoor_rain", "raindrop")
    assert torch.count_nonzero(readout.router.weight) == 0
    for head in readout.heads.values():
        assert torch.count_nonzero(head.output.weight) == 0
        assert torch.count_nonzero(head.output.bias) == 0

    images = torch.rand(8, 3, 17, 19)
    baseline = model.restore_structural(images)
    details = model.restore(images, deterministic=True, return_details=True)
    torch.testing.assert_close(
        details["restored"],
        baseline,
        atol=1e-6,
        rtol=0,
    )
    assert torch.count_nonzero(details["residual"]) == 0
    assert readout.last_router_probabilities is not None
    torch.testing.assert_close(
        readout.last_router_probabilities,
        torch.full((8, 3), 1.0 / 3.0),
        atol=1e-7,
        rtol=0,
    )

    zero_details = model.restore(
        torch.zeros(1, 3, 17, 19),
        deterministic=True,
        return_details=True,
    )
    assert torch.count_nonzero(zero_details["residual"]) == 0


def test_c0prime_breaks_head_symmetry_and_opens_router_gradient() -> None:
    readout = CategoryRoutedReadout(
        16,
        3,
        ("snow", "outdoor_rain", "raindrop"),
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
    assert all(torch.count_nonzero(gradient) for gradient in first_gradients)
    assert all(
        not torch.equal(left, right)
        for index, left in enumerate(first_gradients)
        for right in first_gradients[index + 1 :]
    )
    assert readout.router.weight.grad is not None
    assert torch.count_nonzero(readout.router.weight.grad) == 0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(readout(feature), target).backward()
    assert readout.router.weight.grad is not None
    assert torch.count_nonzero(readout.router.weight.grad) > 0


def test_c0prime_exposes_detached_router_entropy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config = load_configs(tmp_path, monkeypatch)
    model = build_model(config).train()
    image = torch.rand(1, 3, 16, 16)
    outputs = model(image, torch.rand_like(image), training_stage="residual")
    entropy = outputs["adapter_regularization"]

    assert entropy.requires_grad is False
    assert math.isfinite(float(entropy))
    assert float(entropy) == pytest.approx(math.log(3.0), abs=1e-6)


def test_c0prime_preflight_helpers_cover_eight_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("preflight_v27_c0prime.py")
    _, config = load_configs(tmp_path, monkeypatch)
    model = build_model(config)
    samples = [
        {
            "degraded": torch.rand(3, 16 + index % 2, 18 + index % 3),
            "target": torch.rand(3, 16 + index % 2, 18 + index % 3),
            "input_path": f"dev-{index}",
        }
        for index in range(8)
    ]

    structure = module.assert_c0prime_structure(model)
    identity = module.check_c0prime_identity(
        model,
        samples,
        torch.device("cpu"),
    )
    gradients = module.check_c0prime_gradient_path(
        model,
        config,
        samples[0],
        torch.device("cpu"),
    )

    assert structure["trainable_parameters"] == 615_440
    assert identity["maximum_absolute_difference"] <= 1e-6
    assert identity["zero_input_correction_nonzero"] == 0
    assert len(identity["random_dev_images"]) == 8
    assert gradients["initial_loss_relative_difference"] <= 0.01
    assert gradients["first_head_gradients_pairwise_distinct"] is True
    assert gradients["second_router_gradient_norm"] > 0
    assert gradients["foundation_state_unchanged"] is True


def test_c0prime_decision_requires_every_preregistered_gate() -> None:
    module = load_script("analyze_v27_c0prime.py")

    def report(
        snow: tuple[float, list[float]],
        outdoor: tuple[float, list[float]],
        raindrop: tuple[float, list[float]],
        macro: tuple[float, list[float]] = (0.01, [-0.001, 0.02]),
    ) -> dict:
        categories = {
            "snow": snow,
            "outdoor_rain": outdoor,
            "raindrop": raindrop,
        }
        return {
            "macro_weighted": {
                "psnr": {
                    "delta": macro[0],
                    "paired_image_bootstrap_ci95": macro[1],
                }
            },
            "categories": {
                name: {
                    "n": 55 if name == "raindrop" else 400,
                    "psnr": {
                        "delta": values[0],
                        "paired_image_bootstrap_ci95": values[1],
                        "improved": 30,
                    },
                }
                for name, values in categories.items()
            },
        }

    vs_f0 = report(
        (-0.004, [-0.01, 0.001]),
        (-0.002, [-0.01, 0.001]),
        (-0.004, [-0.02, 0.001]),
    )
    vs_c0 = report(
        (0.0, [-0.01, 0.01]),
        (0.001, [-0.01, 0.01]),
        (0.001, [-0.01, 0.01]),
    )
    assert module.decision(vs_f0, vs_c0)["status"] == "PASS"

    vs_f0["macro_weighted"]["psnr"][
        "paired_image_bootstrap_ci95"
    ][0] = -0.0021
    result = module.decision(vs_f0, vs_c0)
    assert result["status"] == "STOP"
    assert "macro_ci_lower_at_least_neg_0p002" in result["failed_checks"]


def test_c0prime_profile_accepts_only_matching_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("profile_v27_residual_bfn.py")
    _, config = load_configs(tmp_path, monkeypatch)
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0prime_routed.yaml"
    )
    preflight_path = tmp_path / "preflight_c0prime.json"
    report = {
        "protocol": "residual_bfn_v27_exploratory",
        "status": "passed",
        "arm": "C0-prime",
        "configs": {
            "C0-prime": {
                "source_sha256": sha256_file(config_path),
                "resolved_sha256": resolved_config_sha256(config),
            }
        },
        "implementation": implementation_fingerprint(),
    }
    preflight_path.write_text(json.dumps(report), encoding="utf-8")
    loaded = module.load_verified_preflight(
        preflight_path,
        arm="C0-prime",
        config_path=config_path,
        config=config,
    )
    assert loaded["arm"] == "C0-prime"

    report["configs"]["C0-prime"]["resolved_sha256"] = "changed"
    preflight_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="config fingerprint changed"):
        module.load_verified_preflight(
            preflight_path,
            arm="C0-prime",
            config_path=config_path,
            config=config,
        )


def test_c0prime_probe_hooks_reproduce_the_soft_mixture() -> None:
    readout = CategoryRoutedReadout(
        16,
        3,
        ("snow", "outdoor_rain", "raindrop"),
        symmetry_break_epsilon=1e-3,
    )
    with torch.no_grad():
        for index, head in enumerate(readout.heads.values(), start=1):
            head.output.weight.normal_(mean=0.0, std=0.01 * index)
    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach()

        return hook

    handles = [readout.router.register_forward_hook(capture("router"))]
    handles.extend(
        head.register_forward_hook(capture(name))
        for name, head in readout.heads.items()
    )
    feature = torch.rand(2, 16, 8, 8)
    try:
        mixed = readout(feature)
    finally:
        for handle in handles:
            handle.remove()
    probabilities = F.softmax(captured["router"].float(), dim=1)
    reconstructed = sum(
        probabilities[:, index, None, None, None] * captured[name]
        for index, name in enumerate(readout.heads)
    )
    torch.testing.assert_close(mixed, reconstructed, atol=1e-7, rtol=0)


def test_c0prime_probe_router_summary_detects_uniform_routing() -> None:
    module = load_script("probe_v27_c0prime_routing.py")
    records = []
    for category in ("snow", "outdoor_rain", "raindrop"):
        records.append(
            {
                "true_category": category,
                "router_probabilities_list": [1 / 3, 1 / 3, 1 / 3],
                "router_entropy": math.log(3),
                "top1_expert": "snow",
            }
        )
    summary = module.summarize_router(
        records,
        ("snow", "outdoor_rain", "raindrop"),
    )

    assert summary["near_uniform"] is True
    assert summary["entropy_deficit_from_ln3"] == pytest.approx(0, abs=1e-12)
    assert summary["classification_accuracy"] == pytest.approx(1 / 3)
    assert summary["balanced_accuracy"] == pytest.approx(1 / 3)
    assert summary["top1_expert_counts"] == {
        "snow": 3,
        "outdoor_rain": 0,
        "raindrop": 0,
    }


def test_c0prime_probe_assigns_one_unique_failure_mode() -> None:
    module = load_script("probe_v27_c0prime_routing.py")

    def diagnostics(cosine, difference, expert_rms=0.001):
        return {
            "overall": {
                "expert_pairs": {
                    "snow__outdoor_rain": {
                        "correction_cosine": cosine,
                        "difference_rms": difference,
                    },
                    "snow__raindrop": {
                        "correction_cosine": cosine,
                        "difference_rms": difference,
                    },
                    "outdoor_rain__raindrop": {
                        "correction_cosine": cosine,
                        "difference_rms": difference,
                    },
                },
                "experts": {
                    category: {"correction_rms": expert_rms}
                    for category in ("snow", "outdoor_rain", "raindrop")
                },
            }
        }

    def report(macro, category_values=(0.0, 0.0, 0.0)):
        return {
            "macro_weighted": {"psnr": {"delta": macro}},
            "categories": {
                category: {"psnr": {"delta": value}}
                for category, value in zip(
                    ("snow", "outdoor_rain", "raindrop"),
                    category_values,
                )
            },
        }

    collapsed_forced = {
        category: report(0.001)
        for category in ("snow", "outdoor_rain", "raindrop")
    }
    result_b = module.determine_attribution(
        {"near_uniform": True},
        diagnostics(0.9999, 1e-8),
        collapsed_forced,
        report(0.0),
        report(0.0),
    )
    assert result_b["code"] == "B"

    specialized_forced = {
        "snow": report(0.0, (0.010, -0.002, -0.002)),
        "outdoor_rain": report(0.002, (0.0, 0.010, 0.0)),
        "raindrop": report(0.001, (0.0, 0.0, 0.010)),
    }
    result_a = module.determine_attribution(
        {"near_uniform": True},
        diagnostics(0.5, 0.001),
        specialized_forced,
        report(0.004),
        report(0.006),
    )
    assert result_a["code"] == "A"

    result_c = module.determine_attribution(
        {"near_uniform": False},
        diagnostics(0.5, 0.001),
        specialized_forced,
        report(0.004),
        report(0.006),
    )
    assert result_c["code"] == "C"

    no_headroom_forced = {
        "snow": report(0.0001, (0.0001, 0.0, 0.0)),
        "outdoor_rain": report(0.0002, (0.0, 0.0002, 0.0)),
        "raindrop": report(0.0003, (0.0, 0.0, 0.0003)),
    }
    result_d = module.determine_attribution(
        {"near_uniform": False},
        diagnostics(0.5, 0.001),
        no_headroom_forced,
        report(0.0002),
        report(0.0004),
    )
    assert result_d["code"] == "D"

    harmful_but_different = {
        "snow": report(-0.25, (-0.10, -0.30, -0.35)),
        "outdoor_rain": report(-0.04, (-0.20, -0.02, -0.15)),
        "raindrop": report(-0.21, (-0.25, -0.18, -0.03)),
    }
    result_harmful = module.determine_attribution(
        {"near_uniform": False},
        diagnostics(0.5, 0.001),
        harmful_but_different,
        report(-0.10),
        report(-0.03),
    )
    assert max(
        result_harmful["forced_category_psnr_spreads_db"].values()
    ) > 0.005
    assert result_harmful["category_spread_is_headroom"] is False
    assert result_harmful["experts_offer_headroom"] is False
    assert result_harmful["code"] == "D"


def test_c0prime_probe_runs_read_only_on_fixed_epoch_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script("probe_v27_c0prime_routing.py")
    _, config = load_configs(tmp_path, monkeypatch)
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
                    "experiment_arm": "C0-prime",
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
    assert summary["checkpoint"]["epoch"] == 3
    assert summary["dataset"]["images"] == 3
    assert summary["router"]["near_uniform"] is True
    assert summary["attribution"]["code"] == "B"
    assert summary["integrity"]["checkpoint_sha256_unchanged"] is True
    assert summary["integrity"]["model_state_unchanged"] is True
    assert summary["integrity"]["maximum_soft_reconstruction_error"] <= 1e-6
    assert sha256_file(checkpoint) == checkpoint_before
    assert len(
        (output_dir / "router_per_image.jsonl").read_text().splitlines()
    ) == 3
    assert report_path.is_file()
