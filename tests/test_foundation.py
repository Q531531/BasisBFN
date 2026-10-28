from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import pytest

from sgbformer.models import (
    LatentConditionedFiLMSpectralAdapter,
    SGBFormer,
)
from sgbformer.models.foundation import CheckpointCompatibleEDFFN, ExternalSSRFoundation


DUMMY_SSR = """
import torch
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
        self.encoder = nn.Conv2d(inp_channels, dim, 3, padding=1, bias=bias)
        self.decoder_level3 = nn.Conv2d(dim, dim, 3, padding=1, bias=bias)
        self.decoder_level2 = nn.Conv2d(dim, dim, 3, padding=1, bias=bias)
        self.decoder_level1 = nn.Conv2d(
            dim, dim, 3, padding=1, bias=bias
        )
        self.output = nn.Conv2d(dim, out_channels, 3, padding=1, bias=bias)

    def forward(self, image):
        feature = self.decoder_level3(self.encoder(image))
        feature = self.decoder_level1(self.decoder_level2(feature))
        return image + self.output(feature)
"""

SPLIT_SSR = """
import torch
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
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_dec_level3 = self.decoder_level3(out_enc_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3) + out_enc_level2
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2) + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        return self.output(out_dec_level1) + inp_img
"""


def _write_dummy_ssr(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "SSR"
    architecture = repo / "basicsr" / "models" / "archs" / "SSR_arch.py"
    architecture.parent.mkdir(parents=True)
    architecture.write_text(DUMMY_SSR, encoding="utf-8")

    spec = spec_from_file_location("_dummy_ssr_checkpoint", architecture)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    network = module.SSR(dim=4)
    checkpoint = tmp_path / "net_g_best.pth"
    torch.save({"params": network.state_dict()}, checkpoint)
    return repo, checkpoint


def _write_split_ssr(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "SplitSSR"
    architecture = repo / "basicsr" / "models" / "archs" / "SSR_arch.py"
    architecture.parent.mkdir(parents=True)
    architecture.write_text(SPLIT_SSR, encoding="utf-8")

    spec = spec_from_file_location("_split_ssr_checkpoint", architecture)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    network = module.SSR(dim=4)
    checkpoint = tmp_path / "split_net_g_best.pth"
    torch.save({"params": network.state_dict()}, checkpoint)
    return repo, checkpoint


def _latent_model_config(
    repo: Path,
    checkpoint: Path,
    *,
    steps: int,
    oracle_z: bool = False,
) -> dict:
    return {
        "semantic": {"encoder": "none", "embedding_dim": 1},
        "foundation": {
            "type": "external_ssr",
            "repo_path": str(repo),
            "checkpoint": str(checkpoint),
            "architecture": {"dim": 4},
            "feature_adapters": {
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
        "bfn": {
            "inference_mode": "none",
            "mask_mode": "none",
        },
        "latent_bfn": {
            "steps": 0 if oracle_z else steps,
            "oracle_z": oracle_z,
            "global_dim": 8,
            "spatial_dim": 2,
            "hidden_channels": 4,
            "feature_channels": 16,
            "num_categories": 3,
            "spatial_downsample": 2,
        },
    }


def test_checkpoint_compatible_edffn_shapes_and_padding() -> None:
    layer = CheckpointCompatibleEDFFN(
        dim=4,
        ffn_expansion_factor=3,
        bias=False,
    )
    assert layer.project_in.weight.shape == (24, 4, 1, 1)
    assert layer.project_out.weight.shape == (4, 12, 1, 1)
    assert layer.fft.shape == (4, 1, 1, 8, 5)

    image = torch.rand(1, 4, 15, 17, requires_grad=True)
    output = layer(image)
    assert output.shape == image.shape
    output.mean().backward()
    assert image.grad is not None


def test_external_ssr_foundation_is_frozen_and_drives_bfn(
    tmp_path: Path,
) -> None:
    repo, checkpoint = _write_dummy_ssr(tmp_path)
    model = SGBFormer(
        semantic={"encoder": "none", "embedding_dim": 1},
        foundation={
            "type": "external_ssr",
            "repo_path": str(repo),
            "checkpoint": str(checkpoint),
            "freeze": True,
            "architecture": {
                "inp_channels": 3,
                "out_channels": 3,
                "dim": 4,
                "num_blocks": [1, 1, 1],
                "ffn_expansion_factor": 2,
                "bias": True,
            },
        },
        bfn={
            "steps": 2,
            "min_variance": 0.01,
            "base_channels": 8,
            "channel_multipliers": (1, 2),
            "blocks_per_level": 1,
            "attention_heads": 2,
            "entropy_kernel": 3,
            "entropy_bins": 4,
            "entropy_threshold": 0.0,
        },
    )
    assert model.backbone.decoder_channels == 4
    assert all(
        not parameter.requires_grad
        for parameter in model.backbone.parameters()
    )

    degraded = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(degraded)
    structural, feature, _, _ = model.structural_forward(degraded)
    direct = model.backbone.model(degraded)
    torch.testing.assert_close(structural, direct)
    assert feature.shape == (1, 4, 16, 16)

    outputs = model(
        degraded,
        target,
        training_stage="bfn",
    )
    outputs["bfn_loss"].backward()
    assert any(
        parameter.grad is not None for parameter in model.receiver.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.backbone.parameters()
    )

    model.zero_grad(set_to_none=True)
    model.refinement_mode = "direct"
    model.mask_mode = "none"
    direct_outputs = model(
        degraded,
        target,
        training_stage="residual",
    )
    (direct_outputs["restored"] - target).abs().mean().backward()
    assert torch.all(direct_outputs["sampling_mask"] == 1)
    assert any(
        parameter.grad is not None for parameter in model.receiver.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.backbone.parameters()
    )
    direct_details = model.restore(
        degraded,
        deterministic=True,
        return_details=True,
        return_trajectory=True,
    )
    assert len(direct_details["trajectory"]) == 1


def test_external_ssr_paper_core_requires_spectral_gate_and_iterative_bfn(
    tmp_path: Path,
) -> None:
    repo, checkpoint = _write_dummy_ssr(tmp_path)
    foundation = {
        "type": "external_ssr",
        "repo_path": str(repo),
        "checkpoint": str(checkpoint),
        "architecture": {"dim": 4},
        "feature_adapters": {
            "decoder_level3": {
                "type": "paper_spectral_gate",
                "channels": 4,
                "initial_gate": 0.999,
            },
        },
    }
    bfn = {
        "steps": 2,
        "inference_mode": "iterative",
        "mask_mode": "entropy",
        "min_variance": 0.01,
        "base_channels": 8,
        "channel_multipliers": (1, 2),
        "blocks_per_level": 1,
        "attention_heads": 2,
        "entropy_kernel": 3,
        "entropy_bins": 4,
        "entropy_threshold": 0.0,
    }
    core = {
        "require_spectral_gate": True,
        "require_iterative_bfn": True,
        "require_entropy_mask": True,
        "bfn_steps": 2,
    }
    model = SGBFormer(
        semantic={"encoder": "none", "embedding_dim": 1},
        foundation=foundation,
        bfn=bfn,
        core=core,
    )
    assert model.backbone.has_paper_spectral_gate
    assert model.receiver is not None

    image = torch.rand(1, 3, 16, 16)
    outputs = model(image, image, training_stage="joint")
    outputs["bfn_loss"].backward()
    gate = model.backbone.adapters["decoder_level3"].spectral_gate.gate
    assert gate.weight.grad is not None
    assert any(
        parameter.grad is not None for parameter in model.receiver.parameters()
    )
    details = model.restore(
        image,
        num_steps=2,
        deterministic=True,
        return_details=True,
        return_trajectory=True,
    )
    assert len(details["trajectory"]) == 2

    without_gate = dict(foundation)
    without_gate.pop("feature_adapters")
    with pytest.raises(ValueError, match="spectral gating"):
        SGBFormer(
            semantic={"encoder": "none", "embedding_dim": 1},
            foundation=without_gate,
            bfn=bfn,
            core=core,
        )
    with pytest.raises(ValueError, match="iterative Bayesian"):
        SGBFormer(
            semantic={"encoder": "none", "embedding_dim": 1},
            foundation=foundation,
            bfn={**bfn, "inference_mode": "none"},
            core=core,
        )


def test_external_ssr_feature_adapter_is_identity_and_trainable(
    tmp_path: Path,
) -> None:
    repo, checkpoint = _write_dummy_ssr(tmp_path)
    model = SGBFormer(
        semantic={"encoder": "none", "embedding_dim": 1},
        foundation={
            "type": "external_ssr",
            "repo_path": str(repo),
            "checkpoint": str(checkpoint),
            "freeze": True,
            "feature_adapters": {
                "decoder_level1": {
                    "channels": 4,
                    "hidden_channels": 4,
                }
            },
            "architecture": {
                "inp_channels": 3,
                "out_channels": 3,
                "dim": 4,
                "num_blocks": [1, 1, 1],
                "ffn_expansion_factor": 2,
                "bias": True,
            },
        },
        bfn={
            "receiver_trainable": False,
            "inference_mode": "direct",
            "mask_mode": "none",
            "base_channels": 8,
            "channel_multipliers": (1, 2),
            "blocks_per_level": 1,
            "attention_heads": 2,
        },
    )
    degraded = torch.rand(1, 3, 16, 16)
    baseline = ExternalSSRFoundation(
        str(repo), str(checkpoint), architecture={"dim": 4}
    )
    expected, _ = baseline(degraded)
    structural, _, _, _ = model.structural_forward(degraded)
    torch.testing.assert_close(structural, expected)

    target = torch.rand_like(degraded)
    outputs = model(degraded, target, training_stage="structure")
    (outputs["structural"] - target).abs().mean().backward()
    assert all(
        parameter.grad is None
        for parameter in model.backbone.model.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.backbone.adapters.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.receiver.parameters()
    )


def test_external_ssr_state_adapter_is_scoped_and_trainable(
    tmp_path: Path,
) -> None:
    repo, checkpoint = _write_dummy_ssr(tmp_path)
    foundation = ExternalSSRFoundation(
        str(repo),
        str(checkpoint),
        architecture={"dim": 4},
        feature_adapters={
            "decoder_level3": {
                "type": "state_spectral",
                "channels": 4,
                "hidden_channels": 4,
            }
        },
    )
    image = torch.rand(1, 3, 16, 16)
    baseline, _ = foundation(image)
    state = {
        "degraded": image,
        "baseline": baseline.detach(),
        "belief_mean": torch.rand_like(image),
        "precision": torch.ones(1, 1, 16, 16),
        "time": torch.tensor([0.5]),
    }
    state_output, _ = foundation.forward_with_state(image, state)
    torch.testing.assert_close(state_output, baseline, atol=0, rtol=0)
    state_output.square().mean().backward()
    adapter = foundation.adapters["decoder_level3"]
    assert adapter.output.weight.grad is not None
    assert adapter.output.weight.grad.abs().sum() > 0
    foundation.zero_grad(set_to_none=True)
    restored, _ = foundation(image)
    torch.testing.assert_close(restored, baseline, atol=0, rtol=0)


def test_split_ssr_cache_matches_full_state_decode(tmp_path: Path) -> None:
    repo, checkpoint = _write_split_ssr(tmp_path)
    foundation = ExternalSSRFoundation(
        str(repo),
        str(checkpoint),
        architecture={"dim": 4},
        feature_adapters={
            "decoder_level3": {
                "type": "state_spectral",
                "channels": 16,
                "hidden_channels": 4,
            }
        },
    )
    image = torch.rand(1, 3, 16, 16)
    baseline, _ = foundation(image)
    cache = foundation.cache_decoder_prefix(image)
    cached, cached_feature = foundation.decode_from_cached_decoder3(cache)
    torch.testing.assert_close(cached, baseline, atol=1e-7, rtol=1e-6)
    assert cached_feature.shape == (1, 4, 16, 16)

    state = {
        "degraded": image,
        "baseline": baseline.detach(),
        "belief_mean": torch.rand_like(image),
        "precision": torch.ones(1, 1, 16, 16),
        "time": torch.tensor([0.5]),
    }
    adapter = foundation.state_conditioned_adapter
    assert adapter is not None
    with torch.no_grad():
        adapter.output.weight.normal_(0, 0.1)
    full_state, _ = foundation.forward_with_state(image, state)
    cached_state, _ = foundation.decode_from_cached_decoder3(cache, state)
    torch.testing.assert_close(full_state, cached_state, atol=1e-6, rtol=1e-5)
    cached_state.square().mean().backward()
    assert adapter.output.weight.grad is not None
    assert adapter.output.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None for parameter in foundation.model.parameters()
    )


def test_decoder_state_bfn_drives_frozen_ssr_decoder(
    tmp_path: Path,
) -> None:
    repo, checkpoint = _write_dummy_ssr(tmp_path)
    foundation = {
        "type": "external_ssr",
        "repo_path": str(repo),
        "checkpoint": str(checkpoint),
        "architecture": {"dim": 4},
        "feature_adapters": {
            "decoder_level3": {
                "type": "state_spectral",
                "channels": 4,
                "hidden_channels": 4,
            }
        },
    }
    common_bfn = {
        "receiver_type": "decoder_state",
        "steps": 2,
        "inference_mode": "iterative",
        "mask_mode": "none",
        "min_variance": 0.01,
        "residual_scale": 0.1,
        "training_final_state": "rollout",
        "training_rollout_steps": 2,
        "rollout_backprop_steps": 1,
    }
    model = SGBFormer(
        semantic={"encoder": "none", "embedding_dim": 1},
        foundation=foundation,
        bfn=common_bfn,
    )
    direct = SGBFormer(
        semantic={"encoder": "none", "embedding_dim": 1},
        foundation=foundation,
        bfn={
            **common_bfn,
            "inference_mode": "direct",
            "state_condition_mode": "neutral",
        },
    )
    assert model.receiver is None
    assert direct.receiver is None
    assert sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    ) == sum(
        parameter.numel() for parameter in direct.parameters()
        if parameter.requires_grad
    )

    image = torch.rand(1, 3, 16, 16)
    outputs = model(image, image, training_stage="joint")
    (outputs["bfn_loss"] + outputs["restored"].mean()).backward()
    adapter = model.backbone.adapters["decoder_level3"]
    assert adapter.output.weight.grad is not None
    assert adapter.output.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None for parameter in model.backbone.model.parameters()
    )
    details = model.restore(
        image,
        deterministic=True,
        return_details=True,
        return_trajectory=True,
    )
    assert len(details["trajectory"]) == 2
    assert details["decoder_state_calls"] == 3
    assert details["decoder_state_updates"] == 2
    assert [entry["time"] for entry in details["state_update_schedule"]] == [
        0.0,
        0.5,
    ]
    assert (
        details["state_update_schedule"][0]["alpha"]
        != details["state_update_schedule"][1]["alpha"]
    )
    assert details["terminal_readout_time"] == 1.0
    torch.testing.assert_close(
        model.restore_structural(image),
        details["structural"],
    )
    reset_details = model.restore(
        image,
        deterministic=True,
        state_ablation="reset-belief",
        return_details=True,
    )
    assert reset_details["decoder_state_calls"] == 3
    assert reset_details["decoder_state_updates"] == 2
    assert reset_details["state_update_schedule"] == details[
        "state_update_schedule"
    ]
    direct_details = direct.restore(
        image,
        deterministic=True,
        return_details=True,
    )
    assert direct_details["decoder_state_calls"] == 1
    assert direct_details["decoder_state_updates"] == 0
    assert direct_details["state_update_schedule"] == []
    with pytest.raises(ValueError, match="batch size"):
        model.restore(
            image,
            deterministic=True,
            state_ablation="shuffle-belief",
        )
    paired = torch.rand(2, 3, 16, 16)
    reset = model.restore(
        paired,
        deterministic=True,
        state_ablation="reset-belief",
    )
    shuffled = model.restore(
        paired,
        deterministic=True,
        state_ablation="shuffle-belief",
    )
    assert reset.shape == paired.shape
    assert shuffled.shape == paired.shape


def test_decoder_state_bfn_uses_cached_ssr_prefix(tmp_path: Path) -> None:
    repo, checkpoint = _write_split_ssr(tmp_path)
    model = SGBFormer(
        semantic={"encoder": "none", "embedding_dim": 1},
        foundation={
            "type": "external_ssr",
            "repo_path": str(repo),
            "checkpoint": str(checkpoint),
            "architecture": {"dim": 4},
            "feature_adapters": {
                "decoder_level3": {
                    "type": "state_spectral",
                    "channels": 16,
                    "hidden_channels": 4,
                }
            },
        },
        bfn={
            "receiver_type": "decoder_state",
            "decoder_state_cache": True,
            "steps": 2,
            "inference_mode": "iterative",
            "mask_mode": "none",
            "min_variance": 0.01,
            "residual_scale": 0.1,
            "training_final_state": "rollout",
            "training_rollout_steps": 2,
            "rollout_backprop_steps": 1,
        },
    )
    image = torch.rand(1, 3, 16, 16)
    outputs = model(image, image, training_stage="joint")
    (outputs["bfn_loss"] + outputs["restored"].mean()).backward()
    adapter = model.backbone.state_conditioned_adapter
    assert adapter is not None
    assert adapter.output.weight.grad is not None
    assert model.restore(image, deterministic=True).shape == image.shape


def test_v3_multilevel_gradients_and_no_receiver(tmp_path: Path) -> None:
    repo, checkpoint = _write_dummy_ssr(tmp_path)
    config = {
        "repo_path": str(repo), "checkpoint": str(checkpoint),
        "architecture": {"dim": 4},
    }
    baseline = ExternalSSRFoundation(**config)
    model = SGBFormer(
        semantic={"encoder": "none", "embedding_dim": 1},
        foundation={
            "type": "external_ssr", **config,
            "trainable_modules": ["decoder_level1", "output"],
            "feature_adapters": {
                name: {"type": "calibrated", "channels": 4, "hidden_channels": 4}
                for name in ("decoder_level3", "decoder_level2", "decoder_level1")
            },
        },
        bfn={"inference_mode": "none", "mask_mode": "none"},
    )
    assert model.receiver is None
    assert not any(key.startswith("receiver.") for key in model.state_dict())
    image = torch.rand(1, 3, 17, 19)
    padded = torch.nn.functional.pad(image, (0, 5, 0, 7), mode="reflect")
    expected = baseline(padded)[0][..., :17, :19]
    outputs = model(image, image, training_stage="structure")
    torch.testing.assert_close(outputs["structural"], expected, rtol=0, atol=0)
    details = model.restore(image, return_details=True, return_trajectory=True)
    torch.testing.assert_close(details["restored"], expected.clamp(0, 1), rtol=0, atol=0)
    assert details["adapter_uncertainty"].shape == (1, 1, 17, 19)
    assert details["trajectory"] == []
    with pytest.raises(ValueError, match="requires training_stage"):
        model(image, image, training_stage="bfn")

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image, image, training_stage="structure")
        loss = (outputs["structural"] - image).square().mean()
        loss = loss + 0.01 * outputs["uncertainty_calibration"]
        loss.backward()
        for adapter in model.backbone.adapters.values():
            assert adapter.mean_head.weight.grad.abs().sum() > 0
            assert adapter.risk_head.weight.grad.abs().sum() > 0
        assert model.backbone.model.decoder_level1.weight.grad.abs().sum() > 0
        assert model.backbone.model.decoder_level2.weight.grad is None
        assert model.backbone.model.encoder.weight.grad is None
        optimizer.step()
    assert model.backbone.adapters["decoder_level3"].project_in.weight.grad.abs().sum() > 0


def test_partial_finetuning_keeps_frozen_modules_in_eval(tmp_path: Path) -> None:
    repo, checkpoint = _write_dummy_ssr(tmp_path)
    foundation = ExternalSSRFoundation(
        str(repo), str(checkpoint), architecture={"dim": 4},
        trainable_modules=["decoder_level1", "output"],
        frozen_modules_eval=True,
    )
    foundation.train()
    assert not foundation.model.encoder.training
    assert not foundation.model.decoder_level2.training
    assert foundation.model.decoder_level1.training
    assert foundation.model.output.training
    foundation.eval()
    assert not any(module.training for module in foundation.modules())


def test_calibrated_adapters_train_through_fully_frozen_foundation(
    tmp_path: Path,
) -> None:
    repo, checkpoint = _write_dummy_ssr(tmp_path)
    model = SGBFormer(
        semantic={"encoder": "none", "embedding_dim": 1},
        foundation={
            "type": "external_ssr",
            "repo_path": str(repo),
            "checkpoint": str(checkpoint),
            "architecture": {"dim": 4},
            "trainable_modules": [],
            "frozen_modules_eval": True,
            "feature_adapters": {
                name: {
                    "type": "calibrated",
                    "channels": 4,
                    "hidden_channels": 4,
                    "uncertainty_mode": "none",
                    "use_spectral": True,
                }
                for name in (
                    "decoder_level3",
                    "decoder_level2",
                    "decoder_level1",
                )
            },
        },
        bfn={"inference_mode": "none", "mask_mode": "none"},
    )
    model.train()
    assert not any(
        parameter.requires_grad
        for parameter in model.backbone.model.parameters()
    )
    assert not any(
        module.training for module in model.backbone.model.modules()
    )

    image = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(image)
    outputs = model(image, target, training_stage="structure")
    (outputs["structural"] - target).square().mean().backward()

    assert all(
        adapter.mean_head.weight.grad is not None
        and adapter.mean_head.weight.grad.abs().sum() > 0
        for adapter in model.backbone.adapters.values()
    )
    assert all(
        parameter.grad is None
        for parameter in model.backbone.model.parameters()
    )


@pytest.mark.parametrize("channels", [8, 16])
def test_latent_adapter_is_strict_identity_at_each_injection(
    channels: int,
) -> None:
    adapter = LatentConditionedFiLMSpectralAdapter(
        channels=channels,
        global_dim=8,
        spatial_dim=2,
        hidden_channels=4,
    )
    feature = torch.randn(2, channels, 8, 12)
    state = {
        "global_mean": torch.randn(2, 8),
        "global_precision": torch.rand(2, 8) + 1,
        "spatial_mean": torch.randn(2, 2, 2, 3),
        "spatial_precision": torch.rand(2, 2, 2, 3) + 1,
    }
    output = adapter(feature, state)
    torch.testing.assert_close(output, feature, atol=0, rtol=0)
    assert adapter.last_gamma_delta == 0
    assert adapter.last_beta == 0
    assert adapter.last_spectral_delta == 0


def test_latent_bfn_rgb_identity_gradients_and_single_decoder(
    tmp_path: Path,
) -> None:
    repo, checkpoint = _write_split_ssr(tmp_path)
    baseline = ExternalSSRFoundation(
        str(repo),
        str(checkpoint),
        architecture={"dim": 4},
    ).eval()
    model = SGBFormer(
        **_latent_model_config(repo, checkpoint, steps=4)
    ).eval()
    image = torch.rand(2, 3, 16, 16)
    expected, _ = baseline(image)

    counts = {"decoder_level3": 0, "decoder_level2": 0, "decoder_level1": 0}
    handles = [
        getattr(model.backbone.model, name).register_forward_pre_hook(
            lambda module, inputs, name=name: counts.__setitem__(
                name,
                counts[name] + 1,
            )
        )
        for name in counts
    ]
    restored = model.restore(image, deterministic=True)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(restored, expected.clamp(0, 1), atol=0, rtol=0)
    assert counts == {
        "decoder_level3": 1,
        "decoder_level2": 1,
        "decoder_level1": 1,
    }

    model.train()
    target = torch.rand_like(image)
    outputs = model(
        image,
        target,
        training_stage="joint",
        category=torch.tensor([0, 2]),
    )
    loss = (
        (outputs["restored"] - target).square().mean()
        + outputs["bfn_loss"]
        + torch.nn.functional.cross_entropy(
            outputs["latent_category_logits"],
            outputs["latent_category_target"],
        )
    )
    loss.backward()
    for adapter in model.backbone.adapters.values():
        assert adapter.beta_projection.weight.grad is not None
        assert adapter.beta_projection.weight.grad.abs().sum() > 0
    global_output = model.latent_flow.global_observer[-1]
    spatial_output = model.latent_flow.spatial_observer[-1]
    assert global_output.weight.grad[model.latent_global_dim :].abs().sum() > 0
    assert spatial_output.weight.grad[
        model.latent_spatial_dim :
    ].abs().sum() > 0


@pytest.mark.parametrize("update_mode", ["deterministic", "bayesian"])
def test_image_only_gradient_reaches_latent_after_zero_projection_bootstrap(
    tmp_path: Path,
    update_mode: str,
) -> None:
    repo, checkpoint = _write_split_ssr(tmp_path)
    config = _latent_model_config(repo, checkpoint, steps=4)
    config["latent_bfn"].update(
        update_mode=update_mode,
        accuracy_schedule="linear_beta",
        loss_mode="final_mean",
        loss_weighting="adaptive_accuracy",
    )
    model = SGBFormer(**config).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-4,
    )
    image = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(image)
    category = torch.tensor([0])

    first = model(
        image,
        target,
        training_stage="joint",
        category=category,
    )
    torch.nn.functional.mse_loss(first["structural"], target).backward()
    first_latent_grad = sum(
        float(parameter.grad.square().sum())
        for parameter in model.latent_flow.parameters()
        if parameter.grad is not None
    )
    assert first_latent_grad == 0
    assert all(
        adapter.beta_projection.weight.grad is not None
        and adapter.beta_projection.weight.grad.abs().sum() > 0
        for adapter in model.backbone.adapters.values()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = model(
        image,
        target,
        training_stage="joint",
        category=category,
    )
    torch.nn.functional.mse_loss(second["structural"], target).backward()
    assert sum(
        float(parameter.grad.square().sum())
        for parameter in model.latent_flow.parameters()
        if parameter.grad is not None
    ) > 0
    assert sum(
        float(parameter.grad.square().sum())
        for parameter in model.latent_flow.degradation_encoder.parameters()
        if parameter.grad is not None
    ) > 0


def test_latent_arms_match_parameters_and_are_equal_at_k1(
    tmp_path: Path,
) -> None:
    repo, checkpoint = _write_split_ssr(tmp_path)
    arm_a = SGBFormer(
        **_latent_model_config(repo, checkpoint, steps=4)
    ).eval()
    arm_b = SGBFormer(
        **_latent_model_config(repo, checkpoint, steps=1)
    ).eval()
    arm_b.load_state_dict(arm_a.state_dict(), strict=True)

    parameters_a = {
        name: (parameter.numel(), parameter.requires_grad)
        for name, parameter in arm_a.named_parameters()
    }
    parameters_b = {
        name: (parameter.numel(), parameter.requires_grad)
        for name, parameter in arm_b.named_parameters()
    }
    assert parameters_a == parameters_b
    assert arm_b.latent_flow.global_observer[-1].out_features == (
        arm_b.latent_global_dim * 2
    )
    assert arm_b.latent_flow.spatial_observer[-1].out_channels == (
        arm_b.latent_spatial_dim * 2
    )

    image = torch.rand(1, 3, 16, 16)
    output_a = arm_a.restore(image, num_steps=1, deterministic=True)
    output_b = arm_b.restore(image, deterministic=True)
    torch.testing.assert_close(output_a, output_b, atol=0, rtol=0)


def test_latent_adapter_precision_can_be_oracle_aligned_without_changing_belief(
    tmp_path: Path,
) -> None:
    repo, checkpoint = _write_split_ssr(tmp_path)
    model = SGBFormer(
        **_latent_model_config(repo, checkpoint, steps=4)
    ).eval()
    image = torch.rand(1, 3, 16, 16)
    cache, _, outputs, _, original_size = model.predict_latent_state(
        image,
        num_steps=4,
    )
    belief_state, belief_guard = model.adapter_state_from_latent_outputs(
        outputs,
        precision_mode="belief",
    )
    unit_state, unit_guard = model.adapter_state_from_latent_outputs(
        outputs,
        precision_mode="unit",
    )
    torch.testing.assert_close(
        belief_state["global_mean"],
        unit_state["global_mean"],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        belief_state["spatial_mean"],
        unit_state["spatial_mean"],
        atol=0,
        rtol=0,
    )
    assert belief_guard["global_mean_abs_log_precision"] > 0
    assert belief_guard["spatial_mean_abs_log_precision"] > 0
    assert unit_guard["global_mean_abs_log_precision"] == 0
    assert unit_guard["spatial_mean_abs_log_precision"] == 0
    assert torch.equal(
        unit_state["global_precision"],
        torch.ones_like(unit_state["global_precision"]),
    )
    assert torch.equal(
        unit_state["spatial_precision"],
        torch.ones_like(unit_state["spatial_precision"]),
    )
    restored = model.decode_cached_latent_state(
        cache,
        unit_state,
        original_size=original_size,
    )
    assert restored.shape == image.shape


def test_oracle_z_omits_flow_is_identity_and_decodes_once(
    tmp_path: Path,
) -> None:
    repo, checkpoint = _write_split_ssr(tmp_path)
    baseline = ExternalSSRFoundation(
        str(repo),
        str(checkpoint),
        architecture={"dim": 4},
    ).eval()
    model = SGBFormer(
        **_latent_model_config(
            repo,
            checkpoint,
            steps=0,
            oracle_z=True,
        )
    ).eval()
    assert model.oracle_z_enabled
    assert model.latent_flow is None
    assert not any(
        name.startswith("latent_flow.")
        for name, _ in model.named_parameters()
    )
    assert all(
        name.startswith("backbone.adapters.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )

    image = torch.rand(2, 3, 16, 16)
    target = torch.rand_like(image)
    category = torch.tensor([0, 2])
    expected, _ = baseline(image)
    counts = {"decoder_level3": 0, "decoder_level2": 0, "decoder_level1": 0}
    handles = [
        getattr(model.backbone.model, name).register_forward_pre_hook(
            lambda module, inputs, name=name: counts.__setitem__(
                name,
                counts[name] + 1,
            )
        )
        for name in counts
    ]
    details = model.restore_oracle_z(
        image,
        target,
        category=category,
        return_details=True,
    )
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(
        details["restored"],
        expected.clamp(0, 1),
        atol=0,
        rtol=0,
    )
    assert counts == {
        "decoder_level3": 1,
        "decoder_level2": 1,
        "decoder_level1": 1,
    }
    assert details["decoder_calls"] == 1
    assert details["deployable"] is False
    assert set(details["modulation_diagnostics"]) == {
        "decoder_level3",
        "decoder_level2",
    }
    for values in details["modulation_diagnostics"].values():
        assert all(value == 0 for value in values.values())
    with pytest.raises(RuntimeError, match="non-deployable"):
        model.restore(image)

    model.train()
    outputs = model(
        image,
        target,
        training_stage="joint",
        category=category,
    )
    assert outputs["bfn_loss"] == 0
    (outputs["restored"] - target).square().mean().backward()
    for adapter in model.backbone.adapters.values():
        assert adapter.beta_projection.weight.grad is not None
        assert adapter.beta_projection.weight.grad.abs().sum() > 0
