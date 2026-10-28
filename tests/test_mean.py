from pathlib import Path

import torch

from sgbformer.models import ConditionalMeanCorrector, SGBFormer
from test_foundation import _write_dummy_ssr


def test_mean_corrector_starts_as_identity_and_exposes_condition() -> None:
    corrector = ConditionalMeanCorrector(
        feature_channels=8,
        hidden_channels=4,
        blocks=1,
        correction_scale=0.1,
    )
    degraded = torch.rand(2, 3, 16, 16, requires_grad=True)
    structural = torch.rand_like(degraded, requires_grad=True)
    feature = torch.rand(2, 8, 8, 8, requires_grad=True)
    correction, condition = corrector(degraded, structural, feature)
    torch.testing.assert_close(
        correction,
        torch.zeros_like(correction),
        rtol=0,
        atol=0,
    )
    assert condition.shape == (2, 4, 16, 16)
    (correction - 0.01).square().mean().backward()
    assert corrector.output.weight.grad.abs().sum() > 0
    assert degraded.grad is None
    assert structural.grad is None
    assert feature.grad is None


def test_stage_a_trains_only_mean_corrector(tmp_path: Path) -> None:
    repo, checkpoint = _write_dummy_ssr(tmp_path)
    model = SGBFormer(
        semantic={"encoder": "none", "embedding_dim": 1},
        foundation={
            "type": "external_ssr",
            "repo_path": str(repo),
            "checkpoint": str(checkpoint),
            "architecture": {"dim": 4},
            "feature_adapters": {
                "decoder_level3": {
                    "type": "paper_spectral_gate",
                    "channels": 4,
                    "initial_gate": 0.999,
                    "trainable": False,
                },
            },
        },
        mean_correction={
            "enabled": True,
            "hidden_channels": 4,
            "blocks": 1,
            "detach_conditions": True,
        },
        bfn={"inference_mode": "none", "mask_mode": "entropy"},
        core={"require_spectral_gate": True},
    )
    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_names
    assert all(name.startswith("mean_corrector.") for name in trainable_names)

    image = torch.rand(2, 3, 16, 16)
    target = torch.rand_like(image)
    outputs = model(image, target, training_stage="mean")
    torch.testing.assert_close(
        outputs["structural"],
        outputs["foundation"],
        rtol=0,
        atol=0,
    )
    (outputs["structural"] - target).square().mean().backward()
    assert model.mean_corrector.output.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for parameter in model.backbone.parameters()
    )
