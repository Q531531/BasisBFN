import math

import pytest
import torch

from sgbformer.models.adapters import (
    CalibratedSpectralAdapter,
    LatentConditionedFiLMSpectralAdapter,
    StateConditionedSpectralAdapter,
)
from sgbformer.training import clip_training_gradients


def test_latent_adapter_fusion_gain_is_one_and_identity_initialized() -> None:
    adapter = LatentConditionedFiLMSpectralAdapter(
        channels=8,
        global_dim=6,
        spatial_dim=2,
        hidden_channels=4,
    )
    assert adapter.fusion_gain.item() == 1.0
    feature = torch.randn(2, 8, 8, 10)
    state = {
        "global_mean": torch.randn(2, 6),
        "global_precision": torch.rand(2, 6) + 1,
        "spatial_mean": torch.randn(2, 2, 2, 3),
        "spatial_precision": torch.rand(2, 2, 2, 3) + 1,
    }
    torch.testing.assert_close(adapter(feature, state), feature, atol=0, rtol=0)
    for projection in (
        adapter.gamma_projection,
        adapter.beta_projection,
        adapter.spectral_projection,
    ):
        assert torch.count_nonzero(projection.weight) == 0
        assert torch.count_nonzero(projection.bias) == 0


@pytest.mark.parametrize("mode", ["none", "auxiliary", "gated", "uniform"])
@pytest.mark.parametrize("spectral", [False, True])
def test_calibrated_adapter_starts_as_exact_identity(mode, spectral) -> None:
    adapter = CalibratedSpectralAdapter(
        8, 4, use_spectral=spectral, uncertainty_mode=mode
    )
    feature = torch.randn(2, 8, 15, 17, requires_grad=True)
    output = adapter(feature)
    torch.testing.assert_close(output, feature, atol=0, rtol=0)
    output.square().mean().backward()
    assert adapter.mean_head.weight.grad.abs().sum() > 0
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters()
    )
    assert adapter.risk_head.weight.grad is None


def test_reference_normalization_matches_legacy_at_training_size() -> None:
    torch.manual_seed(8)
    legacy = CalibratedSpectralAdapter(4, 4, uncertainty_mode="none")
    normalized = CalibratedSpectralAdapter(
        4, 4, uncertainty_mode="none", spectral_reference_size=16
    )
    with torch.no_grad():
        legacy.mean_head.weight.normal_(0, 0.1)
        legacy.evidence_gate.weight.normal_(0, 0.1)
    normalized.load_state_dict(legacy.state_dict())
    feature = torch.randn(1, 4, 16, 16)
    torch.testing.assert_close(legacy(feature), normalized(feature), atol=0, rtol=0)
    for adapter in (legacy, normalized):
        adapter(feature).square().mean().backward()
    for name, parameter in legacy.named_parameters():
        other = dict(normalized.named_parameters())[name]
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=0, rtol=0)


def test_spectral_evidence_normalization_handles_periodic_size_change() -> None:
    torch.manual_seed(11)
    legacy = CalibratedSpectralAdapter(4, 4, uncertainty_mode="none")
    normalized = CalibratedSpectralAdapter(
        4, 4, uncertainty_mode="none", spectral_reference_size=16
    )
    with torch.no_grad():
        legacy.evidence_gate.weight.fill_(0.2)
    normalized.load_state_dict(legacy.state_dict())
    hidden = torch.rand(1, 4, 16, 16) * 0.2
    repeated = hidden.repeat(1, 1, 2, 3)
    expected = normalized.spectral_evidence(hidden).repeat(1, 1, 2, 3)
    torch.testing.assert_close(
        normalized.spectral_evidence(repeated), expected, atol=2e-7, rtol=1e-5
    )
    assert (legacy.spectral_evidence(repeated) - expected).abs().mean() > 1e-3


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_normalized_spectral_odd_sizes_backward(device) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    adapter = CalibratedSpectralAdapter(
        4, 4, uncertainty_mode="none", spectral_reference_size=16
    ).to(device)
    feature = torch.randn(1, 4, 17, 23, device=device, requires_grad=True)
    torch.testing.assert_close(adapter(feature), feature, atol=0, rtol=0)
    with torch.no_grad():
        adapter.mean_head.weight.normal_(0, 0.1)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=device == "cpu"):
        loss = adapter(feature).float().square().mean()
    loss.backward()
    assert adapter.evidence_gate.weight.grad.abs().sum() > 0
    assert torch.isfinite(feature.grad).all()


def test_spectral_reference_size_rejects_invalid_values() -> None:
    for value in (0, -1, 16.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            CalibratedSpectralAdapter(4, 4, spectral_reference_size=value)


def test_spatial_spectral_controls_match_capacity_and_initial_correction() -> None:
    spatial = CalibratedSpectralAdapter(8, 4, use_spectral=False)
    spectral = CalibratedSpectralAdapter(8, 4, use_spectral=True)
    with torch.no_grad():
        spatial.mean_head.weight.normal_(0, 0.1)
    spectral.load_state_dict(spatial.state_dict())
    feature = torch.randn(2, 8, 12, 14)
    torch.testing.assert_close(spatial(feature), spectral(feature), rtol=0, atol=0)
    assert sum(p.numel() for p in spatial.parameters()) == sum(
        p.numel() for p in spectral.parameters()
    )
    spectral(feature).square().mean().backward()
    assert spectral.evidence_gate.weight.grad.abs().sum() > 0
    spatial(feature).square().mean().backward()
    assert spatial.evidence_gate.weight.grad.abs().sum() > 0


def test_calibration_is_absolute_and_does_not_move_restoration_features() -> None:
    adapter = CalibratedSpectralAdapter(8, 4)
    feature = torch.randn(2, 8, 8, 8, requires_grad=True)
    squared_error = torch.full((2, 1, 8, 8), 0.01, requires_grad=True)
    adapter(feature)
    adapter.calibration_loss(squared_error).backward()
    assert adapter.risk_head.bias.grad < 0  # Underestimated variance must increase.
    assert feature.grad is None
    assert squared_error.grad is None
    assert adapter.project_in.weight.grad is None
    assert adapter.mean_head.weight.grad is None
    adapter.zero_grad(set_to_none=True)
    adapter(feature)
    adapter.calibration_loss(squared_error / 16).backward()
    assert adapter.risk_head.bias.grad > 0


def test_calibration_score_prefers_true_variance_and_ignores_padding() -> None:
    adapter = CalibratedSpectralAdapter(8, 4)
    feature = torch.randn(1, 8, 8, 8)
    error = torch.full((1, 1, 8, 8), 0.01)
    adapter(feature)
    before = adapter.calibration_loss(error)
    with torch.no_grad():
        adapter.risk_head.bias.fill_(math.log(0.01 / adapter.reference_variance))
    adapter(feature)
    optimum = adapter.calibration_loss(error)
    assert optimum < before
    valid = torch.ones_like(error)
    valid[..., 6:, :] = 0
    valid[..., :, 5:] = 0
    padded_error = error.clone()
    padded_error[valid == 0] = 100
    torch.testing.assert_close(
        adapter.calibration_loss(padded_error, valid), optimum
    )


def test_risk_gating_is_neutral_bounded_and_detached() -> None:
    adapter = CalibratedSpectralAdapter(8, 4)
    feature = torch.randn(1, 8, 10, 10)
    with torch.no_grad():
        adapter.mean_head.weight.zero_()
        adapter.mean_head.bias.fill_(1)
    neutral = adapter(feature) - feature
    torch.testing.assert_close(neutral, torch.full_like(neutral, 0.1))
    with torch.no_grad():
        adapter.risk_head.bias.fill_(6)
    high_risk = adapter(feature) - feature
    assert torch.all(high_risk < neutral)
    assert torch.all(high_risk > neutral * 0.49)
    with torch.no_grad():
        adapter.risk_head.bias.fill_(-6)
    low_risk = adapter(feature) - feature
    assert torch.all(low_risk > neutral)
    assert torch.all(low_risk < neutral * 1.51)
    adapter(feature).square().mean().backward()
    assert adapter.risk_head.bias.grad is None


def test_state_conditioned_adapter_is_identity_without_state_and_trainable() -> None:
    adapter = StateConditionedSpectralAdapter(8, hidden_channels=4)
    feature = torch.randn(2, 8, 7, 9, requires_grad=True)
    torch.testing.assert_close(adapter(feature), feature, atol=0, rtol=0)
    state = {
        "degraded": torch.rand(2, 3, 15, 17),
        "baseline": torch.rand(2, 3, 15, 17),
        "belief_mean": torch.rand(2, 3, 15, 17),
        "precision": torch.ones(2, 1, 15, 17),
        "time": torch.tensor([0.2, 0.8]),
    }
    output = adapter(feature, state)
    torch.testing.assert_close(output, feature, atol=0, rtol=0)
    output.square().mean().backward()
    assert adapter.output.weight.grad is not None
    assert adapter.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.output.weight.normal_(0, 0.1)
    live = adapter(feature, state)
    neutral = adapter(feature, {**state, "mode": "neutral"})
    assert not torch.equal(live, neutral)


def test_tiny_synthetic_fit_learns_correction_and_risk() -> None:
    torch.manual_seed(7)
    adapter = CalibratedSpectralAdapter(4, 4, use_spectral=True)
    feature = torch.randn(2, 4, 8, 8)
    target = feature + 0.08
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.02)
    initial_error = (adapter(feature) - target).square().mean().item()
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        restored = adapter(feature)
        error = (restored - target).square().mean(dim=1, keepdim=True)
        loss = error.mean() + 0.001 * adapter.calibration_loss(error)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters())
        optimizer.step()
    final_error = (adapter(feature) - target).square().mean().item()
    assert final_error < initial_error * 0.4
    assert adapter.evidence_gate.weight.detach().abs().sum() > 0
    assert adapter.risk_head.weight.detach().abs().sum() > 0


@pytest.mark.parametrize("logit", [-8.0, 8.0])
def test_smooth_risk_can_recover_beyond_old_clamp_bounds(logit) -> None:
    adapter = CalibratedSpectralAdapter(4, 4, risk_log_ratio_mode="smooth")
    with torch.no_grad():
        adapter.risk_head.bias.fill_(logit)
    adapter(torch.randn(1, 4, 8, 8))
    error = torch.full((1, 1, 8, 8), adapter.reference_variance)
    adapter.calibration_loss(error).backward()
    gradient = adapter.risk_head.bias.grad.item()
    assert math.isfinite(gradient)
    assert gradient * logit > 0
    ratio = adapter.last_uncertainty / adapter.reference_variance
    assert torch.all(ratio.log().abs() < 6)


def test_smooth_risk_preserves_neutral_initialization_and_legacy_default() -> None:
    legacy = CalibratedSpectralAdapter(4, 4)
    smooth = CalibratedSpectralAdapter(4, 4, risk_log_ratio_mode="smooth")
    smooth.load_state_dict(legacy.state_dict())
    feature = torch.randn(1, 4, 8, 8, requires_grad=True)
    torch.testing.assert_close(smooth(feature), feature, atol=0, rtol=0)
    torch.testing.assert_close(
        smooth.last_uncertainty,
        torch.full((1, 1, 8, 8), smooth.reference_variance),
        atol=0, rtol=0,
    )
    assert legacy.risk_log_ratio_mode == "clamp"
    assert sum(p.numel() for p in legacy.parameters()) == sum(
        p.numel() for p in smooth.parameters()
    )
    with torch.no_grad():
        smooth.mean_head.bias.fill_(0.2)
    smooth(feature).square().mean().backward()
    assert smooth.risk_head.bias.grad is None


@pytest.mark.parametrize("risk_mode", ["clamp", "smooth"])
def test_auxiliary_risk_does_not_change_restoration_training(risk_mode) -> None:
    torch.manual_seed(42)
    spectral = CalibratedSpectralAdapter(
        4, 4, uncertainty_mode="none", risk_log_ratio_mode=risk_mode
    )
    auxiliary = CalibratedSpectralAdapter(
        4, 4, uncertainty_mode="auxiliary", risk_log_ratio_mode=risk_mode
    )
    auxiliary.load_state_dict(spectral.state_dict())
    adapters = (spectral, auxiliary)
    optimizers = [
        torch.optim.AdamW((p for p in a.parameters() if p.requires_grad), lr=0.01)
        for a in adapters
    ]
    feature = torch.randn(2, 4, 8, 8)
    target = feature + 0.05
    for _ in range(5):
        for adapter, optimizer in zip(adapters, optimizers):
            optimizer.zero_grad(set_to_none=True)
            error = (adapter(feature) - target).square().mean(1, keepdim=True)
            loss = error.mean() + 1e-4 * adapter.last_regularization
            loss = loss + 0.01 * adapter.calibration_loss(error)
            loss.backward()
            clip_training_gradients(adapter, 1e-3, separate_risk=True)
            optimizer.step()
        for name, value in spectral.state_dict().items():
            if not name.startswith("risk_head."):
                torch.testing.assert_close(
                    auxiliary.state_dict()[name], value, atol=0, rtol=0
                )
    assert auxiliary.risk_head.weight.abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_smooth_risk_mixed_precision_and_mps_backward(device) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    adapter = CalibratedSpectralAdapter(
        4, 4, risk_log_ratio_mode="smooth"
    ).to(device)
    with torch.no_grad():
        adapter.risk_head.bias.fill_(8)
    feature = torch.randn(1, 4, 16, 16, device=device, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=device == "cpu"):
        restored = adapter(feature)
        error = restored.float().square().mean(1, keepdim=True)
        loss = error.mean() + 0.01 * adapter.calibration_loss(error)
    loss.backward()
    assert torch.isfinite(feature.grad).all()
    assert adapter.risk_head.bias.grad.abs().sum() > 0
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters()
    )
