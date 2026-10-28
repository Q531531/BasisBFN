import pytest
import torch

from sgbformer.losses import (
    SGBFormerLoss, CharbonnierLoss, frequency_amplitude_loss, pearson_loss,
)


@pytest.mark.parametrize("constant_target", [False, True])
def test_pearson_constant_prediction_has_finite_gradients(constant_target) -> None:
    prediction = torch.zeros(2, 3, 8, 8, requires_grad=True)
    target = torch.zeros_like(prediction) if constant_target else torch.rand_like(prediction)
    loss = pearson_loss(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    if constant_target:
        assert loss == 0


def test_pearson_identical_nonconstant_images() -> None:
    image = torch.rand(2, 3, 8, 8)
    assert pearson_loss(image, image).item() == pytest.approx(0, abs=1e-6)


def test_frequency_loss_has_finite_zero_image_gradients() -> None:
    prediction = torch.zeros(1, 3, 15, 17, requires_grad=True)
    frequency_amplitude_loss(prediction, torch.zeros_like(prediction)).backward()
    assert torch.isfinite(prediction.grad).all()


def test_mse_structure_loss_uses_float32_and_exact_pixel_gradient() -> None:
    prediction = torch.full((1, 3, 8, 8), 0.0001, dtype=torch.float16, requires_grad=True)
    target = torch.zeros_like(prediction)
    outputs = {
        "structural": prediction, "restored": prediction,
        "bfn_loss": prediction.new_zeros(()),
    }
    criterion = SGBFormerLoss(structural_loss_type="mse", bfn_weight=0)
    loss, parts = criterion(outputs, target)
    expected = prediction.float().square().mean()
    assert loss.dtype == torch.float32
    assert loss > 0
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    torch.testing.assert_close(
        prediction.grad, (2 * prediction.float() / prediction.numel()).half()
    )
    torch.testing.assert_close(parts["structural"], expected.detach())


def test_legacy_charbonnier_is_default_and_mse_name_is_validated() -> None:
    prediction, target = torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
    outputs = {"structural": prediction, "restored": prediction, "bfn_loss": torch.tensor(0.)}
    loss, _ = SGBFormerLoss()(outputs, target)
    torch.testing.assert_close(loss, CharbonnierLoss()(prediction, target), rtol=0, atol=0)
    with pytest.raises(ValueError, match="structural_loss_type"):
        SGBFormerLoss(structural_loss_type="unknown")
    with pytest.raises(ValueError, match="final_loss_type"):
        SGBFormerLoss(final_loss_type="unknown")


def test_final_mse_supervises_deployed_restoration() -> None:
    structural = torch.zeros(1, 3, 4, 4)
    restored = torch.full_like(structural, 0.25, requires_grad=True)
    target = torch.ones_like(structural)
    outputs = {
        "structural": structural,
        "restored": restored,
        "bfn_loss": restored.new_zeros(()),
    }
    loss, parts = SGBFormerLoss(
        structural_weight=0,
        bfn_weight=0,
        final_weight=1,
        final_loss_type="mse",
    )(outputs, target)
    torch.testing.assert_close(loss, (restored - target).square().mean())
    torch.testing.assert_close(parts["final"], loss.detach())
    loss.backward()
    assert restored.grad is not None


def test_final_loss_mask_excludes_non_rollout_samples() -> None:
    target = torch.ones(2, 3, 4, 4)
    restored = torch.stack((
        torch.full_like(target[0], 0.5),
        torch.full_like(target[1], -10.0),
    )).requires_grad_()
    outputs = {
        "structural": torch.zeros_like(target),
        "restored": restored,
        "bfn_loss": restored.new_zeros(()),
        "final_loss_mask": restored.new_tensor([1.0, 0.0]).view(2, 1, 1, 1),
    }
    loss, _ = SGBFormerLoss(
        structural_weight=0,
        bfn_weight=0,
        final_weight=1,
        final_loss_type="mse",
    )(outputs, target)
    assert loss.item() == pytest.approx(0.25)
    loss.backward()
    assert restored.grad[0].abs().sum() > 0
    assert restored.grad[1].abs().sum() == 0


def test_amplitude_loss_does_not_measure_spatial_alignment() -> None:
    torch.manual_seed(5)
    target = torch.rand(1, 3, 16, 16)
    shifted = target.roll(1, dims=-1)
    assert frequency_amplitude_loss(shifted, target) < 1e-6
    assert (shifted - target).square().mean() > 0.01


def test_latent_category_loss_is_explicit_and_weighted() -> None:
    prediction = torch.zeros(2, 3, 4, 4, requires_grad=True)
    target = torch.zeros_like(prediction)
    logits = torch.tensor(
        [[2.0, 0.0, -1.0], [-1.0, 0.0, 2.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 2])
    outputs = {
        "structural": prediction,
        "restored": prediction,
        "bfn_loss": prediction.new_zeros(()),
        "latent_category_logits": logits,
        "latent_category_target": labels,
    }
    criterion = SGBFormerLoss(
        structural_weight=0,
        bfn_weight=0,
        latent_category_weight=0.1,
    )
    loss, parts = criterion(outputs, target)
    expected = 0.1 * torch.nn.functional.cross_entropy(logits, labels)
    torch.testing.assert_close(loss, expected)
    assert parts["latent_category"] > 0
    loss.backward()
    assert logits.grad is not None
