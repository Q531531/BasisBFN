"""Regression coverage for reusable checkpoint-compatible readouts."""

import pytest
import torch
from torch.nn import functional as F

from sgbformer.models.receiver import CategoryRoutedReadout
from sgbformer.models import SignedSpatialBasisReadout


@pytest.mark.parametrize("raw_gate,expected_gate,expected_gradient", [
    (-0.5, 0.0, 0.0), (-0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
    (0.5, 0.5, 1.0), (1.0, 1.0, 1.0), (1.5, 1.0, 0.0),
])
def test_routed_gate_boundary_and_saturation_gradients(raw_gate, expected_gate, expected_gradient):
    readout = CategoryRoutedReadout(
        16, 3, ("snow", "outdoor_rain", "raindrop"),
        acceptance_gate={"enabled": True, "freeze_base_receiver": True},
    )
    with torch.no_grad():
        for head in readout.heads.values():
            head.output.weight.zero_()
            head.output.bias.fill_(1.0)
        readout.acceptance_gate.bias.fill_(raw_gate)
    output = readout(torch.ones(1, 16, 2, 2))
    torch.testing.assert_close(output, torch.full_like(output, expected_gate), atol=1e-7, rtol=0)
    output.mean().backward()
    tolerance = 2 * torch.finfo(output.dtype).eps
    torch.testing.assert_close(
        readout.acceptance_gate.bias.grad, torch.tensor([expected_gradient]),
        atol=tolerance, rtol=0,
    )
    torch.testing.assert_close(
        readout.acceptance_gate.weight.grad,
        torch.full_like(readout.acceptance_gate.weight, expected_gradient),
        atol=tolerance, rtol=0,
    )


def test_signed_basis_identity_initialization_and_gradient_opening():
    torch.manual_seed(1)
    readout = SignedSpatialBasisReadout(16, 3)
    torch.manual_seed(999)
    other = SignedSpatialBasisReadout(16, 3)
    torch.testing.assert_close(readout.basis_projection.weight, other.basis_projection.weight, atol=0, rtol=0)
    feature = torch.rand(2, 16, 8, 8, requires_grad=True)
    target = torch.rand(2, 3, 8, 8)
    output = readout(feature)
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)
    F.mse_loss(output, target).backward()
    assert torch.count_nonzero(readout.coefficient_projection.weight.grad) > 0
    assert torch.count_nonzero(readout.basis_projection.weight.grad) == 0
    assert torch.count_nonzero(feature.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(readout(feature), target).backward()
    assert torch.count_nonzero(readout.basis_projection.weight.grad) > 0
    assert torch.count_nonzero(feature.grad) > 0
