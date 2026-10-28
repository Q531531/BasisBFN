"""Training objectives for SGBformer."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(
            (prediction - target).square() + self.epsilon**2
        ).mean()


def pearson_loss(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = prediction.float().flatten(1)
    target = target.float().flatten(1)
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (prediction * target).sum(dim=1)
    prediction_energy = prediction.square().sum(dim=1)
    target_energy = target.square().sum(dim=1)
    denominator = (
        prediction_energy.clamp_min(1e-8).sqrt()
        * target_energy.clamp_min(1e-8).sqrt()
    )
    # Correlation is undefined for constant targets; pixel loss still supervises them.
    valid = (target_energy > 1e-8).to(prediction.dtype)
    return ((1.0 - numerator / denominator) * valid).sum() / valid.sum().clamp_min(1)


def frequency_amplitude_loss(prediction: Tensor, target: Tensor) -> Tensor:
    prediction_spectrum = torch.fft.rfft2(
        prediction.float(), norm="ortho"
    ).abs()
    target_spectrum = torch.fft.rfft2(
        target.float(), norm="ortho"
    ).abs()
    return F.l1_loss(
        prediction_spectrum.log1p(),
        target_spectrum.log1p(),
    )


class BasisBFNLoss(nn.Module):
    """Joint RGB reconstruction and coefficient-space BFN objective.

    The paper reports Equation (11) for BFN refinement but omits the
    structural-stage objective and coefficients. The configurable defaults
    use a Charbonnier structural loss and the exact BFN term returned by the
    model; final-image supervision is optional.
    """

    def __init__(
        self,
        structural_weight: float = 1.0,
        bfn_weight: float = 1.0,
        final_weight: float = 0.0,
        correlation_weight: float = 0.0,
        frequency_weight: float = 0.0,
        adapter_weight: float = 0.0,
        uncertainty_weight: float = 0.0,
        latent_category_weight: float = 0.0,
        charbonnier_epsilon: float = 1e-3,
        structural_loss_type: str = "charbonnier",
        final_loss_type: str = "l1",
    ) -> None:
        super().__init__()
        self.structural_weight = structural_weight
        self.bfn_weight = bfn_weight
        self.final_weight = final_weight
        self.correlation_weight = correlation_weight
        self.frequency_weight = frequency_weight
        self.adapter_weight = adapter_weight
        self.uncertainty_weight = uncertainty_weight
        self.latent_category_weight = latent_category_weight
        self.charbonnier_epsilon = charbonnier_epsilon
        if structural_loss_type not in {"charbonnier", "mse"}:
            raise ValueError("structural_loss_type must be charbonnier or mse")
        if final_loss_type not in {"l1", "charbonnier", "mse"}:
            raise ValueError(
                "final_loss_type must be l1, charbonnier, or mse"
            )
        self.structural_loss_type = structural_loss_type
        self.final_loss_type = final_loss_type
        self.pixel_loss = (
            nn.MSELoss()
            if structural_loss_type == "mse"
            else CharbonnierLoss(charbonnier_epsilon)
        )
        self.final_pixel_loss = {
            "l1": nn.L1Loss(),
            "charbonnier": CharbonnierLoss(charbonnier_epsilon),
            "mse": nn.MSELoss(),
        }[final_loss_type]

    def final_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        prediction = prediction.float()
        target = target.float()
        if valid_mask is None:
            return self.final_pixel_loss(prediction, target)
        if self.final_loss_type == "mse":
            values = (prediction - target).square()
        elif self.final_loss_type == "l1":
            values = (prediction - target).abs()
        else:
            values = torch.sqrt(
                (prediction - target).square()
                + self.charbonnier_epsilon**2
            )
        valid = valid_mask.to(device=values.device, dtype=values.dtype)
        valid = valid.expand_as(values)
        return (values * valid).sum() / valid.sum().clamp_min(1.0)

    def forward(
        self, outputs: dict[str, Tensor], target: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        prediction = outputs["structural"]
        structural = (
            self.pixel_loss(prediction.float(), target.float())
            if self.structural_loss_type == "mse"
            else self.pixel_loss(prediction, target)
        )
        bfn = outputs["bfn_loss"]
        final = (
            self.final_loss(
                outputs["restored"],
                target,
                outputs.get("final_loss_mask"),
            )
            if self.final_weight > 0
            else target.new_zeros(())
        )
        correlation = (
            pearson_loss(outputs["structural"], target)
            if self.correlation_weight > 0
            else target.new_zeros(())
        )
        frequency = (
            frequency_amplitude_loss(outputs["structural"], target)
            if self.frequency_weight > 0
            else target.new_zeros(())
        )
        adapter = outputs.get(
            "adapter_regularization", target.new_zeros(())
        )
        uncertainty = outputs.get(
            "uncertainty_calibration", target.new_zeros(())
        )
        if self.latent_category_weight > 0:
            category_logits = outputs.get("latent_category_logits")
            category_target = outputs.get("latent_category_target")
            if category_logits is None or category_target is None:
                raise ValueError(
                    "Latent category supervision requires logits and labels"
                )
            category = F.cross_entropy(
                category_logits.float(),
                category_target.long(),
            )
        else:
            category = target.new_zeros(())
        total = (
            self.structural_weight * structural
            + self.bfn_weight * bfn
            + self.final_weight * final
            + self.correlation_weight * correlation
            + self.frequency_weight * frequency
            + self.adapter_weight * adapter
            + self.uncertainty_weight * uncertainty
            + self.latent_category_weight * category
        )
        return total, {
            "total": total.detach(),
            "structural": structural.detach(),
            "bfn": bfn.detach(),
            "final": final.detach(),
            "correlation": correlation.detach(),
            "frequency": frequency.detach(),
            "adapter": adapter.detach(),
            "uncertainty": uncertainty.detach(),
            "latent_category": category.detach(),
        }


SGBFormerLoss = BasisBFNLoss
