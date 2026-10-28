"""Image-restoration metrics."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _crop_border(image: Tensor, crop_border: int) -> Tensor:
    if crop_border < 0:
        raise ValueError("crop_border cannot be negative")
    if crop_border == 0:
        return image
    if min(image.shape[-2:]) <= crop_border * 2:
        raise ValueError("crop_border removes the complete image")
    return image[
        ..., crop_border:-crop_border, crop_border:-crop_border
    ]


def rgb_to_matlab_y(image: Tensor) -> Tensor:
    """Convert RGB [0, 1] tensors to MATLAB-compatible Y in [0, 1]."""

    if image.shape[-3] != 3:
        raise ValueError("Y-channel evaluation expects three-channel RGB input")
    coefficients = image.new_tensor((65.481, 128.553, 24.966))
    y_channel = (
        image.movedim(-3, -1) * coefficients
    ).sum(dim=-1, keepdim=True)
    return (y_channel + 16.0).div(255.0).movedim(-1, -3)


def _prepare_images(
    prediction: Tensor,
    target: Tensor,
    *,
    crop_border: int,
    test_y_channel: bool,
) -> tuple[Tensor, Tensor]:
    prediction = _crop_border(prediction.clamp(0, 1), crop_border)
    target = _crop_border(target.clamp(0, 1), crop_border)
    if test_y_channel:
        prediction = rgb_to_matlab_y(prediction)
        target = rgb_to_matlab_y(target)
    return prediction, target


def psnr(
    prediction: Tensor,
    target: Tensor,
    *,
    crop_border: int = 0,
    test_y_channel: bool = False,
) -> Tensor:
    prediction, target = _prepare_images(
        prediction,
        target,
        crop_border=crop_border,
        test_y_channel=test_y_channel,
    )
    mse = (prediction - target).square().mean(
        dim=(-3, -2, -1)
    )
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def ssim(
    prediction: Tensor,
    target: Tensor,
    window_size: int = 11,
    *,
    crop_border: int = 0,
    test_y_channel: bool = False,
) -> Tensor:
    """Compute BasicSR-compatible SSIM for BCHW tensors in [0, 1]."""

    prediction, target = _prepare_images(
        prediction,
        target,
        crop_border=crop_border,
        test_y_channel=test_y_channel,
    )
    size = min(window_size, prediction.shape[-2], prediction.shape[-1])
    if size % 2 == 0:
        size -= 1
    size = max(size, 1)
    coordinates = torch.arange(
        size, device=prediction.device, dtype=prediction.dtype
    ) - size // 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
    kernel = kernel.repeat(prediction.shape[1], 1, 1, 1)

    def filter_image(image: Tensor) -> Tensor:
        return F.conv2d(image, kernel, groups=prediction.shape[1])

    mean_x = filter_image(prediction)
    mean_y = filter_image(target)
    variance_x = filter_image(prediction.square()) - mean_x.square()
    variance_y = filter_image(target.square()) - mean_y.square()
    covariance = filter_image(prediction * target) - mean_x * mean_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1)
        * (variance_x + variance_y + c2)
    ).clamp_min(1e-12)
    return score.mean(dim=(-3, -2, -1))


@torch.no_grad()
def uncertainty_scores(
    variance: Tensor, prediction: Tensor, target: Tensor, bins: int = 10
) -> dict[str, Tensor]:
    """RGB Gaussian log score and equal-count calibration error per image.

    The constant-variance score uses the image's predicted mean, not its GT
    error, so it tests whether spatial risk information helps.
    """
    if bins < 1 or variance.shape != prediction[:, :1].shape:
        raise ValueError("Expected positive bins and B1HW variance at image resolution")
    error = (prediction.float() - target.float()).square().mean(dim=1).flatten(1)
    variance = variance.float().clamp_min(1e-8).flatten(1)
    uniform = variance.mean(dim=1, keepdim=True)
    nll = 0.5 * (error / variance + variance.log()).mean(dim=1)
    uniform_nll = 0.5 * (error / uniform + uniform.log()).mean(dim=1)
    calibration = []
    for predicted, observed in zip(variance, error):
        order = predicted.argsort()
        value = predicted.new_zeros(())
        for indices in torch.tensor_split(order, min(bins, order.numel())):
            value += (
                predicted[indices].mean() - observed[indices].mean()
            ).abs() * indices.numel() / order.numel()
        calibration.append(value)
    return {
        "risk_nll": nll,
        "risk_uniform_nll": uniform_nll,
        "risk_spatial_nll_delta": nll - uniform_nll,
        "risk_calibration_mae": torch.stack(calibration),
        "risk_mean_variance": variance.mean(dim=1),
        "risk_observed_mse": error.mean(dim=1),
    }


class LPIPSMetric(nn.Module):
    def __init__(self, network: str = "alex") -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS evaluation requires `pip install lpips`."
            ) from exc
        self.metric = lpips.LPIPS(net=network).eval()
        self.metric.requires_grad_(False)

    @torch.no_grad()
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.metric(prediction * 2 - 1, target * 2 - 1).flatten()
