import pytest
import torch

from sgbformer.metrics import psnr, rgb_to_matlab_y, ssim, uncertainty_scores


def test_matlab_y_conversion_matches_reference_coefficients() -> None:
    image = torch.zeros(1, 3, 12, 12)
    image[:, 0] = 1.0
    expected = (65.481 + 16.0) / 255.0
    converted = rgb_to_matlab_y(image)
    assert converted.shape == (1, 1, 12, 12)
    assert float(converted.mean()) == pytest.approx(expected)


def test_all_weather_metrics_support_y_channel_and_crop() -> None:
    target = torch.zeros(1, 3, 16, 16)
    prediction = target.clone()
    prediction[:, 0, 0] = 1.0

    uncropped = psnr(prediction, target, test_y_channel=True)
    cropped = psnr(
        prediction,
        target,
        crop_border=1,
        test_y_channel=True,
    )
    assert float(cropped) > float(uncropped)
    assert torch.isfinite(
        ssim(prediction, target, test_y_channel=True)
    ).all()


def test_crop_border_rejects_empty_image() -> None:
    image = torch.zeros(1, 3, 4, 4)
    with pytest.raises(ValueError, match="complete image"):
        psnr(image, image, crop_border=2)


def test_uncertainty_scores_prefer_calibrated_spatial_variance() -> None:
    target = torch.zeros(1, 3, 10, 10)
    prediction = torch.full_like(target, 0.02)
    prediction[..., 5:, :] = 0.2
    variance = prediction.square().mean(dim=1, keepdim=True)
    scores = uncertainty_scores(variance, prediction, target)
    assert scores["risk_calibration_mae"].item() == pytest.approx(0, abs=1e-8)
    assert scores["risk_spatial_nll_delta"].item() < 0
    torch.testing.assert_close(scores["risk_mean_variance"], scores["risk_observed_mse"])
    wrong = uncertainty_scores(variance * 4, prediction, target)
    assert wrong["risk_nll"] > scores["risk_nll"]
