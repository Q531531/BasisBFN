from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import torch

from sgbformer.losses import SGBFormerLoss


def load_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_v27_residual_geometry.py"
    )
    spec = spec_from_file_location("_audit_v27_residual_geometry", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_signed_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_v27_signed_coefficient_oracle.py"
    )
    spec = spec_from_file_location("_audit_v27_signed_coefficients", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safety_report(
    *,
    snow: float = 0.01,
    outdoor: float = 0.0,
    raindrop: float = 0.0,
) -> dict:
    return {
        "categories": {
            "snow": {
                "psnr": {
                    "delta": snow,
                    "paired_image_bootstrap_ci95": [snow - 0.001, snow + 0.001],
                }
            },
            "outdoor_rain": {
                "psnr": {
                    "delta": outdoor,
                    "paired_image_bootstrap_ci95": [
                        outdoor - 0.001,
                        outdoor + 0.001,
                    ],
                }
            },
            "raindrop": {
                "psnr": {
                    "delta": raindrop,
                    "paired_image_bootstrap_ci95": [
                        raindrop - 0.001,
                        raindrop + 0.001,
                    ],
                }
            },
        }
    }


def attribution_summary(
    *,
    geometry_error: float = 0.0,
    alpha_zero_error: float = 0.0,
    alpha_one_error: float = 0.0,
    snow_alpha: float = 0.75,
    outdoor_alpha: float = 0.75,
    raindrop_alpha: float = 0.75,
    category_gain: float = 0.0,
    global_gain: float = 0.0,
    global_report: dict | None = None,
    current_vs_y: float = 0.8,
    correction_vs_current: float = 0.1,
) -> dict:
    gain = {"macro_weighted": {"psnr": {"delta": category_gain}}}
    return {
        "geometry": {
            "maximum_identity_relative_error": geometry_error,
        },
        "endpoint_reproduction": {
            "alpha_zero_maximum_psnr_error": alpha_zero_error,
            "alpha_one_maximum_metric_error": alpha_one_error,
        },
        "scale_sweep": {
            "best_fixed_alpha_by_category": {
                "snow": {"alpha": snow_alpha},
                "outdoor_rain": {"alpha": outdoor_alpha},
                "raindrop": {"alpha": raindrop_alpha},
            },
            "category_specific_policy": {
                "gain_vs_learned": gain,
            },
            "best_global_fixed_alpha": {
                "gain_vs_learned_macro_psnr": global_gain,
                "analysis": global_report or safety_report(),
            },
        },
        "continuous_mixture": {
            "category_oracle": {
                "gain_vs_learned": {
                    "macro_weighted": {"psnr": {"delta": 0.0}}
                }
            },
            "per_image_oracle": {
                "gain_vs_learned": {
                    "macro_weighted": {"psnr": {"delta": 0.0}}
                }
            },
        },
        "loss_alignment": {
            "overall": {
                "negative_gradient_vs_y_psnr_direction": {
                    "current_total": {"mean": current_vs_y}
                },
                "correction_vs_current_gradient_cosine": {
                    "mean": correction_vs_current
                },
            }
        },
    }


def test_simplex_grid_has_all_231_convex_combinations() -> None:
    module = load_script()
    weights = module.simplex_grid(0.05)

    assert weights.shape == (231, 3)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    assert len({tuple(row) for row in weights}) == 231
    assert (0.0, 0.0, 1.0) in {tuple(row) for row in weights}
    assert (1.0, 0.0, 0.0) in {tuple(row) for row in weights}


def test_signed_l1_grid_is_bounded_and_contains_negative_basis_weights() -> None:
    module = load_signed_script()
    weights = module.signed_l1_grid()

    assert weights.shape == (129, 3)
    assert len({tuple(row) for row in weights}) == 129
    assert np.all(np.abs(weights).sum(axis=1) <= 1.0)
    assert (0.0, 0.0, 0.0) in {tuple(row) for row in weights}
    assert (-1.0, 0.0, 0.0) in {tuple(row) for row in weights}
    assert (0.0, 0.0, 1.0) in {tuple(row) for row in weights}


def test_signed_oracle_recommendation_requires_meaningful_upper_bound() -> None:
    module = load_signed_script()

    def gain(value: float) -> dict:
        return {"macro_weighted": {"psnr": {"delta": value}}}

    insufficient = module.determine_recommendation(
        safety_report(),
        gain(0.01),
        gain(0.049),
    )
    useful = module.determine_recommendation(
        safety_report(),
        gain(0.01),
        gain(0.06),
    )
    conditional = module.determine_recommendation(
        safety_report(outdoor=-0.01),
        gain(0.01),
        gain(0.06),
    )

    assert insufficient["code"] == "S3"
    assert useful["code"] == "S1"
    assert conditional["code"] == "S2"


def test_geometry_identity_uses_effective_y_space_correction() -> None:
    module = load_script()
    generator = torch.Generator().manual_seed(42)
    baseline = torch.rand(1, 1, 8, 8, generator=generator).double()
    target = torch.rand(1, 1, 8, 8, generator=generator).double()
    candidate = (
        baseline
        + 0.05 * torch.randn(1, 1, 8, 8, generator=generator).double()
    )
    result = module.geometry_terms(baseline, candidate, target)

    assert result["identity_relative_error"] < 1e-12
    assert result["delta_mse"] == pytest.approx(
        result["identity_rhs"],
        abs=1e-14,
    )
    expected_alpha = (
        ((target - baseline) * (candidate - baseline)).sum()
        / (candidate - baseline).square().sum()
    )
    assert result["alpha_star"] == pytest.approx(float(expected_alpha))


def test_loss_gradients_distinguish_rgb_training_from_y_metric() -> None:
    module = load_script()
    generator = torch.Generator().manual_seed(7)
    baseline = torch.rand(1, 3, 8, 8, generator=generator)
    target = torch.rand(1, 3, 8, 8, generator=generator)
    prediction = baseline + 0.01 * torch.randn(
        1,
        3,
        8,
        8,
        generator=generator,
    )
    criterion = SGBFormerLoss(
        structural_weight=0.0,
        bfn_weight=0.0,
        final_weight=1.0,
        final_loss_type="mse",
    )
    result = module.loss_gradient_diagnostics(
        prediction,
        baseline,
        target,
        criterion,
    )

    assert result["current_gradient_vs_residual_cosine"] > 0.99
    assert result["current_vs_rgb_mse_gradient_cosine"] == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert result["negative_gradient_vs_y_psnr_direction"][
        "y_channel_mse"
    ] == pytest.approx(1.0, abs=1e-6)
    assert result["gradient_rms"]["current_structural_loss"] == 0.0


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (attribution_summary(geometry_error=1e-3), "E5"),
        (
            attribution_summary(
                snow_alpha=0.75,
                outdoor_alpha=0.0,
                raindrop_alpha=0.125,
                category_gain=0.01,
            ),
            "E2",
        ),
        (
            attribution_summary(
                global_gain=0.01,
                global_report=safety_report(),
            ),
            "E1",
        ),
        (
            attribution_summary(
                current_vs_y=0.3,
                correction_vs_current=0.4,
            ),
            "E3",
        ),
        (
            attribution_summary(
                current_vs_y=0.3,
                correction_vs_current=0.05,
            ),
            "E4",
        ),
    ],
)
def test_geometry_audit_assigns_one_primary_attribution(
    summary: dict,
    expected: str,
) -> None:
    module = load_script()
    result = module.determine_attribution(summary)
    assert result["code"] == expected
