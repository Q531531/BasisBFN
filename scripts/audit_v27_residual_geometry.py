#!/usr/bin/env python3
"""Audit fixed epoch-3 residual geometry without training or checkpoint changes."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_dev_by_category import (
    DEFAULT_CATEGORY_PATTERNS,
    DEFAULT_CATEGORY_WEIGHTS,
    read_name_map,
)
from probe_v27_c0prime_routing import (
    CATEGORIES,
    analyze_variant,
    compare_variants,
    cosine,
    evaluation_payload,
    macro_delta,
    metric_row,
    tensor_state_sha256,
)
from sgbformer.config import load_config
from sgbformer.data import PairedImageDataset, classify_paths
from sgbformer.inference import load_model_for_inference
from sgbformer.losses import SGBFormerLoss
from sgbformer.metrics import rgb_to_matlab_y
from sgbformer.models import CategoryRoutedReadout
from sgbformer.protocol import implementation_fingerprint, sha256_file
from sgbformer.utils import resolve_device, seed_everything
from sgbformer.v27 import (
    V27_C0DOUBLEPRIME_ARM,
    V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS,
    V27_EXPECTED_DEV_PAIRS,
    resolved_config_sha256,
    validate_v27_c0doubleprime_config,
)


ALPHAS = (
    -1.0,
    -0.5,
    -0.25,
    0.0,
    0.125,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
)
MIXTURE_STEP = 0.05
GEOMETRY_RELATIVE_TOLERANCE = 1e-5
METRIC_REPRODUCTION_TOLERANCE = 1e-4
MATERIAL_GAIN_DB = 0.003
PER_IMAGE_GAIN_DB = 0.005
LOSS_METRIC_MISMATCH_COSINE = 0.5
CORRECTION_FOLLOWS_LOSS_COSINE = 0.2
CHARBONNIER_EPSILON = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--registered-evaluation", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--pairs-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "docs"
            / "v27_c0doubleprime_residual_geometry.md"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixture-chunk-size", type=int, default=21)
    return parser.parse_args()


def alpha_key(alpha: float) -> str:
    return f"{alpha:g}"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def simplex_grid(step: float = MIXTURE_STEP) -> np.ndarray:
    divisions = round(1.0 / step)
    if divisions < 1 or not math.isclose(
        divisions * step,
        1.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Simplex step must divide one exactly")
    weights = [
        (left / divisions, middle / divisions, right / divisions)
        for left in range(divisions + 1)
        for middle in range(divisions - left + 1)
        for right in (divisions - left - middle,)
    ]
    return np.asarray(weights, dtype=np.float64)


def psnr_from_mse(mse: Tensor) -> Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def geometry_terms(
    baseline_y: Tensor,
    candidate_y: Tensor,
    target_y: Tensor,
) -> dict[str, float | bool]:
    baseline_y = baseline_y.detach().double()
    candidate_y = candidate_y.detach().double()
    target_y = target_y.detach().double()
    residual = target_y - baseline_y
    correction = candidate_y - baseline_y
    baseline_error = residual.square().mean()
    candidate_error = (residual - correction).square().mean()
    correction_energy = correction.square().mean()
    projection = (residual * correction).mean()
    delta_error = candidate_error - baseline_error
    identity_rhs = correction_energy - 2.0 * projection
    identity_absolute_error = (delta_error - identity_rhs).abs()
    identity_relative_error = identity_absolute_error / max(
        float(delta_error.abs()),
        float(identity_rhs.abs()),
        float(baseline_error.abs()),
        1e-12,
    )
    residual_rms = residual.square().mean().sqrt()
    correction_rms = correction.square().mean().sqrt()
    return {
        "baseline_mse": float(baseline_error),
        "candidate_mse": float(candidate_error),
        "delta_mse": float(delta_error),
        "identity_rhs": float(identity_rhs),
        "identity_absolute_error": float(identity_absolute_error),
        "identity_relative_error": float(identity_relative_error),
        "residual_rms": float(residual_rms),
        "correction_rms": float(correction_rms),
        "correction_residual_rms_ratio": float(
            correction_rms / residual_rms.clamp_min(1e-12)
        ),
        "cosine": cosine(correction, residual),
        "projection_mean": float(projection),
        "correction_energy_mean": float(correction_energy),
        "beneficial": bool(2.0 * projection > correction_energy),
        "alpha_star": float(
            projection / correction_energy.clamp_min(1e-12)
        ),
        "pixel_count": int(residual.numel()),
        "projection_sum": float((residual * correction).sum()),
        "correction_energy_sum": float(correction.square().sum()),
        "residual_energy_sum": float(residual.square().sum()),
    }


def quantile_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "max": None,
        }
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "max": float(array.max()),
    }


def summarize_geometry(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        projection_sum = sum(
            float(record["geometry"]["projection_sum"])
            for record in group
        )
        correction_energy_sum = sum(
            float(record["geometry"]["correction_energy_sum"])
            for record in group
        )
        residual_energy_sum = sum(
            float(record["geometry"]["residual_energy_sum"])
            for record in group
        )
        pixel_count = sum(
            int(record["geometry"]["pixel_count"])
            for record in group
        )
        alpha_star = projection_sum / max(correction_energy_sum, 1e-12)
        return {
            "n": len(group),
            "beneficial_fraction": float(
                np.mean(
                    [
                        float(record["geometry"]["beneficial"])
                        for record in group
                    ]
                )
            ),
            "cosine": quantile_summary(
                [float(record["geometry"]["cosine"]) for record in group]
            ),
            "correction_residual_rms_ratio": quantile_summary(
                [
                    float(
                        record["geometry"][
                            "correction_residual_rms_ratio"
                        ]
                    )
                    for record in group
                ]
            ),
            "projection_mean_per_y_value": (
                projection_sum / max(pixel_count, 1)
            ),
            "correction_energy_mean_per_y_value": (
                correction_energy_sum / max(pixel_count, 1)
            ),
            "residual_energy_mean_per_y_value": (
                residual_energy_sum / max(pixel_count, 1)
            ),
            "delta_mse_mean": float(
                np.mean(
                    [
                        float(record["geometry"]["delta_mse"])
                        for record in group
                    ]
                )
            ),
            "delta_psnr_mean": float(
                np.mean(
                    [
                        float(record["geometry"]["delta_psnr"])
                        for record in group
                    ]
                )
            ),
            "analytic_alpha_star_unclamped": alpha_star,
            "analytic_minimum_delta_mse_unclamped": (
                -projection_sum**2
                / max(correction_energy_sum * max(pixel_count, 1), 1e-12)
            ),
        }

    return {
        "overall": summarize(records),
        "by_category": {
            category: summarize(
                [
                    record
                    for record in records
                    if record["category"] == category
                ]
            )
            for category in CATEGORIES
        },
    }


def negative_gradient(
    prediction: Tensor,
    loss_builder: Callable[[Tensor], Tensor],
) -> Tensor:
    variable = prediction.detach().clone().requires_grad_(True)
    loss = loss_builder(variable)
    gradient = torch.autograd.grad(
        loss,
        variable,
        retain_graph=False,
        create_graph=False,
    )[0]
    return -gradient.detach()


def loss_gradient_diagnostics(
    raw_prediction: Tensor,
    baseline: Tensor,
    target: Tensor,
    criterion: SGBFormerLoss,
) -> dict[str, Any]:
    raw_prediction = raw_prediction.detach().float()
    baseline = baseline.detach().float()
    target = target.detach().float()
    residual = target - baseline
    correction = raw_prediction - baseline
    zero = target.new_zeros(())

    def current_loss(prediction: Tensor) -> Tensor:
        outputs = {
            "restored": prediction,
            "structural": baseline,
            "bfn_loss": zero,
            "adapter_regularization": zero,
            "uncertainty_calibration": zero,
        }
        return criterion(outputs, target)[0]

    def rgb_mse(prediction: Tensor) -> Tensor:
        return F.mse_loss(prediction, target)

    def y_mse(prediction: Tensor) -> Tensor:
        return F.mse_loss(
            rgb_to_matlab_y(prediction.clamp(0, 1)),
            rgb_to_matlab_y(target.clamp(0, 1)),
        )

    def charbonnier(prediction: Tensor) -> Tensor:
        return torch.sqrt(
            (prediction - target).square() + CHARBONNIER_EPSILON**2
        ).mean()

    def residual_cosine(prediction: Tensor) -> Tensor:
        predicted = (prediction - baseline).flatten()
        wanted = residual.flatten()
        return 1.0 - (
            torch.dot(predicted, wanted)
            / (
                predicted.norm().clamp_min(1e-12)
                * wanted.norm().clamp_min(1e-12)
            )
        )

    def projected_residual(prediction: Tensor) -> Tensor:
        return F.mse_loss(prediction - baseline, residual)

    builders = {
        "current_total": current_loss,
        "rgb_mse": rgb_mse,
        "y_channel_mse": y_mse,
        "charbonnier": charbonnier,
        "residual_cosine": residual_cosine,
        "projected_residual": projected_residual,
    }
    gradients = {
        name: negative_gradient(raw_prediction, builder)
        for name, builder in builders.items()
    }
    gradients["current_structural_loss"] = torch.zeros_like(raw_prediction)
    y_direction = gradients["y_channel_mse"]
    current = gradients["current_total"]
    return {
        "current_gradient_vs_residual_cosine": cosine(current, residual),
        "correction_vs_current_gradient_cosine": cosine(correction, current),
        "correction_vs_residual_cosine": cosine(correction, residual),
        "current_vs_rgb_mse_gradient_cosine": cosine(
            current,
            gradients["rgb_mse"],
        ),
        "negative_gradient_vs_y_psnr_direction": {
            name: cosine(gradient, y_direction)
            for name, gradient in gradients.items()
        },
        "gradient_rms": {
            name: float(gradient.square().mean().sqrt())
            for name, gradient in gradients.items()
        },
    }


def summarize_loss_alignment(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    scalar_fields = (
        "current_gradient_vs_residual_cosine",
        "correction_vs_current_gradient_cosine",
        "correction_vs_residual_cosine",
        "current_vs_rgb_mse_gradient_cosine",
    )
    gradient_names = tuple(
        records[0]["loss_alignment"][
            "negative_gradient_vs_y_psnr_direction"
        ]
    )

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            **{
                field: quantile_summary(
                    [
                        float(record["loss_alignment"][field])
                        for record in group
                    ]
                )
                for field in scalar_fields
            },
            "negative_gradient_vs_y_psnr_direction": {
                name: quantile_summary(
                    [
                        float(
                            record["loss_alignment"][
                                "negative_gradient_vs_y_psnr_direction"
                            ][name]
                        )
                        for record in group
                    ]
                )
                for name in gradient_names
            },
            "gradient_rms": {
                name: quantile_summary(
                    [
                        float(
                            record["loss_alignment"]["gradient_rms"][name]
                        )
                        for record in group
                    ]
                )
                for name in gradient_names
            },
        }

    return {
        "overall": summarize(records),
        "by_category": {
            category: summarize(
                [
                    record
                    for record in records
                    if record["category"] == category
                ]
            )
            for category in CATEGORIES
        },
    }


def selected_correction_geometry(
    selected_indices: np.ndarray,
    dot: np.ndarray,
    correction_energy: np.ndarray,
    residual_energy: np.ndarray,
) -> dict[str, float]:
    rows = np.arange(len(selected_indices))
    projection = float(dot[rows, selected_indices].sum())
    correction = float(correction_energy[rows, selected_indices].sum())
    residual = float(residual_energy.sum())
    return {
        "correction_target_cosine": (
            projection / math.sqrt(max(correction * residual, 1e-24))
        ),
        "projection_sum": projection,
        "correction_energy_sum": correction,
        "residual_energy_sum": residual,
    }


def category_safety(report: dict[str, Any]) -> bool:
    snow = report["categories"]["snow"]["psnr"]
    outdoor = report["categories"]["outdoor_rain"]["psnr"]
    raindrop = report["categories"]["raindrop"]["psnr"]
    return bool(
        (
            snow["delta"] >= 0
            or (
                snow["delta"] >= -0.005
                and snow["paired_image_bootstrap_ci95"][1] >= 0
            )
        )
        and outdoor["delta"] >= -0.003
        and outdoor["paired_image_bootstrap_ci95"][1] >= 0
        and raindrop["delta"] >= -0.005
        and raindrop["paired_image_bootstrap_ci95"][1] >= 0
    )


def determine_attribution(summary: dict[str, Any]) -> dict[str, Any]:
    geometry = summary["geometry"]
    endpoints = summary["endpoint_reproduction"]
    scale = summary["scale_sweep"]
    mixture = summary["continuous_mixture"]
    loss = summary["loss_alignment"]["overall"]
    checks = {
        "geometry_identity": (
            geometry["maximum_identity_relative_error"]
            < GEOMETRY_RELATIVE_TOLERANCE
        ),
        "alpha_zero_reproduces_f0": (
            endpoints["alpha_zero_maximum_psnr_error"]
            <= METRIC_REPRODUCTION_TOLERANCE
        ),
        "alpha_one_reproduces_candidate": (
            endpoints["alpha_one_maximum_metric_error"]
            <= METRIC_REPRODUCTION_TOLERANCE
        ),
    }
    if not all(checks.values()):
        code = "E5"
        conclusion = "Implementation/coordinate mismatch"
        next_experiment = "Fix the implementation before training a new model."
    else:
        best = scale["best_fixed_alpha_by_category"]
        category_gain = float(
            scale["category_specific_policy"]["gain_vs_learned"][
                "macro_weighted"
            ]["psnr"]["delta"]
        )
        snow_alpha = float(best["snow"]["alpha"])
        rain_alphas = [
            float(best["outdoor_rain"]["alpha"]),
            float(best["raindrop"]["alpha"]),
        ]
        category_accept_reject = bool(
            category_gain >= MATERIAL_GAIN_DB
            and snow_alpha >= 0.25
            and (
                max(rain_alphas) <= 0.25
                or snow_alpha - max(rain_alphas) >= 0.5
            )
        )
        global_scale = scale["best_global_fixed_alpha"]
        global_scale_fix = bool(
            float(global_scale["gain_vs_learned_macro_psnr"])
            >= MATERIAL_GAIN_DB
            and category_safety(global_scale["analysis"])
        )
        current_vs_y = float(
            loss["negative_gradient_vs_y_psnr_direction"][
                "current_total"
            ]["mean"]
        )
        correction_vs_current = float(
            loss["correction_vs_current_gradient_cosine"]["mean"]
        )
        if category_accept_reject:
            code = "E2"
            conclusion = "Category-dependent accept/reject mismatch"
            next_experiment = (
                "Test one identity-safe correction acceptance gate as the "
                "only behavioral variable."
            )
        elif global_scale_fix:
            code = "E1"
            conclusion = "Correction scale/calibration mismatch"
            next_experiment = (
                "Test one fixed scalar correction calibration as the only "
                "behavioral variable."
            )
        elif (
            current_vs_y < LOSS_METRIC_MISMATCH_COSINE
            and correction_vs_current >= CORRECTION_FOLLOWS_LOSS_COSINE
        ):
            code = "E3"
            conclusion = "Loss/metric mismatch"
            next_experiment = (
                "Test one metric-aligned residual objective as the only "
                "behavioral variable."
            )
        else:
            code = "E4"
            conclusion = "Receiver representation/optimization mismatch"
            next_experiment = (
                "Test one redesigned residual basis/readout as the only "
                "behavioral variable."
            )
    return {
        "code": code,
        "conclusion": conclusion,
        "coordinate_checks": checks,
        "next_single_variable_experiment": next_experiment,
        "continuous_category_mixture_gain_db": float(
            mixture["category_oracle"]["gain_vs_learned"][
                "macro_weighted"
            ]["psnr"]["delta"]
        ),
        "continuous_per_image_mixture_gain_db": float(
            mixture["per_image_oracle"]["gain_vs_learned"][
                "macro_weighted"
            ]["psnr"]["delta"]
        ),
        "thresholds": {
            "geometry_relative_error": GEOMETRY_RELATIVE_TOLERANCE,
            "metric_reproduction": METRIC_REPRODUCTION_TOLERANCE,
            "material_gain_db": MATERIAL_GAIN_DB,
            "per_image_gain_db": PER_IMAGE_GAIN_DB,
            "loss_metric_mismatch_cosine": LOSS_METRIC_MISMATCH_COSINE,
            "correction_follows_loss_cosine": (
                CORRECTION_FOLLOWS_LOSS_COSINE
            ),
        },
        "forbidden": [
            "Do not train in this audit.",
            "Do not evaluate another checkpoint.",
            "Do not access official test sets.",
            "Do not start D4/B4.",
            "Do not report oracle diagnostics as model results.",
            "Do not abandon the BFN research objective; repair the deterministic "
            "receiver contract first.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    geometry = summary["geometry"]
    scale = summary["scale_sweep"]
    mixture = summary["continuous_mixture"]
    loss = summary["loss_alignment"]
    attribution = summary["attribution"]
    endpoint = summary["endpoint_reproduction"]
    lines = [
        "# v27 C0-double-prime Residual Geometry Audit",
        "",
        "## Status",
        "",
        "- Mode: read-only, EMA epoch 3 only",
        f"- Images: {summary['dataset']['images']}",
        f"- Geometry identity passed: `{geometry['identity_passed']}`",
        f"- alpha=0 reproduces F0: `{endpoint['alpha_zero_passed']}`",
        f"- alpha=1 reproduces C0-double-prime: "
        f"`{endpoint['alpha_one_passed']}`",
        f"- Main attribution: **{attribution['code']}. "
        f"{attribution['conclusion']}**",
        "",
        "## Coordinate And Computation Audit",
        "",
        "| Stage | Tensor definition | Space / clamp | Source |",
        "|---|---|---|---|",
    ]
    for row in summary["coordinate_audit"]["data_flow"]:
        lines.append(
            f"| {row['stage']} | {row['definition']} | "
            f"{row['space_and_clamp']} | `{row['source']}` |"
        )
    lines.extend(
        [
            "",
            "Observed mismatches:",
            "",
        ]
    )
    lines.extend(
        f"- {value}"
        for value in summary["coordinate_audit"]["observed_mismatches"]
    )
    lines.extend(
        [
            "",
            "## PSNR/MSE Geometry",
            "",
            "- Maximum identity relative error: "
            f"`{geometry['maximum_identity_relative_error']:.3e}`",
            "- Overall beneficial fraction: "
            f"`{geometry['summary']['overall']['beneficial_fraction']:.6f}`",
            "- Overall correction/residual RMS ratio: "
            f"`{geometry['summary']['overall']['correction_residual_rms_ratio']['mean']:.6f}`",
            "- Overall correction-target cosine: "
            f"`{geometry['summary']['overall']['cosine']['mean']:.6f}`",
            "",
            "| Category | Beneficial fraction | Cosine mean | Cosine median | "
            "RMS ratio | Delta MSE | Delta PSNR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for category in CATEGORIES:
        values = geometry["summary"]["by_category"][category]
        lines.append(
            f"| {category} | {values['beneficial_fraction']:.6f} | "
            f"{values['cosine']['mean']:.6f} | "
            f"{values['cosine']['median']:.6f} | "
            f"{values['correction_residual_rms_ratio']['mean']:.6f} | "
            f"{values['delta_mse_mean']:+.8e} | "
            f"{values['delta_psnr_mean']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed Alpha Sweep",
            "",
            "| alpha | Natural dPSNR | Macro dPSNR | Macro CI95 | Snow | "
            "Outdoor | RainDrop |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for alpha in ALPHAS:
        report = scale["fixed"][alpha_key(alpha)]
        macro = report["macro_weighted"]["psnr"]
        lines.append(
            f"| {alpha:g} | {report['natural_mean']['psnr']['delta']:+.6f} | "
            f"{macro['delta']:+.6f} | "
            f"[{macro['paired_image_bootstrap_ci95'][0]:+.6f}, "
            f"{macro['paired_image_bootstrap_ci95'][1]:+.6f}] | "
            f"{report['categories']['snow']['psnr']['delta']:+.6f} | "
            f"{report['categories']['outdoor_rain']['psnr']['delta']:+.6f} | "
            f"{report['categories']['raindrop']['psnr']['delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "Analytic unclamped alpha stars:",
            "",
            f"- Global: `{scale['analytic_alpha_star']['overall']:.6f}`",
        ]
    )
    lines.extend(
        f"- {category}: "
        f"`{scale['analytic_alpha_star']['by_category'][category]:.6f}`"
        for category in CATEGORIES
    )
    lines.extend(
        [
            "",
            "## Continuous Three-Expert Mixture",
            "",
            f"- Grid step: `{mixture['grid_step']}`",
            f"- Combinations: `{mixture['grid_size']}`",
            "",
            "| Scheme | Weights | Natural | Macro | Snow | Outdoor | RainDrop | "
            "Gain vs learned | Cosine |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("global_fixed", "category_oracle", "per_image_oracle"):
        values = mixture[key]
        weights = (
            f"{len(values['weights'])} selected grid points"
            if key == "per_image_oracle"
            else json.dumps(values["weights"], sort_keys=True)
        )
        lines.append(
            f"| {key} | `{weights}` | "
            f"{values['analysis']['natural_mean']['psnr']['delta']:+.6f} | "
            f"{values['analysis']['macro_weighted']['psnr']['delta']:+.6f} | "
            f"{values['analysis']['categories']['snow']['psnr']['delta']:+.6f} | "
            f"{values['analysis']['categories']['outdoor_rain']['psnr']['delta']:+.6f} | "
            f"{values['analysis']['categories']['raindrop']['psnr']['delta']:+.6f} | "
            f"{values['gain_vs_learned']['macro_weighted']['psnr']['delta']:+.6f} | "
            f"{values['correction_geometry']['correction_target_cosine']:.6f} |"
        )
    learned = mixture["learned_router"]
    lines.append(
        f"| learned_router | `per-image` | "
        f"{learned['analysis']['natural_mean']['psnr']['delta']:+.6f} | "
        f"{learned['analysis']['macro_weighted']['psnr']['delta']:+.6f} | "
        f"{learned['analysis']['categories']['snow']['psnr']['delta']:+.6f} | "
        f"{learned['analysis']['categories']['outdoor_rain']['psnr']['delta']:+.6f} | "
        f"{learned['analysis']['categories']['raindrop']['psnr']['delta']:+.6f} | "
        f"{0.0:+.6f} | "
        f"{learned['correction_geometry']['correction_target_cosine']:.6f} |"
    )
    lines.extend(
        [
            "",
            "The category and per-image rows are dev-label/target oracles. "
            "They are diagnostics, not model results.",
            "",
            "## Loss And Metric Alignment",
            "",
            "| Negative gradient | Cosine with formal Y-PSNR direction |",
            "|---|---:|",
        ]
    )
    for name, values in loss["overall"][
        "negative_gradient_vs_y_psnr_direction"
    ].items():
        lines.append(f"| {name} | {values['mean']:.6f} |")
    lines.extend(
        [
            "",
            "- Current gradient vs RGB residual: "
            f"`{loss['overall']['current_gradient_vs_residual_cosine']['mean']:.6f}`",
            "- Correction vs current gradient: "
            f"`{loss['overall']['correction_vs_current_gradient_cosine']['mean']:.6f}`",
            "- Correction vs RGB residual: "
            f"`{loss['overall']['correction_vs_residual_cosine']['mean']:.6f}`",
            "",
            "## Attribution",
            "",
            f"**{attribution['code']}. {attribution['conclusion']}**",
            "",
            "Only allowed next experiment:",
            "",
            f"> {attribution['next_single_variable_experiment']}",
            "",
            "Forbidden:",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in attribution["forbidden"])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- All alpha and mixture selections use the same dev set they report.",
            "- Per-image alpha and mixture choices directly use targets.",
            "- The simplex grid is discrete at step 0.05 and is not an "
            "unconstrained continuous optimizer.",
            "- Gradient alignment is measured offline on full dev images; "
            "training used augmented 256x256 patches.",
            "- Bootstrap conditions on one fixed checkpoint and excludes "
            "training-seed variance.",
            "- No official test data was accessed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples != 10000 or args.seed != 42:
        raise ValueError(
            "Geometry audit requires bootstrap=10000 and seed=42"
        )
    if args.mixture_chunk_size < 1:
        raise ValueError("mixture-chunk-size must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    registered_path = args.registered_evaluation.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if checkpoint_path.name != "epoch_0003.pt":
        raise ValueError("Only epoch_0003.pt is allowed")
    for path in (checkpoint_path, registered_path, pairs_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if int(raw_checkpoint.get("epoch", -1)) != 2:
        raise ValueError("Checkpoint metadata is not fixed epoch 3")
    if "ema" not in raw_checkpoint:
        raise ValueError("Geometry audit requires registered EMA weights")
    config = raw_checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no embedded config")
    if config.get("protocol", {}).get("arm") != V27_C0DOUBLEPRIME_ARM:
        raise ValueError("Checkpoint is not C0-double-prime")
    reference_c0 = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0_proxy.yaml"
    )
    reference_c0prime = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0prime_routed.yaml"
    )
    validate_v27_c0doubleprime_config(
        config,
        reference_c0,
        reference_c0prime,
    )
    expected_val = config["data"]["val"]
    for actual, expected, label in (
        (args.input_dir, expected_val["input_dir"], "input_dir"),
        (args.target_dir, expected_val["target_dir"], "target_dir"),
        (pairs_file, expected_val["pairs_file"], "pairs_file"),
    ):
        if actual.expanduser().resolve() != Path(expected).expanduser().resolve():
            raise ValueError(f"Audit {label} differs from checkpoint config")

    device = resolve_device(args.device)
    seed_everything(args.seed)
    model, loaded_config = load_model_for_inference(checkpoint_path, device)
    if resolved_config_sha256(loaded_config) != resolved_config_sha256(config):
        raise RuntimeError("Loaded config differs from checkpoint config")
    if sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) != V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("C0-double-prime parameter count changed")
    receiver = model.receiver
    if receiver is None or not receiver.routed_readout_enabled:
        raise RuntimeError("Checkpoint has no routed residual receiver")
    readout = receiver.output
    if not isinstance(readout, CategoryRoutedReadout):
        raise RuntimeError("Unexpected routed readout implementation")
    expert_names = tuple(readout.categories)
    if expert_names != CATEGORIES:
        raise RuntimeError(f"Unexpected expert order: {expert_names!r}")

    dataset = PairedImageDataset(
        args.input_dir,
        args.target_dir,
        pairs_file=pairs_file,
    )
    if len(dataset) != V27_EXPECTED_DEV_PAIRS:
        raise RuntimeError(
            f"Expected {V27_EXPECTED_DEV_PAIRS} dev images, got {len(dataset)}"
        )
    labels = classify_paths(
        [pair[0] for pair in dataset.pairs],
        DEFAULT_CATEGORY_PATTERNS,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    name_map = read_name_map(pairs_file)

    from analyze_dev_by_category import load_evaluation

    registered, _ = load_evaluation(registered_path)
    registered_protocol = registered["payload"].get("protocol", {})
    if (
        registered_protocol.get("experiment_protocol")
        != "residual_bfn_v27_exploratory"
        or registered_protocol.get("experiment_arm")
        != V27_C0DOUBLEPRIME_ARM
    ):
        raise ValueError("Registered evaluation is not C0-double-prime")
    if registered_protocol.get("resolved_config_sha256") != (
        resolved_config_sha256(config)
    ):
        raise ValueError("Registered evaluation config fingerprint changed")
    for key, expected in {
        "test_y_channel": True,
        "crop_border": 0,
        "stochastic": False,
        "tile": None,
        "output_quantization": "float32_clamp_0_1",
    }.items():
        if registered_protocol.get(key) != expected:
            raise ValueError(
                f"Registered evaluation requires {key}={expected!r}"
            )

    weights = simplex_grid()
    weight_tensor = torch.as_tensor(
        weights,
        device=device,
        dtype=torch.float32,
    )
    image_count = len(dataset)
    mixture_count = len(weights)
    mixture_psnr = np.empty(
        (image_count, mixture_count),
        dtype=np.float64,
    )
    mixture_projection = np.empty_like(mixture_psnr)
    mixture_correction_energy = np.empty_like(mixture_psnr)
    residual_energy = np.empty(image_count, dtype=np.float64)
    alpha_rows = {alpha: [] for alpha in ALPHAS}
    alpha_psnr = np.empty((image_count, len(ALPHAS)), dtype=np.float64)
    baseline_rows: list[dict[str, Any]] = []
    learned_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    learned_probabilities: list[list[float]] = []
    criterion = SGBFormerLoss(**config["loss"]).to(device)
    captured: dict[str, Tensor] = {}

    def capture(name: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Tensor, ...],
            output: Tensor,
        ) -> None:
            captured[name] = output.detach()

        return hook

    handles = [readout.router.register_forward_hook(capture("router_logits"))]
    handles.extend(
        head.register_forward_hook(capture(f"head:{name}"))
        for name, head in readout.heads.items()
    )
    checkpoint_sha_before = sha256_file(checkpoint_path)
    model_state_before = tensor_state_sha256(model)
    maximum_identity_relative_error = 0.0
    maximum_alpha_zero_psnr_error = 0.0
    maximum_alpha_one_metric_error = 0.0
    maximum_soft_reconstruction_error = 0.0
    started = time.perf_counter()
    try:
        for index, batch in enumerate(loader):
            captured.clear()
            degraded = batch["degraded"].to(device)
            target = batch["target"].to(device)
            name = str(batch["name"][0])
            category = labels[index]
            with torch.no_grad():
                details = model.restore(
                    degraded,
                    deterministic=True,
                    return_details=True,
                )
                if int(details.get("receiver_calls", -1)) != 1:
                    raise RuntimeError(
                        f"{name}: expected exactly one receiver forward"
                    )
                baseline = details["structural"].float()
                learned = details["restored"].float()
                raw_correction = details["residual"].float()
                effective_correction = learned - baseline
                baseline_y = rgb_to_matlab_y(baseline.double())
                learned_y = rgb_to_matlab_y(learned.double())
                target_y = rgb_to_matlab_y(target.double().clamp(0, 1))
                geometry = geometry_terms(
                    baseline_y,
                    learned_y,
                    target_y,
                )
                baseline_row = metric_row(name, baseline, target)
                learned_row = metric_row(name, learned, target)
                geometry["delta_psnr"] = (
                    learned_row["psnr"] - baseline_row["psnr"]
                )
                maximum_identity_relative_error = max(
                    maximum_identity_relative_error,
                    float(geometry["identity_relative_error"]),
                )
                baseline_rows.append(baseline_row)
                learned_rows.append(learned_row)

                registered_row = registered["rows"].get(name)
                if registered_row is None:
                    raise RuntimeError(
                        f"{name}: absent from registered evaluation"
                    )
                registered_error = max(
                    abs(
                        learned_row[metric]
                        - float(registered_row[metric])
                    )
                    for metric in ("psnr", "ssim")
                )
                maximum_alpha_one_metric_error = max(
                    maximum_alpha_one_metric_error,
                    registered_error,
                )

                for alpha_index, alpha in enumerate(ALPHAS):
                    prediction = (
                        baseline + alpha * effective_correction
                    ).clamp(0, 1)
                    prediction_y = rgb_to_matlab_y(prediction.double())
                    mse = (prediction_y - target_y).square().mean()
                    value = float(psnr_from_mse(mse))
                    alpha_psnr[index, alpha_index] = value
                    alpha_rows[alpha].append(
                        {"name": name, "psnr": value}
                    )
                maximum_alpha_zero_psnr_error = max(
                    maximum_alpha_zero_psnr_error,
                    abs(
                        alpha_psnr[index, ALPHAS.index(0.0)]
                        - baseline_row["psnr"]
                    ),
                )
                maximum_alpha_one_metric_error = max(
                    maximum_alpha_one_metric_error,
                    abs(
                        alpha_psnr[index, ALPHAS.index(1.0)]
                        - learned_row["psnr"]
                    ),
                )

                logits = captured.get("router_logits")
                if logits is None or logits.shape != (1, len(expert_names)):
                    raise RuntimeError(
                        f"{name}: router hook did not capture logits"
                    )
                probabilities = F.softmax(logits.float(), dim=1)[0]
                learned_probabilities.append(
                    [float(value) for value in probabilities]
                )
                height, width = target.shape[-2:]
                expert_corrections = []
                for expert in expert_names:
                    output = captured.get(f"head:{expert}")
                    if output is None:
                        raise RuntimeError(
                            f"{name}: head hook did not capture {expert}"
                        )
                    expert_corrections.append(
                        output[..., :height, :width][0].float()
                        * model.residual_scale
                    )
                expert_stack = torch.stack(expert_corrections, dim=0)
                reconstructed_raw = torch.einsum(
                    "e,echw->chw",
                    probabilities,
                    expert_stack,
                )[None]
                reconstruction_error = float(
                    (reconstructed_raw - raw_correction).abs().max()
                )
                maximum_soft_reconstruction_error = max(
                    maximum_soft_reconstruction_error,
                    reconstruction_error,
                )
                if reconstruction_error > 1e-6:
                    raise RuntimeError(
                        f"{name}: expert mixture reconstruction error "
                        f"{reconstruction_error}"
                    )

                residual_y = target_y - baseline_y
                residual_energy[index] = float(residual_y.square().sum())
                for start in range(
                    0,
                    mixture_count,
                    args.mixture_chunk_size,
                ):
                    end = min(start + args.mixture_chunk_size, mixture_count)
                    corrections = torch.einsum(
                        "ke,echw->kchw",
                        weight_tensor[start:end],
                        expert_stack,
                    )
                    predictions = (baseline + corrections).clamp(0, 1)
                    predictions_y = rgb_to_matlab_y(predictions.double())
                    effective_y = predictions_y - baseline_y
                    mse = (
                        predictions_y - target_y
                    ).square().mean(dim=(-3, -2, -1))
                    mixture_psnr[index, start:end] = (
                        psnr_from_mse(mse).cpu().numpy()
                    )
                    mixture_projection[index, start:end] = (
                        (effective_y * residual_y)
                        .sum(dim=(-3, -2, -1))
                        .cpu()
                        .numpy()
                    )
                    mixture_correction_energy[index, start:end] = (
                        effective_y.square()
                        .sum(dim=(-3, -2, -1))
                        .cpu()
                        .numpy()
                    )

            raw_prediction = (
                baseline.detach().clone() + raw_correction.detach().clone()
            )
            alignment = loss_gradient_diagnostics(
                raw_prediction,
                baseline,
                target,
                criterion,
            )
            records.append(
                {
                    "name": name,
                    "input_path": str(batch["input_path"][0]),
                    "category": category,
                    "geometry": geometry,
                    "alpha_psnr": {
                        alpha_key(alpha): float(
                            alpha_psnr[index, alpha_index]
                        )
                        for alpha_index, alpha in enumerate(ALPHAS)
                    },
                    "analytic_alpha_star_unclamped": float(
                        geometry["alpha_star"]
                    ),
                    "learned_router_probabilities": {
                        expert: float(probabilities[position])
                        for position, expert in enumerate(expert_names)
                    },
                    "loss_alignment": alignment,
                }
            )
    finally:
        for handle in handles:
            handle.remove()

    checkpoint_sha_after = sha256_file(checkpoint_path)
    model_state_after = tensor_state_sha256(model)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise RuntimeError("Checkpoint changed during read-only audit")
    if model_state_before != model_state_after:
        raise RuntimeError("Model state changed during read-only audit")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Model parameters received gradients")
    if len(records) != V27_EXPECTED_DEV_PAIRS:
        raise RuntimeError("Audit did not process the complete dev set")

    baseline_protocol = copy.deepcopy(registered_protocol)
    baseline_protocol["experiment_protocol"] = (
        "reconstructed_f0_from_structural_path"
    )
    baseline_protocol["experiment_arm"] = "F0-reconstructed"
    baseline, baseline_schema = evaluation_payload(
        baseline_rows,
        baseline_protocol,
        "f0_structural",
    )
    learned_report, learned_normalized = analyze_variant(
        baseline,
        baseline_schema,
        learned_rows,
        registered_protocol,
        variant="learned_router",
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    scale_reports = {}
    scale_normalized = {}
    for alpha in ALPHAS:
        report, normalized = analyze_variant(
            baseline,
            baseline_schema,
            alpha_rows[alpha],
            registered_protocol,
            variant=f"alpha_{alpha_key(alpha)}",
            name_map=name_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        scale_reports[alpha_key(alpha)] = report
        scale_normalized[alpha] = normalized

    best_global_alpha = max(
        ALPHAS,
        key=lambda alpha: (
            macro_delta(scale_reports[alpha_key(alpha)]),
            -abs(alpha),
        ),
    )
    best_category_alpha = {
        category: max(
            ALPHAS,
            key=lambda alpha: (
                scale_reports[alpha_key(alpha)]["categories"][category][
                    "psnr"
                ]["delta"],
                -abs(alpha),
            ),
        )
        for category in CATEGORIES
    }
    category_alpha_rows = [
        alpha_rows[best_category_alpha[label]][index]
        for index, label in enumerate(labels)
    ]
    category_alpha_report, category_alpha_normalized = analyze_variant(
        baseline,
        baseline_schema,
        category_alpha_rows,
        registered_protocol,
        variant="oracle_category_alpha",
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    category_alpha_gain = compare_variants(
        learned_normalized,
        category_alpha_normalized,
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    per_image_alpha_indices = alpha_psnr.argmax(axis=1)
    per_image_alpha_rows = [
        alpha_rows[ALPHAS[int(alpha_index)]][index]
        for index, alpha_index in enumerate(per_image_alpha_indices)
    ]
    per_image_alpha_report, per_image_alpha_normalized = analyze_variant(
        baseline,
        baseline_schema,
        per_image_alpha_rows,
        registered_protocol,
        variant="oracle_per_image_fixed_alpha",
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    per_image_alpha_gain = compare_variants(
        learned_normalized,
        per_image_alpha_normalized,
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    category_indices = {
        category: np.asarray(
            [index for index, label in enumerate(labels) if label == category],
            dtype=np.int64,
        )
        for category in CATEGORIES
    }
    natural_scores = mixture_psnr.mean(axis=0)
    macro_scores = sum(
        DEFAULT_CATEGORY_WEIGHTS[category]
        * mixture_psnr[indices].mean(axis=0)
        for category, indices in category_indices.items()
    )
    global_mixture_index = int(np.argmax(macro_scores))
    category_mixture_indices = {
        category: int(np.argmax(mixture_psnr[indices].mean(axis=0)))
        for category, indices in category_indices.items()
    }
    per_image_mixture_indices = mixture_psnr.argmax(axis=1)

    def rows_for_selection(selected: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "name": records[index]["name"],
                "psnr": float(mixture_psnr[index, selected[index]]),
            }
            for index in range(image_count)
        ]

    global_selection = np.full(
        image_count,
        global_mixture_index,
        dtype=np.int64,
    )
    category_selection = np.asarray(
        [category_mixture_indices[label] for label in labels],
        dtype=np.int64,
    )
    selections = {
        "global_fixed": global_selection,
        "category_oracle": category_selection,
        "per_image_oracle": per_image_mixture_indices,
    }
    mixture_results: dict[str, Any] = {}
    for name, selection in selections.items():
        report, normalized = analyze_variant(
            baseline,
            baseline_schema,
            rows_for_selection(selection),
            registered_protocol,
            variant=f"continuous_mixture_{name}",
            name_map=name_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        gain = compare_variants(
            learned_normalized,
            normalized,
            name_map=name_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        if name == "global_fixed":
            selected_weights: Any = weights[global_mixture_index].tolist()
        elif name == "category_oracle":
            selected_weights = {
                category: weights[index].tolist()
                for category, index in category_mixture_indices.items()
            }
        else:
            counts = Counter(
                tuple(float(value) for value in weights[index])
                for index in selection
            )
            selected_weights = {
                json.dumps(list(weight)): count
                for weight, count in sorted(counts.items())
            }
        mixture_results[name] = {
            "selection_is_oracle": name != "global_fixed",
            "weights": selected_weights,
            "analysis": report,
            "gain_vs_learned": gain,
            "correction_geometry": selected_correction_geometry(
                selection,
                mixture_projection,
                mixture_correction_energy,
                residual_energy,
            ),
        }

    learned_projection = np.asarray(
        [
            float(record["geometry"]["projection_sum"])
            for record in records
        ],
        dtype=np.float64,
    )
    learned_correction_energy = np.asarray(
        [
            float(record["geometry"]["correction_energy_sum"])
            for record in records
        ],
        dtype=np.float64,
    )
    learned_geometry = {
        "correction_target_cosine": (
            float(learned_projection.sum())
            / math.sqrt(
                max(
                    float(learned_correction_energy.sum())
                    * float(residual_energy.sum()),
                    1e-24,
                )
            )
        ),
        "projection_sum": float(learned_projection.sum()),
        "correction_energy_sum": float(
            learned_correction_energy.sum()
        ),
        "residual_energy_sum": float(residual_energy.sum()),
    }

    for index, record in enumerate(records):
        record["best_fixed_alpha_from_grid"] = float(
            ALPHAS[int(per_image_alpha_indices[index])]
        )
        mixture_index = int(per_image_mixture_indices[index])
        record["best_simplex_mixture_oracle"] = {
            "weights": weights[mixture_index].tolist(),
            "psnr": float(mixture_psnr[index, mixture_index]),
            "gain_vs_learned_psnr": float(
                mixture_psnr[index, mixture_index]
                - learned_rows[index]["psnr"]
            ),
        }

    geometry_summary = summarize_geometry(records)
    analytic_alpha = {
        "overall": geometry_summary["overall"][
            "analytic_alpha_star_unclamped"
        ],
        "by_category": {
            category: geometry_summary["by_category"][category][
                "analytic_alpha_star_unclamped"
            ]
            for category in CATEGORIES
        },
        "per_image": quantile_summary(
            [
                float(record["analytic_alpha_star_unclamped"])
                for record in records
            ]
        ),
    }
    best_global_gain = compare_variants(
        learned_normalized,
        scale_normalized[best_global_alpha],
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    scale_summary = {
        "alphas": list(ALPHAS),
        "fixed": scale_reports,
        "analytic_alpha_star": analytic_alpha,
        "best_global_fixed_alpha": {
            "alpha": best_global_alpha,
            "analysis": scale_reports[alpha_key(best_global_alpha)],
            "gain_vs_learned": best_global_gain,
            "gain_vs_learned_macro_psnr": macro_delta(best_global_gain),
        },
        "best_fixed_alpha_by_category": {
            category: {
                "alpha": best_category_alpha[category],
                "category_delta_psnr": scale_reports[
                    alpha_key(best_category_alpha[category])
                ]["categories"][category]["psnr"]["delta"],
            }
            for category in CATEGORIES
        },
        "category_specific_policy": {
            "selection_is_dev_leaking": True,
            "analysis": category_alpha_report,
            "gain_vs_learned": category_alpha_gain,
        },
        "per_image_fixed_grid_oracle": {
            "severe_target_leakage": True,
            "analysis": per_image_alpha_report,
            "gain_vs_learned": per_image_alpha_gain,
            "alpha_counts": dict(
                Counter(
                    alpha_key(ALPHAS[int(index)])
                    for index in per_image_alpha_indices
                )
            ),
        },
    }
    probability_array = np.asarray(
        learned_probabilities,
        dtype=np.float64,
    )
    mixture_summary = {
        "grid_step": MIXTURE_STEP,
        "grid_size": mixture_count,
        "selection_metric": "current_dev_macro_y_psnr_delta_vs_f0",
        "global_natural_best_weights": weights[
            int(np.argmax(natural_scores))
        ].tolist(),
        "learned_router": {
            "weights": {
                "overall_mean": {
                    expert: float(probability_array[:, position].mean())
                    for position, expert in enumerate(expert_names)
                },
                "by_category_mean": {
                    category: {
                        expert: float(
                            probability_array[
                                category_indices[category],
                                position,
                            ].mean()
                        )
                        for position, expert in enumerate(expert_names)
                    }
                    for category in CATEGORIES
                },
            },
            "analysis": learned_report,
            "correction_geometry": learned_geometry,
        },
        **mixture_results,
    }
    loss_summary = summarize_loss_alignment(records)
    endpoint_summary = {
        "metric_tolerance": METRIC_REPRODUCTION_TOLERANCE,
        "alpha_zero_maximum_psnr_error": (
            maximum_alpha_zero_psnr_error
        ),
        "alpha_zero_passed": (
            maximum_alpha_zero_psnr_error
            <= METRIC_REPRODUCTION_TOLERANCE
        ),
        "alpha_one_maximum_metric_error": (
            maximum_alpha_one_metric_error
        ),
        "alpha_one_passed": (
            maximum_alpha_one_metric_error
            <= METRIC_REPRODUCTION_TOLERANCE
        ),
        "maximum_soft_expert_reconstruction_error": (
            maximum_soft_reconstruction_error
        ),
    }
    coordinate_audit = {
        "data_flow": [
            {
                "stage": "degraded/GT load",
                "definition": "PIL RGB converted to float32 and divided by 255",
                "space_and_clamp": "RGB [0,1], no additional normalization",
                "source": "sgbformer/data/paired.py:17-21",
            },
            {
                "stage": "baseline y0",
                "definition": "frozen SSR output",
                "space_and_clamp": "RGB, clamped to [0,1] before residual target",
                "source": "sgbformer/models/model.py:1338-1366",
            },
            {
                "stage": "residual target r",
                "definition": "target - structural",
                "space_and_clamp": "raw RGB difference after y0 clamp",
                "source": "sgbformer/models/model.py:1360-1367",
            },
            {
                "stage": "receiver correction c_raw",
                "definition": "identity receiver output multiplied by 0.2",
                "space_and_clamp": "unbounded RGB correction, no output activation",
                "source": "sgbformer/models/model.py:1397-1411",
            },
            {
                "stage": "training output",
                "definition": "y_hat_raw = y0 + c_raw",
                "space_and_clamp": "RGB, not clamped before loss",
                "source": "sgbformer/models/model.py:1410-1414",
            },
            {
                "stage": "training loss",
                "definition": "final_weight=1 RGB MSE; all other weights zero",
                "space_and_clamp": "augmented 256x256 RGB patches, before clamp",
                "source": "sgbformer/losses.py:128-175; train.py:776-796",
            },
            {
                "stage": "deployed output",
                "definition": "y_hat = clamp(y0 + c_raw, 0, 1)",
                "space_and_clamp": "float32 RGB [0,1]",
                "source": "sgbformer/models/model.py:1725-1758",
            },
            {
                "stage": "dev metric",
                "definition": "PSNR after MATLAB-compatible RGB-to-Y",
                "space_and_clamp": "full image, clamp [0,1], crop border 0",
                "source": "sgbformer/metrics.py:13-55; evaluate.py:415-519",
            },
            {
                "stage": "checkpoint weights",
                "definition": "EMA shadow over trainable receiver parameters",
                "space_and_clamp": "EMA used by fixed endpoint inference",
                "source": "sgbformer/inference.py:27-43; sgbformer/checkpoint.py:118-150",
            },
        ],
        "observed_mismatches": [
            "Training optimizes RGB MSE while the registered metric is Y-channel PSNR.",
            "Training MSE sees y0+c_raw before clamp; evaluation clamps before Y conversion.",
            "Training samples augmented 256x256 patches; evaluation uses complete images.",
            "The fixed endpoint uses EMA weights; epoch training diagnostics use online weights.",
            "Residual target and correction share the same raw RGB coordinate scale; "
            "there is no normalized-residual/raw-correction mismatch.",
        ],
    }
    summary = {
        "schema_version": 1,
        "protocol": "v27_c0doubleprime_residual_geometry",
        "status": "completed",
        "read_only": True,
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": 3,
            "sha256": checkpoint_sha_before,
            "ema_weights_used": True,
            "resolved_config_sha256": resolved_config_sha256(config),
        },
        "registered_evaluation": {
            "path": str(registered_path),
            "sha256": sha256_file(registered_path),
        },
        "dataset": {
            "images": len(records),
            "input_dir": str(args.input_dir.expanduser().resolve()),
            "target_dir": str(args.target_dir.expanduser().resolve()),
            "pairs_file": str(pairs_file),
            "pairs_file_sha256": sha256_file(pairs_file),
            "category_counts": dict(Counter(labels)),
            "official_test_accessed": False,
        },
        "coordinate_audit": coordinate_audit,
        "geometry": {
            "identity_formula": (
                "MSE(candidate)-MSE(baseline) = "
                "mean(c^2)-2*mean(r*c)"
            ),
            "formal_space": (
                "clamped float RGB converted to MATLAB Y, crop border 0"
            ),
            "maximum_identity_relative_error": (
                maximum_identity_relative_error
            ),
            "relative_tolerance": GEOMETRY_RELATIVE_TOLERANCE,
            "identity_passed": (
                maximum_identity_relative_error
                < GEOMETRY_RELATIVE_TOLERANCE
            ),
            "summary": geometry_summary,
        },
        "endpoint_reproduction": endpoint_summary,
        "scale_sweep": scale_summary,
        "continuous_mixture": mixture_summary,
        "loss_alignment": loss_summary,
        "integrity": {
            "checkpoint_sha256_unchanged": True,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "model_state_unchanged": True,
            "model_parameter_gradients_absent": True,
            "receiver_forward_passes_per_image": 1,
            "extra_training_steps": 0,
        },
        "runtime": {
            "device": str(device),
            "seconds": time.perf_counter() - started,
            "simplex_candidates_per_image": mixture_count,
        },
        "implementation": implementation_fingerprint(),
    }
    summary["attribution"] = determine_attribution(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "per_image.jsonl"
    per_image_path.write_text(
        "".join(
            json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=True,
                default=json_default,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(summary), encoding="utf-8")

    print(
        "Geometry identity: "
        f"passed={summary['geometry']['identity_passed']} "
        f"max_relative_error={maximum_identity_relative_error:.3e}"
    )
    print(
        "Endpoint reproduction: "
        f"alpha0={endpoint_summary['alpha_zero_passed']} "
        f"alpha1={endpoint_summary['alpha_one_passed']}"
    )
    print(
        "Best fixed alpha by category: "
        + json.dumps(
            {
                category: best_category_alpha[category]
                for category in CATEGORIES
            },
            sort_keys=True,
        )
    )
    print(
        "Continuous mixture gains vs learned: "
        f"global={macro_delta(mixture_results['global_fixed']['gain_vs_learned']):+.6f} "
        f"category={macro_delta(mixture_results['category_oracle']['gain_vs_learned']):+.6f} "
        f"per_image={macro_delta(mixture_results['per_image_oracle']['gain_vs_learned']):+.6f}"
    )
    print(
        "Loss alignment: "
        f"current_vs_Y={loss_summary['overall']['negative_gradient_vs_y_psnr_direction']['current_total']['mean']:.6f} "
        f"correction_vs_current={loss_summary['overall']['correction_vs_current_gradient_cosine']['mean']:.6f}"
    )
    attribution = summary["attribution"]
    print(
        f"Attribution={attribution['code']} "
        f"{attribution['conclusion']}"
    )
    print(
        "Next="
        f"{attribution['next_single_variable_experiment']}"
    )
    print(f"Wrote {per_image_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
