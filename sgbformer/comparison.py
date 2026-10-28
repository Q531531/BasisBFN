"""Paired, dataset-stratified comparisons for frozen evaluation protocols."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


_METRICS = ("psnr", "ssim", "lpips")
_DIRECTIONS = {"psnr": 1, "ssim": 1, "lpips": -1}
_FOUNDATION_TRAINING_KEYS = {
    "feature_adapters",
    "freeze",
    "frozen_modules_eval",
    "trainable_modules",
}


def foundation_signature(config: dict[str, Any]) -> str | None:
    """Identify the underlying foundation independently of training policy."""

    model = config.get("model", config)
    foundation = model.get("foundation")
    if not isinstance(foundation, dict):
        return None
    identity = {
        key: value
        for key, value in foundation.items()
        if key not in _FOUNDATION_TRAINING_KEYS
    }
    return json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_protocol(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    name: str,
) -> None:
    protocol_keys = (
        "test_y_channel",
        "crop_border",
        "lpips_network",
        "tile",
        "stochastic",
    )
    for key in protocol_keys:
        if key not in baseline.get("protocol", {}) or key not in candidate.get(
            "protocol", {}
        ):
            raise ValueError(f"{name}: missing protocol {key}")
        if baseline["protocol"][key] != candidate["protocol"][key]:
            raise ValueError(f"{name}: incompatible protocol {key}")
    if baseline["protocol"]["stochastic"]:
        raise ValueError("Paired benchmark comparison requires deterministic inference")
    if baseline["protocol"]["tile"] is not None:
        for result in (baseline, candidate):
            protocol = result["protocol"]
            tile, overlap = protocol["tile"], protocol.get("overlap")
            if overlap is None or tile < 1 or not 0 <= overlap < tile:
                raise ValueError(f"{name}: tiled result needs a valid overlap")
        if baseline["protocol"]["overlap"] != candidate["protocol"]["overlap"]:
            raise ValueError(f"{name}: incompatible protocol overlap")
    quantization = [
        result["protocol"].get("output_quantization")
        for result in (baseline, candidate)
    ]
    if (
        any(value is not None for value in quantization)
        and quantization[0] != quantization[1]
    ):
        raise ValueError(f"{name}: incompatible or missing output_quantization")
    foundation_signatures = [
        result["protocol"].get("foundation_signature")
        for result in (baseline, candidate)
    ]
    if (
        any(value is not None for value in foundation_signatures)
        and foundation_signatures[0] != foundation_signatures[1]
    ):
        raise ValueError(f"{name}: incompatible frozen foundation")


def _row_maps(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    name: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    maps = []
    for result in (baseline, candidate):
        rows = result.get("images")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{name}: empty image results")
        mapping = {row["name"]: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"{name}: duplicate image names")
        maps.append(mapping)
    if maps[0].keys() != maps[1].keys():
        raise ValueError(f"{name}: image sets differ")
    return maps[0], maps[1]


def _metric_summary(
    names: list[str],
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    generator: np.random.Generator,
) -> dict[str, Any]:
    if not all(
        metric in row
        for mapping in (baseline, candidate)
        for row in mapping.values()
    ):
        raise ValueError(f"Missing metric {metric}")
    left = np.array(
        [baseline[name][metric] for name in names],
        dtype=np.float64,
    )
    right = np.array(
        [candidate[name][metric] for name in names],
        dtype=np.float64,
    )
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"Non-finite {metric}")
    delta = right - left
    indices = generator.integers(
        0,
        len(delta),
        size=(bootstrap_samples, len(delta)),
    )
    bootstrap = delta[indices].mean(axis=1)
    best = int(delta.argmax())
    worst = int(delta.argmin())
    direction = _DIRECTIONS[metric]
    return {
        "baseline": float(left.mean()),
        "candidate": float(right.mean()),
        "delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "improved_images": int((direction * delta > 0).sum()),
        "improved_image_fraction": float((direction * delta > 0).mean()),
        "paired_image_bootstrap_ci95": np.quantile(
            bootstrap,
            [0.025, 0.975],
        ).tolist(),
        "largest_positive_delta": {
            "name": names[best],
            "delta": float(delta[best]),
        },
        "largest_negative_delta": {
            "name": names[worst],
            "delta": float(delta[worst]),
        },
    }


def paired_image_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare one matched evaluation with bootstrap and category summaries."""

    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    _validate_protocol(baseline, candidate, name="evaluation")
    left, right = _row_maps(baseline, candidate, name="evaluation")
    names = sorted(left)
    metrics = [
        metric
        for metric in _METRICS
        if metric in left[names[0]] and metric in right[names[0]]
    ]
    if not metrics:
        raise ValueError("Evaluation contains no supported metrics")
    generator = np.random.default_rng(seed)

    def summarize(group_names: list[str]) -> dict[str, Any]:
        return {
            "images": len(group_names),
            "metrics": {
                metric: _metric_summary(
                    group_names,
                    left,
                    right,
                    metric,
                    bootstrap_samples=bootstrap_samples,
                    generator=generator,
                )
                for metric in metrics
            },
        }

    categories: dict[str, list[str]] = {}
    for image_name in names:
        left_category = left[image_name].get("category")
        right_category = right[image_name].get("category")
        if left_category != right_category:
            raise ValueError(
                f"{image_name}: baseline and candidate categories differ"
            )
        if left_category is not None:
            categories.setdefault(str(left_category), []).append(image_name)

    result = {
        "overall": summarize(names),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "limitation": (
            "Image bootstrap conditions on these checkpoints; it does not "
            "measure training-seed variance."
        ),
    }
    if categories:
        result["categories"] = {
            category: summarize(category_names)
            for category, category_names in sorted(categories.items())
        }
    return result


def paired_comparison(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("Baseline and candidate must contain the same nonempty datasets")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    deltas: dict[str, list[np.ndarray]] = {
        metric: [] for metric in _METRICS
    }
    datasets = {}
    for name in sorted(baseline):
        left, right = baseline[name], candidate[name]
        _validate_protocol(left, right, name=name)
        maps = _row_maps(left, right, name=name)
        summary = {"images": len(maps[0])}
        for metric in _METRICS:
            if not all(
                metric in row
                for mapping in maps
                for row in mapping.values()
            ):
                raise ValueError(f"{name}: missing metric {metric}")
            values = [
                np.array(
                    [mapping[key][metric] for key in sorted(mapping)],
                    dtype=np.float64,
                )
                for mapping in maps
            ]
            if not all(np.isfinite(value).all() for value in values):
                raise ValueError(f"{name}: non-finite {metric}")
            delta = values[1] - values[0]
            deltas[metric].append(delta)
            summary[metric] = {
                "baseline": float(values[0].mean()),
                "candidate": float(values[1].mean()),
                "delta": float(delta.mean()),
                "improved_image_fraction": float(
                    (_DIRECTIONS[metric] * delta > 0).mean()
                ),
            }
        datasets[name] = summary
    generator = np.random.default_rng(seed)
    macro = {}
    for metric, groups in deltas.items():
        samples = np.zeros(bootstrap_samples)
        # Resample paired images within each dataset, then give each set equal weight.
        for delta in groups:
            for start in range(0, bootstrap_samples, 32):
                count = min(32, bootstrap_samples - start)
                indices = generator.integers(0, len(delta), size=(count, len(delta)))
                samples[start:start + count] += delta[indices].mean(axis=1) / len(groups)
        macro[metric] = {
            "delta": float(np.mean([delta.mean() for delta in groups])),
            "paired_image_bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
    return {
        "datasets": datasets,
        "macro": macro,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "limitation": "Image bootstrap conditions on these checkpoints; it does not measure training-seed variance.",
    }
