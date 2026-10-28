"""Shared category comparisons for fixed BasisBFN endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analyze_dev_by_category import (
    DEFAULT_CATEGORY_PATTERNS,
    DEFAULT_CATEGORY_WEIGHTS,
    analyze_dev_by_category,
    load_evaluation,
)

CATEGORIES = ("snow", "outdoor_rain", "raindrop")


def compare(
    baseline_path: Path, candidate_path: Path, *,
    name_map: dict[str, str], bootstrap_samples: int, seed: int,
) -> dict[str, Any]:
    baseline, baseline_schema = load_evaluation(baseline_path)
    candidate, candidate_schema = load_evaluation(candidate_path)
    return analyze_dev_by_category(
        baseline, candidate, baseline_schema=baseline_schema,
        candidate_schema=candidate_schema,
        category_patterns=DEFAULT_CATEGORY_PATTERNS,
        category_weights=DEFAULT_CATEGORY_WEIGHTS, name_map=name_map,
        bootstrap_samples=bootstrap_samples, seed=seed, reported_delta_psnr=None,
    )


def category_psnr(report: dict[str, Any], category: str) -> dict[str, Any]:
    values = report["categories"][category]["psnr"]
    return {
        "delta": values["delta"], "ci95": values["paired_image_bootstrap_ci95"],
        "improved": values["improved"], "n": report["categories"][category]["n"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
