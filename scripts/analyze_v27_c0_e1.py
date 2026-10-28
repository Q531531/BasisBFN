#!/usr/bin/env python3
"""Evaluate the fixed epoch-3 C0-E1 rain-safe exploratory endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_dev_by_category import (
    DEFAULT_CATEGORY_PATTERNS,
    DEFAULT_CATEGORY_WEIGHTS,
    analyze_dev_by_category,
    load_evaluation,
    read_name_map,
)


RAIN_CATEGORIES = ("outdoor_rain", "raindrop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--c0-epoch3", required=True, type=Path)
    parser.add_argument("--candidate-epoch3", required=True, type=Path)
    parser.add_argument("--pairs-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compare(
    baseline_path: Path,
    candidate_path: Path,
    *,
    name_map: dict[str, str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline, baseline_schema = load_evaluation(baseline_path)
    candidate, candidate_schema = load_evaluation(candidate_path)
    return analyze_dev_by_category(
        baseline,
        candidate,
        baseline_schema=baseline_schema,
        candidate_schema=candidate_schema,
        category_patterns=DEFAULT_CATEGORY_PATTERNS,
        category_weights=DEFAULT_CATEGORY_WEIGHTS,
        name_map=name_map,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        reported_delta_psnr=None,
    )


def validate_candidate(path: Path, *, protocol: str, arm: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("protocol", {})
    if (
        recorded.get("experiment_protocol") != protocol
        or recorded.get("experiment_arm") != arm
    ):
        raise ValueError(
            f"{path}: expected {protocol}/{arm}, got "
            f"{recorded.get('experiment_protocol')}/"
            f"{recorded.get('experiment_arm')}"
        )
    return payload


def category_psnr(report: dict[str, Any], category: str) -> dict[str, Any]:
    values = report["categories"][category]["psnr"]
    return {
        "delta": values["delta"],
        "ci95": values["paired_image_bootstrap_ci95"],
        "improved": values["improved"],
        "n": report["categories"][category]["n"],
    }


def decision(
    vs_f0: dict[str, Any],
    vs_c0: dict[str, Any],
) -> dict[str, Any]:
    macro = vs_f0["macro_weighted"]["psnr"]
    checks: dict[str, Any] = {
        "macro_point_positive": macro["delta"] > 0,
    }
    for category in RAIN_CATEGORIES:
        f0 = category_psnr(vs_f0, category)
        c0 = category_psnr(vs_c0, category)
        checks[f"{category}_vs_f0_ci_upper_nonnegative"] = f0["ci95"][1] >= 0
        checks[f"{category}_vs_f0_point_at_least_neg_0p01"] = (
            f0["delta"] >= -0.01
        )
        checks[f"{category}_vs_c0_point_improved"] = c0["delta"] > 0
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "STOP",
        "primary_endpoint": "fixed_epoch_3",
        "checks": checks,
        "failed_checks": [
            name for name, value in checks.items() if not value
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples != 10000 or args.seed != 42:
        raise ValueError("C0-E1 requires 10000 bootstrap samples and seed 42")
    baseline = args.baseline.expanduser().resolve()
    c0 = args.c0_epoch3.expanduser().resolve()
    candidate = args.candidate_epoch3.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    validate_candidate(
        c0,
        protocol="residual_bfn_v27",
        arm="C0",
    )
    candidate_payload = validate_candidate(
        candidate,
        protocol="residual_bfn_v27_exploratory",
        arm="C0-E1",
    )
    name_map = read_name_map(pairs_file)
    vs_f0 = compare(
        baseline,
        candidate,
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    vs_c0 = compare(
        c0,
        candidate,
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result = decision(vs_f0, vs_c0)
    summary = {
        "schema_version": 1,
        "protocol": "residual_bfn_v27_exploratory",
        "arm": "C0-E1",
        "preregistered": False,
        "primary_endpoint": "fixed_epoch_3",
        "candidate_resolved_config_sha256": candidate_payload[
            "protocol"
        ].get("resolved_config_sha256"),
        "baseline": str(baseline),
        "reference_c0_epoch3": str(c0),
        "candidate_epoch3": str(candidate),
        "pairs_file": str(pairs_file),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "vs_f0": vs_f0,
        "vs_c0_epoch3": vs_c0,
        "decision": result,
        "limitations": [
            "This is an explicitly exploratory, fixed-epoch-3 comparison.",
            "Image bootstrap does not include training-seed variance.",
            "RainDrop has only 55 dev images.",
            "Official test sets must not be used for this decision.",
        ],
    }
    write_json(output_dir / "vs_f0.json", vs_f0)
    write_json(output_dir / "vs_c0_epoch3.json", vs_c0)
    write_json(output_dir / "summary.json", summary)

    macro = vs_f0["macro_weighted"]["psnr"]
    ci = macro["paired_image_bootstrap_ci95"]
    print(
        f"C0-E1 epoch3 vs F0: macro={macro['delta']:+.6f} "
        f"CI95=[{ci[0]:+.6f}, {ci[1]:+.6f}]"
    )
    for category in RAIN_CATEGORIES:
        f0 = category_psnr(vs_f0, category)
        c0_delta = category_psnr(vs_c0, category)["delta"]
        print(
            f"{category}: vs_F0={f0['delta']:+.6f} "
            f"CI95=[{f0['ci95'][0]:+.6f}, {f0['ci95'][1]:+.6f}] "
            f"vs_C0_epoch3={c0_delta:+.6f}"
        )
    print(
        f"Exploratory decision: {result['status']} "
        f"failed={result['failed_checks']}"
    )
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
