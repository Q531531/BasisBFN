#!/usr/bin/env python3
"""Evaluate only the fixed epoch-3 v27 C0-accept endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_dev_by_category import read_name_map
from analyze_v27_c0prime import (
    CATEGORIES,
    category_psnr,
    compare,
    decision,
    validate_candidate,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--c0doubleprime-epoch3", required=True, type=Path)
    parser.add_argument("--candidate-epoch3", required=True, type=Path)
    parser.add_argument("--pairs-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples != 10000 or args.seed != 42:
        raise ValueError("C0-accept requires 10000 bootstrap samples and seed 42")
    baseline = args.baseline.expanduser().resolve()
    source = args.c0doubleprime_epoch3.expanduser().resolve()
    candidate = args.candidate_epoch3.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    validate_candidate(
        source,
        protocol="residual_bfn_v27_exploratory",
        arm="C0-double-prime",
    )
    candidate_payload = validate_candidate(
        candidate,
        protocol="residual_bfn_v27_exploratory",
        arm="C0-accept",
    )
    name_map = read_name_map(pairs_file)
    vs_f0 = compare(
        baseline,
        candidate,
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    vs_source = compare(
        source,
        candidate,
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result = decision(vs_f0, vs_source)
    summary = {
        "schema_version": 1,
        "protocol": "residual_bfn_v27_exploratory",
        "arm": "C0-accept",
        "source_arm": "C0-double-prime",
        "preregistered": True,
        "primary_endpoint": "fixed_epoch_3",
        "candidate_resolved_config_sha256": candidate_payload[
            "protocol"
        ].get("resolved_config_sha256"),
        "baseline": str(baseline),
        "reference_c0doubleprime_epoch3": str(source),
        "candidate_epoch3": str(candidate),
        "pairs_file": str(pairs_file),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "vs_f0": vs_f0,
        "vs_c0doubleprime_epoch3": vs_source,
        "decision": result,
        "limitations": [
            "Only epoch_0003 is eligible for the preregistered decision.",
            "Image bootstrap does not include training-seed variance.",
            "RainDrop has only 55 dev images.",
            "Official test sets must not be used for this decision.",
        ],
    }
    write_json(output_dir / "vs_f0.json", vs_f0)
    write_json(output_dir / "vs_c0doubleprime_epoch3.json", vs_source)
    write_json(output_dir / "summary.json", summary)

    macro = vs_f0["macro_weighted"]["psnr"]
    macro_ci = macro["paired_image_bootstrap_ci95"]
    print(
        f"C0-accept epoch3 vs F0: macro={macro['delta']:+.6f} "
        f"CI95=[{macro_ci[0]:+.6f}, {macro_ci[1]:+.6f}]"
    )
    for category in CATEGORIES:
        f0 = category_psnr(vs_f0, category)
        source_delta = category_psnr(vs_source, category)["delta"]
        print(
            f"{category}: vs_F0={f0['delta']:+.6f} "
            f"CI95=[{f0['ci95'][0]:+.6f}, {f0['ci95'][1]:+.6f}] "
            f"vs_C0doubleprime={source_delta:+.6f}"
        )
    print(
        f"Preregistered decision: {result['status']} "
        f"failed={result['failed_checks']}"
    )
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
