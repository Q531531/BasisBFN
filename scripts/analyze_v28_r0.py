#!/usr/bin/env python3
"""Apply the preregistered decision to the fixed v28-R0 epoch-6 endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_dev_by_category import read_name_map
from analyze_v27_c0prime import CATEGORIES, category_psnr, compare, write_json
from sgbformer.protocol import sha256_file
from sgbformer.v27 import resolved_config_sha256
from sgbformer.v28 import (
    V28_EXPECTED_DEV_PAIRS,
    V28_PROTOCOL,
    V28_R0_ARM,
    V28_R0_CONTINUATION_GATE,
    validate_v28_r0_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-epoch6", required=True, type=Path)
    parser.add_argument("--pairs-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_fixed_endpoint(
    checkpoint_path: Path,
    evaluation_path: Path,
) -> dict[str, Any]:
    if checkpoint_path.name != "epoch_0006.pt":
        raise ValueError("v28-R0 decision accepts only epoch_0006.pt")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if int(checkpoint.get("epoch", -1)) != 5:
        raise ValueError("v28-R0 checkpoint must record completed epoch 6")
    if "ema" not in checkpoint:
        raise ValueError("v28-R0 endpoint requires EMA weights")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("v28-R0 checkpoint has no embedded config")
    validate_v28_r0_config(config)

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    protocol = evaluation.get("protocol", {})
    if (
        protocol.get("experiment_protocol") != V28_PROTOCOL
        or protocol.get("experiment_arm") != V28_R0_ARM
    ):
        raise ValueError("Candidate evaluation is not v28-R0")
    if (
        protocol.get("resolved_config_sha256")
        != resolved_config_sha256(config)
    ):
        raise ValueError(
            "Candidate evaluation does not match the checkpoint config"
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if protocol.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(
            "Candidate evaluation does not match the checkpoint hash"
        )
    if protocol.get("checkpoint_epoch") != 5:
        raise ValueError("Candidate evaluation is not from completed epoch 6")
    if protocol.get("checkpoint_weights") != "ema":
        raise ValueError("Candidate evaluation must use EMA weights")
    expected_evaluation = {
        "test_y_channel": True,
        "crop_border": 0,
        "stochastic": False,
        "tile": None,
        "self_ensemble": False,
        "refinement_mode": "direct",
        "receiver_calls_per_trajectory": 1,
    }
    for key, expected in expected_evaluation.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"v28-R0 evaluation requires protocol.{key}={expected!r}"
            )
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "resolved_config_sha256": resolved_config_sha256(config),
        "evaluation": evaluation,
    }


def continuation_decision(vs_f0: dict[str, Any]) -> dict[str, Any]:
    macro = vs_f0["macro_weighted"]["psnr"]
    outdoor = category_psnr(vs_f0, "outdoor_rain")
    raindrop = category_psnr(vs_f0, "raindrop")
    gate = V28_R0_CONTINUATION_GATE
    checks = {
        "macro_delta_at_least_0p05": (
            macro["delta"] >= gate["macro_delta_psnr_min"]
        ),
        "macro_ci95_lower_strictly_positive": (
            macro["paired_image_bootstrap_ci95"][0] > 0
        ),
        "outdoor_rain_point_at_least_neg_0p003": (
            outdoor["delta"] >= gate["outdoor_rain_delta_min"]
        ),
        "outdoor_rain_ci_upper_nonnegative": (
            outdoor["ci95"][1] >= 0
        ),
        "raindrop_point_at_least_neg_0p005": (
            raindrop["delta"] >= gate["raindrop_delta_min"]
        ),
        "raindrop_ci_upper_nonnegative": (
            raindrop["ci95"][1] >= 0
        ),
    }
    passed = all(checks.values())
    return {
        "status": "GO" if passed else "STOP",
        "primary_endpoint": "fixed_epoch_6",
        "checks": checks,
        "failed_checks": [
            name for name, value in checks.items() if not value
        ],
        "next_step": (
            "Design coefficient-state BFN only after a separate approved plan."
            if passed
            else "Stop v28-R0 and do not start coefficient-state BFN."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples != 10000 or args.seed != 42:
        raise ValueError("v28-R0 requires 10000 bootstrap samples and seed 42")
    baseline = args.baseline.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    candidate = args.candidate_epoch6.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    endpoint = validate_fixed_endpoint(checkpoint, candidate)
    name_map = read_name_map(pairs_file)
    if len(name_map) != V28_EXPECTED_DEV_PAIRS:
        raise ValueError(
            f"v28-R0 requires {V28_EXPECTED_DEV_PAIRS} dev pairs, "
            f"got {len(name_map)}"
        )
    vs_f0 = compare(
        baseline,
        candidate,
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    missing_categories = set(CATEGORIES) - set(vs_f0.get("categories", {}))
    if missing_categories:
        raise ValueError(
            "v28-R0 comparison is missing categories: "
            + ", ".join(sorted(missing_categories))
        )
    result = continuation_decision(vs_f0)
    summary = {
        "schema_version": 1,
        "protocol": V28_PROTOCOL,
        "arm": V28_R0_ARM,
        "preregistered": True,
        "primary_endpoint": "fixed_epoch_6",
        "checkpoint": {
            key: endpoint[key]
            for key in (
                "checkpoint_sha256",
                "checkpoint_epoch",
                "checkpoint_global_step",
                "resolved_config_sha256",
            )
        },
        "baseline": str(baseline),
        "candidate_epoch6": str(candidate),
        "pairs_file": str(pairs_file),
        "pairs_file_sha256": sha256_file(pairs_file),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "vs_f0": vs_f0,
        "decision": result,
        "continuation_gate": V28_R0_CONTINUATION_GATE,
        "limitations": [
            "Only EMA epoch_0006.pt is eligible for the decision.",
            "Epochs 1-5 are diagnostic and cannot be selected.",
            "Image bootstrap does not include training-seed variance.",
            "RainDrop has only 55 dev images.",
            "Official test sets must not be used for this decision.",
            "A GO proves deterministic basis capacity, not BFN value.",
        ],
    }
    write_json(output_dir / "vs_f0.json", vs_f0)
    write_json(output_dir / "summary.json", summary)

    macro = vs_f0["macro_weighted"]["psnr"]
    macro_ci = macro["paired_image_bootstrap_ci95"]
    print(
        f"v28-R0 epoch6 vs F0: macro={macro['delta']:+.6f} "
        f"CI95=[{macro_ci[0]:+.6f}, {macro_ci[1]:+.6f}]"
    )
    for category in CATEGORIES:
        category_result = category_psnr(vs_f0, category)
        print(
            f"{category}: vs_F0={category_result['delta']:+.6f} "
            f"CI95=[{category_result['ci95'][0]:+.6f}, "
            f"{category_result['ci95'][1]:+.6f}]"
        )
    print(
        f"Preregistered decision: {result['status']} "
        f"failed={result['failed_checks']}"
    )
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
