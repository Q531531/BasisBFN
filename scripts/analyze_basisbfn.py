#!/usr/bin/env python3
"""Analyze the fixed epoch-6 endpoints of the BasisBFN-state matrix."""

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
from sgbformer.basisbfn import (
    BASISBFN_ARMS,
    BASISBFN_EXPECTED_DEV_PAIRS,
    BASISBFN_PROTOCOL,
    validate_basisbfn_config,
)


CONTRASTS = {
    "bayesian_effect_without_gate": ("D4", "B4"),
    "bayesian_effect_with_gate": ("D4-G", "B4-G"),
    "gate_effect_deterministic": ("D4", "D4-G"),
    "gate_effect_bayesian": ("B4", "B4-G"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="ARM=PATH",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="ARM=EVALUATION_JSON",
    )
    parser.add_argument("--pairs-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_arm_paths(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        arm, separator, raw_path = value.partition("=")
        if not separator or arm not in BASISBFN_ARMS or not raw_path:
            raise ValueError(
                f"{label} must use ARM=PATH for one of "
                + ", ".join(BASISBFN_ARMS)
            )
        if arm in result:
            raise ValueError(f"Duplicate {label} arm: {arm}")
        result[arm] = Path(raw_path).expanduser().resolve()
    if set(result) != set(BASISBFN_ARMS):
        raise ValueError(
            f"{label} must include exactly "
            + ", ".join(BASISBFN_ARMS)
        )
    return result


def validate_fixed_endpoint(
    arm: str,
    checkpoint_path: Path,
    evaluation_path: Path,
) -> dict[str, Any]:
    if checkpoint_path.name != "epoch_0006.pt":
        raise ValueError(f"{arm}: only epoch_0006.pt is eligible")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if int(checkpoint.get("epoch", -1)) != 5:
        raise ValueError(f"{arm}: checkpoint must record completed epoch 6")
    if "ema" not in checkpoint:
        raise ValueError(f"{arm}: endpoint requires EMA weights")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{arm}: checkpoint has no embedded config")
    if validate_basisbfn_config(config) != arm:
        raise ValueError(f"{arm}: checkpoint contains another matrix arm")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    protocol = evaluation.get("protocol", {})
    expected = {
        "experiment_protocol": BASISBFN_PROTOCOL,
        "experiment_arm": arm,
        "resolved_config_sha256": resolved_config_sha256(config),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": 5,
        "checkpoint_weights": "ema",
        "test_y_channel": True,
        "crop_border": 0,
        "stochastic": False,
        "tile": None,
        "self_ensemble": False,
        "refinement_mode": "direct",
        "steps": 4,
        "receiver_calls_per_trajectory": 1,
        "coefficient_state_updates_per_trajectory": 4,
        "basis_evaluations_per_trajectory": 1,
        "coefficient_update_mode": BASISBFN_ARMS[arm][
            "update_mode"
        ],
        "coefficient_gate_enabled": BASISBFN_ARMS[arm][
            "gate_enabled"
        ],
    }
    for key, expected_value in expected.items():
        if protocol.get(key) != expected_value:
            raise ValueError(
                f"{arm}: evaluation protocol.{key} must be "
                f"{expected_value!r}, got {protocol.get(key)!r}"
            )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "resolved_config_sha256": resolved_config_sha256(config),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
    }


def macro_psnr(report: dict[str, Any]) -> dict[str, Any]:
    return report["macro_weighted"]["psnr"]


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples != 10000 or args.seed != 42:
        raise ValueError(
            "BasisBFN matrix requires bootstrap=10000 and seed=42"
        )
    baseline = args.baseline.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoints = parse_arm_paths(args.checkpoint, "checkpoint")
    candidates = parse_arm_paths(args.candidate, "candidate")
    endpoints = {
        arm: validate_fixed_endpoint(
            arm,
            checkpoints[arm],
            candidates[arm],
        )
        for arm in BASISBFN_ARMS
    }
    name_map = read_name_map(pairs_file)
    if len(name_map) != BASISBFN_EXPECTED_DEV_PAIRS:
        raise ValueError(
            f"Expected {BASISBFN_EXPECTED_DEV_PAIRS} dev pairs, "
            f"got {len(name_map)}"
        )

    versus_f0 = {
        arm: compare(
            baseline,
            candidates[arm],
            name_map=name_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for arm in BASISBFN_ARMS
    }
    contrasts = {
        name: compare(
            candidates[reference_arm],
            candidates[candidate_arm],
            name_map=name_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for name, (reference_arm, candidate_arm) in CONTRASTS.items()
    }
    for report in (*versus_f0.values(), *contrasts.values()):
        missing = set(CATEGORIES) - set(report.get("categories", {}))
        if missing:
            raise ValueError(
                "Coefficient comparison is missing categories: "
                + ", ".join(sorted(missing))
            )

    interaction = (
        macro_psnr(contrasts["gate_effect_bayesian"])["delta"]
        - macro_psnr(contrasts["gate_effect_deterministic"])["delta"]
    )
    summary = {
        "schema_version": 1,
        "protocol": BASISBFN_PROTOCOL,
        "arms": list(BASISBFN_ARMS),
        "primary_endpoint": "fixed_epoch_6",
        "endpoints": endpoints,
        "baseline": str(baseline),
        "baseline_sha256": sha256_file(baseline),
        "pairs_file": str(pairs_file),
        "pairs_file_sha256": sha256_file(pairs_file),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "versus_f0": versus_f0,
        "contrasts": contrasts,
        "macro_psnr_interaction_bayesian_x_gate": interaction,
        "selection_policy": (
            "Report all fixed endpoints and registered contrasts; do not scan "
            "checkpoints or substitute diagnostic epochs."
        ),
        "limitations": [
            "Only EMA epoch_0006.pt is eligible for every arm.",
            "Epochs 1-5 are diagnostic and cannot be selected.",
            "Image bootstrap does not include training-seed variance.",
            "RainDrop has only 55 dev images.",
            "Official test sets must not be used for this matrix analysis.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm, report in versus_f0.items():
        write_json(output_dir / f"{arm.lower()}_vs_f0.json", report)
    for name, report in contrasts.items():
        write_json(output_dir / f"{name}.json", report)
    write_json(output_dir / "summary.json", summary)

    for arm in BASISBFN_ARMS:
        macro = macro_psnr(versus_f0[arm])
        interval = macro["paired_image_bootstrap_ci95"]
        print(
            f"{arm} epoch6 vs F0: macro={macro['delta']:+.6f} "
            f"CI95=[{interval[0]:+.6f}, {interval[1]:+.6f}]"
        )
        for category in CATEGORIES:
            values = category_psnr(versus_f0[arm], category)
            print(
                f"  {category}: delta={values['delta']:+.6f} "
                f"CI95=[{values['ci95'][0]:+.6f}, "
                f"{values['ci95'][1]:+.6f}]"
            )
    for name, report in contrasts.items():
        macro = macro_psnr(report)
        interval = macro["paired_image_bootstrap_ci95"]
        print(
            f"{name}: macro={macro['delta']:+.6f} "
            f"CI95=[{interval[0]:+.6f}, {interval[1]:+.6f}]"
        )
    print(f"Bayesian x gate macro interaction={interaction:+.6f} dB")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
