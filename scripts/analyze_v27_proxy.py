#!/usr/bin/env python3
"""Analyze registered v27 proxy arms under the frozen dev protocol."""

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
from sgbformer.v27 import V27_ARMS


def parse_arm(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    arm = name.upper()
    if not separator or arm not in V27_ARMS or not path:
        raise argparse.ArgumentTypeError(
            "--arm must use C0=DIR, D1=DIR, D4=DIR, or B4=DIR"
        )
    return arm, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        type=parse_arm,
        help="Repeat as ARM=DIR; each directory must contain epoch_0001..0006.json",
    )
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


def psnr_summary(report: dict[str, Any]) -> dict[str, Any]:
    macro = report["macro_weighted"]["psnr"]
    natural = report["natural_mean"]["psnr"]
    return {
        "macro_delta": macro["delta"],
        "macro_ci95": macro["paired_image_bootstrap_ci95"],
        "natural_delta": natural["delta"],
        "natural_ci95": natural["paired_image_bootstrap_ci95"],
        "categories": {
            name: {
                "delta": values["psnr"]["delta"],
                "ci95": values["psnr"]["paired_image_bootstrap_ci95"],
            }
            for name, values in report["categories"].items()
            if values.get("n")
        },
    }


def category_safety_failure(summary: dict[str, Any]) -> list[str]:
    failures = []
    for category, values in summary["categories"].items():
        if values["delta"] <= -0.02 and values["ci95"][1] < 0:
            failures.append(category)
    return failures


def capacity_decision(summary: dict[str, Any]) -> dict[str, Any]:
    delta = summary["macro_delta"]
    low, high = summary["macro_ci95"]
    unsafe = category_safety_failure(summary)
    if unsafe:
        status = "STOP"
        reason = f"significant category regression: {unsafe}"
    elif delta > 0 and low > 0:
        status = "GO"
        reason = "epoch-6 macro CI95 is entirely positive"
    elif delta > 0 and low <= 0 <= high:
        status = "CONDITIONAL_GO"
        reason = "epoch-6 macro point estimate is positive but CI95 covers zero"
    else:
        status = "STOP"
        reason = "epoch-6 macro point estimate is non-positive"
    return {"status": status, "reason": reason}


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples != 10000 or args.seed != 42:
        raise ValueError("v27 analysis requires 10000 bootstrap samples and seed 42")
    arm_dirs = dict(args.arm)
    if len(arm_dirs) != len(args.arm):
        raise ValueError("Each v27 arm may be supplied only once")
    baseline = args.baseline.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    name_map = read_name_map(pairs_file)

    arm_reports: dict[str, dict[int, dict[str, Any]]] = {}
    arm_config_fingerprints: dict[str, str] = {}
    for arm, directory in arm_dirs.items():
        directory = directory.expanduser().resolve()
        reports = {}
        for epoch in range(1, 7):
            candidate = directory / f"epoch_{epoch:04d}.json"
            if not candidate.is_file():
                raise FileNotFoundError(f"Missing {arm} evaluation: {candidate}")
            candidate_payload = json.loads(
                candidate.read_text(encoding="utf-8")
            )
            protocol = candidate_payload.get("protocol", {})
            if (
                protocol.get("experiment_protocol") != "residual_bfn_v27"
                or protocol.get("experiment_arm") != arm
            ):
                raise ValueError(
                    f"{candidate}: expected residual_bfn_v27 arm {arm}, got "
                    f"{protocol.get('experiment_protocol')}/"
                    f"{protocol.get('experiment_arm')}"
                )
            config_fingerprint = str(protocol.get("resolved_config_sha256", ""))
            if not config_fingerprint:
                raise ValueError(
                    f"{candidate}: missing resolved_config_sha256"
                )
            previous_fingerprint = arm_config_fingerprints.setdefault(
                arm,
                config_fingerprint,
            )
            if config_fingerprint != previous_fingerprint:
                raise ValueError(
                    f"{arm} evaluation config changed across epochs"
                )
            report = compare(
                baseline,
                candidate,
                name_map=name_map,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            reports[epoch] = report
            write_report(
                output_dir / "vs_f0" / arm / f"epoch_{epoch:04d}.json",
                report,
            )
        arm_reports[arm] = reports

    arms: dict[str, Any] = {}
    for arm, reports in arm_reports.items():
        summaries = {
            epoch: psnr_summary(report)
            for epoch, report in reports.items()
        }
        best_epoch = max(
            summaries,
            key=lambda epoch: summaries[epoch]["macro_delta"],
        )
        arms[arm] = {
            "epoch_6_primary": summaries[6],
            "best_macro_secondary": {
                "epoch": best_epoch,
                **summaries[best_epoch],
                "selection_caveat": (
                    "Selected on the same dev set; image bootstrap does not "
                    "correct checkpoint-selection bias."
                ),
            },
            "all_epochs": summaries,
        }

    stage_a0 = None
    if "C0" in arms:
        stage_a0 = capacity_decision(arms["C0"]["epoch_6_primary"])

    pair_specs = (
        ("D1_minus_C0", "C0", "D1"),
        ("D4_minus_D1", "D1", "D4"),
        ("B4_minus_D4", "D4", "B4"),
    )
    paired = {}
    for label, baseline_arm, candidate_arm in pair_specs:
        if baseline_arm not in arm_dirs or candidate_arm not in arm_dirs:
            continue
        report = compare(
            arm_dirs[baseline_arm].expanduser().resolve() / "epoch_0006.json",
            arm_dirs[candidate_arm].expanduser().resolve() / "epoch_0006.json",
            name_map=name_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        write_report(output_dir / "paired" / f"{label}.json", report)
        paired[label] = psnr_summary(report)

    stage_a1 = None
    if "B4" in arms and "B4_minus_D4" in paired:
        system = arms["B4"]["epoch_6_primary"]
        causal = paired["B4_minus_D4"]
        unsafe = sorted(set(
            category_safety_failure(system)
            + category_safety_failure(causal)
        ))
        if unsafe:
            status = "NO_GO"
            reason = f"significant category regression: {unsafe}"
        elif (
            system["macro_ci95"][0] > 0
            and causal["macro_ci95"][0] > 0
        ):
            status = "BFN_GO"
            reason = "B4 beats both F0 and D4 with entirely positive CI95"
        elif system["macro_ci95"][0] > 0:
            status = "SYSTEM_GAIN_ONLY"
            reason = "B4 beats F0 but does not establish B4>D4"
        else:
            status = "NO_GO"
            reason = "B4 does not significantly beat F0"
        stage_a1 = {"status": status, "reason": reason}

    summary = {
        "schema_version": 1,
        "protocol": "residual_bfn_v27",
        "baseline": str(baseline),
        "pairs_file": str(pairs_file),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "primary_checkpoint": "epoch_0006",
        "arm_resolved_config_sha256": arm_config_fingerprints,
        "arms": arms,
        "paired_epoch_6": paired,
        "stage_a0_decision": stage_a0,
        "stage_a1_decision": stage_a1,
        "limitations": [
            "Best-macro checkpoints are secondary and selection-biased.",
            "Image bootstrap does not measure training-seed variance.",
            "RainDrop has only 55 dev images and a wide interval.",
        ],
    }
    write_report(output_dir / "summary.json", summary)

    for arm, values in arms.items():
        primary = values["epoch_6_primary"]
        ci = primary["macro_ci95"]
        print(
            f"{arm} epoch6 vs F0: macro={primary['macro_delta']:+.6f} "
            f"CI95=[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        )
    for label, values in paired.items():
        ci = values["macro_ci95"]
        print(
            f"{label}: macro={values['macro_delta']:+.6f} "
            f"CI95=[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        )
    if stage_a0 is not None:
        print(f"Stage A0: {stage_a0['status']} - {stage_a0['reason']}")
    if stage_a1 is not None:
        print(f"Stage A1: {stage_a1['status']} - {stage_a1['reason']}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
