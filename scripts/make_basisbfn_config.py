#!/usr/bin/env python3
"""Create one paper-protocol BasisBFN arm/seed overlay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
ARM_CONFIGS = {
    "D4": CONFIG_DIR / "basisbfn_d4.yaml",
    "D4-G": CONFIG_DIR / "basisbfn_d4g.yaml",
    "B4": CONFIG_DIR / "basisbfn_b4.yaml",
    "B4-G": CONFIG_DIR / "basisbfn_b4g.yaml",
}
PAPER_SEEDS = (42, 43, 44, 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ARM_CONFIGS))
    parser.add_argument("--seed", required=True, type=int, choices=PAPER_SEEDS)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    arm_slug = args.arm.lower()
    artifact_root = f"runs/basisbfn/artifacts/{arm_slug}/seed_{args.seed}"
    payload = {
        "_base_": os.path.relpath(
            ARM_CONFIGS[args.arm],
            start=output.parent,
        ),
        "seed": args.seed,
        "training": {
            "output_dir": f"runs/basisbfn/{arm_slug}/seed_{args.seed}",
            "basisbfn_preflight_report": (
                f"{artifact_root}/preflight/preflight_{arm_slug}.json"
            ),
            "basisbfn_profile_report": f"{artifact_root}/profile.json",
        },
    }
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
