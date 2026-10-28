#!/usr/bin/env python3
"""Validate the four official All-Weather benchmark splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgbformer.data import PairedImageDataset


SPLITS = {
    "Snow100K-S": (
        "snow100k/Snow100K-S/synthetic",
        "snow100k/Snow100K-S/gt",
    ),
    "Snow100K-L": (
        "snow100k/Snow100K-L/synthetic",
        "snow100k/Snow100K-L/gt",
    ),
    "Outdoor-Rain Test1": ("Test1/input", "Test1/gt"),
    "RainDrop Test-A": ("raindrop/test_a/data", "raindrop/test_a/gt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Directory containing snow100k/, Test1/, and raindrop/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    total = 0
    for name, (input_relative, target_relative) in SPLITS.items():
        input_dir = root / input_relative
        target_dir = root / target_relative
        dataset = PairedImageDataset(input_dir, target_dir, strict=True)
        sample = dataset[0]
        shape = tuple(sample["degraded"].shape[-2:])
        print(
            f"{name}: pairs={len(dataset)} "
            f"sample={sample['name']} shape={shape[1]}x{shape[0]}"
        )
        total += len(dataset)
    print(f"All four official splits are valid; total pairs={total}")


if __name__ == "__main__":
    main()
