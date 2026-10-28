#!/usr/bin/env python3
"""Create deterministic train/validation manifests without copying images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgbformer.data import IMAGE_SUFFIXES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Directory containing input/ and gt/",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must lie in (0, 1)")
    root = Path(args.root).expanduser().resolve()
    input_dir = root / "input"
    target_dir = root / "gt"
    if not input_dir.is_dir() or not target_dir.is_dir():
        raise FileNotFoundError(f"Expected input/ and gt/ under {root}")

    inputs = sorted(
        path.relative_to(input_dir)
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(inputs) < 2:
        raise ValueError("At least two image pairs are required for train/validation splitting")
    missing = [path for path in inputs if not (target_dir / path).is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise ValueError(f"{len(missing)} targets are missing: {preview}")

    generator = random.Random(args.seed)
    generator.shuffle(inputs)
    validation_count = min(len(inputs) - 1, max(1, round(len(inputs) * args.validation_ratio)))
    validation = sorted(inputs[:validation_count])
    training = sorted(inputs[validation_count:])

    outputs = {
        root / "train_pairs.tsv": training,
        root / "val_pairs.tsv": validation,
    }
    for path in outputs:
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} already exists; pass --force to replace")
    for path, pairs in outputs.items():
        lines = [f"{item}\t{item}\n" for item in pairs]
        path.write_text("".join(lines), encoding="utf-8")

    print(f"root: {root}")
    print(f"total pairs: {len(inputs)}")
    print(f"training pairs: {len(training)} -> {root / 'train_pairs.tsv'}")
    print(f"validation pairs: {len(validation)} -> {root / 'val_pairs.tsv'}")


if __name__ == "__main__":
    main()
