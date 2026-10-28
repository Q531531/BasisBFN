#!/usr/bin/env python3
"""Validate configured image pairs before starting a long training run."""

from __future__ import annotations

import argparse

from sgbformer.config import load_config
from sgbformer.data import PairedImageDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sgbformer_5090.yaml")
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)

    for split in ("train", "val"):
        split_config = config["data"][split]
        dataset = PairedImageDataset(
            input_dir=split_config["input_dir"],
            target_dir=split_config["target_dir"],
            patch_size=None,
            augment=False,
            pairs_file=split_config.get("pairs_file"),
            manifest_target_replace=split_config.get(
                "manifest_target_replace", ("input", "gt")
            ),
            strict=bool(split_config.get("strict", True)),
        )
        count = min(len(dataset), args.samples)
        minimum_height = None
        minimum_width = None
        for index in range(count):
            item = dataset[index]
            height, width = item["degraded"].shape[-2:]
            minimum_height = (
                height if minimum_height is None else min(minimum_height, height)
            )
            minimum_width = (
                width if minimum_width is None else min(minimum_width, width)
            )
        print(
            f"{split}: pairs={len(dataset)} checked={count} "
            f"minimum_checked_size={minimum_width}x{minimum_height}"
        )


if __name__ == "__main__":
    main()
