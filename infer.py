#!/usr/bin/env python3
"""Restore one image or a directory of images with a trained BasisBFN."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sgbformer.data import IMAGE_SUFFIXES, load_rgb
from sgbformer.inference import load_model_for_inference, restore_tensor
from sgbformer.utils import resolve_device, save_image, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image or directory")
    parser.add_argument("--output", required=True, help="Output image or directory")
    parser.add_argument(
        "--config",
        help="Configuration for checkpoints without embedded config or relocated SSR files",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--bfn-gain", type=float, default=1.0)
    parser.add_argument("--tile", type=int)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def enumerate_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, _ = load_model_for_inference(
        args.checkpoint, device, config_path=args.config
    )
    input_path = Path(args.input)
    output_path = Path(args.output)
    images = enumerate_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found under {input_path}")

    for path in images:
        image = load_rgb(path).unsqueeze(0).to(device)
        prediction = restore_tensor(
            model,
            image,
            steps=args.steps,
            deterministic=args.deterministic,
            residual_gain=args.bfn_gain,
            tile=args.tile,
            overlap=args.overlap,
        )
        destination = (
            output_path
            if input_path.is_file()
            else output_path / path.relative_to(input_path)
        )
        save_image(prediction[0], destination)
        print(f"{path} -> {destination}")


if __name__ == "__main__":
    main()
