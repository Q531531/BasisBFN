#!/usr/bin/env python3
"""Evaluate only the deterministic restoration foundation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sgbformer.comparison import foundation_signature
from sgbformer.config import load_config
from sgbformer.data import PairedImageDataset
from sgbformer.metrics import LPIPSMetric, psnr, ssim
from sgbformer.models import build_model
from sgbformer.utils import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--pairs-file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop-border", type=int, default=0)
    parser.add_argument("--test-y-channel", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    config = load_config(args.config)
    model = build_model(config).to(device).eval()
    dataset = PairedImageDataset(
        args.input_dir,
        args.target_dir,
        pairs_file=args.pairs_file,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    lpips_metric = LPIPSMetric().to(device) if args.lpips else None
    metric_options = {
        "crop_border": args.crop_border,
        "test_y_channel": args.test_y_channel,
    }

    rows = []
    inference_seconds = 0.0
    inference_images = 0
    peak_memory_bytes = 0
    peak_incremental_memory_bytes = 0
    with torch.inference_mode():
        for batch in loader:
            degraded = batch["degraded"].to(device)
            target = batch["target"].to(device)
            baseline_memory_bytes = 0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                baseline_memory_bytes = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            inference_start = time.perf_counter()
            structural, _, _, original_size = model.structural_forward(
                degraded
            )
            prediction = structural[
                ..., : original_size[0], : original_size[1]
            ].clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory_bytes = max(
                    peak_memory_bytes,
                    torch.cuda.max_memory_allocated(device),
                )
                peak_incremental_memory_bytes = max(
                    peak_incremental_memory_bytes,
                    torch.cuda.max_memory_allocated(device)
                    - baseline_memory_bytes,
                )
            inference_seconds += time.perf_counter() - inference_start
            inference_images += degraded.shape[0]
            row = {
                "name": batch["name"][0],
                "psnr": float(
                    psnr(prediction, target, **metric_options)[0]
                ),
                "ssim": float(
                    ssim(prediction, target, **metric_options)[0]
                ),
            }
            if lpips_metric is not None:
                row["lpips"] = float(lpips_metric(prediction, target)[0])
            rows.append(row)
            print(
                f"{row['name']}: PSNR={row['psnr']:.4f} "
                f"SSIM={row['ssim']:.5f}"
                + (
                    f" LPIPS={row['lpips']:.5f}"
                    if "lpips" in row
                    else ""
                )
            )

    if not rows:
        raise RuntimeError("The evaluation dataset is empty")
    metric_names = [key for key in rows[0] if key != "name"]
    average = {
        metric: sum(row[metric] for row in rows) / len(rows)
        for metric in metric_names
    }
    print(
        "Average: "
        + " ".join(f"{key.upper()}={value:.5f}" for key, value in average.items())
    )
    runtime = {
        "images": inference_images,
        "restoration_trajectories": inference_images,
        "total_inference_seconds": inference_seconds,
        "average_inference_seconds_per_image": (
            inference_seconds / max(inference_images, 1)
        ),
        "average_restoration_trajectories_per_image": 1.0,
        "average_decoder_calls_per_image": 1.0,
        "peak_memory_allocated_mb": (
            peak_memory_bytes / (1024**2)
            if device.type == "cuda"
            else None
        ),
        "peak_incremental_memory_mb": (
            peak_incremental_memory_bytes / (1024**2)
            if device.type == "cuda"
            else None
        ),
        "device": str(device),
        "scope": (
            "latency excludes metrics; total peak includes resident models; "
            "incremental peak is measured above pre-restoration allocation"
        ),
    }
    print("Runtime: " + json.dumps(runtime, ensure_ascii=True))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "average": average,
                    "images": rows,
                    "runtime": runtime,
                    "protocol": {
                        "test_y_channel": args.test_y_channel,
                        "crop_border": args.crop_border,
                        "lpips_network": "alex" if args.lpips else None,
                        "stochastic": False,
                        "tile": None,
                        "overlap": None,
                        "output_quantization": "float32_clamp_0_1",
                        "refinement_mode": "foundation_only",
                        "steps": 0,
                        "latent_bfn_steps": 0,
                        "decoder_calls_per_trajectory": 1,
                        "foundation_signature": foundation_signature(config),
                    },
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
