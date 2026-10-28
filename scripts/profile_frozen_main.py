#!/usr/bin/env python3
"""Measure exactly 200 optimizer steps for one unlocked frozen D/B arm."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sgbformer.config import load_config
from sgbformer.ema import ExponentialMovingAverage
from sgbformer.losses import SGBFormerLoss
from sgbformer.models import build_model
from sgbformer.protocol import (
    foundation_artifact_fingerprint,
    implementation_fingerprint,
    loss_term_manifest,
    sha256_file,
    validate_frozen_main_contract,
)
from sgbformer.training import clip_training_gradients
from sgbformer.utils import parameter_count, seed_everything
from train import build_dataset, build_optimizer, latent_category_tensor


OPTIMIZER_STEPS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    arm = validate_frozen_main_contract(config)
    training = config["training"]
    if training.get("budget_locked", False):
        raise RuntimeError("Profile the unlocked source config, not a locked config")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Frozen-main throughput profiling requires CUDA")
    seed_everything(int(config["seed"]))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dataset = build_dataset(
        config["data"]["train"],
        training=True,
        sample_seed=int(config["seed"]),
    )
    expected_pairs = int(config["protocol"]["expected_train_pairs"])
    if len(dataset) != expected_pairs:
        raise RuntimeError(
            f"Frozen-main requires {expected_pairs} train pairs, got {len(dataset)}"
        )
    micro_batch = int(training["micro_batch_size"])
    global_batch = int(training["batch_size"])
    if global_batch % micro_batch:
        raise RuntimeError("Global batch must be divisible by micro batch")
    accumulation = global_batch // micro_batch
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=int(config["data"]["workers"]),
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    required_micro_batches = OPTIMIZER_STEPS * accumulation
    if len(loader) < required_micro_batches:
        raise RuntimeError("One data pass cannot supply the 200-step profile")

    model = build_model(config).to(device).train()
    optimizer = build_optimizer(model, training)
    criterion = SGBFormerLoss(**config["loss"]).to(device)
    ema = ExponentialMovingAverage(
        model,
        decay=float(training["ema_decay"]),
        warmup_updates=int(training["ema_warmup_updates"]),
    )
    patterns = config["model"]["latent_bfn"]["category_patterns"]
    gradient_clip = float(training.get("gradient_clip", 1.0))
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()

    for _ in range(OPTIMIZER_STEPS):
        for _ in range(accumulation):
            batch = next(iterator)
            degraded = batch["degraded"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            category = latent_category_tensor(
                batch["input_path"],
                patterns,
                device,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):
                outputs = model(
                    degraded,
                    target,
                    training_stage="joint",
                    category=category,
                )
                loss, _ = criterion(outputs, target)
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss during throughput profile")
            scaled_loss.backward()
        if gradient_clip > 0:
            clip_training_gradients(
                model,
                gradient_clip,
                error_if_nonfinite=True,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

    synchronize(device)
    elapsed = time.perf_counter() - started
    seconds_per_step = elapsed / OPTIMIZER_STEPS
    report = {
        "schema_version": 1,
        "protocol": "frozen_main_db",
        "arm": arm,
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "implementation": implementation_fingerprint(),
        "foundation_artifacts": foundation_artifact_fingerprint(config),
        "optimizer_steps": OPTIMIZER_STEPS,
        "micro_batches_per_optimizer_step": accumulation,
        "global_batch_size": global_batch,
        "micro_batch_size": micro_batch,
        "train_pairs": len(dataset),
        "elapsed_seconds": elapsed,
        "seconds_per_optimizer_step": seconds_per_step,
        "throughput_images_per_second": global_batch / seconds_per_step,
        "peak_memory_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        ),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "trainable_parameters": parameter_count(model),
        "loss_terms": loss_term_manifest(config["loss"]),
        "quality_metrics_observed": False,
        "scope": (
            "200 optimizer steps including data loading, forward, backward, "
            "gradient clipping, optimizer, and EMA; no validation or quality metrics"
        ),
    }
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise RuntimeError("Invalid throughput measurement")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
