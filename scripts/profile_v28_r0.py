#!/usr/bin/env python3
"""Measure 200 optimizer steps for preflighted v28-R0."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sgbformer.config import load_config
from sgbformer.ema import ExponentialMovingAverage
from sgbformer.losses import SGBFormerLoss
from sgbformer.models import SignedSpatialBasisReadout, build_model
from sgbformer.protocol import implementation_fingerprint, sha256_file
from sgbformer.training import clip_training_gradients
from sgbformer.utils import parameter_count, seed_everything
from sgbformer.v27 import resolved_config_sha256
from sgbformer.v28 import (
    V28_EXPECTED_TRAIN_PAIRS,
    V28_PEAK_MEMORY_GATE_MB,
    V28_PROFILE_STEPS,
    V28_PROTOCOL,
    V28_R0_ARM,
    V28_R0_EXPECTED_TRAINABLE_PARAMETERS,
    validate_v28_r0_config,
)
from train import build_dataset, build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_verified_preflight(
    path: Path,
    *,
    config_path: Path,
    config: dict,
) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("protocol") != V28_PROTOCOL
        or report.get("arm") != V28_R0_ARM
        or report.get("status") != "passed"
    ):
        raise RuntimeError("Preflight report is not a passed v28-R0 artifact")
    recorded = report.get("config", {})
    if recorded.get("source_sha256") != sha256_file(config_path):
        raise RuntimeError("v28-R0 preflight source config fingerprint changed")
    if recorded.get("resolved_sha256") != resolved_config_sha256(config):
        raise RuntimeError("v28-R0 preflight resolved config fingerprint changed")
    if (
        report.get("implementation", {}).get("sha256")
        != implementation_fingerprint()["sha256"]
    ):
        raise RuntimeError("Implementation changed after v28-R0 preflight")
    return report


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    preflight_path = Path(args.preflight).expanduser().resolve()
    config = load_config(config_path)
    validate_v28_r0_config(config)
    preflight = load_verified_preflight(
        preflight_path,
        config_path=config_path,
        config=config,
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v28-R0 throughput profiling requires CUDA")

    seed_everything(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    training = config["training"]
    dataset = build_dataset(
        config["data"]["train"],
        training=True,
        sample_seed=42,
    )
    if len(dataset) != V28_EXPECTED_TRAIN_PAIRS:
        raise RuntimeError(
            f"v28-R0 requires {V28_EXPECTED_TRAIN_PAIRS} train pairs, "
            f"got {len(dataset)}"
        )
    micro_batch = int(training["micro_batch_size"])
    global_batch = int(training["batch_size"])
    if global_batch % micro_batch:
        raise RuntimeError("Global batch must be divisible by micro batch")
    accumulation = global_batch // micro_batch
    if training.get("sampling") != {"mode": "shuffle"}:
        raise RuntimeError("v28-R0 profile requires natural shuffle sampling")
    print(
        "sampling_banner="
        + json.dumps(
            {"mode": "shuffle", "category_balanced": False},
            sort_keys=True,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=int(config["data"]["workers"]),
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(42),
    )
    if len(loader) < V28_PROFILE_STEPS * accumulation:
        raise RuntimeError("One data pass cannot supply the 200-step profile")

    model = build_model(config).to(device).train()
    if parameter_count(model) != V28_R0_EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("v28-R0 trainable parameter count changed")
    if model.receiver is None or not isinstance(
        model.receiver.output,
        SignedSpatialBasisReadout,
    ):
        raise RuntimeError("v28-R0 signed basis readout is missing")
    readout = model.receiver.output
    optimizer = build_optimizer(model, training)
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("v28-R0 profile requires one optimizer group")
    criterion = SGBFormerLoss(**config["loss"]).to(device)
    ema = ExponentialMovingAverage(
        model,
        decay=float(training["ema_decay"]),
        warmup_updates=int(training["ema_warmup_updates"]),
    )
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    initial_candidate_loss = None
    initial_f0_baseline_loss = None
    initial_loss_relative_difference = None
    coefficient_absolute_sum = 0.0
    coefficient_maximum = 0.0
    coefficient_values = 0

    for step_index in range(V28_PROFILE_STEPS):
        for micro_index in range(accumulation):
            batch = next(iterator)
            degraded = batch["degraded"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):
                outputs = model(
                    degraded,
                    target,
                    training_stage="residual",
                )
                loss, _ = criterion(outputs, target)
                scaled_loss = loss / accumulation
            if step_index == 0 and micro_index == 0:
                initial_candidate_loss = float(loss.detach())
                initial_f0_baseline_loss = float(
                    F.mse_loss(
                        outputs["structural"].float(),
                        target.float(),
                    )
                )
                initial_loss_relative_difference = abs(
                    initial_candidate_loss - initial_f0_baseline_loss
                ) / max(abs(initial_f0_baseline_loss), 1e-12)
                if initial_loss_relative_difference > 0.01:
                    raise RuntimeError(
                        "v28-R0 initial loss differs from F0 by more than 1%"
                    )
            coefficients = readout.last_coefficients
            if coefficients is None or not torch.isfinite(coefficients).all():
                raise FloatingPointError(
                    "Non-finite or missing signed coefficients during profile"
                )
            coefficient_absolute_sum += float(coefficients.abs().sum())
            coefficient_maximum = max(
                coefficient_maximum,
                float(coefficients.abs().max()),
            )
            coefficient_values += coefficients.numel()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss during v28-R0 profile"
                )
            scaled_loss.backward()
        clip_training_gradients(
            model,
            float(training["gradient_clip"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

    synchronize(device)
    elapsed = time.perf_counter() - started
    seconds_per_step = elapsed / V28_PROFILE_STEPS
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise RuntimeError("Invalid v28-R0 throughput measurement")
    peak_memory_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
    )
    if peak_memory_allocated_mb > V28_PEAK_MEMORY_GATE_MB:
        raise RuntimeError(
            "v28-R0 peak allocated memory exceeds the 2.0 GiB gate: "
            f"{peak_memory_allocated_mb:.2f} MiB"
        )
    coefficient_absolute_mean = (
        coefficient_absolute_sum / max(coefficient_values, 1)
    )
    if coefficient_absolute_mean <= 0:
        raise RuntimeError("v28-R0 coefficients remained inactive")
    report = {
        "schema_version": 1,
        "protocol": V28_PROTOCOL,
        "arm": V28_R0_ARM,
        "status": "passed",
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_config_sha256(config),
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
            "implementation_sha256": preflight["implementation"]["sha256"],
        },
        "implementation": implementation_fingerprint(),
        "optimizer_steps": V28_PROFILE_STEPS,
        "micro_batches_per_optimizer_step": accumulation,
        "global_batch_size": global_batch,
        "micro_batch_size": micro_batch,
        "train_pairs": len(dataset),
        "elapsed_seconds": elapsed,
        "seconds_per_optimizer_step": seconds_per_step,
        "throughput_images_per_second": global_batch / seconds_per_step,
        "peak_memory_allocated_mb": peak_memory_allocated_mb,
        "peak_memory_gate_mb": V28_PEAK_MEMORY_GATE_MB,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "trainable_parameters": parameter_count(model),
        "receiver_calls_per_image": 1,
        "sampling": {"mode": "shuffle"},
        "initial_loss_check": {
            "candidate": initial_candidate_loss,
            "f0_baseline": initial_f0_baseline_loss,
            "relative_difference": initial_loss_relative_difference,
            "maximum_relative_difference": 0.01,
            "passed": True,
        },
        "coefficient_diagnostics": {
            "absolute_mean": coefficient_absolute_mean,
            "absolute_maximum": coefficient_maximum,
            "range": [-1.0, 1.0],
            "observations": coefficient_values,
        },
        "quality_metrics_observed": False,
        "scope": (
            "200 optimizer steps including data loading, forward, backward, "
            "gradient clipping, optimizer, and EMA; no validation or quality "
            "metrics"
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
