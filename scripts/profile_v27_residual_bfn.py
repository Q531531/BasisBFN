#!/usr/bin/env python3
"""Measure 200 optimizer steps for one preflighted v27 arm."""

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
from sgbformer.checkpoint import load_checkpoint
from sgbformer.data import CategoryBalancedSampler, classify_paths
from sgbformer.ema import ExponentialMovingAverage
from sgbformer.losses import SGBFormerLoss
from sgbformer.models import build_model
from sgbformer.protocol import implementation_fingerprint, sha256_file
from sgbformer.training import clip_training_gradients
from sgbformer.training import validate_resume_config
from sgbformer.utils import parameter_count, seed_everything
from sgbformer.v27 import (
    V27_C0ACCEPT_ARM,
    V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS,
    V27_C0DOUBLEPRIME_ARM,
    V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS,
    V27_C0PRIME_ARM,
    V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS,
    V27_EXPECTED_TRAINABLE_PARAMETERS,
    V27_EXPECTED_TRAIN_PAIRS,
    resolved_config_sha256,
    validate_v27_c0_e1_config,
    validate_v27_c0accept_config,
    validate_v27_c0doubleprime_config,
    validate_v27_c0prime_config,
    validate_v27_config,
)
from train import build_dataset, build_optimizer


OPTIMIZER_STEPS = 200
ROUTED_READOUT_ARMS = {
    V27_C0PRIME_ARM,
    V27_C0DOUBLEPRIME_ARM,
    V27_C0ACCEPT_ARM,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initialization-checkpoint")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_verified_preflight(
    path: Path,
    *,
    arm: str,
    config_path: Path,
    config: dict,
    initialization_checkpoint: Path | None = None,
) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_protocol = (
        "residual_bfn_v27_exploratory"
        if arm in {"C0-E1", *ROUTED_READOUT_ARMS}
        else "residual_bfn_v27"
    )
    if report.get("protocol") != expected_protocol:
        raise RuntimeError("Preflight report is not a v27 artifact")
    if report.get("status") != "passed":
        raise RuntimeError("v27 preflight did not pass")
    recorded = report.get("configs", {}).get(arm, {})
    if recorded.get("source_sha256") != sha256_file(config_path):
        raise RuntimeError(f"Preflight source config fingerprint changed for {arm}")
    if recorded.get("resolved_sha256") != resolved_config_sha256(config):
        raise RuntimeError(f"Preflight config fingerprint changed for {arm}")
    current_core = implementation_fingerprint()["sha256"]
    recorded_implementation = (
        report.get("implementation", {}).get("sha256")
        if arm in {"C0-E1", *ROUTED_READOUT_ARMS}
        else report.get("source", {}).get("core", {}).get("sha256")
    )
    if recorded_implementation != current_core:
        raise RuntimeError("Implementation changed after v27 preflight")
    if arm == V27_C0ACCEPT_ARM:
        if initialization_checkpoint is None:
            raise RuntimeError(
                "C0-accept profile requires --initialization-checkpoint"
            )
        recorded_initialization = report.get(
            "initialization_checkpoint",
            {},
        )
        if (
            recorded_initialization.get("sha256")
            != sha256_file(initialization_checkpoint)
        ):
            raise RuntimeError(
                "C0-accept initialization checkpoint changed after preflight"
            )
    return report


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    if (
        config.get("protocol", {}).get("name")
        == "residual_bfn_v27_exploratory"
    ):
        reference = load_config(
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "ssr_residual_v27_c0_proxy.yaml"
        )
        exploratory_arm = config.get("protocol", {}).get("arm")
        if exploratory_arm == "C0-E1":
            arm = validate_v27_c0_e1_config(config, reference)
        elif exploratory_arm == V27_C0PRIME_ARM:
            arm = validate_v27_c0prime_config(config, reference)
        elif exploratory_arm == V27_C0DOUBLEPRIME_ARM:
            reference_c0prime = load_config(
                PROJECT_ROOT
                / "configs"
                / "experiments"
                / "ssr_residual_v27_c0prime_routed.yaml"
            )
            arm = validate_v27_c0doubleprime_config(
                config,
                reference,
                reference_c0prime,
            )
        elif exploratory_arm == V27_C0ACCEPT_ARM:
            reference_c0prime = load_config(
                PROJECT_ROOT
                / "configs"
                / "experiments"
                / "ssr_residual_v27_c0prime_routed.yaml"
            )
            reference_c0doubleprime = load_config(
                PROJECT_ROOT
                / "configs"
                / "experiments"
                / "ssr_residual_v27_c0doubleprime_bottleneck_routed.yaml"
            )
            arm = validate_v27_c0accept_config(
                config,
                reference,
                reference_c0prime,
                reference_c0doubleprime,
            )
        else:
            raise ValueError(
                f"Unsupported v27 exploratory arm: {exploratory_arm!r}"
            )
    else:
        arm = validate_v27_config(config)
    preflight_path = Path(args.preflight).expanduser().resolve()
    initialization_checkpoint = (
        Path(args.initialization_checkpoint).expanduser().resolve()
        if args.initialization_checkpoint
        else None
    )
    preflight = load_verified_preflight(
        preflight_path,
        arm=arm,
        config_path=config_path,
        config=config,
        initialization_checkpoint=initialization_checkpoint,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v27 throughput profiling requires CUDA")
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
    if len(dataset) != V27_EXPECTED_TRAIN_PAIRS:
        raise RuntimeError(
            f"v27 requires {V27_EXPECTED_TRAIN_PAIRS} train pairs, "
            f"got {len(dataset)}"
        )
    micro_batch = int(training["micro_batch_size"])
    global_batch = int(training["batch_size"])
    if global_batch % micro_batch:
        raise RuntimeError("Global batch must be divisible by micro batch")
    accumulation = global_batch // micro_batch
    sampling = training.get("sampling", {"mode": "shuffle"})
    print(
        "sampling_banner="
        + json.dumps(
            {
                "mode": sampling.get("mode"),
                "category_balanced": (
                    sampling.get("mode") == "category_balanced"
                ),
            },
            sort_keys=True,
        )
    )
    sampler = None
    sampler_report = None
    if sampling.get("mode") == "category_balanced":
        labels = classify_paths(
            [pair[0] for pair in dataset.pairs],
            sampling["category_patterns"],
        )
        sampler = CategoryBalancedSampler(
            labels,
            sampling["target_fractions"],
            num_samples=len(dataset),
            seed=42,
        )
        sampler_report = {
            "source_counts": dict(sampler.category_counts),
            "target_counts": dict(sampler.target_counts),
        }
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config["data"]["workers"]),
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(42),
    )
    if len(loader) < OPTIMIZER_STEPS * accumulation:
        raise RuntimeError("One data pass cannot supply the 200-step profile")

    model = build_model(config).to(device).train()
    expected_parameters = {
        V27_C0ACCEPT_ARM: V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS,
        V27_C0PRIME_ARM: V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS,
        V27_C0DOUBLEPRIME_ARM: (
            V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }.get(arm, V27_EXPECTED_TRAINABLE_PARAMETERS)
    if parameter_count(model) != expected_parameters:
        raise RuntimeError("v27 trainable parameter count changed")
    initialization = None
    if arm == V27_C0ACCEPT_ARM:
        assert initialization_checkpoint is not None
        initialization = load_checkpoint(
            initialization_checkpoint,
            model,
            map_location=device,
        )
        validate_resume_config(initialization["config"], config)
        if (
            int(initialization.get("epoch", -2)) != -1
            or int(initialization.get("global_step", -1)) != 0
        ):
            raise RuntimeError(
                "C0-accept initialization must start before epoch 1"
            )
    optimizer = build_optimizer(model, training)
    criterion = SGBFormerLoss(**config["loss"]).to(device)
    ema = ExponentialMovingAverage(
        model,
        decay=float(training["ema_decay"]),
        warmup_updates=int(training["ema_warmup_updates"]),
    )
    stage = (
        "residual"
        if arm in {"C0", "C0-E1", *ROUTED_READOUT_ARMS}
        else "joint"
    )
    gradient_clip = float(training["gradient_clip"])
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    initial_candidate_loss = None
    initial_f0_baseline_loss = None
    initial_loss_relative_difference = None
    adapter_diagnostic_sum = 0.0
    adapter_diagnostic_observations = 0

    for step_index in range(OPTIMIZER_STEPS):
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
                    training_stage=stage,
                )
                loss, _ = criterion(outputs, target)
                scaled_loss = loss / accumulation
            if (
                arm in ROUTED_READOUT_ARMS
                and step_index == 0
                and micro_index == 0
            ):
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
                        f"{arm} initial loss differs from F0 by more than 1%"
                    )
            if arm in ROUTED_READOUT_ARMS:
                adapter_diagnostic_sum += float(
                    outputs["adapter_regularization"]
                )
                adapter_diagnostic_observations += 1
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss during v27 profile")
            scaled_loss.backward()
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
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise RuntimeError("Invalid v27 throughput measurement")
    peak_memory_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
    )
    if (
        arm in ROUTED_READOUT_ARMS
        and peak_memory_allocated_mb > 2.0 * 1024
    ):
        raise RuntimeError(
            f"{arm} peak allocated memory exceeds the 2.0 GiB gate: "
            f"{peak_memory_allocated_mb:.2f} MiB"
        )
    acceptance_gate_mean = (
        adapter_diagnostic_sum / adapter_diagnostic_observations
        if adapter_diagnostic_observations
        and arm == V27_C0ACCEPT_ARM
        else None
    )
    if arm == V27_C0ACCEPT_ARM and (
        acceptance_gate_mean is None
        or not 0.0 < acceptance_gate_mean <= 1.0
    ):
        raise RuntimeError(
            "C0-accept gate remained inactive or left [0,1] during profile: "
            f"{acceptance_gate_mean}"
        )
    report = {
        "schema_version": 1,
        "protocol": config["protocol"]["name"],
        "arm": arm,
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_config_sha256(config),
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
            "source_sha256": (
                preflight["implementation"]["sha256"]
                if arm in {"C0-E1", *ROUTED_READOUT_ARMS}
                else preflight["source"]["sha256"]
            ),
        },
        "initialization_checkpoint": (
            {
                "path": str(initialization_checkpoint),
                "sha256": sha256_file(initialization_checkpoint),
                "epoch": int(initialization["epoch"]),
                "global_step": int(initialization["global_step"]),
            }
            if initialization is not None
            and initialization_checkpoint is not None
            else None
        ),
        "implementation": implementation_fingerprint(),
        "optimizer_steps": OPTIMIZER_STEPS,
        "micro_batches_per_optimizer_step": accumulation,
        "global_batch_size": global_batch,
        "micro_batch_size": micro_batch,
        "train_pairs": len(dataset),
        "elapsed_seconds": elapsed,
        "seconds_per_optimizer_step": seconds_per_step,
        "throughput_images_per_second": global_batch / seconds_per_step,
        "peak_memory_allocated_mb": peak_memory_allocated_mb,
        "peak_memory_gate_mb": (
            2.0 * 1024 if arm in ROUTED_READOUT_ARMS else None
        ),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "trainable_parameters": parameter_count(model),
        "receiver_calls_per_image": (
            1
            if arm in {"C0", "C0-E1", "D1", *ROUTED_READOUT_ARMS}
            else 4
        ),
        "sampling": sampler_report or {"mode": "shuffle"},
        "initial_loss_check": (
            {
                "candidate": initial_candidate_loss,
                "f0_baseline": initial_f0_baseline_loss,
                "relative_difference": initial_loss_relative_difference,
                "maximum_relative_difference": 0.01,
                "passed": True,
            }
            if arm in ROUTED_READOUT_ARMS
            else None
        ),
        "router_entropy_mean": (
            adapter_diagnostic_sum / adapter_diagnostic_observations
            if adapter_diagnostic_observations
            and arm != V27_C0ACCEPT_ARM
            else None
        ),
        "router_entropy_observations": (
            adapter_diagnostic_observations
            if arm != V27_C0ACCEPT_ARM
            else 0
        ),
        "acceptance_gate_mean": (
            acceptance_gate_mean
        ),
        "acceptance_gate_observations": (
            adapter_diagnostic_observations
            if arm == V27_C0ACCEPT_ARM
            else 0
        ),
        "quality_metrics_observed": False,
        "scope": (
            "200 optimizer steps including data loading, forward, backward, "
            "gradient clipping, optimizer, and EMA; no validation or quality metrics"
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
