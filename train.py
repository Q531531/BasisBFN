#!/usr/bin/env python3
"""Train BasisBFN on paired degraded/clean images."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sgbformer.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from sgbformer.config import load_config
from sgbformer.data import (
    CategoryBalancedSampler,
    PairedImageDataset,
    classify_paths,
)
from sgbformer.ema import ExponentialMovingAverage
from sgbformer.losses import BasisBFNLoss
from sgbformer.metrics import LPIPSMetric, psnr, ssim
from sgbformer.models import build_model
from sgbformer.protocol import (
    loss_term_manifest,
    validate_budget_lock,
    validate_frozen_main_contract,
)
from sgbformer.training import (
    build_scheduler,
    clip_training_gradients,
    resolve_training_stages,
    stage_for_epoch,
    stage_loss_config,
    validate_resume_config,
)
from sgbformer.tripwire import require_tripwire_resume, write_tripwire_status
from sgbformer.utils import parameter_count, resolve_device, seed_everything
from sgbformer.v27 import validate_v27_config, validate_v27_launch_artifacts
from sgbformer.v28 import (
    V28_PROTOCOL,
    validate_v28_launch_artifacts,
    validate_v28_r0_config,
)
from sgbformer.basisbfn import (
    BASISBFN_PROTOCOL,
    validate_basisbfn_config,
    validate_basisbfn_launch_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument("--resume", help="Checkpoint to resume")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--seed", type=int, help="Override experiment seed")
    return parser.parse_args()


def setup_distributed(requested_device: str) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training currently requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, world_size
    return resolve_device(requested_device), rank, world_size


def build_dataset(
    config: dict[str, Any], training: bool, sample_seed: int | None = None
) -> PairedImageDataset:
    return PairedImageDataset(
        input_dir=config["input_dir"],
        target_dir=config["target_dir"],
        patch_size=int(config["patch_size"]) if training else None,
        augment=bool(config.get("augment", training)),
        pairs_file=config.get("pairs_file"),
        manifest_target_replace=config.get(
            "manifest_target_replace", ("input", "gt")
        ),
        strict=bool(config.get("strict", True)),
        sample_seed=sample_seed,
    )


def build_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> torch.optim.AdamW:
    base_lr = float(config.get("learning_rate", 2e-4))
    prefix_rates = {
        str(prefix): float(value)
        for prefix, value in config.get("parameter_lrs", {}).items()
    }
    grouped: dict[float, list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized_name = (
            name[len("module.") :] if name.startswith("module.") else name
        )
        learning_rate = base_lr
        matches = [
            (prefix, value)
            for prefix, value in prefix_rates.items()
            if normalized_name == prefix or normalized_name.startswith(prefix + ".")
        ]
        if matches:
            learning_rate = max(matches, key=lambda item: len(item[0]))[1]
        grouped.setdefault(learning_rate, []).append(parameter)
    if not grouped:
        raise ValueError("The model has no trainable parameters")
    parameter_groups = [
        {"params": parameters, "lr": learning_rate}
        for learning_rate, parameters in sorted(grouped.items())
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=base_lr,
        betas=tuple(config.get("betas", (0.9, 0.999))),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )


def latent_category_tensor(
    paths: list[str] | tuple[str, ...],
    patterns: dict[str, str],
    device: torch.device,
) -> torch.Tensor:
    names = tuple(patterns)
    indices = {name: index for index, name in enumerate(names)}
    labels = classify_paths(paths, patterns)
    return torch.tensor(
        [indices[label] for label in labels],
        device=device,
        dtype=torch.long,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    lpips_metric: LPIPSMetric | None = None,
    crop_border: int = 0,
    test_y_channel: bool = False,
    latent_category_patterns: dict[str, str] | None = None,
) -> dict[str, float]:
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    totals = {
        "psnr": 0.0,
        "ssim": 0.0,
        "structural_psnr": 0.0,
        "structural_ssim": 0.0,
        "foundation_psnr": 0.0,
        "foundation_ssim": 0.0,
    }
    if lpips_metric is not None:
        totals.update({
            "lpips": 0.0,
            "structural_lpips": 0.0,
            "foundation_lpips": 0.0,
        })
    count = 0
    value_count = 0
    foundation_error = 0.0
    structural_error = 0.0
    correction_dot = 0.0
    correction_energy = 0.0
    target_energy = 0.0
    mean_improved_images = 0
    output_error = 0.0
    bfn_correction_dot = 0.0
    bfn_correction_energy = 0.0
    bfn_target_energy = 0.0
    bfn_improved_images = 0
    for batch in loader:
        degraded = batch["degraded"].to(device)
        target = batch["target"].to(device)
        if getattr(base_model, "oracle_z_enabled", False):
            patterns = latent_category_patterns or {}
            category = latent_category_tensor(
                batch["input_path"],
                patterns,
                device,
            )
            details = base_model.restore_oracle_z(
                degraded,
                target,
                category=category,
                return_details=True,
            )
        else:
            details = base_model.restore(
                degraded,
                num_steps=steps,
                deterministic=True,
                return_details=True,
            )
        prediction = details["restored"]
        structural = details["structural"]
        foundation = details["foundation"]
        batch_size = degraded.shape[0]
        metric_options = {
            "crop_border": crop_border,
            "test_y_channel": test_y_channel,
        }
        prediction_psnr = psnr(prediction, target, **metric_options)
        totals["psnr"] += float(prediction_psnr.sum())
        totals["ssim"] += float(
            ssim(prediction, target, **metric_options).sum()
        )
        structural_psnr = psnr(structural, target, **metric_options)
        foundation_psnr = psnr(foundation, target, **metric_options)
        totals["structural_psnr"] += float(structural_psnr.sum())
        totals["structural_ssim"] += float(
            ssim(structural, target, **metric_options).sum()
        )
        totals["foundation_psnr"] += float(foundation_psnr.sum())
        totals["foundation_ssim"] += float(
            ssim(foundation, target, **metric_options).sum()
        )
        if lpips_metric is not None:
            totals["lpips"] += float(lpips_metric(prediction, target).sum())
            totals["structural_lpips"] += float(
                lpips_metric(structural, target).sum()
            )
            totals["foundation_lpips"] += float(
                lpips_metric(foundation, target).sum()
            )
        correction = structural.float() - foundation.float()
        target_residual = target.float() - foundation.float()
        foundation_error += float(target_residual.square().sum())
        structural_error += float(
            (target.float() - structural.float()).square().sum()
        )
        correction_dot += float((correction * target_residual).sum())
        correction_energy += float(correction.square().sum())
        target_energy += float(target_residual.square().sum())
        value_count += target.numel()
        mean_improved_images += int(
            (structural_psnr > foundation_psnr).sum()
        )
        bfn_correction = prediction.float() - structural.float()
        bfn_target = target.float() - structural.float()
        output_error += float(
            (target.float() - prediction.float()).square().sum()
        )
        bfn_correction_dot += float((bfn_correction * bfn_target).sum())
        bfn_correction_energy += float(bfn_correction.square().sum())
        bfn_target_energy += float(bfn_target.square().sum())
        bfn_improved_images += int(
            (prediction_psnr > structural_psnr).sum()
        )
        count += batch_size
    model.train()
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    metrics["delta_psnr"] = metrics["psnr"] - metrics["structural_psnr"]
    metrics["delta_ssim"] = metrics["ssim"] - metrics["structural_ssim"]
    metrics["mean_delta_psnr"] = (
        metrics["structural_psnr"] - metrics["foundation_psnr"]
    )
    metrics["mean_delta_ssim"] = (
        metrics["structural_ssim"] - metrics["foundation_ssim"]
    )
    metrics["foundation_aggregate_mse"] = (
        foundation_error / max(value_count, 1)
    )
    metrics["structural_aggregate_mse"] = (
        structural_error / max(value_count, 1)
    )
    metrics["mean_mse_relative_change"] = (
        metrics["structural_aggregate_mse"]
        / max(metrics["foundation_aggregate_mse"], 1e-12)
        - 1.0
    )
    metrics["mean_correction_cosine"] = correction_dot / max(
        math.sqrt(correction_energy * target_energy),
        1e-12,
    )
    metrics["mean_correction_rms_ratio"] = math.sqrt(
        correction_energy / max(target_energy, 1e-12)
    )
    metrics["mean_improved_fraction"] = (
        mean_improved_images / max(count, 1)
    )
    metrics["output_aggregate_mse"] = output_error / max(value_count, 1)
    metrics["bfn_mse_relative_change"] = (
        metrics["output_aggregate_mse"]
        / max(metrics["structural_aggregate_mse"], 1e-12)
        - 1.0
    )
    metrics["bfn_correction_cosine"] = bfn_correction_dot / max(
        math.sqrt(bfn_correction_energy * bfn_target_energy),
        1e-12,
    )
    metrics["bfn_correction_rms_ratio"] = math.sqrt(
        bfn_correction_energy / max(bfn_target_energy, 1e-12)
    )
    metrics["bfn_improved_fraction"] = (
        bfn_improved_images / max(count, 1)
    )
    if lpips_metric is not None:
        metrics["delta_lpips"] = metrics["lpips"] - metrics["structural_lpips"]
        metrics["mean_delta_lpips"] = (
            metrics["structural_lpips"] - metrics["foundation_lpips"]
        )
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    protocol_name = config.get("protocol", {}).get("name")
    v27_launch = None
    v28_launch = None
    if protocol_name == "frozen_main_db":
        validate_frozen_main_contract(config)
    elif protocol_name == "residual_bfn_v27":
        validate_v27_config(config)
        v27_launch = validate_v27_launch_artifacts(args.config, config)
    elif protocol_name == V28_PROTOCOL:
        validate_v28_r0_config(config)
        v28_launch = validate_v28_launch_artifacts(args.config, config)
    elif protocol_name == BASISBFN_PROTOCOL:
        validate_basisbfn_config(config)
        v28_launch = validate_basisbfn_launch_artifacts(
            args.config,
            config,
        )
    budget_manifest = validate_budget_lock(args.config, config)
    if args.seed is not None:
        if protocol_name in {
            "residual_bfn_v27",
            V28_PROTOCOL,
            BASISBFN_PROTOCOL,
        }:
            raise RuntimeError(
                f"The {protocol_name} protocol does not permit --seed overrides"
            )
        if config.get("training", {}).get("require_budget_lock", False):
            raise RuntimeError(
                "A locked protocol does not permit --seed overrides"
            )
        config["seed"] = args.seed
        config["training"]["output_dir"] = (
            f"{config['training']['output_dir']}_seed{args.seed}"
        )
    device, rank, world_size = setup_distributed(args.device)
    is_main = rank == 0
    locked_launch = v28_launch or v27_launch
    if locked_launch is not None:
        if world_size != 1 or device.type != "cuda":
            raise RuntimeError(
                "Registered residual profiles are single-GPU CUDA measurements"
            )
        profiled_device = locked_launch.get("profile_device")
        current_device = torch.cuda.get_device_name(device)
        if profiled_device is not None and current_device != profiled_device:
            raise RuntimeError(
                "Training GPU differs from the registered throughput profile: "
                f"{current_device!r} != {profiled_device!r}"
            )
    if budget_manifest is not None:
        if world_size != 1 or device.type != "cuda":
            raise RuntimeError(
                "Frozen-main profiles are single-GPU CUDA measurements"
            )
        arm = str(config["training"]["budget_arm"])
        profiled_device = budget_manifest["profiles"][arm].get("device")
        current_device = torch.cuda.get_device_name(device)
        if profiled_device is not None and current_device != profiled_device:
            raise RuntimeError(
                "Training GPU differs from the frozen throughput profile: "
                f"{current_device!r} != {profiled_device!r}"
            )
    seed = int(config.get("seed", 42)) + rank
    seed_everything(seed)

    train_config = config["training"]
    output_dir = Path(train_config.get("output_dir", "runs/sgbformer"))
    if not args.resume and (output_dir / "metrics.jsonl").exists():
        raise FileExistsError(
            f"{output_dir} already contains a run; resume it or use a new output_dir"
        )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(
            train_config.get("cudnn_benchmark", True)
        )
        allow_tf32 = bool(train_config.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    stages = resolve_training_stages(train_config)
    data_config = config["data"]
    reproducible_data = bool(train_config.get("reproducible_data", False))
    data_seed = int(config.get("seed", 42))
    train_dataset = build_dataset(
        data_config["train"],
        training=True,
        sample_seed=data_seed if reproducible_data else None,
    )
    if protocol_name in {
        "frozen_main_db",
        "residual_bfn_v27",
        V28_PROTOCOL,
        BASISBFN_PROTOCOL,
    }:
        expected_train = int(config["protocol"]["expected_train_pairs"])
        if len(train_dataset) != expected_train:
            raise RuntimeError(
                f"{protocol_name} requires {expected_train} train pairs, "
                f"got {len(train_dataset)}"
            )
    sampling_config = train_config.get("sampling", {})
    sampling_mode = str(sampling_config.get("mode", "shuffle")).lower()
    sampling_counts = None
    sampling_targets = None
    if sampling_mode == "category_balanced":
        if world_size != 1:
            raise ValueError(
                "category_balanced sampling currently requires world_size=1"
            )
        labels = classify_paths(
            [pair[0] for pair in train_dataset.pairs],
            sampling_config.get("category_patterns", {}),
            sampling_config.get("expected_counts"),
        )
        train_sampler = CategoryBalancedSampler(
            labels,
            sampling_config.get("target_fractions", {}),
            num_samples=len(train_dataset),
            seed=data_seed,
        )
        sampling_counts = dict(train_sampler.category_counts)
        sampling_targets = dict(train_sampler.target_counts)
    elif sampling_mode == "shuffle":
        train_sampler = (
            DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
                seed=data_seed,
            )
            if world_size > 1 or reproducible_data
            else None
        )
    else:
        raise ValueError(
            "training.sampling.mode must be shuffle or category_balanced"
        )
    workers = int(data_config.get("workers", 4))
    global_batch_size = int(train_config.get("batch_size", 16))
    configured_micro_batch = train_config.get("micro_batch_size")
    if configured_micro_batch is not None:
        per_device_batch_size = int(configured_micro_batch)
        denominator = per_device_batch_size * world_size
        if global_batch_size % denominator:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"micro batch {per_device_batch_size} * world size {world_size}"
            )
        accumulation = global_batch_size // denominator
    else:
        accumulation = int(train_config.get("gradient_accumulation", 1))
        denominator = world_size * accumulation
        if global_batch_size % denominator:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"world size {world_size} * accumulation {accumulation}"
            )
        per_device_batch_size = global_batch_size // denominator
    if per_device_batch_size < 1:
        raise ValueError(
            "Per-device micro batch is zero; reduce world size or increase "
            "the global batch"
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=True,
        generator=torch.Generator().manual_seed(data_seed) if reproducible_data else None,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training loader is empty. Reduce the global batch size or add data."
        )
    if config.get("protocol", {}).get("name") == "frozen_main_db":
        optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation)
        if optimizer_steps_per_epoch != 1130:
            raise RuntimeError(
                "Frozen-main requires exactly 1130 optimizer steps per epoch"
            )

    val_loader = None
    if is_main and data_config.get("val"):
        val_dataset = build_dataset(data_config["val"], training=False)
        if protocol_name in {
            "frozen_main_db",
            "residual_bfn_v27",
            V28_PROTOCOL,
            BASISBFN_PROTOCOL,
        }:
            expected_dev = int(config["protocol"]["expected_dev_pairs"])
            if len(val_dataset) != expected_dev:
                raise RuntimeError(
                    f"{protocol_name} requires {expected_dev} dev pairs, "
                    f"got {len(val_dataset)}"
                )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=max(0, workers // 2),
        )

    model = build_model(config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=len(stages) > 1,
        )
    optimizer = build_optimizer(model, train_config)
    epochs = int(train_config.get("epochs", 300))
    scheduler = build_scheduler(optimizer, train_config)
    ema = ExponentialMovingAverage(
        model,
        decay=float(train_config.get("ema_decay", 0.999)),
        warmup_updates=int(train_config.get("ema_warmup_updates", 2_000)),
    )
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    amp_dtype_name = str(train_config.get("amp_dtype", "float16")).lower()
    if amp_dtype_name not in {"float16", "bfloat16"}:
        raise ValueError("training.amp_dtype must be float16 or bfloat16")
    amp_dtype = (
        torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    start_epoch = 0
    global_step = 0
    best_psnr = float("-inf")
    best_lpips = float("inf")
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        if "optimizer" not in checkpoint or "scheduler" not in checkpoint:
            raise ValueError("Resume requires a full latest/epoch training checkpoint")
        validate_resume_config(checkpoint.get("config", {}), config)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        if checkpoint.get("best_psnr") is not None:
            best_psnr = float(checkpoint["best_psnr"])
        if checkpoint.get("best_lpips") is not None:
            best_lpips = float(checkpoint["best_lpips"])
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        if start_epoch >= epochs:
            raise ValueError("Resume checkpoint has already completed the configured epochs")
        history_path = output_dir / "metrics.jsonl"
        if history_path.exists():
            records = [
                json.loads(line) for line in history_path.read_text().splitlines()
                if line.strip()
            ]
            if records and (
                records[-1]["epoch"] != start_epoch
                or records[-1]["global_step"] != global_step
            ):
                raise ValueError(
                    "Resume checkpoint disagrees with existing history; "
                    "use a new output_dir for an earlier checkpoint"
                )
        if budget_manifest is not None:
            require_tripwire_resume(
                config_path=args.config,
                config=config,
                manifest=budget_manifest,
                start_epoch=start_epoch,
            )

    base_model = model.module if hasattr(model, "module") else model
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / "metrics.jsonl"
        if not args.resume:
            history_path.write_text("", encoding="utf-8")
        print(
            f"device={device} world_size={world_size} "
            f"train_images={len(train_dataset)} "
            f"global_batch={global_batch_size} "
            f"per_device_batch={per_device_batch_size} "
            f"gradient_accumulation={accumulation} "
            f"trainable_params={parameter_count(base_model):,}"
        )
        print(
            "optimizer_lrs="
            + ",".join(
                f"{group['lr']:.3e}" for group in optimizer.param_groups
            )
        )
        print(
            "loss_terms="
            + json.dumps(
                {
                    "arm": config.get("protocol", {}).get("arm", "unregistered"),
                    "weights": loss_term_manifest(config.get("loss", {})),
                },
                sort_keys=True,
            )
        )
        if budget_manifest is not None:
            print(
                "budget_lock="
                + json.dumps(
                    {
                        "requested_gpu_hours_per_arm": budget_manifest[
                            "requested_gpu_hours_per_arm"
                        ],
                        "effective_gpu_hours_per_arm": budget_manifest[
                            "effective_gpu_hours_per_arm"
                        ],
                        "hard_gpu_hour_cap_per_arm": budget_manifest[
                            "hard_gpu_hour_cap_per_arm"
                        ],
                        "budget_floor_reached": budget_manifest[
                            "budget_floor_reached"
                        ],
                        "steps_per_epoch": budget_manifest["steps_per_epoch"],
                        "epochs": budget_manifest["epochs"],
                    },
                    sort_keys=True,
                )
            )
        print(
            f"refinement_mode={base_model.refinement_mode} "
            f"bfn_steps={base_model.bfn_steps} "
            f"residual_scale={base_model.residual_scale:g} "
            f"training_final_state={base_model.training_final_state} "
            f"training_rollout_steps={base_model.training_rollout_steps} "
            f"rollout_backprop_steps={base_model.rollout_backprop_steps} "
            f"rollout_batch_fraction={base_model.rollout_batch_fraction:g} "
            f"receiver_output_mode={base_model.receiver_output_mode} "
            f"receiver_type={base_model.receiver_type} "
            f"residual_update_mode={base_model.residual_update_mode} "
            f"residual_readout_mode={base_model.residual_readout_mode} "
            f"residual_base_clamp={base_model.residual_base_clamp} "
            f"oracle_z={base_model.oracle_z_enabled} "
            f"state_condition_mode={base_model.decoder_state_mode} "
            f"decoder_state_cache={base_model.decoder_state_use_cache} "
            f"receiver_condition_mode={base_model.receiver_condition_mode} "
            f"mask_mode={base_model.mask_mode} "
            f"latent_update_mode={base_model.latent_update_mode} "
            f"latent_loss_weighting={base_model.latent_loss_weighting} "
            f"latent_adapter_precision_mode="
            f"{base_model.latent_adapter_precision_mode} "
            f"coefficient_state_readout="
            f"{base_model.coefficient_state_readout_enabled}"
        )
        if base_model.coefficient_state_readout_enabled:
            coefficient_readout = base_model.receiver.output
            print(
                "coefficient_bfn="
                + json.dumps(
                    {
                        "enabled": True,
                        "steps": coefficient_readout.steps,
                        "update_mode": coefficient_readout.update_mode,
                        "acceptance_gate": (
                            coefficient_readout.acceptance_gate_enabled
                        ),
                        "training_objective": "discrete_bfn",
                        "bfn_weight": float(
                            config["loss"]["bfn_weight"]
                        ),
                    },
                    sort_keys=True,
                )
            )
        if sampling_counts is not None:
            print(
                "sampling_source_counts="
                + json.dumps(sampling_counts, sort_keys=True)
            )
            print(
                "sampling_epoch_targets="
                + json.dumps(sampling_targets, sort_keys=True)
            )

    gradient_clip = float(train_config.get("gradient_clip", 1.0))
    log_interval = int(train_config.get("log_interval", 50))
    validate_interval = int(train_config.get("validate_interval", 1))
    save_interval = int(train_config.get("save_interval", 10))
    bfn_steps = int(config.get("model", {}).get("bfn", {}).get("steps", 10))
    latent_config = config.get("model", {}).get("latent_bfn", {})
    latent_category_patterns = latent_config.get("category_patterns", {})
    latent_category_names = tuple(latent_category_patterns)
    if base_model.latent_bfn_enabled:
        bfn_steps = base_model.latent_steps
        if len(latent_category_names) != base_model.latent_num_categories:
            raise ValueError(
                "latent_bfn.category_patterns must define exactly "
                f"{base_model.latent_num_categories} categories"
            )
    validation_config = config.get("validation", {})
    lpips_metric = (
        LPIPSMetric(str(validation_config.get("lpips_network", "alex"))).to(device)
        if validation_config.get("lpips", False) and val_loader is not None
        else None
    )
    active_stage = None
    throughput_started: float | None = None
    throughput_steps = 0
    profiled_seconds_per_step = None
    if budget_manifest is not None:
        arm = str(config["training"]["budget_arm"])
        profiled_seconds_per_step = float(
            budget_manifest["profiles"][arm]["seconds_per_optimizer_step"]
        )
    elif locked_launch is not None:
        profiled_seconds_per_step = float(
            locked_launch["profile_seconds_per_step"]
        )
    if checkpoint is not None:
        rng_states = checkpoint.get("rng_states")
        if rng_states is None:
            warnings.warn(
                "Legacy checkpoint has no RNG state; exact continuation is unavailable",
                RuntimeWarning,
            )
        else:
            if len(rng_states) != world_size:
                raise ValueError("Exact resume requires the original world_size")
            # Restore after model, optimizer and metric initialization consume RNG.
            restore_rng_state(rng_states[rank], device, train_loader.generator)
        if scaler.is_enabled() and "scaler" not in checkpoint:
            warnings.warn("Legacy checkpoint has no AMP scaler state", RuntimeWarning)

    for epoch in range(start_epoch, epochs):
        stage = stage_for_epoch(stages, epoch)
        criterion = BasisBFNLoss(
            **stage_loss_config(config.get("loss", {}), stage)
        ).to(device)
        if is_main and stage.name != active_stage:
            print(
                f"training_stage={stage.name} "
                f"epochs={stage.start_epoch + 1}-{stage.end_epoch}"
            )
            active_stage = stage.name
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.perf_counter()
        running = {
            "total": 0.0,
            "structural": 0.0,
            "bfn": 0.0,
            "final": 0.0,
            "correlation": 0.0,
            "frequency": 0.0,
            "adapter": 0.0,
            "uncertainty": 0.0,
            "latent_category": 0.0,
        }
        training_model = model.module if hasattr(model, "module") else model
        mean_corrector = training_model.mean_corrector
        receiver = training_model.receiver
        state_adapter = getattr(
            training_model.backbone,
            "state_conditioned_adapter",
            None,
        )
        last_mean_grad_norm = 0.0
        last_mean_output_norm = (
            float(mean_corrector.output.weight.detach().float().norm())
            if mean_corrector is not None
            else 0.0
        )
        last_receiver_grad_norm = 0.0
        last_receiver_output_norm = (
            float(receiver.output.weight.detach().float().norm())
            if receiver is not None
            else 0.0
        )
        last_state_adapter_grad_norm = 0.0
        last_state_adapter_output_norm = (
            float(state_adapter.output.weight.detach().float().norm())
            if state_adapter is not None
            else 0.0
        )

        for batch_index, batch in enumerate(train_loader):
            if throughput_started is None and profiled_seconds_per_step is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                throughput_started = time.perf_counter()
            degraded = batch["degraded"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            category = None
            if base_model.latent_bfn_enabled:
                category = latent_category_tensor(
                    batch["input_path"],
                    latent_category_patterns,
                    device,
                )
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(
                    degraded,
                    target,
                    training_stage=stage.name,
                    category=category,
                )
                loss, parts = criterion(outputs, target)
                remainder = len(train_loader) % accumulation
                final_group_start = len(train_loader) - remainder
                divisor = (
                    remainder
                    if remainder and batch_index >= final_group_start
                    else accumulation
                )
                scaled_loss = loss / divisor
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}"
                )
            scaler.scale(scaled_loss).backward()

            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                if (
                    gradient_clip > 0
                    or mean_corrector is not None
                    or receiver is not None
                    or state_adapter is not None
                ):
                    scaler.unscale_(optimizer)
                if mean_corrector is not None:
                    gradient_energy = sum(
                        float(parameter.grad.float().square().sum())
                        for parameter in mean_corrector.parameters()
                        if parameter.grad is not None
                    )
                    last_mean_grad_norm = math.sqrt(gradient_energy)
                if receiver is not None:
                    receiver_gradient_energy = sum(
                        float(parameter.grad.float().square().sum())
                        for parameter in receiver.parameters()
                        if parameter.grad is not None
                    )
                    last_receiver_grad_norm = math.sqrt(
                        receiver_gradient_energy
                    )
                if state_adapter is not None:
                    state_gradient_energy = sum(
                        float(parameter.grad.float().square().sum())
                        for parameter in state_adapter.parameters()
                        if parameter.grad is not None
                    )
                    last_state_adapter_grad_norm = math.sqrt(
                        state_gradient_energy
                    )
                if gradient_clip > 0:
                    clip_training_gradients(
                        model, gradient_clip,
                        separate_risk=bool(train_config.get("separate_risk_gradient_clip", False)),
                        error_if_nonfinite=not scaler.is_enabled(),
                    )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if mean_corrector is not None:
                    last_mean_output_norm = float(
                        mean_corrector.output.weight.detach().float().norm()
                    )
                if receiver is not None:
                    last_receiver_output_norm = float(
                        receiver.output.weight.detach().float().norm()
                    )
                if state_adapter is not None:
                    last_state_adapter_output_norm = float(
                        state_adapter.output.weight.detach().float().norm()
                    )
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= previous_scale:
                    ema.update(model)
                    global_step += 1
                    if profiled_seconds_per_step is not None:
                        throughput_steps += 1
                        if throughput_steps == 200:
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            measured = (
                                time.perf_counter() - throughput_started
                            ) / throughput_steps
                            deviation = abs(
                                measured / profiled_seconds_per_step - 1.0
                            )
                            print(
                                "throughput_guard="
                                + json.dumps(
                                    {
                                        "profile_seconds_per_step": (
                                            profiled_seconds_per_step
                                        ),
                                        "measured_seconds_per_step": measured,
                                        "absolute_deviation": deviation,
                                    },
                                    sort_keys=True,
                                )
                            )
                            if deviation > 0.30:
                                raise RuntimeError(
                                    "Throughput deviated from the frozen "
                                    "200-step profile by more than 30%"
                                )

            for key in running:
                running[key] += float(parts[key])
            if is_main and (batch_index + 1) % log_interval == 0:
                denominator = batch_index + 1
                print(
                    f"epoch={epoch + 1}/{epochs} batch={batch_index + 1}/"
                    f"{len(train_loader)} loss={running['total'] / denominator:.5f} "
                    f"struct={running['structural'] / denominator:.5f} "
                    f"bfn={running['bfn'] / denominator:.5f} "
                    f"final={running['final'] / denominator:.5f} "
                    f"corr={running['correlation'] / denominator:.5f} "
                    f"freq={running['frequency'] / denominator:.5f} "
                    f"unc={running['uncertainty'] / denominator:.5f} "
                    f"latent_cls={running['latent_category'] / denominator:.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}"
                    + (
                        f" mean_grad={last_mean_grad_norm:.3e} "
                        f"mean_out={last_mean_output_norm:.3e}"
                        if mean_corrector is not None
                        else ""
                    )
                    + (
                        f" receiver_grad={last_receiver_grad_norm:.3e} "
                        f"receiver_out={last_receiver_output_norm:.3e}"
                        if receiver is not None
                        else ""
                    )
                    + (
                        f" state_adapter_grad={last_state_adapter_grad_norm:.3e} "
                        f"state_adapter_out={last_state_adapter_output_norm:.3e}"
                        if state_adapter is not None
                        else ""
                    )
                )
        scheduler.step()

        if world_size > 1:
            dist.barrier()
        validation = None
        if is_main and val_loader is not None and (epoch + 1) % validate_interval == 0:
            with ema.average_parameters(model):
                validation = validate(
                    model,
                    val_loader,
                    device,
                    bfn_steps,
                    lpips_metric,
                    crop_border=int(
                        validation_config.get("crop_border", 0)
                    ),
                    test_y_channel=bool(
                        validation_config.get("test_y_channel", False)
                    ),
                    latent_category_patterns=latent_category_patterns,
                )
            print(
                f"validation epoch={epoch + 1} "
                f"psnr={validation['psnr']:.4f} "
                f"ssim={validation['ssim']:.5f} "
                f"mean_delta_psnr={validation['mean_delta_psnr']:+.4f} "
                f"mean_cos={validation['mean_correction_cosine']:+.4f} "
                f"mean_rms_ratio={validation['mean_correction_rms_ratio']:.4f} "
                f"bfn_delta_psnr={validation['delta_psnr']:+.4f} "
                f"bfn_cos={validation['bfn_correction_cosine']:+.4f} "
                f"bfn_rms_ratio={validation['bfn_correction_rms_ratio']:.4f}"
                + (
                    f" lpips={validation['lpips']:.5f} "
                    f"mean_delta_lpips={validation['mean_delta_lpips']:+.5f} "
                    f"bfn_delta_lpips={validation['delta_lpips']:+.5f}"
                    if "lpips" in validation
                    else ""
                )
            )
        if world_size > 1:
            dist.barrier()

        rng_state = capture_rng_state(device, train_loader.generator)
        rng_states = [rng_state]
        if world_size > 1:
            rng_states = [None] * world_size
            dist.all_gather_object(rng_states, rng_state)

        if is_main:
            elapsed = time.perf_counter() - epoch_start
            print(f"epoch={epoch + 1} elapsed={elapsed:.1f}s")
            train_metrics = {
                key: value / max(len(train_loader), 1)
                for key, value in running.items()
            }
            record = {
                "epoch": epoch + 1,
                "stage": stage.name,
                "global_step": global_step,
                "next_learning_rate": optimizer.param_groups[0]["lr"],
                "next_learning_rates": [group["lr"] for group in optimizer.param_groups],
                "elapsed_seconds": elapsed,
                "train": train_metrics,
                "optimization": {
                    "mean_gradient_norm": last_mean_grad_norm,
                    "mean_output_weight_norm": last_mean_output_norm,
                    "receiver_gradient_norm": last_receiver_grad_norm,
                    "receiver_output_weight_norm": last_receiver_output_norm,
                    "state_adapter_gradient_norm": last_state_adapter_grad_norm,
                    "state_adapter_output_weight_norm": (
                        last_state_adapter_output_norm
                    ),
                },
                "validation": validation,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            is_best = validation is not None and validation["psnr"] > best_psnr
            if is_best:
                best_psnr = validation["psnr"]
            is_best_lpips = (
                validation is not None
                and "lpips" in validation
                and validation["lpips"] < best_lpips
            )
            if is_best_lpips:
                best_lpips = validation["lpips"]
            save_checkpoint(
                output_dir / "latest.pt",
                model,
                config=config,
                epoch=epoch,
                global_step=global_step,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                scaler=scaler,
                rng_states=rng_states,
                best_psnr=best_psnr,
                best_lpips=best_lpips,
            )
            if (epoch + 1) % save_interval == 0:
                save_checkpoint(
                    output_dir / f"epoch_{epoch + 1:04d}.pt",
                    model,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    scaler=scaler,
                    rng_states=rng_states,
                    best_psnr=best_psnr,
                    best_lpips=best_lpips,
                )
            if is_best:
                save_checkpoint(
                    output_dir / "best_psnr.pt",
                    model,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    ema=ema,
                    best_psnr=best_psnr,
                    best_lpips=best_lpips,
                )
            if is_best_lpips:
                save_checkpoint(
                    output_dir / "best_lpips.pt",
                    model,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    ema=ema,
                    best_psnr=best_psnr,
                    best_lpips=best_lpips,
                )
            tripwire = config.get("protocol", {}).get("tripwire")
            if (
                budget_manifest is not None
                and isinstance(tripwire, dict)
                and tripwire.get("pause_after_epoch") is True
                and epoch + 1 == int(tripwire["epoch"])
            ):
                if validation is None:
                    raise RuntimeError(
                        "Epoch-4 tripwire requires a validation result"
                    )
                status = write_tripwire_status(
                    config_path=args.config,
                    config=config,
                    validation=validation,
                    checkpoint_path=output_dir / "latest.pt",
                )
                print(
                    "tripwire_pause="
                    + json.dumps(status, sort_keys=True)
                )
                return

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
