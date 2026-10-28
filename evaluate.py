#!/usr/bin/env python3
"""Evaluate BasisBFN on a paired image dataset."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sgbformer.comparison import foundation_signature
from sgbformer.data import PairedImageDataset
from sgbformer.inference import (
    load_model_for_inference,
    restoration_tile_count,
    restore_structural_tensor,
    restore_tensor,
)
from sgbformer.metrics import LPIPSMetric, psnr, ssim, uncertainty_scores
from sgbformer.utils import resolve_device, save_image, seed_everything


def parse_category_patterns(values: list[str]) -> dict[str, re.Pattern[str]]:
    patterns = {}
    for value in values:
        name, separator, pattern = value.partition("=")
        if not separator or not name or not pattern:
            raise ValueError(
                "--category-pattern must have NAME=REGEX form"
            )
        if name in patterns:
            raise ValueError(f"Duplicate category pattern: {name}")
        patterns[name] = re.compile(pattern, flags=re.IGNORECASE)
    return patterns


def category_for_path(
    path: str,
    patterns: dict[str, re.Pattern[str]],
) -> str | None:
    if not patterns:
        return None
    filename = Path(path).name
    matches = [
        name for name, pattern in patterns.items()
        if pattern.fullmatch(filename)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{filename!r} must match exactly one configured category; "
            f"matched {matches}"
        )
    return matches[0]


def correction_diagnostics(
    prediction: torch.Tensor,
    structural: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    metric_options: dict[str, int | bool],
) -> dict[str, float]:
    correction = (prediction - structural).float()
    target_correction = (target - structural).float()
    active = mask.float()
    dot = (correction * target_correction * active).sum()
    correction_norm = (correction.square() * active).sum().sqrt()
    target_norm = (target_correction.square() * active).sum().sqrt()
    cosine = dot / (correction_norm * target_norm).clamp_min(1e-12)
    oracle = (
        structural + active * target_correction
    ).clamp(0, 1)
    return {
        "bfn_active_fraction": float(active.mean()),
        "bfn_correction_rms": float(correction.square().mean().sqrt()),
        "target_residual_rms": float(
            target_correction.square().mean().sqrt()
        ),
        "bfn_correction_cosine": float(cosine),
        "entropy_mask_oracle_psnr": float(
            psnr(oracle, target, **metric_options)[0]
        ),
    }


SELF_ENSEMBLE_TRANSFORMS = 8


def _geometric_view(tensor: torch.Tensor, index: int) -> torch.Tensor:
    """Apply one of the eight dihedral views (4 rotations x optional flip)."""

    if not 0 <= index < SELF_ENSEMBLE_TRANSFORMS:
        raise ValueError("Self-ensemble index must lie in [0, 8)")
    view = torch.flip(tensor, dims=(-1,)) if index >= 4 else tensor
    return torch.rot90(view, index % 4, dims=(-2, -1))


def _inverse_geometric_view(tensor: torch.Tensor, index: int) -> torch.Tensor:
    """Undo :func:`_geometric_view` so outputs share the input orientation."""

    if not 0 <= index < SELF_ENSEMBLE_TRANSFORMS:
        raise ValueError("Self-ensemble index must lie in [0, 8)")
    view = torch.rot90(tensor, -(index % 4), dims=(-2, -1))
    return torch.flip(view, dims=(-1,)) if index >= 4 else view


def self_ensemble_restore(
    restore: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
) -> torch.Tensor:
    """Average eight geometric restorations in the pixel domain."""

    accumulator: torch.Tensor | None = None
    for index in range(SELF_ENSEMBLE_TRANSFORMS):
        restored = restore(_geometric_view(image, index))
        if not isinstance(restored, torch.Tensor):
            raise TypeError("Model restoration must return a tensor")
        restored = _inverse_geometric_view(restored, index)
        if restored.shape != image.shape:
            raise RuntimeError(
                "Self-ensemble restoration changed the image geometry: "
                f"{tuple(restored.shape)} != {tuple(image.shape)}"
            )
        accumulator = (
            restored if accumulator is None else accumulator + restored
        )
    assert accumulator is not None
    return accumulator / SELF_ENSEMBLE_TRANSFORMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--pairs-file")
    parser.add_argument("--config")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--state-ablation",
        choices=("normal", "reset-belief", "shuffle-belief"),
        default="normal",
        help=(
            "Evaluation-only decoder-state ablation. reset-belief keeps the "
            "time schedule but removes belief history; shuffle-belief permutes "
            "beliefs across a batch."
        ),
    )
    parser.add_argument(
        "--ablate-direct",
        action="store_true",
        help=(
            "Evaluate an iterative checkpoint with the same receiver weights "
            "in one-pass direct mode."
        ),
    )
    parser.add_argument(
        "--ablate-state-neutral",
        action="store_true",
        help=(
            "Evaluate a decoder-state BFN with neutralized mu, rho, and time "
            "conditions while retaining the same ten-step update schedule."
        ),
    )
    parser.add_argument("--bfn-gain", type=float, default=1.0)
    parser.add_argument("--tile", type=int)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--category-pattern",
        action="append",
        default=[],
        metavar="NAME=REGEX",
        help="Optional filename classifier for per-category output summaries.",
    )
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--uncertainty", action="store_true", help="Evaluate calibrated adapter risk")
    parser.add_argument("--crop-border", type=int, default=0)
    parser.add_argument(
        "--test-y-channel",
        action="store_true",
        help="Use the MATLAB Y channel required by the All-Weather benchmark",
    )
    parser.add_argument(
        "--compare-structural",
        action="store_true",
        help="Report the deterministic backbone and BFN gain separately",
    )
    parser.add_argument(
        "--self-ensemble",
        action="store_true",
        help=(
            "Average eight geometric restorations (4 rotations x optional "
            "horizontal flip) in the pixel domain before computing metrics. "
            "Costs 8x inference and must be reported as a separate row."
        ),
    )
    parser.add_argument("--save-dir")
    parser.add_argument("--json")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image output while retaining summaries.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.tile is not None and args.batch_size != 1:
        raise ValueError("Tiled inference requires --batch-size 1")
    if args.state_ablation == "shuffle-belief" and args.batch_size < 2:
        raise ValueError("shuffle-belief requires --batch-size at least 2")
    if args.ablate_direct and args.state_ablation != "normal":
        raise ValueError(
            "--ablate-direct cannot be combined with --state-ablation"
        )
    if args.ablate_state_neutral and args.state_ablation != "normal":
        raise ValueError(
            "--ablate-state-neutral cannot be combined with --state-ablation"
        )
    if args.self_ensemble and args.stochastic:
        raise ValueError(
            "--self-ensemble cannot be combined with --stochastic: geometric "
            "averaging and stochastic sampling would mix two variance sources"
        )
    if args.self_ensemble and args.uncertainty:
        raise ValueError(
            "--self-ensemble cannot be combined with --uncertainty: the risk "
            "map is defined for a single deterministic forward pass"
        )
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model, config = load_model_for_inference(
        checkpoint_path, device, config_path=args.config
    )
    category_patterns = parse_category_patterns(args.category_pattern)
    checkpoint_refinement_mode = model.refinement_mode
    checkpoint_state_condition_mode = model.decoder_state_mode
    reported_state_ablation = args.state_ablation
    if model.latent_bfn_enabled and args.stochastic:
        raise ValueError("Latent-BFN paired evaluation is deterministic")
    if model.latent_bfn_enabled and args.ablate_direct:
        raise ValueError(
            "Use the parameter-matched Latent-single checkpoint, not "
            "--ablate-direct"
        )
    if args.ablate_direct:
        if checkpoint_refinement_mode != "iterative":
            raise ValueError(
                "--ablate-direct requires an iterative BFN checkpoint"
            )
        if args.stochastic:
            raise ValueError("--ablate-direct is deterministic")
        model.refinement_mode = "direct"
    if args.ablate_state_neutral:
        if model.receiver_type != "decoder_state":
            raise ValueError(
                "--ablate-state-neutral requires a decoder_state BFN checkpoint"
            )
        if args.stochastic:
            raise ValueError("--ablate-state-neutral is deterministic")
        model.decoder_state_mode = "neutral"
        reported_state_ablation = "neutral"
    if (args.compare_structural or args.uncertainty) and args.tile:
        if args.uncertainty:
            raise ValueError("Risk evaluation requires full-image inference")
    if args.uncertainty and (args.crop_border or model.refinement_mode != "none"):
        raise ValueError("Risk evaluation requires inference_mode=none and crop_border=0")
    dataset = PairedImageDataset(
        args.input_dir,
        args.target_dir,
        pairs_file=args.pairs_file,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    lpips_metric = LPIPSMetric().to(device) if args.lpips else None
    transforms_per_image = (
        SELF_ENSEMBLE_TRANSFORMS if args.self_ensemble else 1
    )

    def restore_prediction(image: torch.Tensor) -> torch.Tensor:
        def single(view: torch.Tensor) -> torch.Tensor:
            return restore_tensor(
                model,
                view,
                steps=args.steps,
                deterministic=not args.stochastic,
                residual_gain=args.bfn_gain,
                state_ablation=args.state_ablation,
                tile=args.tile,
                overlap=args.overlap,
            )

        if args.self_ensemble:
            return self_ensemble_restore(single, image)
        return single(image)

    def restore_structural_path(image: torch.Tensor) -> torch.Tensor:
        def single(view: torch.Tensor) -> torch.Tensor:
            return restore_structural_tensor(
                model,
                view,
                tile=args.tile,
                overlap=args.overlap,
            )

        if args.self_ensemble:
            return self_ensemble_restore(single, image)
        return single(image)

    rows = []
    runtime_decoder_trace: dict | None = None
    inference_seconds = 0.0
    inference_images = 0
    inference_trajectories = 0
    peak_memory_bytes = 0
    peak_incremental_memory_bytes = 0
    for batch in loader:
        degraded = batch["degraded"].to(device)
        target = batch["target"].to(device)
        baseline_memory_bytes = 0
        inference_trajectories += degraded.shape[0] * transforms_per_image * (
            restoration_tile_count(
                degraded.shape[-2],
                degraded.shape[-1],
                args.tile,
                args.overlap,
            )
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline_memory_bytes = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        inference_start = time.perf_counter()
        needs_separate_structural = False
        if args.compare_structural or args.uncertainty:
            if args.tile is not None or args.self_ensemble:
                prediction = restore_prediction(degraded)
                needs_separate_structural = True
            else:
                details = model.restore(
                    degraded,
                    num_steps=args.steps,
                    deterministic=not args.stochastic,
                    residual_gain=args.bfn_gain,
                    state_ablation=args.state_ablation,
                    return_details=True,
                )
                prediction = details["restored"]
                structural = details["structural"]
                foundation = details.get("foundation", structural)
                if "decoder_state_calls" in details:
                    trace = {
                        key: details[key]
                        for key in (
                            "decoder_state_calls",
                            "receiver_calls",
                            "decoder_state_updates",
                            "state_update_schedule",
                            "terminal_readout_time",
                            "residual_update_mode",
                            "residual_readout_mode",
                            "coefficient_state_updates",
                            "basis_evaluations",
                            "coefficient_update_mode",
                            "coefficient_gate_enabled",
                        )
                        if key in details
                    }
                    if runtime_decoder_trace is None:
                        runtime_decoder_trace = trace
                    elif trace != runtime_decoder_trace:
                        raise RuntimeError(
                            "Decoder-state execution trace changed across batches"
                        )
            if not isinstance(prediction, torch.Tensor):
                raise TypeError("Model restoration must return a tensor")
            if not needs_separate_structural:
                if not isinstance(structural, torch.Tensor):
                    raise TypeError("Structural restoration must return a tensor")
                if not isinstance(foundation, torch.Tensor):
                    raise TypeError("Foundation restoration must return a tensor")
        else:
            prediction = restore_prediction(degraded)
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

        if needs_separate_structural:
            structural = restore_structural_path(degraded)
            foundation = structural
            details = {}
        if args.compare_structural or args.uncertainty:
            if not isinstance(structural, torch.Tensor):
                raise TypeError("Structural restoration must return a tensor")
            if not isinstance(foundation, torch.Tensor):
                raise TypeError("Foundation restoration must return a tensor")
        metric_options = {
            "crop_border": args.crop_border,
            "test_y_channel": args.test_y_channel,
        }
        image_psnr = psnr(prediction, target, **metric_options)
        image_ssim = ssim(prediction, target, **metric_options)
        image_lpips = (
            lpips_metric(prediction, target)
            if lpips_metric is not None
            else None
        )
        structural_psnr = (
            psnr(structural, target, **metric_options)
            if args.compare_structural
            else None
        )
        structural_ssim = (
            ssim(structural, target, **metric_options)
            if args.compare_structural
            else None
        )
        foundation_psnr = (
            psnr(foundation, target, **metric_options)
            if args.compare_structural
            else None
        )
        foundation_ssim = (
            ssim(foundation, target, **metric_options)
            if args.compare_structural
            else None
        )
        structural_lpips = (
            lpips_metric(structural, target)
            if args.compare_structural and lpips_metric is not None
            else None
        )
        foundation_lpips = (
            lpips_metric(foundation, target)
            if args.compare_structural and lpips_metric is not None
            else None
        )
        for index, name in enumerate(batch["name"]):
            row = {
                "name": name,
                "psnr": float(image_psnr[index]),
                "ssim": float(image_ssim[index]),
            }
            category = category_for_path(
                batch["input_path"][index],
                category_patterns,
            )
            if category is not None:
                row["category"] = category
            if image_lpips is not None:
                row["lpips"] = float(image_lpips[index])
            if args.uncertainty:
                variance = details.get("adapter_uncertainty")
                if variance is None:
                    raise ValueError("Checkpoint has no calibrated adapter risk map")
                row.update({
                    name: float(value[index])
                    for name, value in uncertainty_scores(
                        variance,
                        prediction,
                        target,
                    ).items()
                })
            if args.compare_structural:
                assert structural_psnr is not None
                assert structural_ssim is not None
                assert foundation_psnr is not None
                assert foundation_ssim is not None
                row["structural_psnr"] = float(structural_psnr[index])
                row["structural_ssim"] = float(structural_ssim[index])
                row["delta_psnr"] = row["psnr"] - row["structural_psnr"]
                row["delta_ssim"] = row["ssim"] - row["structural_ssim"]
                row["foundation_psnr"] = float(foundation_psnr[index])
                row["foundation_ssim"] = float(foundation_ssim[index])
                row["mean_delta_psnr"] = (
                    row["structural_psnr"] - row["foundation_psnr"]
                )
                row["mean_delta_ssim"] = (
                    row["structural_ssim"] - row["foundation_ssim"]
                )
                if structural_lpips is not None and foundation_lpips is not None:
                    row["structural_lpips"] = float(structural_lpips[index])
                    row["delta_lpips"] = (
                        row["lpips"] - row["structural_lpips"]
                    )
                    row["foundation_lpips"] = float(foundation_lpips[index])
                    row["mean_delta_lpips"] = (
                        row["structural_lpips"] - row["foundation_lpips"]
                    )
                sampling_mask = details.get("sampling_mask")
                if isinstance(sampling_mask, torch.Tensor):
                    row.update(
                        correction_diagnostics(
                            prediction[index : index + 1],
                            structural[index : index + 1],
                            target[index : index + 1],
                            sampling_mask[index : index + 1],
                            metric_options,
                        )
                    )
            rows.append(row)
            if not args.quiet:
                print(
                    f"{row['name']}: PSNR={row['psnr']:.4f} "
                    f"SSIM={row['ssim']:.5f}"
                    + (
                        f" LPIPS={row['lpips']:.5f}"
                        if "lpips" in row
                        else ""
                    )
                    + (
                        f" MEAN_DELTA_PSNR={row['mean_delta_psnr']:+.4f}"
                        f" BFN_DELTA_PSNR={row['delta_psnr']:+.4f}"
                        + (
                            f" BFN_DELTA_LPIPS={row['delta_lpips']:+.5f}"
                            if "delta_lpips" in row
                            else ""
                        )
                        if args.compare_structural
                        else ""
                    )
                )
            if args.save_dir:
                save_image(
                    prediction[index],
                    Path(args.save_dir) / f"{row['name']}.png",
                )

    if not rows:
        raise RuntimeError("The evaluation dataset is empty")
    metric_names = [
        key
        for key, value in rows[0].items()
        if key != "name" and isinstance(value, float)
    ]
    average = {
        metric: sum(row[metric] for row in rows) / len(rows)
        for metric in metric_names
    }
    if model.latent_bfn_enabled:
        default_updates = int(args.steps or model.latent_steps)
        default_calls = 1
    else:
        default_updates = (
            args.steps or model.bfn_steps
            if model.refinement_mode == "iterative"
            and model.receiver_type == "decoder_state"
            else 0
        )
        default_calls = (
            default_updates + 1
            if default_updates
            else int(
                model.refinement_mode == "direct"
                and model.receiver_type == "decoder_state"
            )
        )
    decoder_updates = int(
        (runtime_decoder_trace or {}).get(
            "decoder_state_updates",
            default_updates,
        )
    )
    decoder_calls = int(
        (runtime_decoder_trace or {}).get(
            "decoder_state_calls",
            default_calls,
        )
    )
    if model.latent_bfn_enabled or model.refinement_mode == "none":
        default_receiver_calls = 0
    elif model.refinement_mode == "direct":
        default_receiver_calls = 1
    else:
        default_receiver_calls = int(args.steps or model.bfn_steps) + int(
            model.residual_readout_mode == "receiver"
        )
    receiver_calls = int(
        (runtime_decoder_trace or {}).get(
            "receiver_calls",
            default_receiver_calls,
        )
    )
    runtime = {
        "images": inference_images,
        "restoration_trajectories": inference_trajectories,
        "total_inference_seconds": inference_seconds,
        "average_inference_seconds_per_image": (
            inference_seconds / max(inference_images, 1)
        ),
        "average_restoration_trajectories_per_image": (
            inference_trajectories / max(inference_images, 1)
        ),
        "average_decoder_state_calls_per_image": (
            decoder_calls * inference_trajectories
            / max(inference_images, 1)
        ),
        "average_decoder_calls_per_image": (
            decoder_calls * inference_trajectories
            / max(inference_images, 1)
        ),
        "average_receiver_calls_per_image": (
            receiver_calls * inference_trajectories
            / max(inference_images, 1)
        ),
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
    print("Average: " + " ".join(f"{k.upper()}={v:.5f}" for k, v in average.items()))
    if args.self_ensemble:
        print(
            f"Self-ensemble: enabled, transforms={transforms_per_image} "
            "(4 rot90 x horizontal flip, pixel-domain mean)"
        )
    print("Runtime: " + json.dumps(runtime, ensure_ascii=True))
    if runtime_decoder_trace is not None:
        print(
            "Decoder-state execution: "
            + json.dumps(runtime_decoder_trace, ensure_ascii=True)
        )
    if args.json:
        output = {
            "average": average,
            "images": rows,
            "runtime": runtime,
            "protocol": {
                "test_y_channel": args.test_y_channel,
                "crop_border": args.crop_border,
                "lpips_network": "alex" if args.lpips else None,
                "stochastic": (
                    args.stochastic
                    and (
                        (
                            model.refinement_mode == "iterative"
                            and not model.latent_bfn_enabled
                        )
                        or (
                            model.coefficient_state_readout_enabled
                            and model.receiver.output.update_mode == "bayesian"
                        )
                    )
                ),
                "tile": args.tile,
                "overlap": args.overlap if args.tile is not None else None,
                "self_ensemble": args.self_ensemble,
                "self_ensemble_transforms": transforms_per_image,
                "self_ensemble_geometry": (
                    "rot90x4_times_horizontal_flip_pixel_mean"
                    if args.self_ensemble
                    else None
                ),
                "steps": (
                    args.steps or model.bfn_steps
                    if (
                        model.refinement_mode == "iterative"
                        or model.coefficient_state_readout_enabled
                    )
                    else int(model.refinement_mode == "direct")
                ),
                "output_quantization": "float32_clamp_0_1",
                "refinement_mode": model.refinement_mode,
                "checkpoint_refinement_mode": checkpoint_refinement_mode,
                "direct_ablation": args.ablate_direct,
                "checkpoint_state_condition_mode": checkpoint_state_condition_mode,
                "state_neutral_ablation": args.ablate_state_neutral,
                "state_ablation": reported_state_ablation,
                "decoder_state_updates": decoder_updates,
                "decoder_state_calls_per_trajectory": decoder_calls,
                "latent_bfn_steps": (
                    int(args.steps or model.latent_steps)
                    if model.latent_bfn_enabled
                    else 0
                ),
                "decoder_calls_per_trajectory": decoder_calls,
                "receiver_calls_per_trajectory": receiver_calls,
                "coefficient_state_updates_per_trajectory": int(
                    (runtime_decoder_trace or {}).get(
                        "coefficient_state_updates",
                        (
                            model.bfn_steps
                            if model.coefficient_state_readout_enabled
                            else 0
                        ),
                    )
                ),
                "basis_evaluations_per_trajectory": int(
                    (runtime_decoder_trace or {}).get(
                        "basis_evaluations",
                        (
                            1
                            if model.coefficient_state_readout_enabled
                            else 0
                        ),
                    )
                ),
                "coefficient_update_mode": (
                    (runtime_decoder_trace or {}).get(
                        "coefficient_update_mode",
                        (
                            model.receiver.output.update_mode
                            if model.coefficient_state_readout_enabled
                            and model.receiver is not None
                            else None
                        ),
                    )
                ),
                "coefficient_gate_enabled": bool(
                    (runtime_decoder_trace or {}).get(
                        "coefficient_gate_enabled",
                        (
                            model.receiver.output.acceptance_gate_enabled
                            if model.coefficient_state_readout_enabled
                            and model.receiver is not None
                            else False
                        ),
                    )
                ),
                "residual_update_mode": model.residual_update_mode,
                "residual_readout_mode": model.residual_readout_mode,
                "residual_base_clamp": model.residual_base_clamp,
                "experiment_protocol": config.get("protocol", {}).get("name"),
                "experiment_arm": config.get("protocol", {}).get("arm"),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_payload.get("epoch"),
                "checkpoint_global_step": checkpoint_payload.get("global_step"),
                "checkpoint_weights": (
                    "ema" if "ema" in checkpoint_payload else "model"
                ),
                "decoder_state_update_schedule": (
                    (runtime_decoder_trace or {}).get(
                        "state_update_schedule",
                        [],
                    )
                ),
                "decoder_state_terminal_readout_time": (
                    (runtime_decoder_trace or {}).get(
                        "terminal_readout_time",
                    )
                ),
                "foundation_signature": foundation_signature(config),
                "bfn_gain": args.bfn_gain,
                "uncertainty": args.uncertainty,
            },
        }
        if category_patterns:
            output["categories"] = {
                category: {
                    metric: sum(
                        row[metric]
                        for row in rows
                        if row.get("category") == category
                    )
                    / sum(
                        row.get("category") == category
                        for row in rows
                    )
                    for metric in metric_names
                }
                for category in category_patterns
            }
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
