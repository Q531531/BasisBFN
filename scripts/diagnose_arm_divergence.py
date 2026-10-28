#!/usr/bin/env python3
"""Compare frozen-main D/B checkpoints without model construction or data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


PROJECTIONS = (
    "gamma_projection",
    "beta_projection",
    "spectral_projection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-d", required=True, type=Path)
    parser.add_argument("--checkpoint-b", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--identity-rms-threshold", type=float, default=1e-6)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    model = checkpoint.get("model")
    ema = checkpoint.get("ema", {}).get("shadow")
    if not isinstance(model, dict) or not isinstance(ema, dict):
        raise ValueError(f"Checkpoint lacks model or EMA shadow: {path}")
    if not all(isinstance(value, Tensor) for value in model.values()):
        raise TypeError(f"Checkpoint model state is not tensor-only: {path}")
    if not all(isinstance(value, Tensor) for value in ema.values()):
        raise TypeError(f"Checkpoint EMA state is not tensor-only: {path}")
    missing_raw = sorted(set(ema) - set(model))
    if missing_raw:
        raise ValueError(f"EMA names are absent from raw model state: {missing_raw}")
    return checkpoint


def group_name(name: str) -> str:
    if name.startswith("latent_flow."):
        return "latent_flow"
    if name.startswith("backbone.adapters.decoder_level"):
        return "latent_adapters"
    if name.startswith("backbone.adapters.encoder_level3"):
        return "paper_spectral_gate"
    if name.startswith("backbone.model.decoder_level1"):
        return "ssr_decoder_level1"
    if name.startswith("backbone.model.decoder_level2"):
        return "ssr_decoder_level2"
    if name.startswith("backbone.model.output"):
        return "ssr_output"
    return "other_trainable"


def tensor_energy(value: Tensor) -> tuple[float, float, float]:
    flat = value.detach().double().reshape(-1)
    square_sum = float(flat.square().sum())
    maximum = float(flat.abs().max()) if flat.numel() else 0.0
    return square_sum, float(flat.numel()), maximum


def compare_tensor(
    left: Tensor,
    right: Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"Tensor shapes differ: {left.shape} != {right.shape}")
    left = left.detach().double()
    right = right.detach().double()
    difference = left - right
    left_norm = float(left.square().sum().sqrt())
    right_norm = float(right.square().sum().sqrt())
    difference_norm = float(difference.square().sum().sqrt())
    denominator = max(0.5 * (left_norm + right_norm), 1e-30)
    return {
        "shape": list(left.shape),
        "numel": left.numel(),
        "left_l2": left_norm,
        "right_l2": right_norm,
        "difference_l2": difference_norm,
        "relative_l2": difference_norm / denominator,
        "max_abs_difference": (
            float(difference.abs().max()) if difference.numel() else 0.0
        ),
        "exactly_equal": torch.equal(left, right),
        "numerically_equal": torch.allclose(
            left,
            right,
            atol=atol,
            rtol=rtol,
        ),
    }


def compare_states(
    left: dict[str, Tensor],
    right: dict[str, Tensor],
    *,
    names: list[str],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    tensor_results = {
        name: compare_tensor(left[name], right[name], atol=atol, rtol=rtol)
        for name in names
    }
    groups: dict[str, dict[str, float | int | bool]] = {}
    for name in names:
        group = groups.setdefault(
            group_name(name),
            {
                "tensors": 0,
                "numel": 0,
                "left_square_sum": 0.0,
                "right_square_sum": 0.0,
                "difference_square_sum": 0.0,
                "max_abs_difference": 0.0,
                "numerically_equal": True,
            },
        )
        result = tensor_results[name]
        group["tensors"] += 1
        group["numel"] += int(result["numel"])
        group["left_square_sum"] += float(result["left_l2"]) ** 2
        group["right_square_sum"] += float(result["right_l2"]) ** 2
        group["difference_square_sum"] += float(result["difference_l2"]) ** 2
        group["max_abs_difference"] = max(
            float(group["max_abs_difference"]),
            float(result["max_abs_difference"]),
        )
        group["numerically_equal"] = bool(
            group["numerically_equal"]
            and result["numerically_equal"]
        )

    group_results = {}
    for name, values in groups.items():
        left_norm = math.sqrt(float(values.pop("left_square_sum")))
        right_norm = math.sqrt(float(values.pop("right_square_sum")))
        difference_norm = math.sqrt(float(values.pop("difference_square_sum")))
        group_results[name] = {
            **values,
            "left_l2": left_norm,
            "right_l2": right_norm,
            "difference_l2": difference_norm,
            "relative_l2": difference_norm
            / max(0.5 * (left_norm + right_norm), 1e-30),
        }

    left_square_sum = sum(
        float(result["left_l2"]) ** 2 for result in tensor_results.values()
    )
    right_square_sum = sum(
        float(result["right_l2"]) ** 2 for result in tensor_results.values()
    )
    difference_square_sum = sum(
        float(result["difference_l2"]) ** 2
        for result in tensor_results.values()
    )
    left_norm = math.sqrt(left_square_sum)
    right_norm = math.sqrt(right_square_sum)
    difference_norm = math.sqrt(difference_square_sum)
    return {
        "trainable_tensors": len(names),
        "trainable_parameters": sum(left[name].numel() for name in names),
        "numerically_identical": all(
            bool(result["numerically_equal"])
            for result in tensor_results.values()
        ),
        "exactly_identical": all(
            bool(result["exactly_equal"])
            for result in tensor_results.values()
        ),
        "global": {
            "left_l2": left_norm,
            "right_l2": right_norm,
            "difference_l2": difference_norm,
            "relative_l2": difference_norm
            / max(0.5 * (left_norm + right_norm), 1e-30),
            "max_abs_difference": max(
                (
                    float(result["max_abs_difference"])
                    for result in tensor_results.values()
                ),
                default=0.0,
            ),
        },
        "groups": group_results,
        "tensors": tensor_results,
    }


def projection_statistics(
    state: dict[str, Tensor],
    *,
    names: list[str],
    identity_rms_threshold: float,
) -> dict[str, Any]:
    projection_names = [
        name
        for name in names
        if any(f".{projection}." in name for projection in PROJECTIONS)
    ]
    groups: dict[str, list[Tensor]] = {}
    for name in projection_names:
        projection = next(
            projection
            for projection in PROJECTIONS
            if f".{projection}." in name
        )
        adapter = name.split(f".{projection}.", 1)[0]
        groups.setdefault(f"{adapter}.{projection}", []).append(state[name])

    results = {}
    maximum_rms = 0.0
    for name, tensors in sorted(groups.items()):
        square_sum = 0.0
        numel = 0
        max_abs = 0.0
        for tensor in tensors:
            tensor_square, tensor_numel, tensor_max = tensor_energy(tensor)
            square_sum += tensor_square
            numel += int(tensor_numel)
            max_abs = max(max_abs, tensor_max)
        l2 = math.sqrt(square_sum)
        rms = l2 / math.sqrt(max(numel, 1))
        maximum_rms = max(maximum_rms, rms)
        results[name] = {
            "numel": numel,
            "l2": l2,
            "rms": rms,
            "max_abs": max_abs,
            "left_zero_initialisation": rms > identity_rms_threshold,
        }
    fusion_gain = {
        name: float(state[name].detach().float())
        for name in names
        if name.endswith(".fusion_gain")
    }
    return {
        "identity_rms_threshold": identity_rms_threshold,
        "maximum_projection_rms": maximum_rms,
        "left_identity_initialisation": maximum_rms > identity_rms_threshold,
        "projections": results,
        "fusion_gain": fusion_gain,
    }


def arm_metadata(checkpoint: dict[str, Any], path: Path) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "epoch": int(checkpoint.get("epoch", -1)) + 1,
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_psnr": checkpoint.get("best_psnr"),
        "seed": config.get("seed"),
        "arm": config.get("protocol", {}).get("arm"),
        "update_mode": (
            config.get("model", {})
            .get("latent_bfn", {})
            .get("update_mode")
        ),
        "ema_updates": checkpoint.get("ema", {}).get("updates"),
    }


def analyze(
    checkpoint_d_path: Path,
    checkpoint_b_path: Path,
    *,
    atol: float,
    rtol: float,
    identity_rms_threshold: float,
) -> dict[str, Any]:
    checkpoints = {
        "D": load_checkpoint(checkpoint_d_path),
        "B": load_checkpoint(checkpoint_b_path),
    }
    ema_states = {
        arm: checkpoint["ema"]["shadow"]
        for arm, checkpoint in checkpoints.items()
    }
    raw_states = {
        arm: checkpoint["model"]
        for arm, checkpoint in checkpoints.items()
    }
    names_d = set(ema_states["D"])
    names_b = set(ema_states["B"])
    if names_d != names_b:
        raise ValueError(
            "D/B EMA trainable parameter names differ. "
            f"Only D: {sorted(names_d - names_b)}; "
            f"only B: {sorted(names_b - names_d)}"
        )
    names = sorted(names_d)
    comparison = {
        "raw": compare_states(
            raw_states["D"],
            raw_states["B"],
            names=names,
            atol=atol,
            rtol=rtol,
        ),
        "ema": compare_states(
            ema_states["D"],
            ema_states["B"],
            names=names,
            atol=atol,
            rtol=rtol,
        ),
    }
    projections = {
        state_name: {
            arm: projection_statistics(
                states[arm],
                names=names,
                identity_rms_threshold=identity_rms_threshold,
            )
            for arm in ("D", "B")
        }
        for state_name, states in (
            ("raw", raw_states),
            ("ema", ema_states),
        )
    }
    arms_numerically_identical = bool(
        comparison["ema"]["numerically_identical"]
    )
    return {
        "schema_version": 1,
        "scope": (
            "CPU-only checkpoint inspection; trainable names come from EMA "
            "shadow; no model construction, data loading, or forward pass"
        ),
        "atol": atol,
        "rtol": rtol,
        "arms": {
            "D": arm_metadata(checkpoints["D"], checkpoint_d_path),
            "B": arm_metadata(checkpoints["B"], checkpoint_b_path),
        },
        "arms_numerically_identical": arms_numerically_identical,
        "stop_condition": (
            "stop_and_debug"
            if arms_numerically_identical
            else "continue_locked_training"
        ),
        "comparison": comparison,
        "adapter_identity": projections,
        "interpretation_limits": [
            "Weighted latent training loss is not an unweighted posterior metric.",
            "Projection norms show departure from zero initialization, not "
            "the image-space causal effect of modulation.",
            "The epoch-16 fusion_gain=0 paired probe is required for the "
            "image-space causal effect.",
        ],
    }


def main() -> None:
    args = parse_args()
    if min(args.atol, args.rtol, args.identity_rms_threshold) < 0:
        raise ValueError("Tolerance values must be non-negative")
    report = analyze(
        args.checkpoint_d.expanduser().resolve(),
        args.checkpoint_b.expanduser().resolve(),
        atol=args.atol,
        rtol=args.rtol,
        identity_rms_threshold=args.identity_rms_threshold,
    )
    output = args.json_out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "arms_numerically_identical": report[
                    "arms_numerically_identical"
                ],
                "stop_condition": report["stop_condition"],
                "raw_relative_l2": report["comparison"]["raw"]["global"][
                    "relative_l2"
                ],
                "ema_relative_l2": report["comparison"]["ema"]["global"][
                    "relative_l2"
                ],
                "raw_left_identity": {
                    arm: report["adapter_identity"]["raw"][arm][
                        "left_identity_initialisation"
                    ]
                    for arm in ("D", "B")
                },
                "ema_left_identity": {
                    arm: report["adapter_identity"]["ema"][arm][
                        "left_identity_initialisation"
                    ]
                    for arm in ("D", "B")
                },
                "json": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
