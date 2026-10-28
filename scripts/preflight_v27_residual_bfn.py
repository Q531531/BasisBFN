#!/usr/bin/env python3
"""Validate v27 identity, parameter, gradient, and protocol contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sgbformer.config import load_config
from sgbformer.metrics import psnr
from sgbformer.models import build_model
from sgbformer.protocol import (
    foundation_artifact_fingerprint,
    implementation_fingerprint,
    sha256_file,
)
from sgbformer.utils import parameter_count, seed_everything
from sgbformer.v27 import (
    V27_ARMS,
    V27_EXPECTED_DEV_PAIRS,
    V27_EXPECTED_TRAINABLE_PARAMETERS,
    V27_EXPECTED_TRAIN_PAIRS,
    normalized_v27_config,
    resolved_config_sha256,
    trainable_parameter_signature,
    trainable_state_sha256,
    validate_v27_config,
)
from train import build_dataset, build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in V27_ARMS:
        parser.add_argument(f"--config-{arm.lower()}", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def tensor_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    energy = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return energy**0.5


def receiver_gradient_groups(model: torch.nn.Module) -> dict[str, float]:
    receiver = model.receiver
    if receiver is None:
        raise AssertionError("v27 requires a residual receiver")
    output_ids = {id(parameter) for parameter in receiver.output.parameters()}
    output = [
        parameter
        for parameter in receiver.parameters()
        if id(parameter) in output_ids
    ]
    upstream = [
        parameter
        for parameter in receiver.parameters()
        if id(parameter) not in output_ids
    ]
    return {
        "output_projection": grad_norm(output),
        "upstream": grad_norm(upstream),
        "all": grad_norm(list(receiver.parameters())),
    }


def assert_model_structure(model: torch.nn.Module) -> dict[str, Any]:
    if parameter_count(model) != V27_EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError(
            f"Expected {V27_EXPECTED_TRAINABLE_PARAMETERS} trainable parameters, "
            f"got {parameter_count(model)}"
        )
    if model.receiver is None:
        raise AssertionError("v27 receiver is missing")
    if model.latent_bfn_enabled or model.latent_flow is not None:
        raise AssertionError("The v26 degradation-latent path must be disabled")
    if model.mean_corrector is not None:
        raise AssertionError("v27 does not use mean_correction")
    if model.backbone.adapters:
        raise AssertionError("v27 foundation must not contain feature adapters")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise AssertionError("Every SSR foundation parameter must be frozen")
    if not model.backbone._frozen_modules_eval:
        raise AssertionError("Frozen SSR modules must remain in eval mode")
    if torch.count_nonzero(model.receiver.output.weight).item() != 0:
        raise AssertionError("Residual output weight is not zero initialized")
    if torch.count_nonzero(model.receiver.output.bias).item() != 0:
        raise AssertionError("Residual output bias is not zero initialized")
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or any(
        not name.startswith("receiver.") for name in trainable_names
    ):
        raise AssertionError("Only receiver parameters may be trainable")
    return {
        "trainable_parameters": parameter_count(model),
        "trainable_parameter_names": trainable_names,
        "foundation_trainable_parameters": parameter_count(
            model.backbone,
            trainable_only=True,
        ),
        "feature_adapters": [],
        "latent_bfn_enabled": model.latent_bfn_enabled,
        "receiver_output_nonzero": 0,
    }


@torch.inference_mode()
def check_identity(
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    records = []
    for height, width in ((32, 40), (33, 41), (64, 64)):
        image = torch.rand(1, 3, height, width, device=device)
        baseline = model.restore_structural(image)
        details = model.restore(
            image,
            deterministic=True,
            return_details=True,
        )
        restored = details["restored"]
        correction = details["residual"]
        if not torch.equal(restored, baseline):
            difference = float((restored - baseline).abs().max())
            raise AssertionError(
                f"v27 identity failed for {(height, width)}: max_abs={difference}"
            )
        if torch.count_nonzero(correction).item() != 0:
            raise AssertionError("v27 initial correction must be exactly zero")
        metric_delta = float(
            psnr(restored, image, test_y_channel=True)[0]
            - psnr(baseline, image, test_y_channel=True)[0]
        )
        if metric_delta != 0.0:
            raise AssertionError("Identity outputs must have identical PSNR")
        records.append({
            "shape": [1, 3, height, width],
            "output_equal": True,
            "correction_nonzero": 0,
            "psnr_delta": metric_delta,
            "receiver_calls": int(details.get("receiver_calls", 0)),
        })
    return {"cases": records}


def check_gradient_path(
    model: torch.nn.Module,
    config: dict[str, Any],
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.to(device).train()
    if model.backbone.model.training:
        raise AssertionError("Frozen SSR entered train mode")
    degraded = sample["degraded"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    stage = (
        "residual"
        if config["protocol"]["arm"] in {"C0", "C0-E1"}
        else "joint"
    )
    foundation_before = tensor_state_sha256(model.backbone)
    optimizer = build_optimizer(model, config["training"])
    if len(optimizer.param_groups) != 1:
        raise AssertionError("v27 Stage A must have one optimizer group")

    optimizer.zero_grad(set_to_none=True)
    outputs = model(degraded, target, training_stage=stage)
    image_loss = F.mse_loss(outputs["restored"].float(), target.float())
    image_loss.backward()
    first = receiver_gradient_groups(model)
    if first["output_projection"] <= 0:
        raise AssertionError("First image backward did not reach output projection")
    if first["upstream"] != 0:
        raise AssertionError("Zero output projection should block first-step upstream")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise AssertionError("Frozen SSR received gradients")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(degraded, target, training_stage=stage)
    second_loss = F.mse_loss(outputs["restored"].float(), target.float())
    second_loss.backward()
    second = receiver_gradient_groups(model)
    if second["upstream"] <= 0:
        raise AssertionError("Second image backward did not reach receiver upstream")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise AssertionError("Frozen SSR received gradients after optimizer step")
    foundation_after = tensor_state_sha256(model.backbone)
    if foundation_before != foundation_after:
        raise AssertionError("Frozen SSR state changed during gradient preflight")
    return {
        "stage": stage,
        "optimizer_groups": [
            {
                "lr": float(group["lr"]),
                "parameters": sum(
                    parameter.numel() for parameter in group["params"]
                ),
            }
            for group in optimizer.param_groups
        ],
        "first_image_backward": first,
        "second_image_backward": second,
        "foundation_state_unchanged": True,
    }


def source_fingerprint() -> dict[str, Any]:
    core = implementation_fingerprint()
    paths = [
        PROJECT_ROOT / "scripts" / name
        for name in (
            "preflight_v27_residual_bfn.py",
            "profile_v27_residual_bfn.py",
            "analyze_v27_proxy.py",
        )
    ]
    scripts = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in paths
        if path.is_file()
    }
    digest = hashlib.sha256()
    digest.update(core["sha256"].encode("ascii"))
    for name, value in sorted(scripts.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
    return {
        "sha256": digest.hexdigest(),
        "core": core,
        "scripts": scripts,
    }


def main() -> None:
    args = parse_args()
    config_paths = {
        arm: Path(getattr(args, f"config_{arm.lower()}")).expanduser().resolve()
        for arm in V27_ARMS
    }
    configs = {
        arm: load_config(path) for arm, path in config_paths.items()
    }
    for arm, config in configs.items():
        validate_v27_config(config, expected_arm=arm)
    if normalized_v27_config(configs["D4"]) != normalized_v27_config(
        configs["B4"]
    ):
        raise AssertionError("D4/B4 differ outside the registered update mode")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(42)
    train_dataset = build_dataset(
        configs["C0"]["data"]["train"],
        training=True,
        sample_seed=42,
    )
    dev_dataset = build_dataset(
        configs["C0"]["data"]["val"],
        training=False,
    )
    if len(train_dataset) != V27_EXPECTED_TRAIN_PAIRS:
        raise AssertionError(
            f"Expected {V27_EXPECTED_TRAIN_PAIRS} train pairs, "
            f"got {len(train_dataset)}"
        )
    if len(dev_dataset) != V27_EXPECTED_DEV_PAIRS:
        raise AssertionError(
            f"Expected {V27_EXPECTED_DEV_PAIRS} dev pairs, got {len(dev_dataset)}"
        )
    sample = train_dataset[0]

    arm_reports = {}
    signatures = {}
    initial_states = {}
    for arm in V27_ARMS:
        seed_everything(42)
        model = build_model(configs[arm]).to(device)
        signatures[arm] = trainable_parameter_signature(model)
        initial_states[arm] = trainable_state_sha256(model)
        structure = assert_model_structure(model)
        identity = check_identity(model, device)
        gradients = check_gradient_path(
            model,
            configs[arm],
            sample,
            device,
        )
        arm_reports[arm] = {
            "structure": structure,
            "identity": identity,
            "gradients": gradients,
            "trainable_initial_state_sha256": initial_states[arm],
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any(signature != signatures["C0"] for signature in signatures.values()):
        raise AssertionError("v27 trainable parameter signatures are not identical")
    if len(set(initial_states.values())) != 1:
        raise AssertionError("v27 trainable initial parameter values differ")

    report = {
        "schema_version": 1,
        "protocol": "residual_bfn_v27",
        "status": "passed",
        "device": str(device),
        "source": source_fingerprint(),
        "configs": {
            arm: {
                "path": str(config_paths[arm]),
                "source_sha256": sha256_file(config_paths[arm]),
                "resolved_sha256": resolved_config_sha256(configs[arm]),
            }
            for arm in V27_ARMS
        },
        "foundation_artifacts": foundation_artifact_fingerprint(configs["C0"]),
        "manifests": {
            "train": {
                "path": configs["C0"]["data"]["train"]["pairs_file"],
                "sha256": sha256_file(
                    configs["C0"]["data"]["train"]["pairs_file"]
                ),
                "pairs": len(train_dataset),
            },
            "dev": {
                "path": configs["C0"]["data"]["val"]["pairs_file"],
                "sha256": sha256_file(
                    configs["C0"]["data"]["val"]["pairs_file"]
                ),
                "pairs": len(dev_dataset),
            },
        },
        "parameter_contract": {
            "expected_trainable_parameters": (
                V27_EXPECTED_TRAINABLE_PARAMETERS
            ),
            "signature_equal": True,
            "initial_values_equal": True,
            "signature": signatures["C0"],
        },
        "d4_b4_registered_difference_only": True,
        "arms": arm_reports,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "preflight.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
