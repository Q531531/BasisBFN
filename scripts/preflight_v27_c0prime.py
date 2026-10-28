#!/usr/bin/env python3
"""Validate the identity-safe v27 C0-prime routed-readout contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sgbformer.config import load_config
from sgbformer.models import CategoryRoutedReadout, build_model
from sgbformer.protocol import (
    foundation_artifact_fingerprint,
    implementation_fingerprint,
    sha256_file,
)
from sgbformer.utils import parameter_count, seed_everything
from sgbformer.v27 import (
    V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS,
    V27_EXPECTED_DEV_PAIRS,
    V27_EXPECTED_TRAINABLE_PARAMETERS,
    V27_EXPECTED_TRAIN_PAIRS,
    audit_v27_c0prime_configs,
    resolved_config_sha256,
    validate_v27_c0prime_config,
    validate_v27_config,
)
from train import build_dataset, build_optimizer


IDENTITY_IMAGES = 8
IDENTITY_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-c0", required=True)
    parser.add_argument("--config-c0prime", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def tensor_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def routed_readout(model: torch.nn.Module) -> CategoryRoutedReadout:
    receiver = model.receiver
    if receiver is None or not receiver.routed_readout_enabled:
        raise AssertionError("C0-prime category-routed readout is missing")
    if not isinstance(receiver.output, CategoryRoutedReadout):
        raise AssertionError("C0-prime receiver output has the wrong type")
    return receiver.output


def assert_c0prime_structure(model: torch.nn.Module) -> dict[str, Any]:
    count = parameter_count(model)
    if count != V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError(
            "Expected "
            f"{V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS} trainable parameters, "
            f"got {count}"
        )
    if count >= 2 * V27_EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError("C0-prime exceeds the 2x C0 parameter ceiling")
    readout = routed_readout(model)
    for name, head in readout.heads.items():
        if torch.count_nonzero(head.output.weight).item() != 0:
            raise AssertionError(f"{name} output weight is not zero initialized")
        if torch.count_nonzero(head.output.bias).item() != 0:
            raise AssertionError(f"{name} output bias is not zero initialized")
    if torch.count_nonzero(readout.router.weight).item() != 0:
        raise AssertionError("Router weight is not zero initialized")
    expected_bias = math.log(1.0 / len(readout.categories))
    torch.testing.assert_close(
        readout.router.bias,
        torch.full_like(readout.router.bias, expected_bias),
        atol=0,
        rtol=0,
    )
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise AssertionError("Every SSR foundation parameter must be frozen")
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or any(
        not name.startswith("receiver.") for name in trainable_names
    ):
        raise AssertionError("Only receiver parameters may be trainable")
    return {
        "trainable_parameters": count,
        "c0_trainable_parameters": V27_EXPECTED_TRAINABLE_PARAMETERS,
        "ratio_vs_c0": count / V27_EXPECTED_TRAINABLE_PARAMETERS,
        "below_2x_c0": True,
        "trainable_parameter_names": trainable_names,
        "foundation_trainable_parameters": 0,
        "categories": list(readout.categories),
        "head_output_nonzero": {
            name: int(torch.count_nonzero(head.output.weight))
            for name, head in readout.heads.items()
        },
        "router_weight_nonzero": int(torch.count_nonzero(readout.router.weight)),
        "router_bias": readout.router.bias.detach().cpu().tolist(),
    }


@torch.inference_mode()
def check_c0prime_identity(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    if len(samples) != IDENTITY_IMAGES:
        raise AssertionError(f"C0-prime identity requires {IDENTITY_IMAGES} images")
    model.eval()
    readout = routed_readout(model)
    records = []
    for sample in samples:
        image = sample["degraded"].unsqueeze(0).to(device)
        baseline = model.restore_structural(image)
        details = model.restore(
            image,
            deterministic=True,
            return_details=True,
        )
        restored = details["restored"]
        correction = details["residual"]
        max_abs = float((restored - baseline).abs().max())
        if max_abs > IDENTITY_TOLERANCE:
            raise AssertionError(
                f"C0-prime identity max_abs={max_abs} exceeds "
                f"{IDENTITY_TOLERANCE}"
            )
        if torch.count_nonzero(correction).item() != 0:
            raise AssertionError("Initial routed correction must be exactly zero")
        probabilities = readout.last_router_probabilities
        if probabilities is None:
            raise AssertionError("Router probabilities were not recorded")
        expected = torch.full_like(probabilities, 1.0 / 3.0)
        torch.testing.assert_close(probabilities, expected, atol=1e-7, rtol=0)
        records.append(
            {
                "input_path": str(sample.get("input_path", "")),
                "shape": list(image.shape),
                "max_abs_vs_f0": max_abs,
                "correction_nonzero": int(torch.count_nonzero(correction)),
                "router_probabilities": probabilities[0].cpu().tolist(),
            }
        )

    first = samples[0]["degraded"]
    zero = torch.zeros_like(first).unsqueeze(0).to(device)
    zero_details = model.restore(
        zero,
        deterministic=True,
        return_details=True,
    )
    zero_correction = zero_details["residual"]
    if torch.count_nonzero(zero_correction).item() != 0:
        raise AssertionError("Zero-input correction must be exactly zero")
    entropy = readout.last_router_entropy
    if entropy is None or not torch.isfinite(entropy):
        raise AssertionError("Router entropy must be finite")
    entropy_value = float(entropy)
    if not 0.0 <= entropy_value <= math.log(3.0) + 1e-6:
        raise AssertionError("Router entropy lies outside [0, log(3)]")
    return {
        "tolerance": IDENTITY_TOLERANCE,
        "random_dev_images": records,
        "maximum_absolute_difference": max(
            record["max_abs_vs_f0"] for record in records
        ),
        "zero_input_correction_nonzero": int(
            torch.count_nonzero(zero_correction)
        ),
        "router_entropy": entropy_value,
        "router_entropy_upper_bound": math.log(3.0),
    }


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    energy = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return energy**0.5


def check_c0prime_gradient_path(
    model: torch.nn.Module,
    config: dict[str, Any],
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.to(device).train()
    if model.backbone.model.training:
        raise AssertionError("Frozen SSR entered train mode")
    readout = routed_readout(model)
    degraded = sample["degraded"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    optimizer = build_optimizer(model, config["training"])
    foundation_before = tensor_state_sha256(model.backbone)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(degraded, target, training_stage="residual")
    candidate_loss = F.mse_loss(outputs["restored"].float(), target.float())
    baseline_loss = F.mse_loss(outputs["structural"].float(), target.float())
    relative_loss_delta = abs(
        float((candidate_loss - baseline_loss).detach())
    ) / max(
        abs(float(baseline_loss.detach())),
        1e-12,
    )
    if relative_loss_delta > 0.01:
        raise AssertionError("Initial loss differs from F0 by more than 1%")
    candidate_loss.backward()

    head_gradients = {
        name: head.output.weight.grad.detach().cpu().clone()
        for name, head in readout.heads.items()
    }
    if any(torch.count_nonzero(value).item() == 0 for value in head_gradients.values()):
        raise AssertionError("First backward did not reach every routed head")
    pairwise_equal = [
        torch.equal(left, right)
        for index, left in enumerate(head_gradients.values())
        for right in list(head_gradients.values())[index + 1 :]
    ]
    if any(pairwise_equal):
        raise AssertionError("Head symmetry break did not separate gradients")
    first_router_gradient = _gradient_norm(list(readout.router.parameters()))
    if first_router_gradient != 0:
        raise AssertionError("Zero heads should give zero first router gradient")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(degraded, target, training_stage="residual")
    F.mse_loss(outputs["restored"].float(), target.float()).backward()
    second_router_gradient = _gradient_norm(list(readout.router.parameters()))
    if second_router_gradient <= 0:
        raise AssertionError("Router did not receive a second-step gradient")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise AssertionError("Frozen SSR received gradients")
    foundation_after = tensor_state_sha256(model.backbone)
    if foundation_before != foundation_after:
        raise AssertionError("Frozen SSR state changed during preflight")
    entropy = outputs["adapter_regularization"]
    if entropy.requires_grad:
        raise AssertionError("Router entropy diagnostic must be detached")
    return {
        "initial_candidate_loss": float(candidate_loss.detach()),
        "initial_f0_baseline_loss": float(baseline_loss.detach()),
        "initial_loss_relative_difference": relative_loss_delta,
        "first_head_gradient_norms": {
            name: float(value.float().norm())
            for name, value in head_gradients.items()
        },
        "first_head_gradients_pairwise_distinct": True,
        "first_router_gradient_norm": first_router_gradient,
        "second_router_gradient_norm": second_router_gradient,
        "router_entropy_detached": True,
        "router_entropy": float(entropy),
        "foundation_state_unchanged": True,
    }


def main() -> None:
    args = parse_args()
    config_paths = {
        "C0": Path(args.config_c0).expanduser().resolve(),
        "C0-prime": Path(args.config_c0prime).expanduser().resolve(),
    }
    configs = {
        name: load_config(path) for name, path in config_paths.items()
    }
    validate_v27_config(configs["C0"], expected_arm="C0")
    validate_v27_c0prime_config(configs["C0-prime"], configs["C0"])
    audit = audit_v27_c0prime_configs(configs["C0"], configs["C0-prime"])

    sampling = configs["C0-prime"]["training"]["sampling"]
    print(
        "sampling_banner="
        + json.dumps(
            {
                "mode": sampling["mode"],
                "category_balanced": False,
                "matches_c0": sampling == configs["C0"]["training"]["sampling"],
            },
            sort_keys=True,
        )
    )
    if sampling != {"mode": "shuffle"}:
        raise AssertionError("C0-prime must use shuffle sampling")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(42)
    train_dataset = build_dataset(
        configs["C0-prime"]["data"]["train"],
        training=True,
        sample_seed=42,
    )
    dev_dataset = build_dataset(
        configs["C0-prime"]["data"]["val"],
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
    generator = torch.Generator().manual_seed(42)
    identity_indices = torch.randperm(
        len(dev_dataset),
        generator=generator,
    )[:IDENTITY_IMAGES].tolist()
    identity_samples = [dev_dataset[index] for index in identity_indices]

    seed_everything(42)
    model = build_model(configs["C0-prime"]).to(device)
    structure = assert_c0prime_structure(model)
    identity = check_c0prime_identity(model, identity_samples, device)
    gradients = check_c0prime_gradient_path(
        model,
        configs["C0-prime"],
        train_dataset[0],
        device,
    )
    report = {
        "schema_version": 1,
        "protocol": "residual_bfn_v27_exploratory",
        "arm": "C0-prime",
        "status": "passed",
        "device": str(device),
        "implementation": implementation_fingerprint(),
        "configs": {
            name: {
                "path": str(path),
                "source_sha256": sha256_file(path),
                "resolved_sha256": resolved_config_sha256(configs[name]),
            }
            for name, path in config_paths.items()
        },
        "foundation_artifacts": foundation_artifact_fingerprint(
            configs["C0-prime"]
        ),
        "manifests": {
            "train": {
                "path": configs["C0-prime"]["data"]["train"]["pairs_file"],
                "sha256": sha256_file(
                    configs["C0-prime"]["data"]["train"]["pairs_file"]
                ),
                "pairs": len(train_dataset),
            },
            "dev": {
                "path": configs["C0-prime"]["data"]["val"]["pairs_file"],
                "sha256": sha256_file(
                    configs["C0-prime"]["data"]["val"]["pairs_file"]
                ),
                "pairs": len(dev_dataset),
                "identity_indices": identity_indices,
            },
        },
        "causal_audit": audit,
        "sampling": {
            "mode": "shuffle",
            "category_balanced": False,
            "matches_c0": True,
        },
        "structure": structure,
        "identity": identity,
        "gradients": gradients,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "preflight_c0prime.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
