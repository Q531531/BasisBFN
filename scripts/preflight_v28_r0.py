#!/usr/bin/env python3
"""Validate the v28-R0 deterministic signed-basis contract."""

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
from sgbformer.models import SignedSpatialBasisReadout, build_model
from sgbformer.protocol import (
    foundation_artifact_fingerprint,
    implementation_fingerprint,
    sha256_file,
)
from sgbformer.utils import parameter_count, seed_everything
from sgbformer.v27 import (
    V27_EXPECTED_TRAINABLE_PARAMETERS,
    resolved_config_sha256,
)
from sgbformer.v28 import (
    V28_EXPECTED_DEV_PAIRS,
    V28_EXPECTED_TRAIN_PAIRS,
    V28_PROTOCOL,
    V28_R0_ARM,
    V28_R0_EXPECTED_TRAINABLE_PARAMETERS,
    audit_v28_r0_config,
    validate_v28_r0_config,
)
from train import build_dataset, build_optimizer


IDENTITY_IMAGES = 8
IDENTITY_TOLERANCE = 1e-6
MINIMUM_GRADIENT_NORM = 1e-12
MAXIMUM_DCT_GRAM_ERROR = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--reference-c0",
        default=(
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "ssr_residual_v27_c0_proxy.yaml"
        ),
    )
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


def gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    energy = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(energy)


def signed_basis_readout(model: torch.nn.Module) -> SignedSpatialBasisReadout:
    receiver = model.receiver
    if receiver is None or not receiver.signed_basis_readout_enabled:
        raise AssertionError("v28-R0 signed basis readout is missing")
    if receiver.routed_readout_enabled:
        raise AssertionError("v28-R0 must not enable routed readout")
    if not isinstance(receiver.output, SignedSpatialBasisReadout):
        raise AssertionError("v28-R0 receiver output has the wrong type")
    return receiver.output


def assert_r0_structure(model: torch.nn.Module) -> dict[str, Any]:
    count = parameter_count(model)
    if count != V28_R0_EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError(
            f"Expected {V28_R0_EXPECTED_TRAINABLE_PARAMETERS} trainable "
            f"parameters, got {count}"
        )
    if count >= 2 * V27_EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError("v28-R0 exceeds the 2x v27-C0 parameter ceiling")
    if model.receiver is None:
        raise AssertionError("v28-R0 receiver is missing")
    readout = signed_basis_readout(model)
    if model.latent_bfn_enabled or model.latent_flow is not None:
        raise AssertionError("v28-R0 must not enable latent BFN")
    if model.mean_corrector is not None:
        raise AssertionError("v28-R0 must not enable mean correction")
    if model.backbone.adapters:
        raise AssertionError("v28-R0 foundation must not contain adapters")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise AssertionError("Every SSR foundation parameter must be frozen")
    if not model.backbone._frozen_modules_eval:
        raise AssertionError("Frozen SSR modules must remain in eval mode")
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or any(
        not name.startswith("receiver.") for name in trainable_names
    ):
        raise AssertionError("Only receiver parameters may be trainable")

    basis_weight = readout.basis_projection.weight.detach().float()
    coefficient_weight = readout.coefficient_projection.weight.detach()
    coefficient_bias = readout.coefficient_projection.bias.detach()
    if torch.count_nonzero(basis_weight).item() == 0:
        raise AssertionError("Deterministic basis projection is zero")
    if (
        torch.count_nonzero(coefficient_weight).item() != 0
        or torch.count_nonzero(coefficient_bias).item() != 0
    ):
        raise AssertionError("Coefficient projection is not zero initialized")
    flattened = basis_weight.flatten(1)
    gram = flattened @ flattened.T
    expected = torch.eye(
        flattened.shape[0],
        device=gram.device,
        dtype=gram.dtype,
    ) * readout.basis_init_gain**2
    gram_error = float((gram - expected).abs().max())
    if gram_error > MAXIMUM_DCT_GRAM_ERROR:
        raise AssertionError(
            f"Deterministic basis rows are not orthogonal: {gram_error}"
        )
    return {
        "trainable_parameters": count,
        "v27_c0_trainable_parameters": V27_EXPECTED_TRAINABLE_PARAMETERS,
        "additional_parameters_vs_v27_c0": (
            count - V27_EXPECTED_TRAINABLE_PARAMETERS
        ),
        "ratio_vs_v27_c0": count / V27_EXPECTED_TRAINABLE_PARAMETERS,
        "below_2x_v27_c0": True,
        "trainable_parameter_names": trainable_names,
        "foundation_trainable_parameters": 0,
        "num_bases": readout.num_bases,
        "basis_projection_nonzero": int(torch.count_nonzero(basis_weight)),
        "coefficient_weight_nonzero": int(
            torch.count_nonzero(coefficient_weight)
        ),
        "coefficient_bias_nonzero": int(
            torch.count_nonzero(coefficient_bias)
        ),
        "dct_gram_max_abs_error": gram_error,
        "dct_gram_tolerance": MAXIMUM_DCT_GRAM_ERROR,
        "output_scale": readout.output_scale,
    }


@torch.inference_mode()
def check_r0_identity(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    if len(samples) != IDENTITY_IMAGES:
        raise AssertionError(
            f"v28-R0 identity requires {IDENTITY_IMAGES} images"
        )
    model.to(device).eval()
    readout = signed_basis_readout(model)
    records = []
    for sample in samples:
        image = sample["degraded"].unsqueeze(0).to(device)
        baseline = model.restore_structural(image)
        details = model.restore(
            image,
            deterministic=True,
            return_details=True,
        )
        maximum_difference = float(
            (details["restored"] - baseline).abs().max()
        )
        if maximum_difference > IDENTITY_TOLERANCE:
            raise AssertionError(
                f"v28-R0 identity max_abs={maximum_difference} exceeds "
                f"{IDENTITY_TOLERANCE}"
            )
        if torch.count_nonzero(details["residual"]).item() != 0:
            raise AssertionError("v28-R0 initial correction must be zero")
        if details["receiver_calls"] != 1:
            raise AssertionError("v28-R0 direct inference must call receiver once")
        if readout.last_basis_maps is None or (
            torch.count_nonzero(readout.last_basis_maps).item() == 0
        ):
            raise AssertionError("v28-R0 basis maps are inactive")
        if readout.last_coefficients is None or (
            torch.count_nonzero(readout.last_coefficients).item() != 0
        ):
            raise AssertionError("v28-R0 initial coefficients are not zero")
        records.append(
            {
                "input_path": str(sample.get("input_path", "")),
                "shape": list(image.shape),
                "max_abs_vs_f0": maximum_difference,
                "correction_nonzero": int(
                    torch.count_nonzero(details["residual"])
                ),
                "basis_nonzero": int(
                    torch.count_nonzero(readout.last_basis_maps)
                ),
                "coefficient_nonzero": int(
                    torch.count_nonzero(readout.last_coefficients)
                ),
                "receiver_calls": int(details["receiver_calls"]),
            }
        )

    zero = torch.zeros(1, 3, 32, 40, device=device)
    zero_details = model.restore(
        zero,
        deterministic=True,
        return_details=True,
    )
    if torch.count_nonzero(zero_details["residual"]).item() != 0:
        raise AssertionError("v28-R0 zero-input correction must be zero")
    return {
        "images": IDENTITY_IMAGES,
        "tolerance": IDENTITY_TOLERANCE,
        "random_dev_images": records,
        "maximum_absolute_difference": max(
            record["max_abs_vs_f0"] for record in records
        ),
        "zero_input_correction_nonzero": int(
            torch.count_nonzero(zero_details["residual"])
        ),
        "receiver_calls_per_image": 1,
    }


def check_r0_gradient_path(
    model: torch.nn.Module,
    config: dict[str, Any],
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.to(device).train()
    if model.backbone.model.training:
        raise AssertionError("Frozen SSR entered train mode")
    if model.receiver is None:
        raise AssertionError("v28-R0 receiver is missing")
    readout = signed_basis_readout(model)
    degraded = sample["degraded"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    coefficient_parameters = list(readout.coefficient_projection.parameters())
    basis_parameters = list(readout.basis_projection.parameters())
    readout_ids = {id(parameter) for parameter in readout.parameters()}
    upstream_parameters = [
        parameter
        for parameter in model.receiver.parameters()
        if id(parameter) not in readout_ids
    ]
    optimizer = build_optimizer(model, config["training"])
    if len(optimizer.param_groups) != 1:
        raise AssertionError("v28-R0 must have one optimizer group")
    foundation_before = tensor_state_sha256(model.backbone)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(degraded, target, training_stage="residual")
    candidate_loss = F.mse_loss(outputs["restored"].float(), target.float())
    baseline_loss = F.mse_loss(outputs["structural"].float(), target.float())
    relative_loss_delta = abs(
        float((candidate_loss - baseline_loss).detach())
    ) / max(abs(float(baseline_loss.detach())), 1e-12)
    if relative_loss_delta > 1e-7:
        raise AssertionError("v28-R0 initial loss is not the exact F0 loss")
    candidate_loss.backward()
    first = {
        "coefficient_projection": gradient_norm(coefficient_parameters),
        "basis_projection": gradient_norm(basis_parameters),
        "receiver_upstream": gradient_norm(upstream_parameters),
    }
    if first["coefficient_projection"] <= MINIMUM_GRADIENT_NORM:
        raise AssertionError("First backward did not reach coefficients")
    if first["basis_projection"] != 0:
        raise AssertionError("Zero coefficients should block first basis gradient")
    if first["receiver_upstream"] != 0:
        raise AssertionError("Zero coefficients should block first trunk gradient")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise AssertionError("Frozen SSR received gradients")

    optimizer.step()
    coefficient_nonzero_after_step = int(
        torch.count_nonzero(readout.coefficient_projection.weight)
        + torch.count_nonzero(readout.coefficient_projection.bias)
    )
    if coefficient_nonzero_after_step == 0:
        raise AssertionError("First update did not open signed coefficients")
    optimizer.zero_grad(set_to_none=True)
    outputs = model(degraded, target, training_stage="residual")
    second_loss = F.mse_loss(outputs["restored"].float(), target.float())
    second_loss.backward()
    second = {
        "coefficient_projection": gradient_norm(coefficient_parameters),
        "basis_projection": gradient_norm(basis_parameters),
        "receiver_upstream": gradient_norm(upstream_parameters),
    }
    if second["basis_projection"] <= MINIMUM_GRADIENT_NORM:
        raise AssertionError("Second backward did not reach basis projection")
    if second["receiver_upstream"] <= MINIMUM_GRADIENT_NORM:
        raise AssertionError("Second backward did not reach receiver trunk")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise AssertionError("Frozen SSR received gradients after optimizer step")
    foundation_after = tensor_state_sha256(model.backbone)
    if foundation_before != foundation_after:
        raise AssertionError("Frozen SSR state changed during gradient preflight")
    return {
        "optimizer_groups": [
            {
                "lr": float(group["lr"]),
                "parameters": sum(
                    parameter.numel() for parameter in group["params"]
                ),
            }
            for group in optimizer.param_groups
        ],
        "initial_candidate_loss": float(candidate_loss.detach()),
        "initial_f0_baseline_loss": float(baseline_loss.detach()),
        "initial_loss_relative_difference": relative_loss_delta,
        "first_image_backward": first,
        "coefficient_nonzero_after_first_step": (
            coefficient_nonzero_after_step
        ),
        "second_image_backward": second,
        "minimum_gradient_norm": MINIMUM_GRADIENT_NORM,
        "foundation_state_unchanged": True,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    reference_path = Path(args.reference_c0).expanduser().resolve()
    config = load_config(config_path)
    reference = load_config(reference_path)
    validate_v28_r0_config(config, reference)
    causal_audit = audit_v28_r0_config(config, reference)

    sampling = config["training"]["sampling"]
    print(
        "sampling_banner="
        + json.dumps(
            {
                "mode": sampling["mode"],
                "category_balanced": False,
                "matches_v27_c0": (
                    sampling == reference["training"]["sampling"]
                ),
            },
            sort_keys=True,
        )
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(42)
    train_dataset = build_dataset(
        config["data"]["train"],
        training=True,
        sample_seed=42,
    )
    dev_dataset = build_dataset(config["data"]["val"], training=False)
    if len(train_dataset) != V28_EXPECTED_TRAIN_PAIRS:
        raise AssertionError(
            f"Expected {V28_EXPECTED_TRAIN_PAIRS} train pairs, "
            f"got {len(train_dataset)}"
        )
    if len(dev_dataset) != V28_EXPECTED_DEV_PAIRS:
        raise AssertionError(
            f"Expected {V28_EXPECTED_DEV_PAIRS} dev pairs, "
            f"got {len(dev_dataset)}"
        )
    generator = torch.Generator().manual_seed(42)
    identity_indices = torch.randperm(
        len(dev_dataset),
        generator=generator,
    )[:IDENTITY_IMAGES].tolist()
    identity_samples = [dev_dataset[index] for index in identity_indices]

    seed_everything(42)
    model = build_model(config).to(device)
    structure = assert_r0_structure(model)
    identity = check_r0_identity(model, identity_samples, device)
    gradients = check_r0_gradient_path(
        model,
        config,
        train_dataset[0],
        device,
    )
    report = {
        "schema_version": 1,
        "protocol": V28_PROTOCOL,
        "arm": V28_R0_ARM,
        "status": "passed",
        "device": str(device),
        "implementation": implementation_fingerprint(),
        "config": {
            "path": str(config_path),
            "source_sha256": sha256_file(config_path),
            "resolved_sha256": resolved_config_sha256(config),
        },
        "reference_c0": {
            "path": str(reference_path),
            "source_sha256": sha256_file(reference_path),
            "resolved_sha256": resolved_config_sha256(reference),
        },
        "foundation_artifacts": foundation_artifact_fingerprint(config),
        "manifests": {
            "train": {
                "path": config["data"]["train"]["pairs_file"],
                "sha256": sha256_file(
                    config["data"]["train"]["pairs_file"]
                ),
                "pairs": len(train_dataset),
            },
            "dev": {
                "path": config["data"]["val"]["pairs_file"],
                "sha256": sha256_file(
                    config["data"]["val"]["pairs_file"]
                ),
                "pairs": len(dev_dataset),
                "identity_indices": identity_indices,
            },
        },
        "causal_audit": causal_audit,
        "sampling": {
            "mode": "shuffle",
            "category_balanced": False,
            "matches_v27_c0": True,
        },
        "structure": structure,
        "identity": identity,
        "gradients": gradients,
        "scope": {
            "training_started": False,
            "quality_metrics_observed": False,
            "official_test_accessed": False,
        },
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "preflight_r0.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
