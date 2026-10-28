#!/usr/bin/env python3
"""Validate one arm of the matched BasisBFN-state matrix."""

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
from sgbformer.models import CoefficientStateBasisReadout, build_model
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
from sgbformer.basisbfn import (
    BASISBFN_ARMS,
    BASISBFN_EXPECTED_DEV_PAIRS,
    BASISBFN_EXPECTED_TRAINABLE_PARAMETERS,
    BASISBFN_EXPECTED_TRAIN_PAIRS,
    BASISBFN_PROTOCOL,
    audit_basisbfn_config,
    audit_basisbfn_matrix,
    validate_basisbfn_config,
)
from train import build_dataset, build_optimizer


IDENTITY_IMAGES = 8
IDENTITY_TOLERANCE = 1e-6
MINIMUM_GRADIENT_NORM = 1e-12
MAXIMUM_DCT_GRAM_ERROR = 1e-5
MATRIX_CONFIGS = {
    "D4": "basisbfn_d4.yaml",
    "D4-G": "basisbfn_d4g.yaml",
    "B4": "basisbfn_b4.yaml",
    "B4-G": "basisbfn_b4g.yaml",
}


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
    parser.add_argument(
        "--matrix-config-dir",
        default=PROJECT_ROOT / "configs" / "experiments",
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


def coefficient_readout(
    model: torch.nn.Module,
) -> CoefficientStateBasisReadout:
    receiver = model.receiver
    if receiver is None or not receiver.coefficient_state_readout_enabled:
        raise AssertionError("BasisBFN-state readout is missing")
    if receiver.routed_readout_enabled or receiver.signed_basis_readout_enabled:
        raise AssertionError("BasisBFN arm enabled another readout")
    if not isinstance(receiver.output, CoefficientStateBasisReadout):
        raise AssertionError("BasisBFN readout has the wrong type")
    return receiver.output


def assert_structure(
    model: torch.nn.Module,
    arm: str,
) -> dict[str, Any]:
    count = parameter_count(model)
    if count != BASISBFN_EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError(
            f"Expected {BASISBFN_EXPECTED_TRAINABLE_PARAMETERS} "
            f"trainable parameters, got {count}"
        )
    if count >= 2 * V27_EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError("BasisBFN arm exceeds the 2x C0 ceiling")
    readout = coefficient_readout(model)
    expected = BASISBFN_ARMS[arm]
    if readout.update_mode != expected["update_mode"]:
        raise AssertionError("Readout update mode differs from its matrix arm")
    if readout.acceptance_gate_enabled != expected["gate_enabled"]:
        raise AssertionError("Readout gate state differs from its matrix arm")
    if readout.steps != 4 or model.bfn_steps != 4:
        raise AssertionError("Coefficient-state arm must use four updates")
    if model.latent_bfn_enabled or model.latent_flow is not None:
        raise AssertionError("Coefficient-state arm must not enable latent BFN")
    if model.mean_corrector is not None:
        raise AssertionError("Coefficient-state arm must not use mean correction")
    if model.backbone.adapters:
        raise AssertionError("Coefficient-state foundation has adapters")
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
    flattened = basis_weight.flatten(1)
    gram = flattened @ flattened.T
    expected_gram = torch.eye(
        flattened.shape[0],
        device=gram.device,
        dtype=gram.dtype,
    ) * readout.basis_init_gain**2
    gram_error = float((gram - expected_gram).abs().max())
    if gram_error > MAXIMUM_DCT_GRAM_ERROR:
        raise AssertionError(
            f"Coefficient basis is not DCT-orthogonal: {gram_error}"
        )
    if (
        torch.count_nonzero(readout.observer_output.weight).item() != 0
        or torch.count_nonzero(readout.observer_output.bias).item() != 0
    ):
        raise AssertionError("Coefficient observer is not zero initialized")
    if (
        torch.count_nonzero(readout.gate_output.weight).item() != 0
        or torch.count_nonzero(readout.gate_output.bias).item() != 0
    ):
        raise AssertionError("Posterior gate is not neutral initialized")
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
        "coefficient_state_steps": readout.steps,
        "update_mode": readout.update_mode,
        "acceptance_gate_enabled": readout.acceptance_gate_enabled,
        "dct_gram_max_abs_error": gram_error,
        "dct_gram_tolerance": MAXIMUM_DCT_GRAM_ERROR,
    }


@torch.inference_mode()
def check_identity(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    if len(samples) != IDENTITY_IMAGES:
        raise AssertionError(
            f"Coefficient identity requires {IDENTITY_IMAGES} images"
        )
    model.to(device).eval()
    readout = coefficient_readout(model)
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
                f"Coefficient identity max_abs={maximum_difference} exceeds "
                f"{IDENTITY_TOLERANCE}"
            )
        if torch.count_nonzero(details["residual"]).item() != 0:
            raise AssertionError("Initial coefficient correction must be zero")
        if details["receiver_calls"] != 1:
            raise AssertionError("Coefficient arm must call receiver once")
        if details["coefficient_state_updates"] != 4:
            raise AssertionError("Coefficient arm must update state four times")
        if details["basis_evaluations"] != 1:
            raise AssertionError("Coefficient arm must build one shared basis")
        if readout.last_posterior_precision is None or not torch.all(
            readout.last_posterior_precision > 1
        ):
            raise AssertionError("Coefficient posterior precision did not grow")
        if readout.last_acceptance is None or not torch.all(
            (readout.last_acceptance >= 0)
            & (readout.last_acceptance <= 1)
        ):
            raise AssertionError("Posterior gate left [0, 1]")
        records.append(
            {
                "input_path": str(sample.get("input_path", "")),
                "shape": list(image.shape),
                "max_abs_vs_f0": maximum_difference,
                "receiver_calls": int(details["receiver_calls"]),
                "coefficient_state_updates": int(
                    details["coefficient_state_updates"]
                ),
                "basis_evaluations": int(details["basis_evaluations"]),
                "posterior_precision_mean": float(
                    readout.last_posterior_precision.mean()
                ),
                "acceptance_mean": float(readout.last_acceptance.mean()),
            }
        )
    return {
        "images": IDENTITY_IMAGES,
        "tolerance": IDENTITY_TOLERANCE,
        "random_dev_images": records,
        "maximum_absolute_difference": max(
            record["max_abs_vs_f0"] for record in records
        ),
        "receiver_calls_per_image": 1,
        "coefficient_state_updates_per_image": 4,
        "basis_evaluations_per_image": 1,
    }


def check_gradient_path(
    model: torch.nn.Module,
    config: dict[str, Any],
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.to(device).train()
    if model.backbone.model.training:
        raise AssertionError("Frozen SSR entered train mode")
    if model.receiver is None:
        raise AssertionError("Coefficient receiver is missing")
    readout = coefficient_readout(model)
    degraded = sample["degraded"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    observer_parameters = list(readout.observer_output.parameters())
    basis_parameters = list(readout.basis_projection.parameters())
    gate_parameters = list(readout.gate_output.parameters())
    readout_ids = {id(parameter) for parameter in readout.parameters()}
    upstream_parameters = [
        parameter
        for parameter in model.receiver.parameters()
        if id(parameter) not in readout_ids
    ]
    optimizer = build_optimizer(model, config["training"])
    if len(optimizer.param_groups) != 1:
        raise AssertionError("Coefficient arm must have one optimizer group")
    foundation_before = tensor_state_sha256(model.backbone)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(degraded, target, training_stage="residual")
    candidate_loss = F.mse_loss(outputs["restored"].float(), target.float())
    baseline_loss = F.mse_loss(outputs["structural"].float(), target.float())
    relative_loss_delta = abs(
        float((candidate_loss - baseline_loss).detach())
    ) / max(abs(float(baseline_loss.detach())), 1e-12)
    if relative_loss_delta > 1e-7:
        raise AssertionError("Initial coefficient loss is not exact F0 loss")
    bfn_loss = outputs["bfn_loss"]
    if not isinstance(bfn_loss, torch.Tensor) or not torch.isfinite(bfn_loss):
        raise AssertionError("Coefficient BFN loss is missing or non-finite")
    if float(bfn_loss.detach()) <= 0:
        raise AssertionError("Coefficient BFN loss is inactive")
    training_loss = (
        candidate_loss
        + float(config["loss"]["bfn_weight"]) * bfn_loss
    )
    training_loss.backward()
    first = {
        "observer_output": gradient_norm(observer_parameters),
        "basis_projection": gradient_norm(basis_parameters),
        "gate_output": gradient_norm(gate_parameters),
        "receiver_upstream": gradient_norm(upstream_parameters),
    }
    if first["observer_output"] <= MINIMUM_GRADIENT_NORM:
        raise AssertionError("First backward did not reach observer output")
    if first["basis_projection"] != 0 or first["receiver_upstream"] != 0:
        raise AssertionError("Zero coefficients did not isolate first gradient")
    if first["gate_output"] != 0:
        raise AssertionError("Zero coefficients should block first gate gradient")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise AssertionError("Frozen SSR received gradients")

    optimizer.step()
    observer_nonzero_after_step = int(
        torch.count_nonzero(readout.observer_output.weight)
        + torch.count_nonzero(readout.observer_output.bias)
    )
    if observer_nonzero_after_step == 0:
        raise AssertionError("First update did not open coefficient observations")
    optimizer.zero_grad(set_to_none=True)
    outputs = model(degraded, target, training_stage="residual")
    F.mse_loss(outputs["restored"].float(), target.float()).backward()
    second = {
        "observer_output": gradient_norm(observer_parameters),
        "basis_projection": gradient_norm(basis_parameters),
        "gate_output": gradient_norm(gate_parameters),
        "receiver_upstream": gradient_norm(upstream_parameters),
    }
    if second["basis_projection"] <= MINIMUM_GRADIENT_NORM:
        raise AssertionError("Second backward did not reach the shared basis")
    if second["receiver_upstream"] <= MINIMUM_GRADIENT_NORM:
        raise AssertionError("Second backward did not reach receiver trunk")
    if readout.acceptance_gate_enabled:
        if second["gate_output"] <= MINIMUM_GRADIENT_NORM:
            raise AssertionError("Second backward did not reach posterior gate")
    elif second["gate_output"] != 0:
        raise AssertionError("Disabled gate unexpectedly received a gradient")
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise AssertionError("Frozen SSR received gradients after update")
    if foundation_before != tensor_state_sha256(model.backbone):
        raise AssertionError("Frozen SSR state changed during preflight")
    return {
        "initial_candidate_loss": float(candidate_loss.detach()),
        "initial_f0_baseline_loss": float(baseline_loss.detach()),
        "initial_loss_relative_difference": relative_loss_delta,
        "initial_coefficient_bfn_loss": float(bfn_loss.detach()),
        "coefficient_bfn_enabled": True,
        "first_image_backward": first,
        "observer_nonzero_after_first_step": observer_nonzero_after_step,
        "second_image_backward": second,
        "minimum_gradient_norm": MINIMUM_GRADIENT_NORM,
        "foundation_state_unchanged": True,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    reference_path = Path(args.reference_c0).expanduser().resolve()
    matrix_dir = Path(args.matrix_config_dir).expanduser().resolve()
    config = load_config(config_path)
    reference = load_config(reference_path)
    arm = validate_basisbfn_config(config, reference)
    causal_audit = audit_basisbfn_config(config, reference)
    matrix_configs = {
        matrix_arm: load_config(matrix_dir / filename)
        for matrix_arm, filename in MATRIX_CONFIGS.items()
    }
    matrix_audit = audit_basisbfn_matrix(matrix_configs, reference)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    experiment_seed = int(config["seed"])
    seed_everything(experiment_seed)
    train_dataset = build_dataset(
        config["data"]["train"],
        training=True,
        sample_seed=experiment_seed,
    )
    dev_dataset = build_dataset(config["data"]["val"], training=False)
    if len(train_dataset) != BASISBFN_EXPECTED_TRAIN_PAIRS:
        raise AssertionError(
            f"Expected {BASISBFN_EXPECTED_TRAIN_PAIRS} train pairs, "
            f"got {len(train_dataset)}"
        )
    if len(dev_dataset) != BASISBFN_EXPECTED_DEV_PAIRS:
        raise AssertionError(
            f"Expected {BASISBFN_EXPECTED_DEV_PAIRS} dev pairs, "
            f"got {len(dev_dataset)}"
        )
    generator = torch.Generator().manual_seed(experiment_seed)
    identity_indices = torch.randperm(
        len(dev_dataset),
        generator=generator,
    )[:IDENTITY_IMAGES].tolist()
    identity_samples = [dev_dataset[index] for index in identity_indices]

    seed_everything(experiment_seed)
    model = build_model(config).to(device)
    structure = assert_structure(model, arm)
    identity = check_identity(model, identity_samples, device)
    gradients = check_gradient_path(
        model,
        config,
        train_dataset[0],
        device,
    )
    report = {
        "schema_version": 1,
        "protocol": BASISBFN_PROTOCOL,
        "arm": arm,
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
        "matrix_audit": matrix_audit,
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
    output = output_dir / f"preflight_{arm.lower()}.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
