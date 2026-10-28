#!/usr/bin/env python3
"""Build and validate the frozen-basis v27 C0-accept initialization."""

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

from sgbformer.checkpoint import load_checkpoint, save_checkpoint
from sgbformer.config import load_config
from sgbformer.ema import ExponentialMovingAverage
from sgbformer.inference import load_model_for_inference
from sgbformer.models import CategoryRoutedReadout, build_model
from sgbformer.protocol import (
    foundation_artifact_fingerprint,
    implementation_fingerprint,
    sha256_file,
)
from sgbformer.training import build_scheduler, validate_resume_config
from sgbformer.utils import parameter_count, seed_everything
from sgbformer.v27 import (
    V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS,
    V27_EXPECTED_DEV_PAIRS,
    V27_EXPECTED_TRAIN_PAIRS,
    audit_v27_c0accept_configs,
    resolved_config_sha256,
    validate_v27_c0accept_config,
)
from train import build_dataset, build_optimizer


IDENTITY_IMAGES = 8
IDENTITY_TOLERANCE = 1e-6
MINIMUM_GATE_GRADIENT_NORM = 1e-12
GATE_PARAMETER_NAMES = {
    "receiver.output.acceptance_gate.weight",
    "receiver.output.acceptance_gate.bias",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-c0", required=True, type=Path)
    parser.add_argument("--config-c0prime", required=True, type=Path)
    parser.add_argument("--config-c0doubleprime", required=True, type=Path)
    parser.add_argument("--config-c0accept", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def tensor_state_sha256(
    module: torch.nn.Module,
    *,
    exclude: set[str] | None = None,
) -> str:
    excluded = exclude or set()
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if name in excluded:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def acceptance_readout(model: torch.nn.Module) -> CategoryRoutedReadout:
    receiver = model.receiver
    if receiver is None or not receiver.routed_readout_enabled:
        raise AssertionError("C0-accept routed receiver is missing")
    readout = receiver.output
    if not isinstance(readout, CategoryRoutedReadout):
        raise AssertionError("C0-accept receiver output has the wrong type")
    if not readout.acceptance_gate_enabled or readout.acceptance_gate is None:
        raise AssertionError("C0-accept gate is missing")
    return readout


def transplant_c0doubleprime_ema(
    candidate: torch.nn.Module,
    source: torch.nn.Module,
) -> dict[str, Any]:
    result = candidate.load_state_dict(source.state_dict(), strict=False)
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    if missing != GATE_PARAMETER_NAMES or unexpected:
        raise RuntimeError(
            f"Warm-start mismatch: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    readout = acceptance_readout(candidate)
    gate = readout.acceptance_gate
    assert gate is not None
    torch.nn.init.zeros_(gate.weight)
    torch.nn.init.zeros_(gate.bias)
    return {
        "missing_initialized_parameters": sorted(missing),
        "unexpected_parameters": sorted(unexpected),
        "source_and_candidate_shared_state_equal": (
            tensor_state_sha256(candidate, exclude=GATE_PARAMETER_NAMES)
            == tensor_state_sha256(source)
        ),
    }


def assert_structure(model: torch.nn.Module) -> dict[str, Any]:
    readout = acceptance_readout(model)
    gate = readout.acceptance_gate
    assert gate is not None
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if trainable_names != GATE_PARAMETER_NAMES:
        raise AssertionError(
            f"Only gate parameters may train, got {sorted(trainable_names)}"
        )
    count = parameter_count(model)
    if count != V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS:
        raise AssertionError(
            f"Expected {V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS} "
            f"trainable parameters, got {count}"
        )
    if torch.count_nonzero(gate.weight).item() != 0:
        raise AssertionError("Acceptance gate weight is not zero initialized")
    if torch.count_nonzero(gate.bias).item() != 0:
        raise AssertionError("Acceptance gate bias is not zero initialized")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise AssertionError("SSR foundation must remain frozen")
    return {
        "trainable_parameters": count,
        "trainable_parameter_names": sorted(trainable_names),
        "gate_input_features": int(gate.in_features),
        "gate_outputs": int(gate.out_features),
        "gate_range": [0.0, 1.0],
        "gate_activation": "hard_clamp",
        "gate_weight_nonzero": int(torch.count_nonzero(gate.weight)),
        "gate_bias_nonzero": int(torch.count_nonzero(gate.bias)),
        "base_receiver_frozen": True,
        "foundation_frozen": True,
    }


@torch.no_grad()
def check_identity_and_basis(
    candidate: torch.nn.Module,
    source: torch.nn.Module,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    if len(samples) != IDENTITY_IMAGES:
        raise AssertionError(f"Identity check requires {IDENTITY_IMAGES} images")
    candidate.eval()
    source.eval()
    readout = acceptance_readout(candidate)
    records = []
    maximum_identity_error = 0.0
    maximum_basis_error = 0.0
    source_correction_energy = 0.0
    for sample in samples:
        degraded = sample["degraded"].unsqueeze(0).to(device)
        baseline = candidate.restore_structural(degraded)
        source_details = source.restore(
            degraded,
            deterministic=True,
            return_details=True,
        )
        candidate_details = candidate.restore(
            degraded,
            deterministic=True,
            return_details=True,
        )
        acceptance = readout.last_acceptance
        ungated = readout.last_ungated_mixture
        if acceptance is None or ungated is None:
            raise AssertionError("Gate diagnostics were not recorded")
        identity_error = float(
            (candidate_details["restored"] - baseline).abs().max()
        )
        basis_error = float(
            (
                ungated[..., : baseline.shape[-2], : baseline.shape[-1]]
                * candidate.residual_scale
                - source_details["residual"]
            )
            .abs()
            .max()
        )
        maximum_identity_error = max(maximum_identity_error, identity_error)
        maximum_basis_error = max(maximum_basis_error, basis_error)
        source_rms = float(
            source_details["residual"].float().square().mean().sqrt()
        )
        source_correction_energy += source_rms
        if identity_error > IDENTITY_TOLERANCE:
            raise AssertionError(
                f"Acceptance-gated identity error {identity_error}"
            )
        if basis_error > IDENTITY_TOLERANCE:
            raise AssertionError(
                f"Warm-start basis reproduction error {basis_error}"
            )
        if torch.count_nonzero(acceptance).item() != 0:
            raise AssertionError("Initial acceptance must be exactly zero")
        records.append(
            {
                "input_path": str(sample.get("input_path", "")),
                "identity_max_abs": identity_error,
                "basis_max_abs": basis_error,
                "source_correction_rms": source_rms,
                "acceptance": float(acceptance.mean()),
            }
        )
    if source_correction_energy <= 0:
        raise AssertionError("Warm-start source correction is identically zero")
    return {
        "tolerance": IDENTITY_TOLERANCE,
        "maximum_identity_error": maximum_identity_error,
        "maximum_basis_reproduction_error": maximum_basis_error,
        "source_basis_nonzero": True,
        "records": records,
    }


def check_gate_gradient(
    model: torch.nn.Module,
    sample: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    readout = acceptance_readout(model)
    gate = readout.acceptance_gate
    assert gate is not None
    degraded = sample["degraded"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    frozen_before = tensor_state_sha256(model, exclude=GATE_PARAMETER_NAMES)
    model.zero_grad(set_to_none=True)
    outputs = model(degraded, target, training_stage="residual")
    loss = F.mse_loss(outputs["restored"].float(), target.float())
    baseline_loss = F.mse_loss(
        outputs["structural"].float(),
        target.float(),
    )
    loss.backward()
    gate_gradient_energy = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in gate.parameters()
        if parameter.grad is not None
    )
    gate_gradient_norm = math.sqrt(gate_gradient_energy)
    if gate_gradient_norm <= MINIMUM_GATE_GRADIENT_NORM:
        raise AssertionError(
            f"Acceptance gate gradient is too small: {gate_gradient_norm}"
        )
    non_gate_gradients = [
        name
        for name, parameter in model.named_parameters()
        if name not in GATE_PARAMETER_NAMES and parameter.grad is not None
    ]
    if non_gate_gradients:
        raise AssertionError(
            f"Frozen parameters received gradients: {non_gate_gradients}"
        )
    acceptance = readout.last_acceptance
    ungated = readout.last_ungated_mixture
    if acceptance is None or torch.count_nonzero(acceptance).item() != 0:
        raise AssertionError("First-step gate output must remain exactly zero")
    if ungated is None or torch.count_nonzero(ungated).item() == 0:
        raise AssertionError("Warm-start correction basis is inactive")
    diagnostic = outputs["adapter_regularization"]
    if diagnostic.requires_grad or float(diagnostic) != 0.0:
        raise AssertionError(
            "Initial train.adapter diagnostic must be detached zero acceptance"
        )
    frozen_after = tensor_state_sha256(model, exclude=GATE_PARAMETER_NAMES)
    if frozen_before != frozen_after:
        raise AssertionError("Frozen basis changed during gradient check")
    model.zero_grad(set_to_none=True)
    return {
        "initial_loss": float(loss.detach()),
        "f0_baseline_loss": float(baseline_loss.detach()),
        "initial_loss_relative_difference": abs(
            float(loss.detach() - baseline_loss.detach())
        )
        / max(abs(float(baseline_loss.detach())), 1e-12),
        "gate_gradient_norm": gate_gradient_norm,
        "minimum_gate_gradient_norm": MINIMUM_GATE_GRADIENT_NORM,
        "non_gate_parameters_with_gradients": non_gate_gradients,
        "initial_acceptance": float(acceptance.mean()),
        "ungated_basis_rms": float(
            ungated.float().square().mean().sqrt()
        ),
        "adapter_diagnostic": float(diagnostic),
        "frozen_state_unchanged": True,
    }


def save_initialization_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
) -> dict[str, Any]:
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    ema = ExponentialMovingAverage(
        model,
        decay=float(config["training"]["ema_decay"]),
        warmup_updates=int(config["training"]["ema_warmup_updates"]),
    )
    save_checkpoint(
        path,
        model,
        config=config,
        epoch=-1,
        global_step=0,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
    )
    restored = build_model(config)
    restored_optimizer = build_optimizer(restored, config["training"])
    restored_scheduler = build_scheduler(
        restored_optimizer,
        config["training"],
    )
    checkpoint = load_checkpoint(
        path,
        restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        map_location="cpu",
    )
    validate_resume_config(checkpoint["config"], config)
    if int(checkpoint["epoch"]) != -1 or int(checkpoint["global_step"]) != 0:
        raise AssertionError("Initialization checkpoint must resume at epoch 1")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "optimizer_present": "optimizer" in checkpoint,
        "scheduler_present": "scheduler" in checkpoint,
        "ema_present": "ema" in checkpoint,
        "resume_config_matches": True,
    }


def main() -> None:
    args = parse_args()
    config_paths = {
        "C0": args.config_c0.expanduser().resolve(),
        "C0-prime": args.config_c0prime.expanduser().resolve(),
        "C0-double-prime": args.config_c0doubleprime.expanduser().resolve(),
        "C0-accept": args.config_c0accept.expanduser().resolve(),
    }
    configs = {
        name: load_config(path) for name, path in config_paths.items()
    }
    validate_v27_c0accept_config(
        configs["C0-accept"],
        configs["C0"],
        configs["C0-prime"],
        configs["C0-double-prime"],
    )
    causal_audit = audit_v27_c0accept_configs(
        configs["C0"],
        configs["C0-prime"],
        configs["C0-double-prime"],
        configs["C0-accept"],
    )
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    if source_checkpoint.name != "epoch_0003.pt":
        raise ValueError("C0-accept source must be epoch_0003.pt")
    source_sha_before = sha256_file(source_checkpoint)
    source_raw = torch.load(
        source_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if int(source_raw.get("epoch", -1)) != 2 or "ema" not in source_raw:
        raise ValueError("Source must be the C0-double-prime epoch-3 EMA")
    source_config = source_raw.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("Source checkpoint has no embedded config")
    if resolved_config_sha256(source_config) != resolved_config_sha256(
        configs["C0-double-prime"]
    ):
        raise ValueError("Source checkpoint config is not registered C0-double-prime")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(42)
    source_model, _ = load_model_for_inference(source_checkpoint, device)
    candidate = build_model(configs["C0-accept"]).to(device)
    transplant = transplant_c0doubleprime_ema(candidate, source_model)
    if not transplant["source_and_candidate_shared_state_equal"]:
        raise AssertionError("Warm-start state differs from C0-double-prime EMA")
    structure = assert_structure(candidate)

    train_dataset = build_dataset(
        configs["C0-accept"]["data"]["train"],
        training=True,
        sample_seed=42,
    )
    dev_dataset = build_dataset(
        configs["C0-accept"]["data"]["val"],
        training=False,
    )
    if len(train_dataset) != V27_EXPECTED_TRAIN_PAIRS:
        raise AssertionError(
            f"Expected {V27_EXPECTED_TRAIN_PAIRS} train pairs"
        )
    if len(dev_dataset) != V27_EXPECTED_DEV_PAIRS:
        raise AssertionError(f"Expected {V27_EXPECTED_DEV_PAIRS} dev pairs")
    generator = torch.Generator().manual_seed(42)
    identity_indices = torch.randperm(
        len(dev_dataset),
        generator=generator,
    )[:IDENTITY_IMAGES].tolist()
    identity_samples = [dev_dataset[index] for index in identity_indices]
    identity = check_identity_and_basis(
        candidate,
        source_model,
        identity_samples,
        device,
    )
    gradient = check_gate_gradient(
        candidate,
        train_dataset[0],
        device,
    )
    if gradient["initial_loss_relative_difference"] > 0.01:
        raise AssertionError("Initial gate loss differs from F0 by more than 1%")
    if sha256_file(source_checkpoint) != source_sha_before:
        raise AssertionError("Source checkpoint changed during preflight")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initialization_path = output_dir / "c0accept_init.pt"
    configured_initialization = Path(
        configs["C0-accept"]["training"]["initialization_checkpoint"]
    ).expanduser().resolve()
    if initialization_path != configured_initialization:
        raise ValueError(
            "Preflight output-dir does not match "
            "training.initialization_checkpoint"
        )
    initialization = save_initialization_checkpoint(
        initialization_path,
        candidate.cpu(),
        configs["C0-accept"],
    )
    source_sha_after = sha256_file(source_checkpoint)
    report = {
        "schema_version": 1,
        "protocol": "residual_bfn_v27_exploratory",
        "arm": "C0-accept",
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
        "source_checkpoint": {
            "path": str(source_checkpoint),
            "sha256": source_sha_before,
            "sha256_unchanged": source_sha_after == source_sha_before,
            "epoch": 3,
            "weights": "ema",
            "resolved_config_sha256": resolved_config_sha256(source_config),
        },
        "initialization_checkpoint": initialization,
        "foundation_artifacts": foundation_artifact_fingerprint(
            configs["C0-accept"]
        ),
        "manifests": {
            "train": {
                "path": configs["C0-accept"]["data"]["train"]["pairs_file"],
                "sha256": sha256_file(
                    configs["C0-accept"]["data"]["train"]["pairs_file"]
                ),
                "pairs": len(train_dataset),
            },
            "dev": {
                "path": configs["C0-accept"]["data"]["val"]["pairs_file"],
                "sha256": sha256_file(
                    configs["C0-accept"]["data"]["val"]["pairs_file"]
                ),
                "pairs": len(dev_dataset),
                "identity_indices": identity_indices,
            },
        },
        "causal_audit": causal_audit,
        "transplant": transplant,
        "structure": structure,
        "identity": identity,
        "gradient": gradient,
        "diagnostic_semantics": {
            "train.adapter": (
                "detached mean acceptance in [0,1]; adapter_weight remains zero"
            )
        },
    }
    output = output_dir / "preflight_c0accept.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {initialization_path}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
