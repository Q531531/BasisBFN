#!/usr/bin/env python3
"""Validate the single-variable C0-E1 rain-safe exploratory protocol."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from preflight_v27_residual_bfn import (
    assert_model_structure,
    check_gradient_path,
    check_identity,
)
from sgbformer.config import load_config
from sgbformer.data import CategoryBalancedSampler, classify_paths
from sgbformer.models import build_model
from sgbformer.protocol import (
    foundation_artifact_fingerprint,
    implementation_fingerprint,
    sha256_file,
)
from sgbformer.utils import seed_everything
from sgbformer.v27 import (
    V27_C0_E1_TARGET_COUNTS,
    V27_EXPECTED_DEV_PAIRS,
    V27_EXPECTED_TRAIN_PAIRS,
    audit_v27_c0_e1_configs,
    resolved_config_sha256,
    trainable_parameter_signature,
    trainable_state_sha256,
    validate_v27_c0_e1_config,
    validate_v27_config,
)
from train import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-c0", required=True)
    parser.add_argument("--config-e1", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = {
        "C0": Path(args.config_c0).expanduser().resolve(),
        "C0-E1": Path(args.config_e1).expanduser().resolve(),
    }
    configs = {
        name: load_config(path) for name, path in config_paths.items()
    }
    validate_v27_config(configs["C0"], expected_arm="C0")
    validate_v27_c0_e1_config(configs["C0-E1"], configs["C0"])
    audit = audit_v27_c0_e1_configs(configs["C0"], configs["C0-E1"])

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(42)

    datasets = {
        name: build_dataset(
            config["data"]["train"],
            training=True,
            sample_seed=42,
        )
        for name, config in configs.items()
    }
    if any(len(dataset) != V27_EXPECTED_TRAIN_PAIRS for dataset in datasets.values()):
        raise AssertionError(
            f"C0/C0-E1 must both contain {V27_EXPECTED_TRAIN_PAIRS} train pairs"
        )
    if datasets["C0"].pairs != datasets["C0-E1"].pairs:
        raise AssertionError("C0-E1 changed the ordered training pairs")
    dev_dataset = build_dataset(
        configs["C0-E1"]["data"]["val"],
        training=False,
    )
    if len(dev_dataset) != V27_EXPECTED_DEV_PAIRS:
        raise AssertionError(
            f"Expected {V27_EXPECTED_DEV_PAIRS} dev pairs, got {len(dev_dataset)}"
        )

    sampling = configs["C0-E1"]["training"]["sampling"]
    labels = classify_paths(
        [pair[0] for pair in datasets["C0-E1"].pairs],
        sampling["category_patterns"],
    )
    sampler = CategoryBalancedSampler(
        labels,
        sampling["target_fractions"],
        num_samples=len(datasets["C0-E1"]),
        seed=42,
    )
    if len(sampler) != len(datasets["C0"]):
        raise AssertionError("C0-E1 changed the number of samples per epoch")
    if dict(sampler.target_counts) != V27_C0_E1_TARGET_COUNTS:
        raise AssertionError(
            f"Unexpected C0-E1 targets: {dict(sampler.target_counts)}"
        )
    sampled_counts = Counter(
        labels[index] for index, _ in iter(sampler)
    )
    if dict(sampled_counts) != V27_C0_E1_TARGET_COUNTS:
        raise AssertionError(
            f"Sampler realization differs from targets: {dict(sampled_counts)}"
        )

    structures = {}
    identities = {}
    gradients = {}
    signatures = {}
    initial_states = {}
    for name in ("C0", "C0-E1"):
        seed_everything(42)
        model = build_model(configs[name]).to(device)
        structures[name] = assert_model_structure(model)
        identities[name] = check_identity(model, device)
        signatures[name] = trainable_parameter_signature(model)
        initial_states[name] = trainable_state_sha256(model)
        gradients[name] = check_gradient_path(
            model,
            configs[name],
            datasets[name][0],
            device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if signatures["C0"] != signatures["C0-E1"]:
        raise AssertionError("C0-E1 changed the trainable parameter signature")
    if initial_states["C0"] != initial_states["C0-E1"]:
        raise AssertionError("C0-E1 changed trainable initialization")

    report = {
        "schema_version": 1,
        "protocol": "residual_bfn_v27_exploratory",
        "arm": "C0-E1",
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
            configs["C0-E1"]
        ),
        "manifests": {
            "train": {
                "path": configs["C0-E1"]["data"]["train"]["pairs_file"],
                "sha256": sha256_file(
                    configs["C0-E1"]["data"]["train"]["pairs_file"]
                ),
                "pairs": len(datasets["C0-E1"]),
            },
            "dev": {
                "path": configs["C0-E1"]["data"]["val"]["pairs_file"],
                "sha256": sha256_file(
                    configs["C0-E1"]["data"]["val"]["pairs_file"]
                ),
                "pairs": len(dev_dataset),
            },
        },
        "causal_audit": audit,
        "sampler": {
            "source_counts": dict(Counter(labels)),
            "target_counts": dict(sampler.target_counts),
            "realized_counts_epoch_0": dict(sampled_counts),
            "samples_per_epoch": len(sampler),
            "reference_samples_per_epoch": len(datasets["C0"]),
        },
        "parameter_contract": {
            "signature_equal": True,
            "initial_values_equal": True,
            "trainable_parameters": structures["C0-E1"][
                "trainable_parameters"
            ],
        },
        "identity": identities,
        "gradients": gradients,
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
