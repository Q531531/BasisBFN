"""Validation helpers for the preregistered v27 residual-BFN protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import load_config
from .protocol import implementation_fingerprint, sha256_file


V27_ARMS = ("C0", "D1", "D4", "B4")
V27_EXPECTED_TRAINABLE_PARAMETERS = 614_455
V27_EXPECTED_TRAIN_PAIRS = 17_166
V27_EXPECTED_DEV_PAIRS = 903
V27_C0_E1_ARM = "C0-E1"
V27_C0PRIME_ARM = "C0-prime"
V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS = 615_440
V27_C0DOUBLEPRIME_ARM = "C0-double-prime"
V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS = 622_496
V27_C0ACCEPT_ARM = "C0-accept"
V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS = 17
V27_C0PRIME_ROUTED_READOUT = {
    "enabled": True,
    "categories": ["snow", "outdoor_rain", "raindrop"],
    "router_type": "global_average_linear",
    "router_supervision": "none",
    "router_bias_init": "uniform_log_probability",
    "head_type": "groupnorm_silu_conv3x3",
    "head_symmetry_break_epsilon": 0.001,
    "zero_init_output": True,
}
V27_C0DOUBLEPRIME_ROUTED_READOUT = {
    **V27_C0PRIME_ROUTED_READOUT,
    "head_type": "groupnorm_conv3x3_silu_groupnorm_silu_conv3x3",
}
V27_C0ACCEPT_GATE = {
    "enabled": True,
    "type": "global_average_linear_hard_clamp",
    "range": [0.0, 1.0],
    "zero_init": True,
    "freeze_base_receiver": True,
}
V27_C0ACCEPT_ROUTED_READOUT = {
    **V27_C0DOUBLEPRIME_ROUTED_READOUT,
    "acceptance_gate": V27_C0ACCEPT_GATE,
}
V27_C0_E1_PATTERNS = {
    "snow": r".*\.jpe?g",
    "outdoor_rain": r"im_\d+_s\d+_a0[456]\.png",
    "raindrop": r"\d+_rain\.png",
}
V27_C0_E1_FRACTIONS = {
    "snow": 0.30,
    "outdoor_rain": 0.45,
    "raindrop": 0.25,
}
V27_C0_E1_TARGET_COUNTS = {
    "snow": 5_150,
    "outdoor_rain": 7_725,
    "raindrop": 4_291,
}


def resolved_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"v27 requires {label}={expected!r}, got {actual!r}")


def validate_v27_config(
    config: dict[str, Any],
    *,
    expected_arm: str | None = None,
) -> str:
    """Validate every field frozen by the v27 design contract."""

    protocol = config.get("protocol", {})
    _require_equal(protocol.get("name"), "residual_bfn_v27", "protocol.name")
    arm = str(protocol.get("arm", ""))
    if arm not in V27_ARMS:
        raise ValueError(f"v27 arm must be one of {V27_ARMS}, got {arm!r}")
    if expected_arm is not None:
        _require_equal(arm, expected_arm, "protocol.arm")
    _require_equal(config.get("seed"), 42, "seed")
    _require_equal(
        protocol.get("expected_train_pairs"),
        V27_EXPECTED_TRAIN_PAIRS,
        "protocol.expected_train_pairs",
    )
    _require_equal(
        protocol.get("expected_dev_pairs"),
        V27_EXPECTED_DEV_PAIRS,
        "protocol.expected_dev_pairs",
    )
    _require_equal(
        protocol.get("expected_trainable_parameters"),
        V27_EXPECTED_TRAINABLE_PARAMETERS,
        "protocol.expected_trainable_parameters",
    )
    _require_equal(
        protocol.get("long_training_locked"),
        True,
        "protocol.long_training_locked",
    )
    _require_equal(
        protocol.get("official_test_locked"),
        True,
        "protocol.official_test_locked",
    )

    model = config["model"]
    foundation = model["foundation"]
    for key, expected in {
        "freeze": True,
        "frozen_modules_eval": True,
        "trainable_modules": [],
        "feature_adapters": {},
        "strict_load": True,
    }.items():
        _require_equal(foundation.get(key), expected, f"model.foundation.{key}")
    _require_equal(
        model.get("semantic", {}).get("encoder"),
        "none",
        "model.semantic.encoder",
    )
    _require_equal(
        model.get("mean_correction", {}).get("enabled"),
        False,
        "model.mean_correction.enabled",
    )
    _require_equal(
        model.get("latent_bfn", {}).get("enabled"),
        False,
        "model.latent_bfn.enabled",
    )

    bfn = model["bfn"]
    common_bfn = {
        "receiver_type": "unet",
        "receiver_trainable": True,
        "condition_mode": "observation_rich",
        "receiver_output_mode": "identity",
        "residual_base_clamp": True,
        "residual_scale": 0.2,
        "min_variance": 0.01,
        "accuracy_schedule": "linear_beta",
        "mask_mode": "none",
        "min_accuracy_scale": 1.0,
        "base_channels": 16,
        "channel_multipliers": [1, 2, 4],
        "blocks_per_level": 2,
        "attention_heads": 4,
        "dropout": 0.0,
        "training_final_state": "rollout",
        "rollout_backprop_steps": 1,
        "rollout_batch_fraction": 1.0,
    }
    for key, expected in common_bfn.items():
        _require_equal(bfn.get(key), expected, f"model.bfn.{key}")

    arm_fields = {
        "C0": {
            "inference_mode": "direct",
            "update_mode": "deterministic",
            "readout_mode": "receiver",
            "steps": 1,
            "training_rollout_steps": 1,
        },
        "D1": {
            "inference_mode": "iterative",
            "update_mode": "deterministic",
            "readout_mode": "posterior_mean",
            "steps": 1,
            "training_rollout_steps": 1,
        },
        "D4": {
            "inference_mode": "iterative",
            "update_mode": "deterministic",
            "readout_mode": "posterior_mean",
            "steps": 4,
            "training_rollout_steps": 4,
        },
        "B4": {
            "inference_mode": "iterative",
            "update_mode": "bayesian",
            "readout_mode": "posterior_mean",
            "steps": 4,
            "training_rollout_steps": 4,
        },
    }[arm]
    for key, expected in arm_fields.items():
        _require_equal(bfn.get(key), expected, f"model.bfn.{key}")

    loss = config["loss"]
    expected_loss = {
        "structural_weight": 0.0,
        "structural_loss_type": "mse",
        "final_weight": 1.0,
        "final_loss_type": "mse",
        "bfn_weight": 0.0 if arm == "C0" else 0.01,
        "frequency_weight": 0.0,
        "correlation_weight": 0.0,
        "adapter_weight": 0.0,
        "uncertainty_weight": 0.0,
        "latent_category_weight": 0.0,
    }
    for key, expected in expected_loss.items():
        _require_equal(loss.get(key), expected, f"loss.{key}")

    data = config["data"]
    _require_equal(data.get("workers"), 12, "data.workers")
    _require_equal(data["train"].get("patch_size"), 256, "data.train.patch_size")
    _require_equal(data["train"].get("augment"), True, "data.train.augment")
    if not str(data["train"].get("pairs_file", "")).endswith("train_pairs.tsv"):
        raise ValueError("v27 train data must use train_pairs.tsv")
    if not str(data["val"].get("pairs_file", "")).endswith("val_pairs.tsv"):
        raise ValueError("v27 validation data must use val_pairs.tsv")

    training = config["training"]
    expected_training = {
        "epochs": 6,
        "batch_size": 16,
        "micro_batch_size": 2,
        "learning_rate": 0.0002,
        "minimum_learning_rate_ratio": 0.1,
        "schedule_minimum_mode": "per_group",
        "warmup_epochs": 1,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "ema_decay": 0.999,
        "ema_warmup_updates": 2000,
        "amp": True,
        "amp_dtype": "bfloat16",
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "reproducible_data": True,
        "validate_interval": 1,
        "save_interval": 1,
        "require_budget_lock": False,
        "require_v27_preflight": True,
    }
    for key, expected in expected_training.items():
        _require_equal(training.get(key), expected, f"training.{key}")
    _require_equal(
        training.get("sampling"),
        {"mode": "shuffle"},
        "training.sampling",
    )
    expected_stage = "residual" if arm == "C0" else "joint"
    _require_equal(
        training.get("stages"),
        [{"name": expected_stage, "epochs": 6}],
        "training.stages",
    )
    if training.get("parameter_lrs"):
        raise ValueError("v27 Stage A must use one optimizer learning rate")
    if not training.get("v27_preflight_report"):
        raise ValueError("v27 requires training.v27_preflight_report")
    if not training.get("v27_profile_report"):
        raise ValueError("v27 requires training.v27_profile_report")
    if arm != "C0" and not training.get("stage_a0_decision_report"):
        raise ValueError(f"v27 {arm} requires a Stage A0 decision report")
    return arm


def validate_v27_launch_artifacts(
    config_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Require matching preflight/profile artifacts and the staged A0 gate."""

    arm = validate_v27_config(config)
    config_path = Path(config_path).expanduser().resolve()
    config_source_sha256 = sha256_file(config_path)
    config_resolved_sha256 = resolved_config_sha256(config)
    training = config["training"]
    preflight_path = Path(
        training["v27_preflight_report"]
    ).expanduser().resolve()
    profile_path = Path(training["v27_profile_report"]).expanduser().resolve()
    if not preflight_path.is_file():
        raise FileNotFoundError(f"v27 preflight report is missing: {preflight_path}")
    if not profile_path.is_file():
        raise FileNotFoundError(f"v27 profile report is missing: {profile_path}")

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("protocol") != "residual_bfn_v27"
        or preflight.get("status") != "passed"
    ):
        raise RuntimeError("v27 preflight report did not pass")
    recorded_config = preflight.get("configs", {}).get(arm, {})
    if recorded_config.get("source_sha256") != config_source_sha256:
        raise RuntimeError(f"v27 {arm} source config changed after preflight")
    if recorded_config.get("resolved_sha256") != config_resolved_sha256:
        raise RuntimeError(f"v27 {arm} config changed after preflight")
    current_implementation = implementation_fingerprint()["sha256"]
    if (
        preflight.get("source", {}).get("core", {}).get("sha256")
        != current_implementation
    ):
        raise RuntimeError("v27 implementation changed after preflight")

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("protocol") != "residual_bfn_v27":
        raise RuntimeError("v27 profile report has the wrong protocol")
    if profile.get("arm") != arm:
        raise RuntimeError(f"v27 profile arm mismatch: {profile.get('arm')} != {arm}")
    if profile.get("source_config_sha256") != config_source_sha256:
        raise RuntimeError(f"v27 {arm} source config changed after profiling")
    if profile.get("resolved_config_sha256") != config_resolved_sha256:
        raise RuntimeError(f"v27 {arm} config changed after profiling")
    if profile.get("optimizer_steps") != 200:
        raise RuntimeError("v27 profile must contain exactly 200 optimizer steps")
    if (
        profile.get("trainable_parameters")
        != V27_EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("v27 profile trainable parameter count changed")
    if profile.get("preflight", {}).get("sha256") != sha256_file(preflight_path):
        raise RuntimeError("v27 preflight report changed after profiling")
    if profile.get("implementation", {}).get("sha256") != current_implementation:
        raise RuntimeError("v27 implementation changed after profiling")
    seconds_per_step = profile.get("seconds_per_optimizer_step")
    if (
        not isinstance(seconds_per_step, (int, float))
        or seconds_per_step <= 0
    ):
        raise RuntimeError("v27 profile has an invalid throughput measurement")

    stage_a0 = None
    if arm != "C0":
        decision_path = Path(
            training["stage_a0_decision_report"]
        ).expanduser().resolve()
        if not decision_path.is_file():
            raise FileNotFoundError(
                f"Stage A0 decision report is missing: {decision_path}"
            )
        decision_report = json.loads(decision_path.read_text(encoding="utf-8"))
        c0_config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "experiments"
            / "ssr_residual_v27_c0_proxy.yaml"
        )
        c0_config = load_config(c0_config_path)
        recorded_c0 = decision_report.get(
            "arm_resolved_config_sha256", {}
        ).get("C0")
        if recorded_c0 != resolved_config_sha256(c0_config):
            raise RuntimeError(
                "Stage A0 decision does not match the registered C0 config"
            )
        stage_a0 = decision_report.get("stage_a0_decision")
        if not isinstance(stage_a0, dict) or stage_a0.get("status") not in {
            "GO",
            "CONDITIONAL_GO",
        }:
            raise RuntimeError("Stage A0 did not authorize residual-flow arms")
    return {
        "arm": arm,
        "preflight_path": str(preflight_path),
        "profile_path": str(profile_path),
        "profile_device": profile.get("device"),
        "profile_seconds_per_step": seconds_per_step,
        "stage_a0_decision": stage_a0,
    }


def normalized_v27_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove the registered arm differences used for pairwise audits."""

    normalized = copy.deepcopy(config)
    normalized["protocol"].pop("arm", None)
    normalized["protocol"].pop("arm_description", None)
    normalized["training"].pop("output_dir", None)
    normalized["training"].pop("v27_profile_report", None)
    normalized["model"]["bfn"].pop("update_mode", None)
    return normalized


def audit_v27_c0_e1_configs(
    reference_c0: dict[str, Any],
    exploratory: dict[str, Any],
) -> dict[str, Any]:
    """Assert that C0-E1 changes only sampling plus administrative fields."""

    validate_v27_config(reference_c0, expected_arm="C0")
    protocol = exploratory.get("protocol", {})
    expected_protocol = {
        "name": "residual_bfn_v27_exploratory",
        "arm": V27_C0_E1_ARM,
        "preregistered": False,
        "primary_endpoint": "fixed_epoch_3",
        "checkpoint_selection": "fixed_epoch_3_only",
    }
    for key, expected in expected_protocol.items():
        _require_equal(protocol.get(key), expected, f"protocol.{key}")

    sampling = exploratory.get("training", {}).get("sampling")
    expected_sampling = {
        "mode": "category_balanced",
        "category_patterns": V27_C0_E1_PATTERNS,
        "target_fractions": V27_C0_E1_FRACTIONS,
    }
    _require_equal(sampling, expected_sampling, "training.sampling")
    if not str(
        exploratory.get("training", {}).get("output_dir", "")
    ).startswith("runs/residual_bfn_v27_exploratory/"):
        raise ValueError("C0-E1 must use a new exploratory output directory")

    normalized = copy.deepcopy(exploratory)
    normalized["protocol"] = copy.deepcopy(reference_c0["protocol"])
    for key in (
        "output_dir",
        "v27_preflight_report",
        "v27_profile_report",
        "sampling",
    ):
        normalized["training"][key] = copy.deepcopy(
            reference_c0["training"][key]
        )
    if normalized != reference_c0:
        raise ValueError(
            "C0-E1 changes fields outside protocol metadata, artifact paths, "
            "output_dir, and training.sampling"
        )
    return {
        "reference_arm": "C0",
        "exploratory_arm": V27_C0_E1_ARM,
        "behavioral_differences": ["training.sampling"],
        "administrative_differences": [
            "protocol",
            "training.output_dir",
            "training.v27_preflight_report",
            "training.v27_profile_report",
        ],
        "target_fractions": dict(V27_C0_E1_FRACTIONS),
        "target_counts": dict(V27_C0_E1_TARGET_COUNTS),
    }


def validate_v27_c0_e1_config(
    config: dict[str, Any],
    reference_c0: dict[str, Any],
) -> str:
    audit_v27_c0_e1_configs(reference_c0, config)
    _require_equal(config.get("seed"), 42, "seed")
    _require_equal(
        config["protocol"].get("expected_trainable_parameters"),
        V27_EXPECTED_TRAINABLE_PARAMETERS,
        "protocol.expected_trainable_parameters",
    )
    _require_equal(config["training"].get("epochs"), 6, "training.epochs")
    _require_equal(
        config["training"].get("stages"),
        [{"name": "residual", "epochs": 6}],
        "training.stages",
    )
    return V27_C0_E1_ARM


def audit_v27_c0prime_configs(
    reference_c0: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Assert that C0-prime changes only the registered routed readout."""

    validate_v27_config(reference_c0, expected_arm="C0")
    protocol = candidate.get("protocol", {})
    expected_protocol = {
        "name": "residual_bfn_v27_exploratory",
        "arm": V27_C0PRIME_ARM,
        "preregistered": True,
        "primary_endpoint": "fixed_epoch_3",
        "checkpoint_selection": "fixed_epoch_3_only",
        "expected_trainable_parameters": (
            V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }
    for key, expected in expected_protocol.items():
        _require_equal(protocol.get(key), expected, f"protocol.{key}")

    bfn = candidate.get("model", {}).get("bfn", {})
    _require_equal(
        bfn.get("routed_readout"),
        V27_C0PRIME_ROUTED_READOUT,
        "model.bfn.routed_readout",
    )
    _require_equal(
        candidate.get("training", {}).get("sampling"),
        {"mode": "shuffle"},
        "training.sampling",
    )
    if not str(
        candidate.get("training", {}).get("output_dir", "")
    ).startswith("runs/residual_bfn_v27_exploratory/c0prime_routed"):
        raise ValueError("C0-prime must use its independent output directory")

    normalized = copy.deepcopy(candidate)
    normalized["protocol"] = copy.deepcopy(reference_c0["protocol"])
    for key in (
        "output_dir",
        "v27_preflight_report",
        "v27_profile_report",
    ):
        normalized["training"][key] = copy.deepcopy(
            reference_c0["training"][key]
        )
    normalized["model"]["bfn"].pop("routed_readout", None)
    if normalized != reference_c0:
        raise ValueError(
            "C0-prime changes fields outside protocol metadata, artifact "
            "paths, output_dir, and model.bfn.routed_readout"
        )
    return {
        "reference_arm": "C0",
        "exploratory_arm": V27_C0PRIME_ARM,
        "behavioral_differences": ["model.bfn.routed_readout"],
        "administrative_differences": [
            "protocol",
            "training.output_dir",
            "training.v27_preflight_report",
            "training.v27_profile_report",
        ],
        "sampling": {"mode": "shuffle"},
        "expected_trainable_parameters": (
            V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }


def validate_v27_c0prime_config(
    config: dict[str, Any],
    reference_c0: dict[str, Any],
) -> str:
    audit_v27_c0prime_configs(reference_c0, config)
    _require_equal(config.get("seed"), 42, "seed")
    _require_equal(config["training"].get("epochs"), 6, "training.epochs")
    _require_equal(
        config["training"].get("stages"),
        [{"name": "residual", "epochs": 6}],
        "training.stages",
    )
    bfn = config["model"]["bfn"]
    for key, expected in {
        "inference_mode": "direct",
        "update_mode": "deterministic",
        "readout_mode": "receiver",
        "steps": 1,
        "training_rollout_steps": 1,
    }.items():
        _require_equal(bfn.get(key), expected, f"model.bfn.{key}")
    _require_equal(config["loss"].get("bfn_weight"), 0.0, "loss.bfn_weight")
    _require_equal(
        config["loss"].get("adapter_weight"),
        0.0,
        "loss.adapter_weight",
    )
    return V27_C0PRIME_ARM


def audit_v27_c0doubleprime_configs(
    reference_c0: dict[str, Any],
    reference_c0prime: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Assert that C0-double-prime changes only the private head type."""

    validate_v27_c0prime_config(reference_c0prime, reference_c0)
    protocol = candidate.get("protocol", {})
    expected_protocol = {
        "name": "residual_bfn_v27_exploratory",
        "arm": V27_C0DOUBLEPRIME_ARM,
        "preregistered": True,
        "primary_endpoint": "fixed_epoch_3",
        "checkpoint_selection": "fixed_epoch_3_only",
        "expected_trainable_parameters": (
            V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }
    for key, expected in expected_protocol.items():
        _require_equal(protocol.get(key), expected, f"protocol.{key}")

    routed_readout = candidate.get("model", {}).get("bfn", {}).get(
        "routed_readout"
    )
    _require_equal(
        routed_readout,
        V27_C0DOUBLEPRIME_ROUTED_READOUT,
        "model.bfn.routed_readout",
    )
    _require_equal(
        candidate.get("training", {}).get("sampling"),
        {"mode": "shuffle"},
        "training.sampling",
    )
    if not str(
        candidate.get("training", {}).get("output_dir", "")
    ).startswith(
        "runs/residual_bfn_v27_exploratory/c0doubleprime_bottleneck_routed"
    ):
        raise ValueError(
            "C0-double-prime must use its independent output directory"
        )

    normalized = copy.deepcopy(candidate)
    normalized["protocol"] = copy.deepcopy(reference_c0prime["protocol"])
    for key in (
        "output_dir",
        "v27_preflight_report",
        "v27_profile_report",
    ):
        normalized["training"][key] = copy.deepcopy(
            reference_c0prime["training"][key]
        )
    normalized["model"]["bfn"]["routed_readout"]["head_type"] = (
        V27_C0PRIME_ROUTED_READOUT["head_type"]
    )
    if normalized != reference_c0prime:
        raise ValueError(
            "C0-double-prime changes fields outside protocol metadata, "
            "artifact paths, output_dir, and "
            "model.bfn.routed_readout.head_type"
        )
    return {
        "reference_arm": V27_C0PRIME_ARM,
        "exploratory_arm": V27_C0DOUBLEPRIME_ARM,
        "behavioral_differences": [
            "model.bfn.routed_readout.head_type"
        ],
        "administrative_differences": [
            "protocol",
            "training.output_dir",
            "training.v27_preflight_report",
            "training.v27_profile_report",
        ],
        "sampling": {"mode": "shuffle"},
        "expected_trainable_parameters": (
            V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }


def validate_v27_c0doubleprime_config(
    config: dict[str, Any],
    reference_c0: dict[str, Any],
    reference_c0prime: dict[str, Any],
) -> str:
    audit_v27_c0doubleprime_configs(
        reference_c0,
        reference_c0prime,
        config,
    )
    _require_equal(config.get("seed"), 42, "seed")
    _require_equal(config["training"].get("epochs"), 6, "training.epochs")
    _require_equal(
        config["training"].get("stages"),
        [{"name": "residual", "epochs": 6}],
        "training.stages",
    )
    bfn = config["model"]["bfn"]
    for key, expected in {
        "inference_mode": "direct",
        "update_mode": "deterministic",
        "readout_mode": "receiver",
        "steps": 1,
        "training_rollout_steps": 1,
    }.items():
        _require_equal(bfn.get(key), expected, f"model.bfn.{key}")
    _require_equal(config["loss"].get("bfn_weight"), 0.0, "loss.bfn_weight")
    _require_equal(
        config["loss"].get("adapter_weight"),
        0.0,
        "loss.adapter_weight",
    )
    return V27_C0DOUBLEPRIME_ARM


def audit_v27_c0accept_configs(
    reference_c0: dict[str, Any],
    reference_c0prime: dict[str, Any],
    reference_c0doubleprime: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Assert that C0-accept adds only the frozen-basis acceptance gate."""

    validate_v27_c0doubleprime_config(
        reference_c0doubleprime,
        reference_c0,
        reference_c0prime,
    )
    protocol = candidate.get("protocol", {})
    expected_protocol = {
        "name": "residual_bfn_v27_exploratory",
        "arm": V27_C0ACCEPT_ARM,
        "source_arm": V27_C0DOUBLEPRIME_ARM,
        "source_checkpoint": "epoch_0003.pt",
        "warm_start_weights": "ema",
        "base_receiver_frozen": True,
        "preregistered": True,
        "primary_endpoint": "fixed_epoch_3",
        "checkpoint_selection": "fixed_epoch_3_only",
        "expected_trainable_parameters": (
            V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }
    for key, expected in expected_protocol.items():
        _require_equal(protocol.get(key), expected, f"protocol.{key}")

    routed_readout = candidate.get("model", {}).get("bfn", {}).get(
        "routed_readout"
    )
    _require_equal(
        routed_readout,
        V27_C0ACCEPT_ROUTED_READOUT,
        "model.bfn.routed_readout",
    )
    training = candidate.get("training", {})
    _require_equal(
        training.get("sampling"),
        {"mode": "shuffle"},
        "training.sampling",
    )
    if not str(training.get("output_dir", "")).startswith(
        "runs/residual_bfn_v27_exploratory/c0accept_gate"
    ):
        raise ValueError("C0-accept must use its independent output directory")
    if not str(training.get("initialization_checkpoint", "")).endswith(
        "c0accept_init.pt"
    ):
        raise ValueError(
            "C0-accept requires its registered initialization checkpoint"
        )

    normalized = copy.deepcopy(candidate)
    normalized["protocol"] = copy.deepcopy(
        reference_c0doubleprime["protocol"]
    )
    for key in (
        "output_dir",
        "v27_preflight_report",
        "v27_profile_report",
    ):
        normalized["training"][key] = copy.deepcopy(
            reference_c0doubleprime["training"][key]
        )
    normalized["training"].pop("initialization_checkpoint", None)
    normalized["model"]["bfn"]["routed_readout"].pop(
        "acceptance_gate",
        None,
    )
    if normalized != reference_c0doubleprime:
        raise ValueError(
            "C0-accept changes fields outside protocol metadata, artifact "
            "paths, output_dir, initialization checkpoint, and "
            "model.bfn.routed_readout.acceptance_gate"
        )
    return {
        "reference_arm": V27_C0DOUBLEPRIME_ARM,
        "exploratory_arm": V27_C0ACCEPT_ARM,
        "behavioral_differences": [
            "model.bfn.routed_readout.acceptance_gate"
        ],
        "initialization_difference": (
            "warm start from fixed C0-double-prime epoch-3 EMA"
        ),
        "frozen_components": [
            "SSR foundation",
            "receiver trunk",
            "three private bottleneck experts",
            "soft router",
        ],
        "trainable_component": (
            "image-level Linear(16,1) hard-clamped acceptance gate"
        ),
        "expected_trainable_parameters": (
            V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }


def validate_v27_c0accept_config(
    config: dict[str, Any],
    reference_c0: dict[str, Any],
    reference_c0prime: dict[str, Any],
    reference_c0doubleprime: dict[str, Any],
) -> str:
    audit_v27_c0accept_configs(
        reference_c0,
        reference_c0prime,
        reference_c0doubleprime,
        config,
    )
    _require_equal(config.get("seed"), 42, "seed")
    _require_equal(config["training"].get("epochs"), 6, "training.epochs")
    _require_equal(
        config["training"].get("stages"),
        [{"name": "residual", "epochs": 6}],
        "training.stages",
    )
    bfn = config["model"]["bfn"]
    for key, expected in {
        "inference_mode": "direct",
        "update_mode": "deterministic",
        "readout_mode": "receiver",
        "steps": 1,
        "training_rollout_steps": 1,
    }.items():
        _require_equal(bfn.get(key), expected, f"model.bfn.{key}")
    _require_equal(config["loss"].get("bfn_weight"), 0.0, "loss.bfn_weight")
    _require_equal(
        config["loss"].get("adapter_weight"),
        0.0,
        "loss.adapter_weight",
    )
    return V27_C0ACCEPT_ARM


def trainable_parameter_signature(module: nn.Module) -> dict[str, list[int]]:
    return {
        name: list(parameter.shape)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def trainable_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
