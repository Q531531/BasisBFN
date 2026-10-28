"""Protocol guards for the matched BasisBFN-state experiment matrix."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .config import load_config
from .protocol import implementation_fingerprint, sha256_file
from .v27 import resolved_config_sha256, validate_v27_config


BASISBFN_PROTOCOL = "basisbfn"
BASISBFN_STEPS = 4
BASISBFN_SEEDS = (42, 43, 44, 45)
BASISBFN_ARMS = {
    "D4": {"update_mode": "deterministic", "gate_enabled": False},
    "D4-G": {"update_mode": "deterministic", "gate_enabled": True},
    "B4": {"update_mode": "bayesian", "gate_enabled": False},
    "B4-G": {"update_mode": "bayesian", "gate_enabled": True},
}
BASISBFN_EXPECTED_TRAINABLE_PARAMETERS = 624_217
BASISBFN_EXPECTED_TRAIN_PAIRS = 17_166
BASISBFN_EXPECTED_DEV_PAIRS = 903
BASISBFN_PROFILE_STEPS = 200
BASISBFN_PEAK_MEMORY_GATE_MB = 2.0 * 1024
BASISBFN_GATE = {
    "type": "posterior_spatial_sigmoid",
    "condition": ["context", "posterior_mean", "log_precision"],
    "range": [0.0, 1.0],
    "initial_probability": 0.5,
}
BASISBFN_READOUT_COMMON = {
    "enabled": True,
    "type": "spatial_coefficient_flow",
    "num_bases": 4,
    "steps": BASISBFN_STEPS,
    "hidden_channels": 16,
    "kernel_size": 3,
    "basis_init": "dct_orthogonal",
    "basis_init_gain": 1.0,
    "observation_activation": "tanh",
    "coefficient_activation": "tanh",
    "coefficient_range": [-1.0, 1.0],
    "normalization": "inverse_sqrt_num_bases",
    "zero_init_observer": True,
    "min_variance": 0.01,
    "accuracy_schedule": "linear_beta",
    "min_accuracy_scale": 0.25,
    "target_ridge": 0.001,
}


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"BasisBFN protocol requires {label}={expected!r}, "
            f"got {actual!r}"
        )


def canonical_v27_c0_config() -> dict[str, Any]:
    return load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0_proxy.yaml"
    )


def expected_readout(arm: str) -> dict[str, Any]:
    if arm not in BASISBFN_ARMS:
        raise ValueError(f"Unknown BasisBFN arm: {arm!r}")
    matrix = BASISBFN_ARMS[arm]
    readout = copy.deepcopy(BASISBFN_READOUT_COMMON)
    readout["update_mode"] = matrix["update_mode"]
    readout["acceptance_gate"] = {
        "enabled": matrix["gate_enabled"],
        **BASISBFN_GATE,
    }
    return readout


def audit_basisbfn_config(
    candidate: dict[str, Any],
    reference_c0: dict[str, Any],
) -> dict[str, Any]:
    """Limit each candidate to the registered coefficient-state surface."""

    validate_v27_config(reference_c0, expected_arm="C0")
    protocol = candidate.get("protocol", {})
    arm = str(protocol.get("arm", ""))
    if arm not in BASISBFN_ARMS:
        raise ValueError(f"Unknown BasisBFN arm: {arm!r}")
    expected_protocol = {
        "name": BASISBFN_PROTOCOL,
        "preregistered": True,
        "primary_endpoint": "fixed_epoch_6",
        "diagnostic_endpoints": [
            "fixed_epoch_1",
            "fixed_epoch_2",
            "fixed_epoch_3",
            "fixed_epoch_4",
            "fixed_epoch_5",
        ],
        "checkpoint_selection": "fixed_epoch_6_only",
        "expected_train_pairs": BASISBFN_EXPECTED_TRAIN_PAIRS,
        "expected_dev_pairs": BASISBFN_EXPECTED_DEV_PAIRS,
        "expected_trainable_parameters": (
            BASISBFN_EXPECTED_TRAINABLE_PARAMETERS
        ),
        "official_test_locked": True,
        "long_training_locked": True,
    }
    for key, expected in expected_protocol.items():
        _require_equal(protocol.get(key), expected, f"protocol.{key}")

    bfn = candidate.get("model", {}).get("bfn", {})
    _require_equal(
        bfn.get("coefficient_state_readout"),
        expected_readout(arm),
        "model.bfn.coefficient_state_readout",
    )
    if bfn.get("routed_readout") is not None:
        raise ValueError("BasisBFN arms must not configure routed_readout")
    if bfn.get("signed_basis_readout") is not None:
        raise ValueError(
            "BasisBFN arms must not configure signed_basis_readout"
        )

    training = candidate.get("training", {})
    _require_equal(
        training.get("sampling"),
        {"mode": "shuffle"},
        "training.sampling",
    )
    _require_equal(
        training.get("require_v27_preflight"),
        False,
        "training.require_v27_preflight",
    )
    _require_equal(
        training.get("v27_preflight_report"),
        None,
        "training.v27_preflight_report",
    )
    _require_equal(
        training.get("v27_profile_report"),
        None,
        "training.v27_profile_report",
    )
    if not training.get("basisbfn_preflight_report"):
        raise ValueError(
            "BasisBFN arms require "
            "training.basisbfn_preflight_report"
        )
    if not training.get("basisbfn_profile_report"):
        raise ValueError(
            "BasisBFN arms require "
            "training.basisbfn_profile_report"
        )
    output_prefix = f"runs/basisbfn/{arm.lower()}"
    if not str(training.get("output_dir", "")).startswith(output_prefix):
        raise ValueError(
            f"BasisBFN arm {arm} must use output prefix {output_prefix}"
        )

    seed = int(candidate.get("seed", -1))
    if seed not in BASISBFN_SEEDS:
        raise ValueError(
            f"BasisBFN seed must be one of {BASISBFN_SEEDS}, got {seed}"
        )

    normalized = copy.deepcopy(candidate)
    normalized["seed"] = reference_c0["seed"]
    normalized["protocol"] = copy.deepcopy(reference_c0["protocol"])
    normalized["model"]["bfn"].pop("coefficient_state_readout")
    normalized["model"]["bfn"]["steps"] = reference_c0["model"]["bfn"]["steps"]
    normalized["loss"]["bfn_weight"] = reference_c0["loss"]["bfn_weight"]
    normalized_training = normalized["training"]
    reference_training = reference_c0["training"]
    for key in (
        "output_dir",
        "require_v27_preflight",
        "v27_preflight_report",
        "v27_profile_report",
    ):
        normalized_training[key] = copy.deepcopy(reference_training[key])
    normalized_training.pop("basisbfn_preflight_report")
    normalized_training.pop("basisbfn_profile_report")
    if normalized != reference_c0:
        raise ValueError(
            "BasisBFN arm changes fields outside protocol metadata, "
            "artifact paths, output_dir, loss.bfn_weight, model.bfn.steps, "
            "and model.bfn.coefficient_state_readout"
        )
    return {
        "reference_arm": "v27-C0",
        "candidate_arm": arm,
        "behavioral_differences": [
            "loss.bfn_weight",
            "model.bfn.steps",
            "model.bfn.coefficient_state_readout",
        ],
        "matrix_coordinates": copy.deepcopy(BASISBFN_ARMS[arm]),
        "sampling": {"mode": "shuffle"},
        "expected_trainable_parameters": (
            BASISBFN_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }


def validate_basisbfn_config(
    config: dict[str, Any],
    reference_c0: dict[str, Any] | None = None,
) -> str:
    reference = reference_c0 or canonical_v27_c0_config()
    audit = audit_basisbfn_config(config, reference)
    arm = str(audit["candidate_arm"])
    seed = int(config.get("seed", -1))
    if seed not in BASISBFN_SEEDS:
        raise ValueError(
            f"BasisBFN seed must be one of {BASISBFN_SEEDS}, got {seed}"
        )
    training = config["training"]
    _require_equal(training.get("epochs"), 6, "training.epochs")
    _require_equal(
        training.get("stages"),
        [{"name": "residual", "epochs": 6}],
        "training.stages",
    )
    if training.get("parameter_lrs"):
        raise ValueError(
            "BasisBFN arms must use one optimizer learning rate"
        )
    bfn = config["model"]["bfn"]
    for key, expected in {
        "inference_mode": "direct",
        "receiver_type": "unet",
        "receiver_trainable": True,
        "condition_mode": "observation_rich",
        "receiver_output_mode": "identity",
        "update_mode": "deterministic",
        "readout_mode": "receiver",
        "residual_base_clamp": True,
        "steps": BASISBFN_STEPS,
        "training_rollout_steps": 1,
        "rollout_backprop_steps": 1,
    }.items():
        _require_equal(bfn.get(key), expected, f"model.bfn.{key}")
    expected_losses = {
        "structural_weight": 0.0,
        "final_weight": 1.0,
        "final_loss_type": "mse",
        "bfn_weight": 0.00004,
        "frequency_weight": 0.0,
        "correlation_weight": 0.0,
        "adapter_weight": 0.0,
        "uncertainty_weight": 0.0,
        "latent_category_weight": 0.0,
    }
    for key, expected in expected_losses.items():
        _require_equal(config["loss"].get(key), expected, f"loss.{key}")
    return arm


def _matrix_normal_form(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["protocol"]["arm"] = "<arm>"
    normalized["protocol"]["arm_description"] = "<arm_description>"
    readout = normalized["model"]["bfn"]["coefficient_state_readout"]
    readout["update_mode"] = "<update_mode>"
    readout["acceptance_gate"]["enabled"] = "<gate_enabled>"
    training = normalized["training"]
    training["output_dir"] = "<output_dir>"
    training["basisbfn_preflight_report"] = "<preflight>"
    training["basisbfn_profile_report"] = "<profile>"
    return normalized


def audit_basisbfn_matrix(
    configs: dict[str, dict[str, Any]],
    reference_c0: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove that the four arms vary only update formula and gate use."""

    if set(configs) != set(BASISBFN_ARMS):
        raise ValueError(
            "Coefficient matrix must contain exactly "
            + ", ".join(BASISBFN_ARMS)
        )
    reference = reference_c0 or canonical_v27_c0_config()
    normalized: dict[str, dict[str, Any]] = {}
    for arm, config in configs.items():
        validated_arm = validate_basisbfn_config(config, reference)
        if validated_arm != arm:
            raise ValueError(
                f"Coefficient matrix key {arm!r} contains arm {validated_arm!r}"
            )
        normalized[arm] = _matrix_normal_form(config)
    baseline = normalized["D4"]
    for arm, candidate in normalized.items():
        if candidate != baseline:
            raise ValueError(
                f"Coefficient matrix arm {arm} changes an unregistered field"
            )
    return {
        "arms": copy.deepcopy(BASISBFN_ARMS),
        "controlled_variables": [
            "model.bfn.coefficient_state_readout.update_mode",
            (
                "model.bfn.coefficient_state_readout."
                "acceptance_gate.enabled"
            ),
        ],
        "parameter_matched": True,
        "expected_trainable_parameters_per_arm": (
            BASISBFN_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }


def validate_basisbfn_launch_artifacts(
    config_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Require arm-bound preflight and 200-step CUDA profile artifacts."""

    arm = validate_basisbfn_config(config)
    config_path = Path(config_path).expanduser().resolve()
    source_sha256 = sha256_file(config_path)
    resolved_sha256 = resolved_config_sha256(config)
    implementation_sha256 = implementation_fingerprint()["sha256"]
    training = config["training"]
    preflight_path = Path(
        training["basisbfn_preflight_report"]
    ).expanduser().resolve()
    profile_path = Path(
        training["basisbfn_profile_report"]
    ).expanduser().resolve()
    if not preflight_path.is_file():
        raise FileNotFoundError(
            f"BasisBFN preflight report is missing: {preflight_path}"
        )
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"BasisBFN profile report is missing: {profile_path}"
        )

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("protocol") != BASISBFN_PROTOCOL
        or preflight.get("arm") != arm
        or preflight.get("status") != "passed"
    ):
        raise RuntimeError("BasisBFN preflight report did not pass")
    recorded = preflight.get("config", {})
    if recorded.get("source_sha256") != source_sha256:
        raise RuntimeError("Source config changed after coefficient preflight")
    if recorded.get("resolved_sha256") != resolved_sha256:
        raise RuntimeError("Resolved config changed after coefficient preflight")
    if preflight.get("implementation", {}).get("sha256") != implementation_sha256:
        raise RuntimeError("Implementation changed after coefficient preflight")

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if (
        profile.get("protocol") != BASISBFN_PROTOCOL
        or profile.get("arm") != arm
        or profile.get("status") != "passed"
    ):
        raise RuntimeError("BasisBFN profile report did not pass")
    if profile.get("source_config_sha256") != source_sha256:
        raise RuntimeError("Source config changed after coefficient profile")
    if profile.get("resolved_config_sha256") != resolved_sha256:
        raise RuntimeError("Resolved config changed after coefficient profile")
    if profile.get("optimizer_steps") != BASISBFN_PROFILE_STEPS:
        raise RuntimeError(
            "BasisBFN profile must contain exactly 200 optimizer steps"
        )
    if (
        profile.get("trainable_parameters")
        != BASISBFN_EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError(
            "BasisBFN profile trainable parameter count changed"
        )
    if profile.get("receiver_calls_per_image") != 1:
        raise RuntimeError(
            "BasisBFN profile must use one receiver call per image"
        )
    if profile.get("coefficient_state_updates_per_image") != 4:
        raise RuntimeError(
            "BasisBFN profile must use four state updates per image"
        )
    expected_training_calls = 1
    if (
        profile.get("bfn_training_observer_calls_per_image")
        != expected_training_calls
    ):
        raise RuntimeError(
            "BasisBFN profile has the wrong BFN training call count"
        )
    if profile.get("basis_evaluations_per_image") != 1:
        raise RuntimeError(
            "BasisBFN profile must build the basis once per image"
        )
    if profile.get("quality_metrics_observed") is not False:
        raise RuntimeError(
            "BasisBFN profile must not inspect quality metrics"
        )
    if profile.get("preflight", {}).get("sha256") != sha256_file(preflight_path):
        raise RuntimeError("Coefficient preflight changed after profiling")
    if profile.get("implementation", {}).get("sha256") != implementation_sha256:
        raise RuntimeError("Implementation changed after coefficient profile")
    seconds_per_step = profile.get("seconds_per_optimizer_step")
    if (
        not isinstance(seconds_per_step, (int, float))
        or not math.isfinite(seconds_per_step)
        or seconds_per_step <= 0
    ):
        raise RuntimeError("Coefficient profile has invalid throughput")
    peak_memory = profile.get("peak_memory_allocated_mb")
    if (
        not isinstance(peak_memory, (int, float))
        or not math.isfinite(peak_memory)
        or peak_memory > BASISBFN_PEAK_MEMORY_GATE_MB
    ):
        raise RuntimeError("Coefficient profile exceeds the 2.0 GiB memory gate")
    return {
        "arm": arm,
        "preflight_path": str(preflight_path),
        "profile_path": str(profile_path),
        "profile_device": profile.get("device"),
        "profile_seconds_per_step": float(seconds_per_step),
    }
