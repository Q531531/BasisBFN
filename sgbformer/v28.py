"""Validation helpers for the v28-R0 deterministic-basis protocol."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .config import load_config
from .protocol import implementation_fingerprint, sha256_file
from .v27 import resolved_config_sha256, validate_v27_config


V28_PROTOCOL = "deterministic_basis_v28"
V28_R0_ARM = "R0"
V28_EXPECTED_TRAIN_PAIRS = 17_166
V28_EXPECTED_DEV_PAIRS = 903
V28_R0_EXPECTED_TRAINABLE_PARAMETERS = 616_340
V28_PROFILE_STEPS = 200
V28_PEAK_MEMORY_GATE_MB = 2.0 * 1024
V28_R0_SIGNED_BASIS_READOUT = {
    "enabled": True,
    "type": "spatial_rgb_basis",
    "num_bases": 4,
    "kernel_size": 3,
    "basis_init": "dct_orthogonal",
    "basis_init_gain": 1.0,
    "coefficient_activation": "tanh",
    "coefficient_range": [-1.0, 1.0],
    "normalization": "inverse_sqrt_num_bases",
    "zero_init_coefficients": True,
}
V28_R0_CONTINUATION_GATE = {
    "macro_delta_psnr_min": 0.05,
    "macro_ci95_lower_strictly_positive": True,
    "outdoor_rain_delta_min": -0.003,
    "outdoor_rain_ci95_upper_nonnegative": True,
    "raindrop_delta_min": -0.005,
    "raindrop_ci95_upper_nonnegative": True,
}


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"v28-R0 requires {label}={expected!r}, got {actual!r}")


def canonical_v27_c0_config() -> dict[str, Any]:
    return load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0_proxy.yaml"
    )


def audit_v28_r0_config(
    candidate: dict[str, Any],
    reference_c0: dict[str, Any],
) -> dict[str, Any]:
    """Assert that R0 changes only the final readout and protocol metadata."""

    validate_v27_config(reference_c0, expected_arm="C0")
    protocol = candidate.get("protocol", {})
    expected_protocol = {
        "name": V28_PROTOCOL,
        "arm": V28_R0_ARM,
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
        "expected_train_pairs": V28_EXPECTED_TRAIN_PAIRS,
        "expected_dev_pairs": V28_EXPECTED_DEV_PAIRS,
        "expected_trainable_parameters": (
            V28_R0_EXPECTED_TRAINABLE_PARAMETERS
        ),
        "official_test_locked": True,
        "long_training_locked": True,
        "continuation_gate": V28_R0_CONTINUATION_GATE,
    }
    for key, expected in expected_protocol.items():
        _require_equal(protocol.get(key), expected, f"protocol.{key}")

    bfn = candidate.get("model", {}).get("bfn", {})
    _require_equal(
        bfn.get("signed_basis_readout"),
        V28_R0_SIGNED_BASIS_READOUT,
        "model.bfn.signed_basis_readout",
    )
    if bfn.get("routed_readout") is not None:
        raise ValueError("v28-R0 must not configure routed_readout")

    training = candidate.get("training", {})
    _require_equal(training.get("sampling"), {"mode": "shuffle"}, "training.sampling")
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
    if not training.get("v28_preflight_report"):
        raise ValueError("v28-R0 requires training.v28_preflight_report")
    if not training.get("v28_profile_report"):
        raise ValueError("v28-R0 requires training.v28_profile_report")
    if not str(training.get("output_dir", "")).startswith(
        "runs/deterministic_basis_v28/r0"
    ):
        raise ValueError("v28-R0 must use its independent output directory")

    normalized = copy.deepcopy(candidate)
    normalized["protocol"] = copy.deepcopy(reference_c0["protocol"])
    normalized["model"]["bfn"].pop("signed_basis_readout")
    normalized_training = normalized["training"]
    reference_training = reference_c0["training"]
    for key in (
        "output_dir",
        "require_v27_preflight",
        "v27_preflight_report",
        "v27_profile_report",
    ):
        normalized_training[key] = copy.deepcopy(reference_training[key])
    normalized_training.pop("v28_preflight_report")
    normalized_training.pop("v28_profile_report")
    if normalized != reference_c0:
        raise ValueError(
            "v28-R0 changes fields outside protocol metadata, artifact paths, "
            "output_dir, and model.bfn.signed_basis_readout"
        )
    return {
        "reference_arm": "v27-C0",
        "candidate_arm": V28_R0_ARM,
        "behavioral_differences": ["model.bfn.signed_basis_readout"],
        "administrative_differences": [
            "protocol",
            "training.output_dir",
            "training.require_v27_preflight",
            "training.v27_preflight_report",
            "training.v27_profile_report",
            "training.v28_preflight_report",
            "training.v28_profile_report",
        ],
        "sampling": {"mode": "shuffle"},
        "expected_trainable_parameters": (
            V28_R0_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }


def validate_v28_r0_config(
    config: dict[str, Any],
    reference_c0: dict[str, Any] | None = None,
) -> str:
    reference = reference_c0 or canonical_v27_c0_config()
    audit_v28_r0_config(config, reference)
    _require_equal(config.get("seed"), 42, "seed")
    training = config["training"]
    _require_equal(training.get("epochs"), 6, "training.epochs")
    _require_equal(
        training.get("stages"),
        [{"name": "residual", "epochs": 6}],
        "training.stages",
    )
    if training.get("parameter_lrs"):
        raise ValueError("v28-R0 must use one optimizer learning rate")
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
        "steps": 1,
        "training_rollout_steps": 1,
        "rollout_backprop_steps": 1,
    }.items():
        _require_equal(bfn.get(key), expected, f"model.bfn.{key}")
    for key, expected in {
        "structural_weight": 0.0,
        "final_weight": 1.0,
        "final_loss_type": "mse",
        "bfn_weight": 0.0,
        "frequency_weight": 0.0,
        "correlation_weight": 0.0,
        "adapter_weight": 0.0,
        "uncertainty_weight": 0.0,
        "latent_category_weight": 0.0,
    }.items():
        _require_equal(config["loss"].get(key), expected, f"loss.{key}")
    return V28_R0_ARM


def validate_v28_launch_artifacts(
    config_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Require matching successful preflight and 200-step CUDA profile."""

    arm = validate_v28_r0_config(config)
    config_path = Path(config_path).expanduser().resolve()
    source_sha256 = sha256_file(config_path)
    resolved_sha256 = resolved_config_sha256(config)
    implementation_sha256 = implementation_fingerprint()["sha256"]
    training = config["training"]
    preflight_path = Path(
        training["v28_preflight_report"]
    ).expanduser().resolve()
    profile_path = Path(training["v28_profile_report"]).expanduser().resolve()
    if not preflight_path.is_file():
        raise FileNotFoundError(
            f"v28-R0 preflight report is missing: {preflight_path}"
        )
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"v28-R0 profile report is missing: {profile_path}"
        )

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("protocol") != V28_PROTOCOL
        or preflight.get("arm") != arm
        or preflight.get("status") != "passed"
    ):
        raise RuntimeError("v28-R0 preflight report did not pass")
    recorded = preflight.get("config", {})
    if recorded.get("source_sha256") != source_sha256:
        raise RuntimeError("v28-R0 source config changed after preflight")
    if recorded.get("resolved_sha256") != resolved_sha256:
        raise RuntimeError("v28-R0 resolved config changed after preflight")
    if preflight.get("implementation", {}).get("sha256") != implementation_sha256:
        raise RuntimeError("v28-R0 implementation changed after preflight")

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if (
        profile.get("protocol") != V28_PROTOCOL
        or profile.get("arm") != arm
        or profile.get("status") != "passed"
    ):
        raise RuntimeError("v28-R0 profile report did not pass")
    if profile.get("source_config_sha256") != source_sha256:
        raise RuntimeError("v28-R0 source config changed after profiling")
    if profile.get("resolved_config_sha256") != resolved_sha256:
        raise RuntimeError("v28-R0 resolved config changed after profiling")
    if profile.get("optimizer_steps") != V28_PROFILE_STEPS:
        raise RuntimeError("v28-R0 profile must contain exactly 200 optimizer steps")
    if (
        profile.get("trainable_parameters")
        != V28_R0_EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("v28-R0 profile trainable parameter count changed")
    if profile.get("receiver_calls_per_image") != 1:
        raise RuntimeError("v28-R0 profile must use one receiver call per image")
    if profile.get("quality_metrics_observed") is not False:
        raise RuntimeError("v28-R0 profile must not inspect quality metrics")
    if profile.get("preflight", {}).get("sha256") != sha256_file(preflight_path):
        raise RuntimeError("v28-R0 preflight report changed after profiling")
    if profile.get("implementation", {}).get("sha256") != implementation_sha256:
        raise RuntimeError("v28-R0 implementation changed after profiling")
    seconds_per_step = profile.get("seconds_per_optimizer_step")
    if (
        not isinstance(seconds_per_step, (int, float))
        or not math.isfinite(seconds_per_step)
        or seconds_per_step <= 0
    ):
        raise RuntimeError("v28-R0 profile has an invalid throughput measurement")
    peak_memory = profile.get("peak_memory_allocated_mb")
    if (
        not isinstance(peak_memory, (int, float))
        or not math.isfinite(peak_memory)
        or peak_memory > V28_PEAK_MEMORY_GATE_MB
    ):
        raise RuntimeError("v28-R0 profile exceeds the 2.0 GiB memory gate")
    return {
        "arm": arm,
        "preflight_path": str(preflight_path),
        "profile_path": str(profile_path),
        "profile_device": profile.get("device"),
        "profile_seconds_per_step": float(seconds_per_step),
    }
