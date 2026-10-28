"""Independent contract for the four parameter-matched BasisBFN ablations."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .config import load_config
from .protocol import (
    foundation_artifact_fingerprint, implementation_fingerprint,
    manifest_fingerprints, resolved_config_sha256, sha256_file,
)

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


def canonical_basisbfn_config(arm: str = "B4-G") -> dict[str, Any]:
    if arm not in BASISBFN_ARMS:
        raise ValueError(f"Unknown BasisBFN arm: {arm!r}")
    config = load_config(Path(__file__).parent / "configs" / "basisbfn.yaml")
    config["protocol"]["arm"] = arm
    readout = config["model"]["bfn"]["coefficient_state_readout"]
    readout["update_mode"] = BASISBFN_ARMS[arm]["update_mode"]
    readout["acceptance_gate"]["enabled"] = BASISBFN_ARMS[arm]["gate_enabled"]
    config["training"]["output_dir"] = f"runs/basisbfn/{arm.lower()}"
    return config


def _matrix_normal_form(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["protocol"]["arm"] = "<arm>"
    normalized["protocol"]["arm_description"] = "<description>"
    readout = normalized["model"]["bfn"]["coefficient_state_readout"]
    readout["update_mode"] = "<update_mode>"
    readout["acceptance_gate"]["enabled"] = "<gate_enabled>"
    for key in ("output_dir", "basisbfn_preflight_report", "basisbfn_profile_report"):
        normalized["training"][key] = f"<{key}>"
    return normalized


def _contract_normal_form(config: dict[str, Any]) -> dict[str, Any]:
    normalized = _matrix_normal_form(config)
    normalized["seed"] = "<seed>"
    foundation = normalized["model"]["foundation"]
    for key in ("repo_path", "checkpoint"):
        foundation[key] = f"<{key}>"
    for split in ("train", "val"):
        for key in ("input_dir", "target_dir", "pairs_file"):
            normalized["data"][split][key] = f"<{split}.{key}>"
    return normalized


def _differences(actual: Any, expected: Any, path: str = "") -> list[str]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences = []
        for key in sorted(set(actual) | set(expected)):
            label = f"{path}.{key}" if path else key
            if key not in actual or key not in expected:
                differences.append(label)
            else:
                differences.extend(_differences(actual[key], expected[key], label))
        return differences
    return [] if actual == expected else [path]


def audit_basisbfn_config(candidate: dict[str, Any]) -> dict[str, Any]:
    arm = str(candidate.get("protocol", {}).get("arm", ""))
    if arm not in BASISBFN_ARMS:
        raise ValueError(f"Unknown BasisBFN arm: {arm!r}")
    if candidate.get("seed") not in BASISBFN_SEEDS:
        raise ValueError(f"BasisBFN seed must be one of {BASISBFN_SEEDS}")
    reference = canonical_basisbfn_config(arm)
    readout = candidate.get("model", {}).get("bfn", {}).get("coefficient_state_readout")
    if readout != reference["model"]["bfn"]["coefficient_state_readout"]:
        raise ValueError("BasisBFN model.bfn.coefficient_state_readout differs from its arm")
    training = candidate.get("training", {})
    prefix = f"runs/basisbfn/{arm.lower()}"
    output = str(training.get("output_dir", ""))
    if output != prefix and not output.startswith(prefix + "/"):
        raise ValueError(f"BasisBFN arm {arm} must use output prefix {prefix}")
    for key in ("basisbfn_preflight_report", "basisbfn_profile_report"):
        if not training.get(key):
            raise ValueError(f"BasisBFN requires training.{key}")
    for split in ("train", "val"):
        for key in ("input_dir", "target_dir", "pairs_file"):
            if not candidate.get("data", {}).get(split, {}).get(key):
                raise ValueError(f"BasisBFN requires data.{split}.{key}")
    for key in ("repo_path", "checkpoint"):
        if not candidate.get("model", {}).get("foundation", {}).get(key):
            raise ValueError(f"BasisBFN requires model.foundation.{key}")
    try:
        changes = _differences(
            _contract_normal_form(candidate), _contract_normal_form(reference),
        )
    except KeyError as error:
        raise ValueError(f"BasisBFN contract is missing {error}") from error
    if changes:
        raise ValueError("BasisBFN contract changed unregistered fields: " + ", ".join(changes))
    return {
        "reference": "basisbfn",
        "candidate_arm": arm,
        "controlled_variables": [
            "model.bfn.coefficient_state_readout.update_mode",
            "model.bfn.coefficient_state_readout.acceptance_gate.enabled",
        ],
        "matrix_coordinates": copy.deepcopy(BASISBFN_ARMS[arm]),
        "expected_trainable_parameters": BASISBFN_EXPECTED_TRAINABLE_PARAMETERS,
    }


def validate_basisbfn_config(config: dict[str, Any]) -> str:
    return str(audit_basisbfn_config(config)["candidate_arm"])


def audit_basisbfn_matrix(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(configs) != set(BASISBFN_ARMS):
        raise ValueError("Coefficient matrix must contain exactly " + ", ".join(BASISBFN_ARMS))
    for arm, config in configs.items():
        if validate_basisbfn_config(config) != arm:
            raise ValueError(f"Coefficient matrix key {arm!r} contains another arm")
    baseline = _matrix_normal_form(configs["D4"])
    for arm, config in configs.items():
        if _matrix_normal_form(config) != baseline:
            raise ValueError(f"Coefficient matrix arm {arm} changes an unregistered field")
    return {
        "arms": copy.deepcopy(BASISBFN_ARMS),
        "controlled_variables": audit_basisbfn_config(configs["D4"])["controlled_variables"],
        "parameter_matched": True,
        "expected_trainable_parameters_per_arm": BASISBFN_EXPECTED_TRAINABLE_PARAMETERS,
    }


def _positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value > 0
    )


def load_verified_preflight(
    path: str | Path, *, config_path: str | Path,
    config: dict[str, Any], arm: str,
) -> dict[str, Any]:
    """Bind CUDA evidence to current code, config, SSR and both manifests."""
    if validate_basisbfn_config(config) != arm:
        raise RuntimeError("Preflight arm differs from config")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != 2 or report.get("protocol") != BASISBFN_PROTOCOL
        or report.get("arm") != arm or report.get("status") != "passed"
    ):
        raise RuntimeError("BasisBFN preflight report did not pass")
    if report.get("device_type") != "cuda" or not report.get("device"):
        raise RuntimeError("BasisBFN preflight requires real CUDA evidence")
    recorded = report.get("config", {})
    if recorded.get("source_sha256") != sha256_file(config_path):
        raise RuntimeError("Source config changed after BasisBFN preflight")
    if recorded.get("resolved_sha256") != resolved_config_sha256(config):
        raise RuntimeError("Resolved config changed after BasisBFN preflight")
    if report.get("implementation", {}).get("sha256") != implementation_fingerprint()["sha256"]:
        raise RuntimeError("Implementation changed after BasisBFN preflight")
    if report.get("foundation_artifacts") != foundation_artifact_fingerprint(config):
        raise RuntimeError("SSR artifacts changed after BasisBFN preflight")
    current_manifests = manifest_fingerprints(config)
    for label, expected in (("train", BASISBFN_EXPECTED_TRAIN_PAIRS), ("dev", BASISBFN_EXPECTED_DEV_PAIRS)):
        current = current_manifests[label]
        recorded_manifest = report.get("manifests", {}).get(label, {})
        if current["pairs"] != expected or any(recorded_manifest.get(k) != v for k, v in current.items()):
            raise RuntimeError(f"BasisBFN {label} manifest changed after preflight")
    if report.get("scope") != {
        "training_started": False, "quality_metrics_observed": False,
        "official_test_accessed": False,
    }:
        raise RuntimeError("BasisBFN preflight scope changed")
    if report.get("structure", {}).get("trainable_parameters") != BASISBFN_EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("BasisBFN preflight parameter count changed")
    identity = report.get("identity", {})
    difference = identity.get("maximum_absolute_difference")
    if (
        identity.get("images") != 8 or not isinstance(difference, (int, float))
        or not math.isfinite(difference) or not 0 <= difference <= 1e-6
    ):
        raise RuntimeError("BasisBFN preflight identity check is invalid")
    if report.get("gradients", {}).get("foundation_state_unchanged") is not True:
        raise RuntimeError("BasisBFN preflight did not preserve frozen SSR")
    return report


def validate_basisbfn_launch_artifacts(
    config_path: str | Path, config: dict[str, Any],
) -> dict[str, Any]:
    arm = validate_basisbfn_config(config)
    training = config["training"]
    preflight_path = Path(training["basisbfn_preflight_report"]).expanduser().resolve()
    profile_path = Path(training["basisbfn_profile_report"]).expanduser().resolve()
    for path in (preflight_path, profile_path):
        if not path.is_file():
            raise FileNotFoundError(f"BasisBFN report is missing: {path}")
    preflight = load_verified_preflight(
        preflight_path, config_path=config_path, config=config, arm=arm,
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2, "protocol": BASISBFN_PROTOCOL, "arm": arm, "status": "passed",
        "source_config_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_config_sha256(config),
        "device_type": "cuda", "device": preflight["device"],
        "optimizer_steps": BASISBFN_PROFILE_STEPS,
        "trainable_parameters": BASISBFN_EXPECTED_TRAINABLE_PARAMETERS,
        "receiver_calls_per_image": 1, "coefficient_state_updates_per_image": 4,
        "bfn_training_observer_calls_per_image": 1, "basis_evaluations_per_image": 1,
        "quality_metrics_observed": False,
        "global_batch_size": training["batch_size"],
        "micro_batch_size": training["micro_batch_size"],
        "micro_batches_per_optimizer_step": training["batch_size"] // training["micro_batch_size"],
        "train_pairs": BASISBFN_EXPECTED_TRAIN_PAIRS,
        "update_mode": BASISBFN_ARMS[arm]["update_mode"],
        "acceptance_gate_enabled": BASISBFN_ARMS[arm]["gate_enabled"],
        "sampling": {"mode": "shuffle"},
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            raise RuntimeError(f"BasisBFN profile.{key} must be {value!r}")
    if profile.get("preflight", {}).get("sha256") != sha256_file(preflight_path):
        raise RuntimeError("BasisBFN preflight changed after profiling")
    if profile.get("implementation", {}).get("sha256") != preflight["implementation"]["sha256"]:
        raise RuntimeError("Implementation changed after BasisBFN profile")
    seconds = profile.get("seconds_per_optimizer_step")
    memory = profile.get("peak_memory_allocated_mb")
    if not _positive(seconds):
        raise RuntimeError("BasisBFN profile has invalid throughput")
    if not _positive(memory) or memory > BASISBFN_PEAK_MEMORY_GATE_MB:
        raise RuntimeError("BasisBFN profile has invalid memory or exceeds the 2.0 GiB gate")
    elapsed = profile.get("elapsed_seconds")
    throughput = profile.get("throughput_images_per_second")
    if (
        not _positive(elapsed) or not _positive(throughput)
        or not math.isclose(elapsed, seconds * BASISBFN_PROFILE_STEPS, rel_tol=1e-6)
        or not math.isclose(throughput, training["batch_size"] / seconds, rel_tol=1e-6)
    ):
        raise RuntimeError("BasisBFN profile timing is inconsistent")
    return {
        "arm": arm, "preflight_path": str(preflight_path),
        "profile_path": str(profile_path), "profile_device": profile["device"],
        "profile_seconds_per_step": float(seconds),
    }
