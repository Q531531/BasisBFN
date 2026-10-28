#!/usr/bin/env python3
"""Freeze a common D/B epoch budget from 200-step throughput profiles."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sgbformer.config import load_config
from sgbformer.comparison import foundation_signature
from sgbformer.protocol import (
    FROZEN_MAIN_BUDGET_POLICY,
    foundation_artifact_fingerprint,
    implementation_fingerprint,
    sha256_file,
    validate_frozen_main_contract,
)


STEPS_PER_EPOCH = math.ceil(18069 / 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-d", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--profile-d", required=True)
    parser.add_argument("--profile-b", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_profile(
    profile: dict[str, Any],
    *,
    arm: str,
    source_config: Path,
) -> float:
    if (
        profile.get("schema_version") != 1
        or profile.get("protocol") != "frozen_main_db"
        or profile.get("arm") != arm
        or profile.get("optimizer_steps") != 200
        or profile.get("global_batch_size") != 16
        or profile.get("train_pairs") != 18069
        or profile.get("quality_metrics_observed") is not False
    ):
        raise ValueError(f"Invalid 200-step profile for arm {arm}")
    if profile.get("source_config_sha256") != sha256_file(source_config):
        raise ValueError(f"Arm {arm} profile does not match its source config")
    seconds = float(profile.get("seconds_per_optimizer_step", 0))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Arm {arm} profile has invalid seconds/step")
    return seconds


def validate_baseline(
    baseline: dict[str, Any],
    *,
    baseline_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = baseline.get("images")
    protocol = baseline.get("protocol", {})
    average = baseline.get("average", {})
    if not isinstance(rows, list) or len(rows) != 903:
        raise ValueError("Tripwire baseline must contain exactly 903 images")
    names = [row.get("name") for row in rows if isinstance(row, dict)]
    if len(names) != 903 or len(set(names)) != 903 or None in names:
        raise ValueError("Tripwire baseline image names are incomplete or duplicated")
    manifest_path = Path(
        str(config["data"]["val"]["pairs_file"])
    ).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frozen dev manifest is missing: {manifest_path}"
        )
    manifest_names = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        manifest_names.append(str(Path(line.split("\t")[0]).with_suffix("")))
    if len(manifest_names) != 903 or set(manifest_names) != set(names):
        raise ValueError(
            "Tripwire baseline images differ from the frozen dev manifest"
        )
    required_protocol = {
        "test_y_channel": True,
        "crop_border": 0,
        "stochastic": False,
        "tile": None,
        "overlap": None,
        "output_quantization": "float32_clamp_0_1",
        "refinement_mode": "foundation_only",
        "steps": 0,
        "latent_bfn_steps": 0,
        "decoder_calls_per_trajectory": 1,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"Tripwire baseline protocol mismatch: {key}={protocol.get(key)!r}"
            )
    expected_foundation = foundation_signature(config)
    if protocol.get("foundation_signature") != expected_foundation:
        raise ValueError("Tripwire baseline uses a different SSR foundation")
    psnr = float(average.get("psnr", float("nan")))
    if not math.isfinite(psnr):
        raise ValueError("Tripwire baseline has no finite average PSNR")
    image_psnr = [
        float(row.get("psnr", float("nan")))
        for row in rows
    ]
    recomputed_psnr = sum(image_psnr) / len(image_psnr)
    if (
        not all(math.isfinite(value) for value in image_psnr)
        or not math.isclose(
            psnr,
            recomputed_psnr,
            rel_tol=0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Tripwire baseline average PSNR does not match its rows")
    regression = float(
        config["protocol"]["tripwire"]["maximum_regression_db"]
    )
    return {
        "path": str(baseline_path),
        "sha256": sha256_file(baseline_path),
        "images": len(rows),
        "dev_manifest_path": str(manifest_path.resolve()),
        "dev_manifest_sha256": sha256_file(manifest_path),
        "baseline_y_psnr": psnr,
        "maximum_regression_db": regression,
        "threshold_y_psnr": psnr - regression,
        "protocol_verified": True,
        "foundation_signature": expected_foundation,
    }


def compute_budget_decision(
    seconds_per_step: dict[str, float],
    policy: dict[str, Any],
) -> dict[str, Any]:
    requested_hours = float(policy["requested_gpu_hours_per_arm"])
    minimum_epochs = int(policy["minimum_epochs"])
    maximum_epochs = int(policy["maximum_epochs"])
    hard_cap_hours = float(policy["hard_gpu_hour_cap_per_arm"])
    if not (
        0 < requested_hours <= hard_cap_hours
        and 1 <= minimum_epochs <= maximum_epochs
    ):
        raise ValueError("Invalid frozen-main budget policy")
    if set(seconds_per_step) != {"D", "B"} or any(
        not math.isfinite(value) or value <= 0
        for value in seconds_per_step.values()
    ):
        raise ValueError("D/B seconds-per-step values must be finite and positive")

    def supported(hours: float, seconds: float) -> int:
        raw = hours * 3600 / (seconds * STEPS_PER_EPOCH)
        return math.floor(raw + 1e-12)

    base_supported = {
        arm: supported(requested_hours, seconds)
        for arm, seconds in seconds_per_step.items()
    }
    base_common = min(base_supported.values())
    required_hours_for_minimum = {
        arm: minimum_epochs * seconds * STEPS_PER_EPOCH / 3600
        for arm, seconds in seconds_per_step.items()
    }
    common_hours_for_minimum = max(required_hours_for_minimum.values())
    effective_hours = requested_hours
    if base_common < minimum_epochs:
        effective_hours = min(
            hard_cap_hours,
            max(requested_hours, common_hours_for_minimum),
        )
    effective_supported = {
        arm: supported(effective_hours, seconds)
        for arm, seconds in seconds_per_step.items()
    }
    uncapped_common_epochs = min(effective_supported.values())
    epochs = min(uncapped_common_epochs, maximum_epochs)
    if epochs < 1:
        raise RuntimeError(
            "The 16 GPU-hour hard cap cannot fund one matched epoch per arm"
        )
    return {
        "requested_gpu_hours_per_arm": requested_hours,
        "effective_gpu_hours_per_arm": effective_hours,
        "hard_gpu_hour_cap_per_arm": hard_cap_hours,
        "minimum_epochs": minimum_epochs,
        "maximum_epochs": maximum_epochs,
        "base_supported_epochs_by_arm": base_supported,
        "supported_epochs_by_arm": effective_supported,
        "uncapped_common_epochs": uncapped_common_epochs,
        "epochs": epochs,
        "budget_escalated": effective_hours > requested_hours,
        "budget_floor_reached": epochs < minimum_epochs,
        "epoch_cap_applied": uncapped_common_epochs > maximum_epochs,
        "required_gpu_hours_for_minimum_by_arm": required_hours_for_minimum,
        "required_common_gpu_hours_for_minimum": common_hours_for_minimum,
        "required_gpu_hours_for_final_epochs_by_arm": {
            arm: epochs * seconds * STEPS_PER_EPOCH / 3600
            for arm, seconds in seconds_per_step.items()
        },
    }


def locked_config(
    config: dict[str, Any],
    *,
    arm: str,
    epochs: int,
    manifest_path: Path,
    budget_decision: dict[str, Any],
    seconds_per_step: float,
    profile_path: Path,
    preflight_path: Path,
    device_name: str,
    implementation_sha256: str,
    foundation_artifacts: dict[str, dict[str, str]],
    tripwire: dict[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    training["epochs"] = epochs
    training["budget_locked"] = True
    training["budget_manifest"] = str(manifest_path)
    training["budget_arm"] = arm
    if len(training.get("stages", [])) != 1:
        raise ValueError("Frozen-main config must have exactly one training stage")
    training["stages"][0]["epochs"] = epochs
    config["protocol"]["tripwire"].update(
        {
            "baseline_y_psnr": tripwire["baseline_y_psnr"],
            "threshold_y_psnr": tripwire["threshold_y_psnr"],
            "baseline_sha256": tripwire["sha256"],
            "dev_manifest_sha256": tripwire["dev_manifest_sha256"],
            "baseline_protocol_verified": True,
        }
    )
    config["budget"] = {
        "requested_gpu_hours_per_arm": budget_decision[
            "requested_gpu_hours_per_arm"
        ],
        "effective_gpu_hours_per_arm": budget_decision[
            "effective_gpu_hours_per_arm"
        ],
        "hard_gpu_hour_cap_per_arm": budget_decision[
            "hard_gpu_hour_cap_per_arm"
        ],
        "minimum_epochs": budget_decision["minimum_epochs"],
        "maximum_epochs": budget_decision["maximum_epochs"],
        "budget_escalated": budget_decision["budget_escalated"],
        "budget_floor_reached": budget_decision["budget_floor_reached"],
        "epoch_cap_applied": budget_decision["epoch_cap_applied"],
        "steps_per_epoch": STEPS_PER_EPOCH,
        "epochs": epochs,
        "profile_seconds_per_optimizer_step": seconds_per_step,
        "profile_sha256": sha256_file(profile_path),
        "preflight_sha256": sha256_file(preflight_path),
        "device": device_name,
        "implementation_sha256": implementation_sha256,
        "foundation_artifacts": foundation_artifacts,
        "formula": (
            "floor(effective_gpu_hours_per_arm * 3600 / "
            "(seconds_per_optimizer_step * steps_per_epoch))"
        ),
    }
    return config


def main() -> None:
    args = parse_args()
    paths = {
        "D": Path(args.config_d).expanduser().resolve(),
        "B": Path(args.config_b).expanduser().resolve(),
    }
    profile_paths = {
        "D": Path(args.profile_d).expanduser().resolve(),
        "B": Path(args.profile_b).expanduser().resolve(),
    }
    preflight_path = Path(args.preflight).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    configs = {arm: load_config(path) for arm, path in paths.items()}
    for arm, config in configs.items():
        validate_frozen_main_contract(config, expected_arm=arm)
        if config["training"].get("budget_locked", False):
            raise ValueError("Budget freezer requires unlocked source configs")

    preflight = read_json(preflight_path)
    if (
        preflight.get("schema_version") != 1
        or preflight.get("protocol") != "frozen_main_db"
        or preflight.get("status") != "passed"
    ):
        raise ValueError("Preflight report is not a passing frozen-main archive")
    for arm, path in paths.items():
        expected = preflight.get("source_config_sha256", {}).get(arm)
        if expected != sha256_file(path):
            raise ValueError(f"Preflight report does not match arm {arm} config")

    profiles = {
        arm: read_json(profile_path)
        for arm, profile_path in profile_paths.items()
    }
    seconds = {
        arm: validate_profile(
            profiles[arm],
            arm=arm,
            source_config=paths[arm],
        )
        for arm in ("D", "B")
    }
    implementation = implementation_fingerprint()
    if preflight.get("implementation") != implementation or any(
        profiles[arm].get("implementation") != implementation
        for arm in ("D", "B")
    ):
        raise ValueError("Implementation changed between preflight and profiles")
    foundation_artifacts = foundation_artifact_fingerprint(configs["D"])
    if foundation_artifact_fingerprint(configs["B"]) != foundation_artifacts:
        raise ValueError("D/B configs resolve to different SSR artifacts")
    if (
        preflight.get("foundation_artifacts") != foundation_artifacts
        or any(
            profiles[arm].get("foundation_artifacts")
            != foundation_artifacts
            for arm in ("D", "B")
        )
    ):
        raise ValueError("SSR artifacts changed between preflight and profiles")
    profile_devices = {profiles[arm].get("device") for arm in ("D", "B")}
    if len(profile_devices) != 1 or None in profile_devices:
        raise ValueError("D/B profiles must use the same recorded GPU")
    if preflight.get("device") not in profile_devices:
        raise ValueError("Preflight and throughput profiles must use the same GPU")
    baseline = validate_baseline(
        read_json(baseline_path),
        baseline_path=baseline_path,
        config=configs["D"],
    )
    policy = configs["D"]["protocol"]["budget_policy"]
    if policy != configs["B"]["protocol"]["budget_policy"]:
        raise ValueError("D/B budget policies differ")
    if policy != FROZEN_MAIN_BUDGET_POLICY:
        raise ValueError("Frozen-main budget policy does not match the lock")
    budget_decision = compute_budget_decision(seconds, policy)
    epochs = int(budget_decision["epochs"])

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to replace non-empty freeze directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "budget_manifest.json"
    locked_paths = {
        "D": output_dir / "frozen_D.yaml",
        "B": output_dir / "frozen_B.yaml",
    }
    for arm in ("D", "B"):
        frozen = locked_config(
            configs[arm],
            arm=arm,
            epochs=epochs,
            manifest_path=manifest_path,
            budget_decision=budget_decision,
            seconds_per_step=seconds[arm],
            profile_path=profile_paths[arm],
            preflight_path=preflight_path,
            device_name=str(profiles[arm]["device"]),
            implementation_sha256=str(implementation["sha256"]),
            foundation_artifacts=foundation_artifacts,
            tripwire=baseline,
        )
        locked_paths[arm].write_text(
            yaml.safe_dump(frozen, sort_keys=False),
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 2,
        "protocol": "frozen_main_db",
        "budget_policy": policy,
        "steps_per_epoch": STEPS_PER_EPOCH,
        **budget_decision,
        "equal_epoch_policy": "minimum_supported_epoch_count_across_D_and_B",
        "ablation_epoch_policy": {
            "A0": epochs,
            "A1": epochs,
            "budget_locked": True,
            "launch_now": False,
            "rule": "reuse_frozen_main_epochs",
        },
        "tripwire": {
            **configs["D"]["protocol"]["tripwire"],
            **baseline,
            "decision_file": str(output_dir / "tripwire_decision.json"),
        },
        "implementation": implementation,
        "foundation_artifacts": foundation_artifacts,
        "profiles": {
            arm: {
                "path": str(profile_paths[arm]),
                "sha256": sha256_file(profile_paths[arm]),
                "seconds_per_optimizer_step": seconds[arm],
                "device": profiles[arm].get("device"),
                "peak_memory_allocated_mb": profiles[arm].get(
                    "peak_memory_allocated_mb"
                ),
            }
            for arm in ("D", "B")
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
        },
        "configs": {
            arm: {
                "path": str(locked_paths[arm]),
                "sha256": sha256_file(locked_paths[arm]),
            }
            for arm in ("D", "B")
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in (*locked_paths.values(), manifest_path):
        path.chmod(0o444)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
