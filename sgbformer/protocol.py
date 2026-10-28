"""Guards for the preregistered frozen D/B training protocol."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .comparison import foundation_signature


LOSS_WEIGHT_KEYS = (
    "structural_weight",
    "bfn_weight",
    "final_weight",
    "correlation_weight",
    "frequency_weight",
    "adapter_weight",
    "uncertainty_weight",
    "latent_category_weight",
)

FROZEN_MAIN_BUDGET_POLICY = {
    "requested_gpu_hours_per_arm": 10.0,
    "minimum_epochs": 12,
    "maximum_epochs": 30,
    "hard_gpu_hour_cap_per_arm": 16.0,
    "shared_epoch_rule": "min_supported_across_D_and_B",
    "ablations_reuse_main_epochs": True,
}

FROZEN_MAIN_TRIPWIRE_POLICY = {
    "epoch": 4,
    "metric": "dev_y_psnr",
    "maximum_regression_db": 0.05,
    "require_both_arms": True,
    "use_b_minus_d": False,
    "pause_after_epoch": True,
    "evaluation_weights": "ema",
    "baseline_source": "verified_frozen_ssr_dev_json",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    paths = sorted((project_root / "sgbformer").rglob("*.py"))
    paths.extend(
        project_root / relative
        for relative in (
            "train.py",
            "scripts/profile_frozen_main.py",
            "scripts/preflight_frozen_main.py",
            "scripts/freeze_frozen_main_budget.py",
            "scripts/decide_frozen_main_tripwire.py",
            "scripts/preflight_v27_residual_bfn.py",
            "scripts/preflight_v27_c0_e1.py",
            "scripts/preflight_v27_c0prime.py",
            "scripts/preflight_v27_c0doubleprime.py",
            "scripts/preflight_v27_c0accept.py",
            "scripts/preflight_v27_c0accept.py",
            "scripts/profile_v27_residual_bfn.py",
            "scripts/analyze_v27_proxy.py",
            "scripts/analyze_v27_c0_e1.py",
            "scripts/analyze_v27_c0prime.py",
            "scripts/analyze_v27_c0doubleprime.py",
            "scripts/analyze_v27_c0accept.py",
            "scripts/analyze_v27_c0accept.py",
            "scripts/probe_v27_c0prime_routing.py",
            "scripts/audit_v27_residual_geometry.py",
            "scripts/preflight_v28_r0.py",
            "scripts/profile_v28_r0.py",
            "scripts/analyze_v28_r0.py",
            "scripts/preflight_basisbfn.py",
            "scripts/profile_basisbfn.py",
            "scripts/analyze_basisbfn.py",
            "scripts/make_basisbfn_config.py",
        )
    )
    file_hashes = {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in sorted(paths)
    }
    digest = hashlib.sha256()
    for relative, file_hash in file_hashes.items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "files": file_hashes,
    }


def foundation_artifact_fingerprint(
    config: dict[str, Any],
) -> dict[str, dict[str, str]]:
    foundation = config.get("model", {}).get("foundation", {})
    repo = Path(str(foundation.get("repo_path", ""))).expanduser().resolve()
    architecture = repo / str(foundation.get("architecture_file", ""))
    checkpoint = Path(str(foundation.get("checkpoint", ""))).expanduser().resolve()
    for label, path in (("architecture", architecture), ("checkpoint", checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(
                f"Frozen-main SSR {label} artifact is missing: {path}"
            )
    return {
        "architecture": {
            "path": str(architecture),
            "sha256": sha256_file(architecture),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
    }


def loss_term_manifest(loss_config: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(loss_config.get(key, 0.0))
        for key in LOSS_WEIGHT_KEYS
    }


def validate_frozen_main_contract(
    config: dict[str, Any],
    *,
    expected_arm: str | None = None,
) -> str:
    """Validate every field frozen before D/B throughput measurement."""

    protocol = config.get("protocol", {})
    arm = str(protocol.get("arm", ""))
    if protocol.get("name") != "frozen_main_db" or arm not in {"D", "B"}:
        raise ValueError("Config is not a registered frozen-main D/B arm")
    if expected_arm is not None and arm != expected_arm:
        raise ValueError(f"Expected arm {expected_arm}, got {arm}")
    if int(config.get("seed", -1)) != 42:
        raise ValueError("Frozen-main seed must be 42")
    if (
        int(protocol.get("expected_train_pairs", -1)) != 18069
        or int(protocol.get("expected_dev_pairs", -1)) != 903
        or protocol.get("checkpoint_selection") != "dev_y_psnr"
    ):
        raise ValueError("Frozen-main data counts or checkpoint rule changed")
    if protocol.get("budget_policy") != FROZEN_MAIN_BUDGET_POLICY:
        raise ValueError("Frozen-main budget policy changed")
    tripwire = protocol.get("tripwire", {})
    for key, expected in FROZEN_MAIN_TRIPWIRE_POLICY.items():
        if tripwire.get(key) != expected:
            raise ValueError(f"Frozen-main tripwire policy changed: {key}")
    generated_tripwire_keys = set(tripwire) - set(FROZEN_MAIN_TRIPWIRE_POLICY)
    if generated_tripwire_keys and not config.get("training", {}).get(
        "budget_locked",
        False,
    ):
        raise ValueError("Unlocked source config contains generated tripwire fields")

    model = config.get("model", {})
    core = model.get("core", {})
    if (
        core.get("require_spectral_gate") is not True
        or core.get("require_iterative_bfn") is not True
        or core.get("require_entropy_mask") is not False
        or core.get("bfn_steps") != 4
    ):
        raise ValueError("Frozen-main core requirements changed")
    foundation = model.get("foundation", {})
    if (
        foundation.get("freeze") is not True
        or foundation.get("strict_load") is not True
        or foundation.get("frozen_modules_eval") is not True
    ):
        raise ValueError("Frozen-main SSR initialization contract changed")
    if foundation.get("trainable_modules") != [
        "decoder_level1",
        "decoder_level2",
        "output",
    ]:
        raise ValueError("Frozen-main SSR trainable suffix changed")
    adapters = foundation.get("feature_adapters", {})
    if set(adapters) != {
        "encoder_level3",
        "decoder_level3",
        "decoder_level2",
    }:
        raise ValueError("Frozen-main adapter insertion points changed")
    if adapters["encoder_level3"].get("type") != "paper_spectral_gate":
        raise ValueError("Frozen-main paper spectral gate is missing")
    if (
        adapters["encoder_level3"].get("channels") != 192
        or adapters["encoder_level3"].get("initial_gate") != 0.999
    ):
        raise ValueError("Frozen-main paper spectral gate settings changed")
    for name in ("decoder_level3", "decoder_level2"):
        if adapters[name].get("type") != "latent_film_spectral":
            raise ValueError("Frozen-main latent adapter type changed")
        expected_channels = 192 if name == "decoder_level3" else 96
        if (
            adapters[name].get("channels") != expected_channels
            or adapters[name].get("global_dim") != 32
            or adapters[name].get("spatial_dim") != 8
            or adapters[name].get("hidden_channels") != 64
            or adapters[name].get("trainable") is not True
        ):
            raise ValueError(f"Frozen-main latent adapter settings changed: {name}")

    latent = model.get("latent_bfn", {})
    expected_update = "deterministic" if arm == "D" else "bayesian"
    required_latent = {
        "steps": 4,
        "global_dim": 32,
        "spatial_dim": 8,
        "hidden_channels": 64,
        "feature_channels": 192,
        "spatial_downsample": 4,
        "num_categories": 3,
        "min_variance": 0.01,
        "min_accuracy_scale": 0.25,
        "target_residual_scale": 0.2,
        "accuracy_schedule": "linear_beta",
        "loss_mode": "final_mean",
        "loss_weighting": "adaptive_accuracy",
        "adapter_precision_mode": "belief",
        "update_mode": expected_update,
    }
    for key, expected in required_latent.items():
        if latent.get(key) != expected:
            raise ValueError(
                f"Frozen-main latent field changed: {key}={latent.get(key)!r}"
            )
    if set(latent.get("category_patterns", {})) != {
        "snow",
        "outdoor_rain",
        "raindrop",
    }:
        raise ValueError("Frozen-main latent category definitions changed")
    bfn = model.get("bfn", {})
    if (
        bfn.get("inference_mode") != "none"
        or bfn.get("receiver_trainable") is not False
        or bfn.get("mask_mode") != "none"
    ):
        raise ValueError("Frozen-main unused RGB-BFN path changed")

    data = config.get("data", {})
    train = data.get("train", {})
    if (
        int(data.get("workers", -1)) != 12
        or train.get("pairs_file") is not None
        or int(train.get("patch_size", -1)) != 256
        or train.get("augment") is not True
    ):
        raise ValueError("Frozen-main training data protocol changed")
    validation_data = data.get("val", {})
    if (
        validation_data.get("pairs_file") is None
        or validation_data.get("strict") is not True
    ):
        raise ValueError("Frozen-main dev manifest contract changed")

    loss = loss_term_manifest(config.get("loss", {}))
    expected_loss = {key: 0.0 for key in LOSS_WEIGHT_KEYS}
    expected_loss["structural_weight"] = 1.0
    expected_loss["bfn_weight"] = 0.01
    expected_loss["latent_category_weight"] = 0.01
    if loss != expected_loss:
        raise ValueError(
            f"Frozen-main {arm} loss terms changed: {loss}"
        )
    if config.get("loss", {}).get("structural_loss_type") != "mse":
        raise ValueError("Frozen-main structural loss must be MSE")
    validation = config.get("validation", {})
    if (
        validation.get("test_y_channel") is not True
        or int(validation.get("crop_border", -1)) != 0
        or validation.get("lpips") is not False
    ):
        raise ValueError("Frozen-main checkpoint validation protocol changed")

    training = config.get("training", {})
    required_training = {
        "batch_size": 16,
        "micro_batch_size": 4,
        "learning_rate": 0.0002,
        "minimum_learning_rate_ratio": 0.1,
        "warmup_epochs": 1,
        "schedule_minimum_mode": "per_group",
        "amp": True,
        "amp_dtype": "bfloat16",
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "validate_interval": 1,
        "log_interval": 250,
        "ema_decay": 0.999,
        "ema_warmup_updates": 2000,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "reproducible_data": True,
        "require_budget_lock": True,
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ValueError(
                f"Frozen-main training field changed: {key}={training.get(key)!r}"
            )
    expected_rates = {
        "backbone.model.decoder_level1": 0.00002,
        "backbone.model.decoder_level2": 0.00002,
        "backbone.model.output": 0.00002,
    }
    if training.get("parameter_lrs") != expected_rates:
        raise ValueError("Frozen-main parameter LR groups changed")
    if training.get("betas") != [0.9, 0.999]:
        raise ValueError("Frozen-main AdamW betas changed")
    if training.get("sampling") != {"mode": "shuffle"}:
        raise ValueError("Frozen-main sampling protocol changed")
    stages = training.get("stages", [])
    if len(stages) != 1 or stages[0].get("name") != "joint":
        raise ValueError("Frozen-main must use one joint training stage")
    return arm


def validate_budget_lock(
    config_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Reject an unlocked or modified preregistered long-run config."""

    training = config.get("training", {})
    if not training.get("require_budget_lock", False):
        return None
    if not training.get("budget_locked", False):
        raise RuntimeError(
            "Long training is locked. Run the 200-step profiler, preflight, "
            "and freeze_frozen_main_budget.py before train.py."
        )
    arm = str(training.get("budget_arm", ""))
    manifest_value = training.get("budget_manifest")
    if not arm or not manifest_value:
        raise RuntimeError("Locked training requires budget_arm and budget_manifest")
    manifest_path = Path(str(manifest_value)).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Budget manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise RuntimeError("Unsupported budget manifest schema")
    config_record = manifest.get("configs", {}).get(arm)
    if not isinstance(config_record, dict):
        raise RuntimeError(f"Budget manifest has no config record for arm {arm}")
    requested_path = Path(config_path).expanduser().resolve()
    if sha256_file(requested_path) != config_record.get("sha256"):
        raise RuntimeError(
            "Locked config SHA256 differs from the budget manifest"
        )
    if int(training.get("epochs", 0)) != int(manifest.get("epochs", -1)):
        raise RuntimeError("Locked config epochs differ from the budget manifest")
    if int(manifest.get("steps_per_epoch", -1)) != 1130:
        raise RuntimeError("Frozen protocol requires exactly 1130 optimizer steps/epoch")
    if manifest.get("budget_policy") != FROZEN_MAIN_BUDGET_POLICY:
        raise RuntimeError("Frozen protocol budget policy mismatch")
    requested_hours = float(manifest.get("requested_gpu_hours_per_arm", -1))
    effective_hours = float(manifest.get("effective_gpu_hours_per_arm", -1))
    hard_cap_hours = float(manifest.get("hard_gpu_hour_cap_per_arm", -1))
    minimum_epochs = int(manifest.get("minimum_epochs", -1))
    maximum_epochs = int(manifest.get("maximum_epochs", -1))
    epochs = int(manifest.get("epochs", -1))
    if (
        requested_hours != 10.0
        or hard_cap_hours != 16.0
        or not requested_hours <= effective_hours <= hard_cap_hours
        or minimum_epochs != 12
        or maximum_epochs != 30
        or epochs > maximum_epochs
    ):
        raise RuntimeError("Frozen protocol budget bounds changed")
    profiles = manifest.get("profiles", {})
    seconds_per_step = {
        candidate: float(
            profiles.get(candidate, {}).get("seconds_per_optimizer_step", 0)
        )
        for candidate in ("D", "B")
    }
    if any(
        not math.isfinite(seconds) or seconds <= 0
        for seconds in seconds_per_step.values()
    ):
        raise RuntimeError("Frozen protocol profile timing is invalid")

    def supported(hours: float, seconds: float) -> int:
        return math.floor(hours * 3600 / (seconds * 1130) + 1e-12)

    base_supported = {
        candidate: supported(requested_hours, seconds)
        for candidate, seconds in seconds_per_step.items()
    }
    required_hours_for_minimum = {
        candidate: minimum_epochs * seconds * 1130 / 3600
        for candidate, seconds in seconds_per_step.items()
    }
    expected_effective_hours = requested_hours
    if min(base_supported.values()) < minimum_epochs:
        expected_effective_hours = min(
            hard_cap_hours,
            max(
                requested_hours,
                max(required_hours_for_minimum.values()),
            ),
        )
    if not math.isclose(
        effective_hours,
        expected_effective_hours,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Frozen protocol effective GPU-hour budget mismatch")
    supported_epochs = {
        candidate: supported(effective_hours, seconds)
        for candidate, seconds in seconds_per_step.items()
    }
    uncapped_common_epochs = min(supported_epochs.values())
    if (
        base_supported != manifest.get("base_supported_epochs_by_arm")
        or supported_epochs != manifest.get("supported_epochs_by_arm")
        or uncapped_common_epochs != manifest.get("uncapped_common_epochs")
        or epochs != min(uncapped_common_epochs, maximum_epochs)
    ):
        raise RuntimeError("Frozen protocol epoch calculation mismatch")
    floor_reached = bool(manifest.get("budget_floor_reached"))
    if floor_reached != (epochs < minimum_epochs):
        raise RuntimeError("Frozen protocol budget_floor_reached is inconsistent")
    if floor_reached and effective_hours != hard_cap_hours:
        raise RuntimeError("Epoch floor may be missed only at the 16-hour hard cap")
    if bool(manifest.get("budget_escalated")) != (
        effective_hours > requested_hours
    ):
        raise RuntimeError("Frozen protocol budget_escalated is inconsistent")
    if bool(manifest.get("epoch_cap_applied")) != (
        uncapped_common_epochs > maximum_epochs
    ):
        raise RuntimeError("Frozen protocol epoch_cap_applied is inconsistent")
    expected_ablation_policy = {
        "A0": epochs,
        "A1": epochs,
        "budget_locked": True,
        "launch_now": False,
        "rule": "reuse_frozen_main_epochs",
    }
    if manifest.get("ablation_epoch_policy") != expected_ablation_policy:
        raise RuntimeError("A0/A1 epoch reuse policy mismatch")
    tripwire = manifest.get("tripwire", {})
    for key, expected in FROZEN_MAIN_TRIPWIRE_POLICY.items():
        if tripwire.get(key) != expected:
            raise RuntimeError(f"Frozen tripwire policy mismatch: {key}")
    baseline_psnr = float(tripwire.get("baseline_y_psnr", float("nan")))
    threshold_psnr = float(tripwire.get("threshold_y_psnr", float("nan")))
    if (
        not math.isfinite(baseline_psnr)
        or not math.isclose(
            threshold_psnr,
            baseline_psnr - 0.05,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or tripwire.get("baseline_protocol_verified") is not True
        or tripwire.get("images") != 903
        or tripwire.get("foundation_signature") != foundation_signature(config)
    ):
        raise RuntimeError("Frozen tripwire baseline is invalid")
    baseline_path = Path(str(tripwire.get("path", ""))).expanduser()
    if (
        not baseline_path.is_file()
        or sha256_file(baseline_path) != tripwire.get("sha256")
    ):
        raise RuntimeError("Frozen tripwire baseline artifact mismatch")
    dev_manifest_path = Path(
        str(config.get("data", {}).get("val", {}).get("pairs_file", ""))
    ).expanduser()
    if (
        not dev_manifest_path.is_file()
        or sha256_file(dev_manifest_path)
        != tripwire.get("dev_manifest_sha256")
    ):
        raise RuntimeError("Frozen tripwire dev manifest mismatch")
    expected_decision_path = manifest_path.parent / "tripwire_decision.json"
    if Path(str(tripwire.get("decision_file", ""))) != expected_decision_path:
        raise RuntimeError("Frozen tripwire decision path mismatch")
    config_tripwire = config.get("protocol", {}).get("tripwire", {})
    for key in (
        *FROZEN_MAIN_TRIPWIRE_POLICY,
        "baseline_y_psnr",
        "threshold_y_psnr",
        "baseline_sha256",
        "dev_manifest_sha256",
        "baseline_protocol_verified",
    ):
        expected = (
            tripwire.get("sha256")
            if key == "baseline_sha256"
            else tripwire.get(key)
        )
        if config_tripwire.get(key) != expected:
            raise RuntimeError(f"Locked config tripwire mismatch: {key}")
    budget = config.get("budget", {})
    profile = manifest.get("profiles", {}).get(arm, {})
    expected_budget = {
        "requested_gpu_hours_per_arm": requested_hours,
        "effective_gpu_hours_per_arm": effective_hours,
        "hard_gpu_hour_cap_per_arm": hard_cap_hours,
        "minimum_epochs": minimum_epochs,
        "maximum_epochs": maximum_epochs,
        "budget_escalated": manifest.get("budget_escalated"),
        "budget_floor_reached": floor_reached,
        "epoch_cap_applied": manifest.get("epoch_cap_applied"),
        "steps_per_epoch": manifest.get("steps_per_epoch"),
        "epochs": manifest.get("epochs"),
        "profile_seconds_per_optimizer_step": profile.get(
            "seconds_per_optimizer_step"
        ),
        "profile_sha256": profile.get("sha256"),
        "preflight_sha256": manifest.get("preflight", {}).get("sha256"),
        "device": profile.get("device"),
    }
    for key, expected in expected_budget.items():
        if budget.get(key) != expected:
            raise RuntimeError(
                f"Locked config budget field differs from manifest: {key}"
            )
    if config.get("protocol", {}).get("name") == "frozen_main_db":
        current_implementation = implementation_fingerprint()
        if (
            budget.get("implementation_sha256")
            != current_implementation["sha256"]
            or manifest.get("implementation", {}).get("sha256")
            != current_implementation["sha256"]
        ):
            raise RuntimeError("Frozen-main implementation SHA256 mismatch")
        current_foundation = foundation_artifact_fingerprint(config)
        if (
            budget.get("foundation_artifacts") != current_foundation
            or manifest.get("foundation_artifacts") != current_foundation
        ):
            raise RuntimeError("Frozen-main SSR artifact SHA256 mismatch")
    preflight = manifest.get("preflight", {})
    preflight_path = Path(str(preflight.get("path", ""))).expanduser()
    if not preflight_path.is_file():
        raise FileNotFoundError("Frozen preflight archive is missing")
    if sha256_file(preflight_path) != preflight.get("sha256"):
        raise RuntimeError("Frozen preflight archive SHA256 mismatch")
    report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError("Frozen preflight archive is not passing")
    return manifest
