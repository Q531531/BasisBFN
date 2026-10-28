import json
from pathlib import Path

import pytest

from sgbformer.config import load_config
from sgbformer.comparison import foundation_signature
from sgbformer.protocol import (
    FROZEN_MAIN_BUDGET_POLICY,
    FROZEN_MAIN_TRIPWIRE_POLICY,
    foundation_artifact_fingerprint,
    implementation_fingerprint,
    sha256_file,
    validate_budget_lock,
    validate_frozen_main_contract,
)
from scripts.freeze_frozen_main_budget import (
    STEPS_PER_EPOCH,
    compute_budget_decision,
    main as freeze_main,
)
from scripts.preflight_frozen_main import normalize_arm_config


def test_frozen_main_contract_and_unlocked_guard() -> None:
    root = Path(__file__).resolve().parents[1]
    for arm in ("D", "B"):
        path = root / "configs" / "experiments" / f"ssr_frozen_main_{arm.lower()}.yaml"
        config = load_config(path)
        assert validate_frozen_main_contract(config, expected_arm=arm) == arm
        with pytest.raises(RuntimeError, match="Long training is locked"):
            validate_budget_lock(path, config)
    assert STEPS_PER_EPOCH == 1130


def test_preflight_normalization_keeps_common_supervision_strict() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    d = load_config(root / "ssr_frozen_main_d.yaml")
    b = load_config(root / "ssr_frozen_main_b.yaml")
    assert normalize_arm_config(d) == normalize_arm_config(b)
    d["loss"]["bfn_weight"] = 0.0
    assert normalize_arm_config(d) != normalize_arm_config(b)


def test_budget_policy_clamps_escalates_and_records_floor_failure() -> None:
    escalated = compute_budget_decision(
        {"D": 2.0, "B": 3.0},
        FROZEN_MAIN_BUDGET_POLICY,
    )
    assert escalated["base_supported_epochs_by_arm"] == {"D": 15, "B": 10}
    assert escalated["effective_gpu_hours_per_arm"] == pytest.approx(11.3)
    assert escalated["supported_epochs_by_arm"] == {"D": 18, "B": 12}
    assert escalated["epochs"] == 12
    assert escalated["budget_escalated"] is True
    assert escalated["budget_floor_reached"] is False

    capped = compute_budget_decision(
        {"D": 0.5, "B": 0.5},
        FROZEN_MAIN_BUDGET_POLICY,
    )
    assert capped["epochs"] == 30
    assert capped["epoch_cap_applied"] is True
    assert capped["effective_gpu_hours_per_arm"] == 10

    floor_missed = compute_budget_decision(
        {"D": 5.0, "B": 5.0},
        FROZEN_MAIN_BUDGET_POLICY,
    )
    assert floor_missed["effective_gpu_hours_per_arm"] == 16
    assert floor_missed["supported_epochs_by_arm"] == {"D": 10, "B": 10}
    assert floor_missed["epochs"] == 10
    assert floor_missed["budget_floor_reached"] is True


def test_budget_lock_checks_config_and_preflight_hashes(tmp_path: Path) -> None:
    config_path = tmp_path / "frozen_D.yaml"
    config_path.write_text("frozen: true\n", encoding="utf-8")
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("{}", encoding="utf-8")
    dev_manifest_path = tmp_path / "val_pairs.tsv"
    dev_manifest_path.write_text("test.png\ttest.png\n", encoding="utf-8")
    manifest_path = tmp_path / "budget_manifest.json"
    manifest = {
        "schema_version": 2,
        "budget_policy": FROZEN_MAIN_BUDGET_POLICY,
        "requested_gpu_hours_per_arm": 10.0,
        "effective_gpu_hours_per_arm": 10.0,
        "hard_gpu_hour_cap_per_arm": 16.0,
        "minimum_epochs": 12,
        "maximum_epochs": 30,
        "budget_escalated": False,
        "budget_floor_reached": False,
        "epoch_cap_applied": False,
        "epochs": 12,
        "steps_per_epoch": 1130,
        "configs": {
            "D": {
                "sha256": sha256_file(config_path),
            }
        },
        "profiles": {
            "D": {
                "seconds_per_optimizer_step": 2.6,
                "sha256": "profile-sha",
                "device": "Test GPU",
            },
            "B": {
                "seconds_per_optimizer_step": 2.6,
                "sha256": "profile-sha-b",
                "device": "Test GPU",
            },
        },
        "base_supported_epochs_by_arm": {"D": 12, "B": 12},
        "supported_epochs_by_arm": {"D": 12, "B": 12},
        "uncapped_common_epochs": 12,
        "ablation_epoch_policy": {
            "A0": 12,
            "A1": 12,
            "budget_locked": True,
            "launch_now": False,
            "rule": "reuse_frozen_main_epochs",
        },
        "tripwire": {
            **FROZEN_MAIN_TRIPWIRE_POLICY,
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "images": 903,
            "baseline_y_psnr": 34.769,
            "threshold_y_psnr": 34.719,
            "baseline_protocol_verified": True,
            "foundation_signature": None,
            "dev_manifest_sha256": sha256_file(dev_manifest_path),
            "decision_file": str(tmp_path / "tripwire_decision.json"),
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "training": {
            "require_budget_lock": True,
            "budget_locked": True,
            "budget_arm": "D",
            "budget_manifest": str(manifest_path),
            "epochs": 12,
        },
        "protocol": {
            "tripwire": {
                **FROZEN_MAIN_TRIPWIRE_POLICY,
                "baseline_y_psnr": 34.769,
                "threshold_y_psnr": 34.719,
                "baseline_sha256": sha256_file(baseline_path),
                "dev_manifest_sha256": sha256_file(dev_manifest_path),
                "baseline_protocol_verified": True,
            }
        },
        "data": {
            "val": {
                "pairs_file": str(dev_manifest_path),
            }
        },
        "budget": {
            "requested_gpu_hours_per_arm": 10.0,
            "effective_gpu_hours_per_arm": 10.0,
            "hard_gpu_hour_cap_per_arm": 16.0,
            "minimum_epochs": 12,
            "maximum_epochs": 30,
            "budget_escalated": False,
            "budget_floor_reached": False,
            "epoch_cap_applied": False,
            "steps_per_epoch": 1130,
            "epochs": 12,
            "profile_seconds_per_optimizer_step": 2.6,
            "profile_sha256": "profile-sha",
            "preflight_sha256": sha256_file(preflight_path),
            "device": "Test GPU",
        },
    }
    assert validate_budget_lock(config_path, config) == manifest

    config_path.write_text("frozen: modified\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA256"):
        validate_budget_lock(config_path, config)


def test_budget_freezer_uses_common_supported_epoch_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    configs = {
        arm: (
            root
            / "configs"
            / "experiments"
            / f"ssr_frozen_main_{arm.lower()}.yaml"
        )
        for arm in ("D", "B")
    }
    ssr_repo = tmp_path / "ssr"
    architecture = ssr_repo / "basicsr" / "models" / "archs" / "SSR_arch.py"
    architecture.parent.mkdir(parents=True)
    architecture.write_text("# test SSR architecture\n", encoding="utf-8")
    checkpoint = tmp_path / "ssr.pt"
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setenv("SSR_REPO", str(ssr_repo))
    monkeypatch.setenv("SSR_CHECKPOINT", str(checkpoint))
    data_root = tmp_path / "data"
    data_root.mkdir()
    dev_names = [f"dev/{index:04d}" for index in range(903)]
    (data_root / "val_pairs.tsv").write_text(
        "".join(f"{name}.png\t{name}.png\n" for name in dev_names),
        encoding="utf-8",
    )
    monkeypatch.setenv("SGB_DATA_ROOT", str(data_root))
    resolved_d = load_config(configs["D"])
    implementation = implementation_fingerprint()
    foundation_artifacts = foundation_artifact_fingerprint(resolved_d)
    profile_paths = {}
    for arm, seconds in (("D", 2.0), ("B", 3.0)):
        path = tmp_path / f"profile_{arm}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol": "frozen_main_db",
                    "arm": arm,
                    "source_config_sha256": sha256_file(configs[arm]),
                    "optimizer_steps": 200,
                    "global_batch_size": 16,
                    "train_pairs": 18069,
                    "quality_metrics_observed": False,
                    "seconds_per_optimizer_step": seconds,
                    "device": "Test GPU",
                    "implementation": implementation,
                    "foundation_artifacts": foundation_artifacts,
                }
            ),
            encoding="utf-8",
        )
        profile_paths[arm] = path
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "frozen_main_db",
                "status": "passed",
                "device": "Test GPU",
                "implementation": implementation,
                "foundation_artifacts": foundation_artifacts,
                "source_config_sha256": {
                    arm: sha256_file(path)
                    for arm, path in configs.items()
                },
            }
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "average": {"psnr": 34.769, "ssim": 0.94},
                "images": [
                    {"name": name, "psnr": 34.769, "ssim": 0.94}
                    for name in dev_names
                ],
                "protocol": {
                    "test_y_channel": True,
                    "crop_border": 0,
                    "lpips_network": None,
                    "stochastic": False,
                    "tile": None,
                    "overlap": None,
                    "output_quantization": "float32_clamp_0_1",
                    "refinement_mode": "foundation_only",
                    "steps": 0,
                    "latent_bfn_steps": 0,
                    "decoder_calls_per_trajectory": 1,
                    "foundation_signature": foundation_signature(resolved_d),
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "locked"
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze_frozen_main_budget.py",
            "--config-d",
            str(configs["D"]),
            "--config-b",
            str(configs["B"]),
            "--profile-d",
            str(profile_paths["D"]),
            "--profile-b",
            str(profile_paths["B"]),
            "--preflight",
            str(preflight_path),
            "--baseline",
            str(baseline_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    freeze_main()

    manifest = json.loads(
        (output_dir / "budget_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["base_supported_epochs_by_arm"] == {"D": 15, "B": 10}
    assert manifest["supported_epochs_by_arm"] == {"D": 18, "B": 12}
    assert manifest["effective_gpu_hours_per_arm"] == pytest.approx(11.3)
    assert manifest["epochs"] == 12
    assert manifest["budget_floor_reached"] is False
    assert manifest["tripwire"]["baseline_y_psnr"] == pytest.approx(34.769)
    assert manifest["tripwire"]["threshold_y_psnr"] == pytest.approx(34.719)
    assert manifest["tripwire"]["dev_manifest_sha256"] == sha256_file(
        data_root / "val_pairs.tsv"
    )
    assert manifest["ablation_epoch_policy"] == {
        "A0": 12,
        "A1": 12,
        "budget_locked": True,
        "launch_now": False,
        "rule": "reuse_frozen_main_epochs",
    }
    for arm in ("D", "B"):
        path = output_dir / f"frozen_{arm}.yaml"
        config = load_config(path)
        assert config["training"]["epochs"] == 12
        assert config["training"]["stages"][0]["epochs"] == 12
        assert config["training"]["budget_locked"] is True
        assert validate_budget_lock(path, config) == manifest
