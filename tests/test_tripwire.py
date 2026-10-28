import json
from pathlib import Path

import pytest

from sgbformer.protocol import sha256_file
from sgbformer.tripwire import (
    require_tripwire_resume,
    tripwire_status_path,
    write_tripwire_status,
)


def _config(tmp_path: Path, arm: str) -> tuple[Path, dict]:
    config_path = tmp_path / f"frozen_{arm}.yaml"
    config_path.write_text(f"arm: {arm}\n", encoding="utf-8")
    config = {
        "protocol": {
            "arm": arm,
            "tripwire": {
                "epoch": 4,
                "metric": "dev_y_psnr",
                "maximum_regression_db": 0.05,
                "baseline_y_psnr": 34.769,
                "threshold_y_psnr": 34.719,
                "evaluation_weights": "ema",
            },
        },
        "training": {
            "output_dir": str(tmp_path / arm),
        },
    }
    return config_path, config


def test_tripwire_status_and_paired_resume_guard(tmp_path: Path) -> None:
    configs = {}
    config_paths = {}
    statuses = {}
    for arm, psnr in (("D", 34.72), ("B", 34.73)):
        config_path, config = _config(tmp_path, arm)
        checkpoint = tmp_path / arm / "latest.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(f"checkpoint-{arm}".encode("ascii"))
        status = write_tripwire_status(
            config_path=config_path,
            config=config,
            validation={"psnr": psnr},
            checkpoint_path=checkpoint,
        )
        assert status["passed"] is True
        configs[arm] = config
        config_paths[arm] = config_path
        status_path = tripwire_status_path(config)
        statuses[arm] = {
            "path": str(status_path),
            "sha256": sha256_file(status_path),
            "passed": True,
        }

    decision_path = tmp_path / "tripwire_decision.json"
    manifest = {
        "tripwire": {
            "decision_file": str(decision_path),
        }
    }
    with pytest.raises(RuntimeError, match="decision is missing"):
        require_tripwire_resume(
            config_path=config_paths["D"],
            config=configs["D"],
            manifest=manifest,
            start_epoch=4,
        )

    decision = {
        "schema_version": 1,
        "protocol": "frozen_main_db",
        "passed": True,
        "b_minus_d_used": False,
        "epoch": 4,
        "threshold_y_psnr": 34.719,
        "configs": {
            arm: sha256_file(path)
            for arm, path in config_paths.items()
        },
        "statuses": statuses,
    }
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    require_tripwire_resume(
        config_path=config_paths["D"],
        config=configs["D"],
        manifest=manifest,
        start_epoch=4,
    )
    decision["passed"] = False
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not authorize"):
        require_tripwire_resume(
            config_path=config_paths["B"],
            config=configs["B"],
            manifest=manifest,
            start_epoch=4,
        )


def test_tripwire_marks_threshold_failure_without_b_minus_d(
    tmp_path: Path,
) -> None:
    config_path, config = _config(tmp_path, "D")
    checkpoint = tmp_path / "D" / "latest.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    status = write_tripwire_status(
        config_path=config_path,
        config=config,
        validation={"psnr": 34.7189},
        checkpoint_path=checkpoint,
    )
    assert status["passed"] is False
    assert status["b_minus_d_used"] is False
