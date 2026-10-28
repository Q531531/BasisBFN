"""Paired epoch tripwire state and resume guards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import sha256_file


def tripwire_status_path(config: dict[str, Any]) -> Path:
    epoch = int(config["protocol"]["tripwire"]["epoch"])
    output_dir = Path(config["training"]["output_dir"]).expanduser().resolve()
    return output_dir / f"tripwire_epoch_{epoch:04d}.json"


def write_tripwire_status(
    *,
    config_path: str | Path,
    config: dict[str, Any],
    validation: dict[str, float],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    tripwire = config["protocol"]["tripwire"]
    epoch = int(tripwire["epoch"])
    metric = str(tripwire["metric"])
    if metric != "dev_y_psnr":
        raise RuntimeError(f"Unsupported tripwire metric: {metric}")
    psnr = float(validation["psnr"])
    threshold = float(tripwire["threshold_y_psnr"])
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Tripwire checkpoint is missing: {checkpoint}")
    status = {
        "schema_version": 1,
        "protocol": "frozen_main_db",
        "arm": str(config["protocol"]["arm"]),
        "epoch": epoch,
        "metric": metric,
        "dev_y_psnr": psnr,
        "baseline_y_psnr": float(tripwire["baseline_y_psnr"]),
        "maximum_regression_db": float(tripwire["maximum_regression_db"]),
        "threshold_y_psnr": threshold,
        "passed": psnr >= threshold,
        "b_minus_d_used": False,
        "evaluation_weights": str(tripwire["evaluation_weights"]),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    path = tripwire_status_path(config)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return status


def require_tripwire_resume(
    *,
    config_path: str | Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    start_epoch: int,
) -> None:
    tripwire = config["protocol"]["tripwire"]
    if start_epoch < int(tripwire["epoch"]):
        return
    decision_path = Path(
        str(manifest["tripwire"]["decision_file"])
    ).expanduser()
    if not decision_path.is_file():
        raise RuntimeError(
            "Epoch-4 tripwire decision is missing; run "
            "scripts/decide_frozen_main_tripwire.py after both arms pause"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    arm = str(config["protocol"]["arm"])
    if (
        decision.get("schema_version") != 1
        or decision.get("protocol") != "frozen_main_db"
        or decision.get("passed") is not True
        or decision.get("b_minus_d_used") is not False
        or decision.get("epoch") != int(tripwire["epoch"])
        or decision.get("threshold_y_psnr")
        != float(tripwire["threshold_y_psnr"])
        or decision.get("configs", {}).get(arm)
        != sha256_file(config_path)
    ):
        raise RuntimeError("Epoch-4 paired tripwire did not authorize resume")
    for candidate in ("D", "B"):
        status_record = decision.get("statuses", {}).get(candidate, {})
        status_path = Path(str(status_record.get("path", ""))).expanduser()
        if (
            status_record.get("passed") is not True
            or not status_path.is_file()
            or sha256_file(status_path) != status_record.get("sha256")
        ):
            raise RuntimeError("Epoch-4 paired tripwire status is invalid")
