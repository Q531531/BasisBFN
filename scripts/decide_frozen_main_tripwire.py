#!/usr/bin/env python3
"""Authorize or stop both frozen-main arms after their epoch-4 pause."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sgbformer.config import load_config
from sgbformer.protocol import sha256_file, validate_budget_lock
from sgbformer.tripwire import tripwire_status_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-d", required=True)
    parser.add_argument("--config-b", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> None:
    args = parse_args()
    config_paths = {
        "D": Path(args.config_d).expanduser().resolve(),
        "B": Path(args.config_b).expanduser().resolve(),
    }
    configs = {
        arm: load_config(path)
        for arm, path in config_paths.items()
    }
    manifests = {
        arm: validate_budget_lock(config_paths[arm], configs[arm])
        for arm in ("D", "B")
    }
    if manifests["D"] != manifests["B"]:
        raise RuntimeError("D/B locked configs do not share one budget manifest")
    manifest = manifests["D"]
    if not isinstance(manifest, dict):
        raise RuntimeError("D/B configs are not budget locked")

    statuses = {}
    for arm in ("D", "B"):
        path = tripwire_status_path(configs[arm])
        if not path.is_file():
            raise FileNotFoundError(f"Arm {arm} tripwire status is missing: {path}")
        status = read_json(path)
        checkpoint = Path(str(status.get("checkpoint_path", ""))).expanduser()
        expected = configs[arm]["protocol"]["tripwire"]
        measured_psnr = float(status.get("dev_y_psnr", float("nan")))
        expected_pass = measured_psnr >= float(expected["threshold_y_psnr"])
        if (
            status.get("schema_version") != 1
            or status.get("protocol") != "frozen_main_db"
            or status.get("arm") != arm
            or status.get("epoch") != expected["epoch"]
            or status.get("metric") != "dev_y_psnr"
            or not math.isfinite(measured_psnr)
            or status.get("threshold_y_psnr") != expected["threshold_y_psnr"]
            or status.get("baseline_y_psnr") != expected["baseline_y_psnr"]
            or status.get("passed") is not expected_pass
            or status.get("b_minus_d_used") is not False
            or status.get("evaluation_weights") != "ema"
            or status.get("config_sha256")
            != manifest["configs"][arm]["sha256"]
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != status.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"Arm {arm} tripwire status is inconsistent")
        statuses[arm] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "dev_y_psnr": measured_psnr,
            "passed": bool(status["passed"]),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": status["checkpoint_sha256"],
        }

    tripwire = manifest["tripwire"]
    passed = all(status["passed"] for status in statuses.values())
    decision = {
        "schema_version": 1,
        "protocol": "frozen_main_db",
        "epoch": tripwire["epoch"],
        "metric": "dev_y_psnr",
        "baseline_y_psnr": tripwire["baseline_y_psnr"],
        "maximum_regression_db": tripwire["maximum_regression_db"],
        "threshold_y_psnr": tripwire["threshold_y_psnr"],
        "passed": passed,
        "b_minus_d_used": False,
        "evaluation_weights": "ema",
        "configs": {
            arm: manifest["configs"][arm]["sha256"]
            for arm in ("D", "B")
        },
        "statuses": statuses,
        "action": (
            "resume_both_arms"
            if passed
            else "stop_both_arms_and_debug_without_architecture_change"
        ),
    }
    output = Path(str(tripwire["decision_file"])).expanduser()
    if output.exists():
        raise FileExistsError(f"Tripwire decision already exists: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    output.chmod(0o444)
    print(json.dumps(decision, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
