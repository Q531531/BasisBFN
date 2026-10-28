"""Run records for BasisBFN preflight, profiling and training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOSS_WEIGHT_KEYS = (
    "structural_weight", "bfn_weight", "final_weight", "correlation_weight",
    "frequency_weight", "adapter_weight", "uncertainty_weight",
    "latent_category_weight",
)


def resolved_config_record(config: dict[str, Any]) -> dict[str, Any]:
    """Return the resolved config in the JSON form stored by run reports."""
    return json.loads(json.dumps(config, sort_keys=True))


def foundation_artifact_paths(config: dict[str, Any]) -> dict[str, Any]:
    foundation = config.get("model", {}).get("foundation", {})
    repo = Path(str(foundation.get("repo_path", ""))).expanduser().resolve()
    paths = {
        "architecture": repo / str(foundation.get("architecture_file", "")),
        "checkpoint": Path(str(foundation.get("checkpoint", ""))).expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"BasisBFN SSR {label} artifact is missing: {path}")
    return {label: {"path": str(path)} for label, path in paths.items()}


def manifest_summary(config: dict[str, Any]) -> dict[str, Any]:
    records = {}
    for label, split in (("train", "train"), ("dev", "val")):
        value = config["data"][split].get("pairs_file")
        if not value:
            raise ValueError(f"BasisBFN requires an explicit {label} pairs_file")
        path = Path(value).expanduser().resolve()
        lines = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if any(len(line.split("\t")) not in (1, 2) for line in lines):
            raise ValueError(f"Invalid {label} manifest: {path}")
        names = [str(Path(line.split("\t")[0]).with_suffix("")) for line in lines]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate image names in {label} manifest: {path}")
        records[label] = {"path": str(path), "pairs": len(lines)}
    return records


def loss_term_manifest(loss_config: dict[str, Any]) -> dict[str, float]:
    return {key: float(loss_config.get(key, 0.0)) for key in LOSS_WEIGHT_KEYS}
