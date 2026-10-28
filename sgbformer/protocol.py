"""Fingerprints for reproducible BasisBFN runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LOSS_WEIGHT_KEYS = (
    "structural_weight", "bfn_weight", "final_weight", "correlation_weight",
    "frequency_weight", "adapter_weight", "uncertainty_weight",
    "latent_category_weight",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def implementation_fingerprint() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    paths = list((project_root / "sgbformer").rglob("*.py"))
    paths.extend((project_root / "sgbformer" / "configs").glob("*.yaml"))
    paths.extend(
        project_root / relative
        for relative in (
            "train.py", "infer.py", "evaluate.py", "evaluate_foundation.py",
            "scripts/preflight_basisbfn.py", "scripts/profile_basisbfn.py",
            "scripts/analyze_basisbfn.py", "scripts/analysis_helpers.py",
            "scripts/analyze_dev_by_category.py", "scripts/make_basisbfn_config.py",
        )
    )
    file_hashes = {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in sorted(set(paths))
    }
    digest = hashlib.sha256()
    for relative, file_hash in file_hashes.items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "files": file_hashes}


def foundation_artifact_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    foundation = config.get("model", {}).get("foundation", {})
    repo = Path(str(foundation.get("repo_path", ""))).expanduser().resolve()
    paths = {
        "architecture": repo / str(foundation.get("architecture_file", "")),
        "checkpoint": Path(str(foundation.get("checkpoint", ""))).expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"BasisBFN SSR {label} artifact is missing: {path}")
    return {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in paths.items()
    }


def manifest_fingerprints(config: dict[str, Any]) -> dict[str, Any]:
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
        records[label] = {
            "path": str(path), "sha256": sha256_file(path), "pairs": len(lines),
        }
    return records


def loss_term_manifest(loss_config: dict[str, Any]) -> dict[str, float]:
    return {key: float(loss_config.get(key, 0.0)) for key in LOSS_WEIGHT_KEYS}
