"""YAML configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    base_path = config.pop("_base_", None)
    if base_path is not None:
        base_config = load_config(path.parent / str(base_path))
        config = _deep_merge(base_config, config)
    return _expand(config)
