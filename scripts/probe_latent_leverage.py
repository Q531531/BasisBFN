#!/usr/bin/env python3
"""Measure what a trained latent BFN changes without training another model.

The primary decomposition uses one fixed checkpoint and image set:

* F0: the original, unmodified SSR checkpoint;
* Fback: EMA-tuned SSR weights with all three SGBFormer adapters bypassed;
* F1: the EMA SGBFormer checkpoint with both decoder latent fusion gains zero;
* F2: the normal EMA SGBFormer checkpoint.

It writes per-image PSNR/SSIM evaluations plus paired natural and stratified
macro bootstrap comparisons for each causal contribution.  Adapter state is
changed only in memory and is restored even when an inference call fails.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_dev_by_category import (  # noqa: E402
    DEFAULT_CATEGORY_PATTERNS,
    DEFAULT_CATEGORY_WEIGHTS,
    analyze_dev_by_category,
    load_evaluation,
    read_name_map,
)
from sgbformer.comparison import foundation_signature  # noqa: E402
from sgbformer.data import PairedImageDataset  # noqa: E402
from sgbformer.inference import load_model_for_inference  # noqa: E402
from sgbformer.metrics import psnr, ssim  # noqa: E402
from sgbformer.models import (  # noqa: E402
    ExternalSSRFoundation,
    LatentConditionedFiLMSpectralAdapter,
    PaperSpectralGateAdapter,
    degradation_latent_targets,
)
from sgbformer.models.foundation import build_foundation  # noqa: E402
from sgbformer.protocol import sha256_file  # noqa: E402
from sgbformer.utils import resolve_device, seed_everything  # noqa: E402


DECODER_ADAPTERS = ("decoder_level3", "decoder_level2")
ENCODER_GATE_ADAPTER = "encoder_level3"
DEFAULT_FUSION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


class ProbeError(ValueError):
    """Raised when a requested probe cannot preserve the stated protocol."""


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", config)
    if not isinstance(model, dict):
        raise ProbeError("Checkpoint config has no model mapping")
    return model


def _parse_named_values(
    values: list[str],
    *,
    option: str,
    value_type: type[float] | type[str],
) -> dict[str, float] | dict[str, str]:
    parsed: dict[str, float] | dict[str, str] = {}
    for value in values:
        name, separator, payload = value.partition("=")
        if not separator or not name or not payload:
            raise ProbeError(f"{option} must have NAME=VALUE form, got {value!r}")
        if name in parsed:
            raise ProbeError(f"Duplicate {option} entry: {name}")
        parsed[name] = value_type(payload)
    return parsed


def _category_protocol(
    config: dict[str, Any],
    pattern_values: list[str],
    weight_values: list[str],
) -> tuple[dict[str, str], dict[str, float]]:
    model = _model_config(config)
    latent = model.get("latent_bfn", {})
    training = config.get("training", {})
    sampling = training.get("sampling", {}) if isinstance(training, dict) else {}
    configured_patterns = (
        latent.get("category_patterns", {})
        if isinstance(latent, dict)
        else {}
    )
    configured_weights = (
        sampling.get("target_fractions", {})
        if isinstance(sampling, dict)
        else {}
    )
    patterns = (
        _parse_named_values(
            pattern_values,
            option="--category-pattern",
            value_type=str,
        )
        if pattern_values
        else dict(configured_patterns or DEFAULT_CATEGORY_PATTERNS)
    )
    weights = (
        _parse_named_values(
            weight_values,
            option="--category-weight",
            value_type=float,
        )
        if weight_values
        else dict(configured_weights or DEFAULT_CATEGORY_WEIGHTS)
    )
    if not patterns:
        raise ProbeError("At least one category pattern is required")
    if set(patterns) != set(weights):
        raise ProbeError(
            "Category patterns and macro weights must name the same categories; "
            f"patterns={sorted(patterns)}, weights={sorted(weights)}"
        )
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ProbeError("Macro weights must be finite and non-negative")
    if sum(weights.values()) <= 0:
        raise ProbeError("Macro weights must sum to a positive value")
    return dict(patterns), dict(weights)


def _compile_category_patterns(
    patterns: dict[str, str],
) -> dict[str, re.Pattern[str]]:
    return {
        name: re.compile(pattern, flags=re.IGNORECASE)
        for name, pattern in patterns.items()
    }


def _category_for_path(
    path: str,
    patterns: dict[str, re.Pattern[str]],
) -> str:
    filename = Path(path).name
    matches = [
        category
        for category, pattern in patterns.items()
        if pattern.fullmatch(filename)
    ]
    if len(matches) != 1:
        raise ProbeError(
            f"{filename!r} must match exactly one category; matched {matches}"
        )
    return matches[0]


def _pad_to_multiple(
    image: Tensor,
    multiple: int,
) -> tuple[Tensor, tuple[int, int]]:
    if multiple < 1:
        raise ProbeError("Foundation required_multiple must be positive")
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if not pad_height and not pad_width:
        return image, (height, width)
    mode = "reflect" if height > pad_height and width > pad_width else "replicate"
    return F.pad(image, (0, pad_width, 0, pad_height), mode=mode), (height, width)


def build_raw_foundation(
    config: dict[str, Any],
    device: torch.device,
) -> ExternalSSRFoundation:
    """Build F0 from the original SSR artifact, without trained adapters."""

    foundation_config = deepcopy(_model_config(config).get("foundation"))
    if not isinstance(foundation_config, dict):
        raise ProbeError("F0 requires model.foundation in the checkpoint config")
    foundation_config.update(
        feature_adapters={},
        freeze=True,
        trainable_modules=[],
        frozen_modules_eval=False,
    )
    foundation = build_foundation(foundation_config).to(device).eval()
    if not isinstance(foundation, ExternalSSRFoundation):
        raise ProbeError("F0 must be an ExternalSSRFoundation")
    if foundation.adapters:
        raise RuntimeError("F0 unexpectedly contains feature adapters")
    if any(parameter.requires_grad for parameter in foundation.model.parameters()):
        raise RuntimeError("F0 original SSR weights must be frozen")
    return foundation


def restore_raw_foundation(
    foundation: ExternalSSRFoundation,
    degraded: Tensor,
) -> Tensor:
    """Mirror SGBFormer padding/clipping for a raw SSR forward."""

    padded, original_size = _pad_to_multiple(
        degraded,
        foundation.required_multiple,
    )
    restored, _ = foundation(padded)
    return restored.clamp(0, 1)[..., : original_size[0], : original_size[1]]


def decoder_latent_adapters(
    model: Any,
) -> dict[str, LatentConditionedFiLMSpectralAdapter]:
    """Return exactly the two decoder adapters that define this probe."""

    backbone = getattr(model, "backbone", None)
    adapters = getattr(backbone, "adapters", None)
    if adapters is None:
        raise ProbeError("Checkpoint has no external SSR adapter collection")
    found: dict[str, LatentConditionedFiLMSpectralAdapter] = {}
    all_latent: set[str] = set()
    for name, adapter in adapters.items():
        module_name = name.replace("__", ".")
        if isinstance(adapter, LatentConditionedFiLMSpectralAdapter):
            all_latent.add(module_name)
    if all_latent != set(DECODER_ADAPTERS):
        raise ProbeError(
            "This probe requires exactly decoder_level3 and decoder_level2 "
            f"latent adapters, found {sorted(all_latent)}"
        )
    for module_name in DECODER_ADAPTERS:
        adapter = adapters[module_name]
        if not isinstance(adapter, LatentConditionedFiLMSpectralAdapter):
            raise ProbeError(f"{module_name} is not a latent FiLM adapter")
        found[module_name] = adapter
    return found


def current_fusion_gains(model: Any) -> dict[str, float]:
    return {
        name: float(adapter.fusion_gain.detach().cpu())
        for name, adapter in decoder_latent_adapters(model).items()
    }


def encoder_gate_adapter(model: Any) -> PaperSpectralGateAdapter:
    """Return the sole spectral gate that must be bypassed for Fback."""

    backbone = getattr(model, "backbone", None)
    adapters = getattr(backbone, "adapters", None)
    if adapters is None or ENCODER_GATE_ADAPTER not in adapters:
        raise ProbeError(
            f"Checkpoint has no {ENCODER_GATE_ADAPTER} encoder spectral gate"
        )
    adapter = adapters[ENCODER_GATE_ADAPTER]
    if not isinstance(adapter, PaperSpectralGateAdapter):
        raise ProbeError(
            f"{ENCODER_GATE_ADAPTER} is not a PaperSpectralGateAdapter"
        )
    paper_gates = {
        name.replace("__", ".")
        for name, candidate in adapters.items()
        if isinstance(candidate, PaperSpectralGateAdapter)
    }
    if paper_gates != {ENCODER_GATE_ADAPTER}:
        raise ProbeError(
            "This probe requires exactly one paper spectral gate at "
            f"{ENCODER_GATE_ADAPTER}, found {sorted(paper_gates)}"
        )
    return adapter


@contextmanager
def temporary_fusion_gains(
    model: Any,
    gains: dict[str, float],
) -> Iterator[None]:
    """Set decoder gains only for one probe and restore them unconditionally."""

    if set(gains) != set(DECODER_ADAPTERS):
        raise ProbeError(
            "Fusion gains must specify decoder_level3 and decoder_level2 exactly"
        )
    if any(not math.isfinite(value) for value in gains.values()):
        raise ProbeError("Fusion gains must be finite")
    adapters = decoder_latent_adapters(model)
    originals = {
        name: adapter.fusion_gain.detach().clone()
        for name, adapter in adapters.items()
    }
    try:
        with torch.no_grad():
            for name, value in gains.items():
                adapters[name].fusion_gain.copy_(
                    torch.as_tensor(
                        value,
                        device=adapters[name].fusion_gain.device,
                        dtype=adapters[name].fusion_gain.dtype,
                    )
                )
        yield
    finally:
        with torch.no_grad():
            for name, original in originals.items():
                adapters[name].fusion_gain.copy_(original)


@contextmanager
def temporary_encoder_gate_bypass(
    model: Any,
    *,
    enabled: bool,
) -> Iterator[None]:
    """Temporarily make the encoder spectral gate an exact identity mapping."""

    if not enabled:
        yield
        return
    adapter = encoder_gate_adapter(model)
    had_instance_forward = "forward" in adapter.__dict__
    original_instance_forward = adapter.__dict__.get("forward")
    original_regularization = adapter.last_regularization

    def identity(feature: Tensor) -> Tensor:
        adapter.last_regularization = feature.new_zeros(())
        return feature

    adapter.forward = identity  # type: ignore[method-assign]
    try:
        yield
    finally:
        if had_instance_forward:
            adapter.forward = original_instance_forward  # type: ignore[method-assign]
        else:
            del adapter.forward
        adapter.last_regularization = original_regularization


def _diagnostics_as_floats(
    details: dict[str, Any],
) -> dict[str, dict[str, float]]:
    diagnostics = details.get("modulation_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    output: dict[str, dict[str, float]] = {}
    for module_name, values in diagnostics.items():
        if not isinstance(values, dict):
            raise TypeError(f"Unexpected modulation diagnostics for {module_name}")
        output[str(module_name)] = {
            str(name): float(value.detach().float().mean().cpu())
            for name, value in values.items()
            if isinstance(value, Tensor)
        }
    return output


def restore_model_variant(
    model: Any,
    degraded: Tensor,
    *,
    gains: dict[str, float] | None,
    bypass_encoder_gate: bool = False,
) -> tuple[Tensor, dict[str, dict[str, float]]]:
    gain_context = (
        temporary_fusion_gains(model, gains) if gains is not None else nullcontext()
    )
    with temporary_encoder_gate_bypass(
        model,
        enabled=bypass_encoder_gate,
    ), gain_context:
        details = model.restore(degraded, deterministic=True, return_details=True)
    if not isinstance(details, dict):
        raise TypeError("Model restore(return_details=True) did not return a mapping")
    restored = details.get("restored")
    if not isinstance(restored, Tensor):
        raise TypeError("Model restore details contain no tensor 'restored'")
    return restored, _diagnostics_as_floats(details)


def oracle_latent_state(
    model: Any,
    degraded: Tensor,
    target: Tensor,
    *,
    category: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build GT latent means aligned to the deployable predictor's state shape."""

    if getattr(model, "oracle_z_enabled", False) or getattr(
        model, "latent_flow", None
    ) is None:
        raise ProbeError(
            "Oracle probe requires a deployable checkpoint with latent_flow"
        )
    _, _, predicted, padded_degraded, degraded_size = model.predict_latent_state(
        degraded
    )
    padded_target, target_size = _pad_to_multiple(
        target,
        int(model.backbone.required_multiple),
    )
    if degraded_size != target_size:
        raise ProbeError("Oracle probe requires matching degraded/target geometry")
    predicted_global_precision = predicted.get("global_precision")
    predicted_spatial_mean = predicted.get("spatial_mean")
    predicted_spatial_precision = predicted.get("spatial_precision")
    if not all(
        isinstance(value, Tensor)
        for value in (
            predicted_global_precision,
            predicted_spatial_mean,
            predicted_spatial_precision,
        )
    ):
        raise TypeError("Latent-BFN prediction returned an incomplete precision state")
    global_target, spatial_target = degradation_latent_targets(
        padded_degraded,
        padded_target,
        global_dim=model.latent_global_dim,
        spatial_dim=model.latent_spatial_dim,
        spatial_size=predicted_spatial_mean.shape[-2:],
        category=category,
        num_categories=model.latent_num_categories,
        residual_scale=model.latent_target_residual_scale,
    )
    return (
        global_target,
        spatial_target,
        predicted_global_precision,
        predicted_spatial_precision,
    )


def oracle_restore(
    model: Any,
    degraded: Tensor,
    target: Tensor,
    *,
    category: Tensor,
    precision_source: str,
    precision_scale: float = 1.0,
) -> tuple[Tensor, dict[str, dict[str, float]]]:
    """Decode GT-derived means under a declared precision condition."""

    if precision_source not in {"predicted", "unit"}:
        raise ProbeError("Oracle precision source must be predicted or unit")
    if not math.isfinite(precision_scale) or precision_scale <= 0:
        raise ProbeError("Oracle precision scale must be finite and positive")
    (
        global_target,
        spatial_target,
        predicted_global_precision,
        predicted_spatial_precision,
    ) = oracle_latent_state(
        model,
        degraded,
        target,
        category=category,
    )
    if precision_source == "unit":
        global_precision = torch.ones_like(global_target)
        spatial_precision = torch.ones_like(spatial_target)
    else:
        global_precision = (
            predicted_global_precision * precision_scale
        ).clamp_min(1e-6)
        spatial_precision = (
            predicted_spatial_precision * precision_scale
        ).clamp_min(1e-6)
    details = model.restore_from_latent_state(
        degraded,
        global_mean=global_target,
        global_precision=global_precision,
        spatial_mean=spatial_target,
        spatial_precision=spatial_precision,
        return_details=True,
    )
    if not isinstance(details, dict):
        raise TypeError(
            "restore_from_latent_state(return_details=True) did not return a mapping"
        )
    restored = details.get("restored")
    if not isinstance(restored, Tensor):
        raise TypeError("Oracle details contain no tensor 'restored'")
    return restored, _diagnostics_as_floats(details)


def _evaluation_payload(
    *,
    rows: list[dict[str, Any]],
    variant: str,
    config: dict[str, Any],
    crop_border: int,
    test_y_channel: bool,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    gains: dict[str, float] | None,
    oracle: bool = False,
    oracle_precision: str | None = None,
    encoder_gate_bypassed: bool = False,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Probe evaluation produced no images")
    average = {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in ("psnr", "ssim")
    }
    categories = {
        category: {
            metric: sum(
                float(row[metric])
                for row in rows
                if row["category"] == category
            )
            / sum(row["category"] == category for row in rows)
            for metric in ("psnr", "ssim")
        }
        for category in sorted({str(row["category"]) for row in rows})
    }
    return {
        "average": average,
        "categories": categories,
        "images": rows,
        "protocol": {
            "test_y_channel": test_y_channel,
            "crop_border": crop_border,
            "lpips_network": None,
            "tile": None,
            "overlap": None,
            "stochastic": False,
            "output_quantization": "float32_clamp_0_1",
            "foundation_signature": foundation_signature(config),
            "checkpoint_state": "ema_loaded",
            "variant": variant,
            "oracle": oracle,
            "oracle_precision": oracle_precision,
            "encoder_gate_bypassed": encoder_gate_bypassed,
        },
        "probe": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "fusion_gains": gains,
            "variant": variant,
            "oracle_precision": oracle_precision,
            "encoder_gate_bypassed": encoder_gate_bypassed,
        },
    }


@torch.inference_mode()
def evaluate_variant(
    *,
    dataset: PairedImageDataset,
    category_patterns: dict[str, re.Pattern[str]],
    category_indices: dict[str, int] | None,
    variant: str,
    config: dict[str, Any],
    crop_border: int,
    test_y_channel: bool,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    device: torch.device,
    restore: Any,
    gains: dict[str, float] | None,
    oracle: bool = False,
    oracle_precision: str | None = None,
    encoder_gate_bypassed: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Evaluate one fixed restoration path image by image."""

    rows: list[dict[str, Any]] = []
    metric_options = {
        "crop_border": crop_border,
        "test_y_channel": test_y_channel,
    }
    for index in range(len(dataset)):
        sample = dataset[index]
        degraded = sample["degraded"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)
        category = _category_for_path(sample["input_path"], category_patterns)
        category_tensor = None
        if oracle:
            if category_indices is None or category not in category_indices:
                raise ProbeError(
                    f"Oracle category {category!r} is absent from latent config"
                )
            category_tensor = torch.tensor(
                [category_indices[category]],
                device=device,
                dtype=torch.long,
            )
        prediction, diagnostics = restore(
            degraded,
            target,
            category_tensor,
        )
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"{sample['name']}: prediction shape {tuple(prediction.shape)} "
                f"does not match target {tuple(target.shape)}"
            )
        row: dict[str, Any] = {
            "name": sample["name"],
            "category": category,
            "psnr": float(psnr(prediction, target, **metric_options)[0]),
            "ssim": float(ssim(prediction, target, **metric_options)[0]),
        }
        if diagnostics:
            row["modulation_diagnostics"] = diagnostics
        rows.append(row)
        if not quiet:
            print(
                f"{variant} {index + 1}/{len(dataset)} {row['name']}: "
                f"PSNR={row['psnr']:.4f} SSIM={row['ssim']:.5f}"
            )
    return _evaluation_payload(
        rows=rows,
        variant=variant,
        config=config,
        crop_border=crop_border,
        test_y_channel=test_y_channel,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        gains=gains,
        oracle=oracle,
        oracle_precision=oracle_precision,
        encoder_gate_bypassed=encoder_gate_bypassed,
    )


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; use a fresh --output-dir or --overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _slug_alpha(alpha: float) -> str:
    return f"{alpha:.6g}".replace("-", "neg").replace(".", "p")


def _comparison(
    baseline_path: Path,
    candidate_path: Path,
    *,
    category_patterns: dict[str, str],
    category_weights: dict[str, float],
    pairs_file: Path | None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_payload, baseline_schema = load_evaluation(baseline_path)
    candidate_payload, candidate_schema = load_evaluation(candidate_path)
    return analyze_dev_by_category(
        baseline_payload,
        candidate_payload,
        baseline_schema=baseline_schema,
        candidate_schema=candidate_schema,
        category_patterns=category_patterns,
        category_weights=category_weights,
        name_map=read_name_map(pairs_file) if pairs_file is not None else None,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        reported_delta_psnr=None,
    )


def _oracle_category_indices(
    config: dict[str, Any],
    patterns: dict[str, str],
    model: Any,
) -> dict[str, int]:
    latent = _model_config(config).get("latent_bfn", {})
    if not isinstance(latent, dict):
        raise ProbeError("Oracle probe requires model.latent_bfn")
    latent_patterns = latent.get("category_patterns")
    if not isinstance(latent_patterns, dict):
        raise ProbeError("Oracle probe requires latent_bfn.category_patterns")
    if set(latent_patterns) != set(patterns):
        raise ProbeError(
            "Oracle evaluation categories must match latent_bfn.category_patterns; "
            f"evaluation={sorted(patterns)}, latent={sorted(latent_patterns)}"
        )
    if len(latent_patterns) != int(model.latent_num_categories):
        raise ProbeError(
            "latent_bfn.category_patterns length does not match num_categories"
        )
    return {name: index for index, name in enumerate(latent_patterns)}


def _fusion_variant_restore(
    model: Any,
    gains: dict[str, float] | None,
    *,
    bypass_encoder_gate: bool = False,
) -> Any:
    def restore(
        degraded: Tensor,
        target: Tensor,
        category: Tensor | None,
    ) -> tuple[Tensor, dict[str, dict[str, float]]]:
        del target, category
        return restore_model_variant(
            model,
            degraded,
            gains=gains,
            bypass_encoder_gate=bypass_encoder_gate,
        )

    return restore


def _raw_variant_restore(foundation: ExternalSSRFoundation) -> Any:
    def restore(
        degraded: Tensor,
        target: Tensor,
        category: Tensor | None,
    ) -> tuple[Tensor, dict[str, dict[str, float]]]:
        del target, category
        return restore_raw_foundation(foundation, degraded), {}

    return restore


def _oracle_variant_restore(
    model: Any,
    *,
    precision_source: str,
    precision_scale: float = 1.0,
) -> Any:
    def restore(
        degraded: Tensor,
        target: Tensor,
        category: Tensor | None,
    ) -> tuple[Tensor, dict[str, dict[str, float]]]:
        if category is None:
            raise RuntimeError("Oracle restoration received no category label")
        return oracle_restore(
            model,
            degraded,
            target,
            category=category,
            precision_source=precision_source,
            precision_scale=precision_scale,
        )

    return restore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--pairs-file", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--crop-border", type=int, default=0)
    channel_group = parser.add_mutually_exclusive_group()
    channel_group.add_argument(
        "--test-y-channel",
        dest="test_y_channel",
        action="store_true",
        default=True,
        help="Use the All-Weather MATLAB Y channel (default).",
    )
    channel_group.add_argument(
        "--rgb-metrics",
        dest="test_y_channel",
        action="store_false",
        help="Use RGB PSNR/SSIM instead of the default Y-channel protocol.",
    )
    parser.add_argument(
        "--fusion-alpha",
        action="append",
        type=float,
        default=[],
        help=(
            "Fusion-strength alpha for the both-decoder sweep. Repeat to "
            "replace the default 0,0.25,0.5,0.75,1 sequence."
        ),
    )
    parser.add_argument(
        "--fusion-alpha-mode",
        choices=("relative", "absolute"),
        default="relative",
        help=(
            "Interpret alpha relative to each checkpoint fusion gain (default, "
            "so alpha=1 is exactly F2) or as an absolute gain."
        ),
    )
    parser.add_argument(
        "--fusion-sweep",
        action="store_true",
        help=(
            "After the F0/Fback/F1/F2 decomposition, evaluate the requested "
            "both-decoder alpha values plus the two single-decoder-off ablations."
        ),
    )
    parser.add_argument(
        "--single-injection-sweep",
        action="store_true",
        help=(
            "Also evaluate every requested alpha with only decoder_level3 or "
            "decoder_level2 enabled. This costs ten extra inference passes with "
            "the default five alpha values."
        ),
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help=(
            "Evaluate GT-derived latent means with deployed predicted precision "
            "and, separately, unit precision; never calls restore_oracle_z."
        ),
    )
    parser.add_argument(
        "--oracle-precision-scale",
        action="append",
        type=float,
        default=[],
        help=(
            "Optional positive multiplier for deployed predicted precision in "
            "additional Oracle conditions; requires --oracle."
        ),
    )
    parser.add_argument(
        "--category-pattern",
        action="append",
        default=[],
        metavar="NAME=REGEX",
        help="Repeatable category override; otherwise use the checkpoint config.",
    )
    parser.add_argument(
        "--category-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help="Repeatable macro-weight override; otherwise use training sampling.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.crop_border < 0:
        raise ProbeError("--crop-border cannot be negative")
    if args.bootstrap_samples < 100:
        raise ProbeError("--bootstrap-samples must be at least 100")
    if args.single_injection_sweep and not args.fusion_sweep:
        raise ProbeError("--single-injection-sweep requires --fusion-sweep")
    if args.fusion_alpha and not args.fusion_sweep:
        raise ProbeError("--fusion-alpha requires --fusion-sweep")
    if args.oracle_precision_scale and not args.oracle:
        raise ProbeError("--oracle-precision-scale requires --oracle")
    alphas = tuple(args.fusion_alpha or DEFAULT_FUSION_ALPHAS)
    if not alphas:
        raise ProbeError("At least one fusion alpha is required")
    if any(not math.isfinite(alpha) or not 0 <= alpha <= 1 for alpha in alphas):
        raise ProbeError("Every --fusion-alpha must be finite and lie in [0, 1]")
    if len(set(alphas)) != len(alphas):
        raise ProbeError("Fusion alpha values must be unique")
    oracle_precision_scales = tuple(args.oracle_precision_scale)
    if any(
        not math.isfinite(scale) or scale <= 0
        for scale in oracle_precision_scales
    ):
        raise ProbeError(
            "Every --oracle-precision-scale must be finite and positive"
        )
    if len(set(oracle_precision_scales)) != len(oracle_precision_scales):
        raise ProbeError("Oracle precision scales must be unique")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint_payload, dict) or "ema" not in checkpoint_payload:
        raise ProbeError(
            "This probe requires a checkpoint carrying EMA weights so F1 and "
            "F2 use the requested fixed EMA state"
        )
    model, config = load_model_for_inference(
        checkpoint_path,
        device,
        config_path=args.config,
    )
    if not getattr(model, "latent_bfn_enabled", False):
        raise ProbeError("This probe requires a latent-BFN checkpoint")
    if getattr(model, "oracle_z_enabled", False):
        raise ProbeError("This probe requires a deployable latent-BFN checkpoint")
    decoder_latent_adapters(model)
    encoder_gate_adapter(model)
    normal_gains = current_fusion_gains(model)
    category_patterns, category_weights = _category_protocol(
        config,
        args.category_pattern,
        args.category_weight,
    )
    compiled_patterns = _compile_category_patterns(category_patterns)
    category_indices = (
        _oracle_category_indices(config, category_patterns, model)
        if args.oracle
        else None
    )
    dataset = PairedImageDataset(
        args.input_dir,
        args.target_dir,
        pairs_file=args.pairs_file,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    common = {
        "dataset": dataset,
        "category_patterns": compiled_patterns,
        "category_indices": category_indices,
        "config": config,
        "crop_border": args.crop_border,
        "test_y_channel": args.test_y_channel,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "device": device,
        "quiet": args.quiet,
    }

    f0_path = output_dir / "f0_foundation.json"
    fback_path = output_dir / "fback_ema_ssr_no_adapters.json"
    f1_path = output_dir / "f1_fusion_off.json"
    f2_path = output_dir / "f2_normal.json"
    sweep_dir = output_dir / "fusion_sweep"

    raw_foundation = build_raw_foundation(config, device)
    f0 = evaluate_variant(
        **common,
        variant="F0_raw_ssr_foundation",
        restore=_raw_variant_restore(raw_foundation),
        gains=None,
    )
    _write_json(f0_path, f0, overwrite=args.overwrite)
    del raw_foundation
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fusion_off = {name: 0.0 for name in DECODER_ADAPTERS}
    fback = evaluate_variant(
        **common,
        variant="Fback_ema_ssr_weights_all_adapters_bypassed",
        restore=_fusion_variant_restore(
            model,
            fusion_off,
            bypass_encoder_gate=True,
        ),
        gains=fusion_off,
        encoder_gate_bypassed=True,
    )
    _write_json(fback_path, fback, overwrite=args.overwrite)
    f1 = evaluate_variant(
        **common,
        variant="F1_ema_v26d_fusion_off",
        restore=_fusion_variant_restore(model, fusion_off),
        gains=fusion_off,
    )
    _write_json(f1_path, f1, overwrite=args.overwrite)
    f2 = evaluate_variant(
        **common,
        variant="F2_ema_v26d_normal",
        restore=_fusion_variant_restore(model, None),
        gains=normal_gains,
    )
    _write_json(f2_path, f2, overwrite=args.overwrite)
    restored_gains = current_fusion_gains(model)
    if restored_gains != normal_gains:
        raise RuntimeError(
            f"Fusion gains were not restored: before={normal_gains}, after={restored_gains}"
        )

    comparisons = {
        "bfn_latent_F2_minus_F1": _comparison(
            f1_path,
            f2_path,
            category_patterns=category_patterns,
            category_weights=category_weights,
            pairs_file=args.pairs_file,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "ema_ssr_weight_drift_Fback_minus_F0": _comparison(
            f0_path,
            fback_path,
            category_patterns=category_patterns,
            category_weights=category_weights,
            pairs_file=args.pairs_file,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "encoder_spectral_gate_F1_minus_Fback": _comparison(
            fback_path,
            f1_path,
            category_patterns=category_patterns,
            category_weights=category_weights,
            pairs_file=args.pairs_file,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "total_F2_minus_F0": _comparison(
            f0_path,
            f2_path,
            category_patterns=category_patterns,
            category_weights=category_weights,
            pairs_file=args.pairs_file,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    sweep_index_path: Path | None = None
    if args.fusion_sweep:
        sweep_paths: dict[str, Path] = {}
        sweep_specs: dict[str, dict[str, float]] = {
            "decoder_level3_off": {
                "decoder_level3": 0.0,
                "decoder_level2": normal_gains["decoder_level2"],
            },
            "decoder_level2_off": {
                "decoder_level3": normal_gains["decoder_level3"],
                "decoder_level2": 0.0,
            },
        }
        for alpha in alphas:
            absolute_alpha = (
                alpha
                if args.fusion_alpha_mode == "absolute"
                else None
            )
            sweep_specs[
                f"both_{args.fusion_alpha_mode}_alpha_{_slug_alpha(alpha)}"
            ] = {
                name: (
                    absolute_alpha
                    if absolute_alpha is not None
                    else alpha * normal_gains[name]
                )
                for name in DECODER_ADAPTERS
            }
        if args.single_injection_sweep:
            for alpha in alphas:
                suffix = _slug_alpha(alpha)
                level3_gain = (
                    alpha
                    if args.fusion_alpha_mode == "absolute"
                    else alpha * normal_gains["decoder_level3"]
                )
                level2_gain = (
                    alpha
                    if args.fusion_alpha_mode == "absolute"
                    else alpha * normal_gains["decoder_level2"]
                )
                sweep_specs[
                    f"decoder_level3_only_{args.fusion_alpha_mode}_alpha_{suffix}"
                ] = {
                    "decoder_level3": level3_gain,
                    "decoder_level2": 0.0,
                }
                sweep_specs[
                    f"decoder_level2_only_{args.fusion_alpha_mode}_alpha_{suffix}"
                ] = {
                    "decoder_level3": 0.0,
                    "decoder_level2": level2_gain,
                }

        for name, gains in sweep_specs.items():
            if gains == fusion_off:
                source = f1
            elif gains == normal_gains:
                source = f2
            else:
                source = evaluate_variant(
                    **common,
                    variant=name,
                    restore=_fusion_variant_restore(model, gains),
                    gains=gains,
                )
            path = sweep_dir / f"{name}.json"
            _write_json(path, source, overwrite=args.overwrite)
            sweep_paths[name] = path
        restored_gains = current_fusion_gains(model)
        if restored_gains != normal_gains:
            raise RuntimeError(
                f"Fusion gains were not restored after sweep: {restored_gains}"
            )

        sweep_comparisons = {
            name: _comparison(
                f1_path,
                path,
                category_patterns=category_patterns,
                category_weights=category_weights,
                pairs_file=args.pairs_file,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            for name, path in sweep_paths.items()
        }
        sweep_index = {
            "baseline": str(f1_path),
            "alpha_mode": args.fusion_alpha_mode,
            "normal_checkpoint_fusion_gains": normal_gains,
            "variants": {
                name: {
                    "path": str(path),
                    "fusion_gains": sweep_specs[name],
                    "comparison_vs_F1": sweep_comparisons[name],
                }
                for name, path in sweep_paths.items()
            },
        }
        sweep_index_path = sweep_dir / "index.json"
        _write_json(
            sweep_index_path,
            sweep_index,
            overwrite=args.overwrite,
        )

    oracle_paths: dict[str, Path] = {}
    if args.oracle:
        oracle_conditions = [
            (
                "oracle_mean_predicted_precision",
                "predicted",
                1.0,
                "predicted",
                "oracle_mean_predicted_precision_minus_F1",
            ),
            (
                "oracle_mean_unit_precision",
                "unit",
                1.0,
                "unit_condition",
                "oracle_mean_unit_precision_condition_minus_F1",
            ),
        ]
        for scale in oracle_precision_scales:
            if scale == 1.0:
                continue
            suffix = _slug_alpha(scale)
            oracle_conditions.append(
                (
                    f"oracle_mean_predicted_precision_scale_{suffix}",
                    "predicted",
                    scale,
                    f"predicted_scaled_{scale:.6g}",
                    f"oracle_mean_predicted_precision_scale_{suffix}_minus_F1",
                )
            )
        for (
            artifact_name,
            precision_source,
            precision_scale,
            precision_label,
            comparison_name,
        ) in oracle_conditions:
            oracle_path = output_dir / f"{artifact_name}.json"
            oracle = evaluate_variant(
                **common,
                variant=artifact_name,
                restore=_oracle_variant_restore(
                    model,
                    precision_source=precision_source,
                    precision_scale=precision_scale,
                ),
                gains=normal_gains,
                oracle=True,
                oracle_precision=precision_label,
            )
            _write_json(oracle_path, oracle, overwrite=args.overwrite)
            oracle_paths[artifact_name] = oracle_path
            comparisons[comparison_name] = _comparison(
                f1_path,
                oracle_path,
                category_patterns=category_patterns,
                category_weights=category_weights,
                pairs_file=args.pairs_file,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        restored_gains = current_fusion_gains(model)
        if restored_gains != normal_gains:
            raise RuntimeError(
                f"Fusion gains were not restored after Oracle probe: {restored_gains}"
            )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_state": "EMA verified and loaded by load_model_for_inference",
        "images": len(dataset),
        "metric_protocol": {
            "test_y_channel": args.test_y_channel,
            "crop_border": args.crop_border,
            "macro_weights": category_weights,
            "category_patterns": category_patterns,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "normal_checkpoint_fusion_gains": normal_gains,
        "artifacts": {
            "F0_raw_ssr_foundation": str(f0_path),
            "Fback_ema_ssr_weights_all_adapters_bypassed": str(fback_path),
            "F1_ema_v26d_fusion_off": str(f1_path),
            "F2_ema_v26d_normal": str(f2_path),
            "fusion_sweep": str(sweep_index_path) if sweep_index_path else None,
            "oracle_conditions": {
                name: str(path) for name, path in oracle_paths.items()
            },
        },
        "comparisons": comparisons,
        "guardrails": {
            "f0_feature_adapters": "none",
            "fback": (
                "EMA SSR weights with encoder_level3 paper gate bypassed and "
                "decoder_level3/decoder_level2 latent gains set to zero"
            ),
            "f1_fusion_gains": fusion_off,
            "f2_fusion_gains": "checkpoint values without temporary override",
            "oracle": (
                "GT means from degradation_latent_targets plus "
                "restore_from_latent_state under predicted and unit precision; "
                "unit is a condition, not an upper bound"
                if args.oracle
                else "not requested"
            ),
            "fusion_gains_restored": current_fusion_gains(model) == normal_gains,
        },
    }
    _write_json(output_dir / "summary.json", summary, overwrite=args.overwrite)

    for name, report in comparisons.items():
        psnr_summary = report["macro_weighted"]["psnr"]
        low, high = psnr_summary["paired_image_bootstrap_ci95"]
        print(
            f"{name}: macro ΔY-PSNR={psnr_summary['delta']:+.6f} "
            f"CI95=[{low:+.6f}, {high:+.6f}]"
        )
    print(f"Wrote probe artifacts to {output_dir}")


if __name__ == "__main__":
    main()
