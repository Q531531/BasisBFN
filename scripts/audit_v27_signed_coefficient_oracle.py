#!/usr/bin/env python3
"""Read-only signed-coefficient oracle audit for the fixed C0'' basis."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_dev_by_category import (
    DEFAULT_CATEGORY_PATTERNS,
    DEFAULT_CATEGORY_WEIGHTS,
    analyze_dev_by_category,
    load_evaluation,
    read_name_map,
)
from analyze_v27_c0prime import CATEGORIES, validate_candidate
from audit_v27_residual_geometry import category_safety, psnr_from_mse
from probe_v27_c0prime_routing import (
    macro_delta,
    metric_row,
    tensor_state_sha256,
)
from sgbformer.config import load_config
from sgbformer.data import PairedImageDataset, classify_paths
from sgbformer.inference import load_model_for_inference
from sgbformer.metrics import rgb_to_matlab_y
from sgbformer.models import CategoryRoutedReadout
from sgbformer.protocol import implementation_fingerprint, sha256_file
from sgbformer.utils import parameter_count, resolve_device, seed_everything
from sgbformer.v27 import (
    V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS,
    V27_EXPECTED_DEV_PAIRS,
    resolved_config_sha256,
    validate_v27_c0doubleprime_config,
)


COEFFICIENT_STEP = 0.25
L1_RADIUS = 1.0
MIXTURE_CHUNK_SIZE = 21
METRIC_REPRODUCTION_TOLERANCE = 1e-4
MATERIAL_GAIN_DB = 0.003
PREDICTOR_WORTHWHILE_UPPER_BOUND_DB = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--registered-evaluation", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--pairs-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixture-chunk-size",
        type=int,
        default=MIXTURE_CHUNK_SIZE,
    )
    return parser.parse_args()


def signed_l1_grid(
    step: float = COEFFICIENT_STEP,
    radius: float = L1_RADIUS,
) -> np.ndarray:
    divisions = round(radius / step)
    if divisions < 1 or not np.isclose(divisions * step, radius):
        raise ValueError("Coefficient step must divide the L1 radius exactly")
    values = range(-divisions, divisions + 1)
    rows = [
        (left * step, middle * step, right * step)
        for left in values
        for middle in values
        for right in values
        if abs(left) + abs(middle) + abs(right) <= divisions
    ]
    rows.sort(key=lambda row: (sum(abs(value) for value in row), row))
    return np.asarray(rows, dtype=np.float64)


def psnr_payload(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    variant: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "images": rows,
        "protocol": {
            **copy.deepcopy(protocol),
            "diagnostic_variant": variant,
            "oracle": variant.startswith("oracle_"),
        },
    }
    normalized = {
        "payload": payload,
        "rows": {str(row["name"]): dict(row) for row in rows},
    }
    schema = {
        "path": f"in-memory:{variant}",
        "images": len(rows),
        "name_key": "name",
        "metrics": ["psnr"],
        "has_embedded_category": False,
        "has_protocol": True,
        "average": None,
    }
    return normalized, schema


def analyze_rows(
    baseline: dict[str, Any],
    baseline_schema: dict[str, Any],
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    variant: str,
    name_map: dict[str, str],
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], dict[str, Any]]]:
    candidate, candidate_schema = psnr_payload(rows, protocol, variant)
    report = analyze_dev_by_category(
        baseline,
        candidate,
        baseline_schema=baseline_schema,
        candidate_schema=candidate_schema,
        category_patterns=DEFAULT_CATEGORY_PATTERNS,
        category_weights=DEFAULT_CATEGORY_WEIGHTS,
        name_map=name_map,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        reported_delta_psnr=None,
    )
    return report, (candidate, candidate_schema)


def compare_payloads(
    left: tuple[dict[str, Any], dict[str, Any]],
    right: tuple[dict[str, Any], dict[str, Any]],
    *,
    name_map: dict[str, str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    return analyze_dev_by_category(
        left[0],
        right[0],
        baseline_schema=left[1],
        candidate_schema=right[1],
        category_patterns=DEFAULT_CATEGORY_PATTERNS,
        category_weights=DEFAULT_CATEGORY_WEIGHTS,
        name_map=name_map,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        reported_delta_psnr=None,
    )


def choose_index(scores: np.ndarray, coefficients: np.ndarray) -> int:
    maximum = float(scores.max())
    candidates = np.flatnonzero(np.isclose(scores, maximum, atol=1e-12))
    return min(
        (int(index) for index in candidates),
        key=lambda index: (
            float(np.abs(coefficients[index]).sum()),
            tuple(float(value) for value in coefficients[index]),
        ),
    )


def coefficient_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def determine_recommendation(
    category_analysis: dict[str, Any],
    category_gain_vs_learned: dict[str, Any],
    per_image_gain_vs_learned: dict[str, Any],
) -> dict[str, Any]:
    category_gain = macro_delta(category_gain_vs_learned)
    per_image_gain = macro_delta(per_image_gain_vs_learned)
    recovers_safety = category_safety(category_analysis)
    has_material_category_gain = category_gain >= MATERIAL_GAIN_DB
    has_predictor_headroom = (
        per_image_gain >= PREDICTOR_WORTHWHILE_UPPER_BOUND_DB
    )
    if not has_predictor_headroom:
        code = "S3"
        conclusion = "Current expert basis has insufficient signed headroom"
        next_action = (
            "Redesign the residual basis/readout before training any "
            "coefficient predictor or Bayesian flow."
        )
    elif recovers_safety and has_material_category_gain:
        code = "S1"
        conclusion = "Current basis supports useful signed coefficients"
        next_action = (
            "Build a matched deterministic signed-coefficient predictor "
            "before introducing Bayesian coefficient flow."
        )
    else:
        code = "S2"
        conclusion = "Basis has per-image headroom but no robust category policy"
        next_action = (
            "Redesign the predictor input or basis conditioning; do not add "
            "Bayesian flow until a deterministic predictor realizes the "
            "headroom."
        )
    return {
        "code": code,
        "conclusion": conclusion,
        "post_hoc": True,
        "checks": {
            "category_policy_recovers_f0_safety": bool(recovers_safety),
            "category_gain_vs_learned_is_material": bool(
                has_material_category_gain
            ),
            "per_image_upper_bound_is_worth_predicting": bool(
                has_predictor_headroom
            ),
        },
        "measurements": {
            "category_macro_gain_vs_learned_db": category_gain,
            "per_image_macro_gain_vs_learned_db": per_image_gain,
        },
        "thresholds": {
            "material_category_gain_db": MATERIAL_GAIN_DB,
            "predictor_worthwhile_upper_bound_db": (
                PREDICTOR_WORTHWHILE_UPPER_BOUND_DB
            ),
        },
        "next_action": next_action,
        "forbidden": [
            "Do not train in this audit.",
            "Do not evaluate another checkpoint.",
            "Do not access official test sets.",
            "Do not start D4/B4.",
            "Do not report oracle results as deployable model results.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    evaluations = summary["evaluations"]
    recommendation = summary["recommendation"]
    lines = [
        "# v27 C0-double-prime Signed Coefficient Oracle Audit",
        "",
        "## Status",
        "",
        "- Mode: read-only, EMA epoch 3 only",
        f"- Images: {summary['dataset']['images']}",
        f"- Coefficient grid size: {summary['grid']['size']}",
        f"- Constraint: `{summary['grid']['constraint']}`",
        f"- Recommendation: **{recommendation['code']}. "
        f"{recommendation['conclusion']}**",
        "",
        "## Selected Policies",
        "",
        "| Policy | Coefficients | Macro vs F0 | Gain vs learned | "
        "F0 safety |",
        "|---|---|---:|---:|---:|",
    ]
    for key in ("global_fixed", "category_oracle", "per_image_oracle"):
        values = evaluations[key]
        coefficients = values.get("coefficients", "per-image")
        if isinstance(coefficients, dict):
            rendered = json.dumps(coefficients, sort_keys=True)
        elif isinstance(coefficients, list):
            rendered = json.dumps(coefficients)
        else:
            rendered = str(coefficients)
        lines.append(
            f"| {key} | `{rendered}` | "
            f"{macro_delta(values['analysis']):+.6f} | "
            f"{macro_delta(values['gain_vs_learned']):+.6f} | "
            f"{category_safety(values['analysis'])} |"
        )
    lines.extend(
        [
            "",
            "## Category Oracle Detail",
            "",
            "| Category | Coefficients | Delta PSNR vs F0 |",
            "|---|---|---:|",
        ]
    )
    category = evaluations["category_oracle"]
    for name in CATEGORIES:
        lines.append(
            f"| {name} | "
            f"`{json.dumps(category['coefficients'][name])}` | "
            f"{category['analysis']['categories'][name]['psnr']['delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All oracle policies use current dev targets and are diagnostic upper "
            "bounds, not model results.",
            "",
        ]
    )
    lines.extend(
        f"- {name}: `{value}`"
        for name, value in recommendation["checks"].items()
    )
    lines.extend(
        [
            "",
            f"Next action: {recommendation['next_action']}",
            "",
            "Forbidden:",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in recommendation["forbidden"])
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Checkpoint unchanged: "
            f"`{summary['integrity']['checkpoint_sha256_unchanged']}`",
            f"- Model state unchanged: "
            f"`{summary['integrity']['model_state_unchanged']}`",
            f"- Parameter gradients absent: "
            f"`{summary['integrity']['parameter_gradients_absent']}`",
            f"- Maximum baseline metric error: "
            f"`{summary['integrity']['maximum_baseline_metric_error']:.3e}`",
            f"- Maximum learned metric error: "
            f"`{summary['integrity']['maximum_registered_metric_error']:.3e}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples != 10000 or args.seed != 42:
        raise ValueError(
            "Signed coefficient audit requires bootstrap=10000 and seed=42"
        )
    if args.mixture_chunk_size < 1:
        raise ValueError("--mixture-chunk-size must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    registered_path = args.registered_evaluation.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = (
        args.report
        if args.report is not None
        else PROJECT_ROOT / "docs" / "v27_signed_coefficient_oracle.md"
    ).expanduser().resolve()
    for path in (
        checkpoint_path,
        baseline_path,
        registered_path,
        pairs_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if checkpoint_path.name != "epoch_0003.pt":
        raise ValueError("Only epoch_0003.pt is allowed")

    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if int(raw_checkpoint.get("epoch", -1)) != 2:
        raise ValueError(
            "Checkpoint metadata epoch index must be 2 for epoch_0003.pt"
        )
    if "ema" not in raw_checkpoint:
        raise ValueError("Signed coefficient audit requires EMA weights")
    config = raw_checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no embedded config")
    reference_c0 = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0_proxy.yaml"
    )
    reference_c0prime = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0prime_routed.yaml"
    )
    validate_v27_c0doubleprime_config(
        config,
        reference_c0,
        reference_c0prime,
    )
    expected_val = config["data"]["val"]
    for actual, expected, label in (
        (args.input_dir, expected_val["input_dir"], "input_dir"),
        (args.target_dir, expected_val["target_dir"], "target_dir"),
        (pairs_file, expected_val["pairs_file"], "pairs_file"),
    ):
        if actual.expanduser().resolve() != Path(expected).expanduser().resolve():
            raise ValueError(
                f"Signed coefficient audit {label} differs from checkpoint config"
            )
    validate_candidate(
        registered_path,
        protocol="residual_bfn_v27_exploratory",
        arm="C0-double-prime",
    )

    baseline, baseline_schema = load_evaluation(baseline_path)
    registered, _ = load_evaluation(registered_path)
    if set(baseline["rows"]) != set(registered["rows"]):
        raise ValueError("Baseline and registered image sets differ")
    baseline_protocol = baseline["payload"].get("protocol")
    if not isinstance(baseline_protocol, dict):
        raise ValueError("Baseline lacks protocol metadata")
    if (
        baseline_protocol.get("test_y_channel") is not True
        or baseline_protocol.get("crop_border") != 0
        or baseline_protocol.get("stochastic") is not False
        or baseline_protocol.get("tile") is not None
    ):
        raise ValueError(
            "Audit requires deterministic full-image Y/crop-0 protocol"
        )

    device = resolve_device(args.device)
    seed_everything(args.seed)
    model, loaded_config = load_model_for_inference(
        checkpoint_path,
        device,
    )
    if resolved_config_sha256(loaded_config) != resolved_config_sha256(config):
        raise RuntimeError("Loaded checkpoint config changed")
    if (
        parameter_count(model)
        != V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS
    ):
        raise RuntimeError("C0-double-prime parameter count changed")
    receiver = model.receiver
    if receiver is None or not receiver.routed_readout_enabled:
        raise RuntimeError("Checkpoint has no routed receiver")
    readout = receiver.output
    if not isinstance(readout, CategoryRoutedReadout):
        raise RuntimeError("Unexpected routed readout implementation")
    expert_names = tuple(readout.categories)
    if expert_names != CATEGORIES:
        raise RuntimeError("Unexpected expert order")
    if readout.acceptance_gate_enabled:
        raise RuntimeError("Signed coefficient audit requires C0-double-prime")

    dataset = PairedImageDataset(
        args.input_dir,
        args.target_dir,
        pairs_file=pairs_file,
    )
    if len(dataset) != V27_EXPECTED_DEV_PAIRS:
        raise RuntimeError(
            f"Expected {V27_EXPECTED_DEV_PAIRS} dev images, got {len(dataset)}"
        )
    labels = classify_paths(
        [pair[0] for pair in dataset.pairs],
        DEFAULT_CATEGORY_PATTERNS,
    )
    name_map = read_name_map(pairs_file)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    coefficients = signed_l1_grid()
    coefficient_tensor = torch.from_numpy(coefficients).to(
        device=device,
        dtype=torch.float32,
    )
    grid_psnr = np.empty(
        (len(dataset), len(coefficients)),
        dtype=np.float64,
    )

    captured: dict[str, Tensor] = {}

    def capture(name: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Tensor, ...],
            output: Tensor,
        ) -> None:
            captured[name] = output.detach()

        return hook

    handles = [
        head.register_forward_hook(capture(name))
        for name, head in readout.heads.items()
    ]
    checkpoint_sha_before = sha256_file(checkpoint_path)
    model_state_before = tensor_state_sha256(model)
    maximum_baseline_metric_error = 0.0
    maximum_registered_metric_error = 0.0
    learned_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                captured.clear()
                degraded = batch["degraded"].to(device)
                target = batch["target"].to(device)
                name = str(batch["name"][0])
                details = model.restore(
                    degraded,
                    deterministic=True,
                    return_details=True,
                )
                height, width = target.shape[-2:]
                structural = details["structural"].float()
                learned = details["restored"].float()
                basis = []
                for expert in expert_names:
                    output = captured.get(expert)
                    if output is None:
                        raise RuntimeError(
                            f"{name}: expert hook did not capture {expert}"
                        )
                    basis.append(
                        output[..., :height, :width].float()[0]
                        * model.residual_scale
                    )
                basis_tensor = torch.stack(basis, dim=0)
                baseline_y = rgb_to_matlab_y(structural.double())
                target_y = rgb_to_matlab_y(target.double())
                for start in range(
                    0,
                    len(coefficients),
                    args.mixture_chunk_size,
                ):
                    end = min(
                        start + args.mixture_chunk_size,
                        len(coefficients),
                    )
                    corrections = torch.einsum(
                        "mk,kchw->mchw",
                        coefficient_tensor[start:end],
                        basis_tensor,
                    )
                    predictions = (
                        structural + corrections
                    ).clamp(0, 1)
                    predictions_y = rgb_to_matlab_y(predictions.double())
                    mse = (
                        predictions_y - target_y
                    ).square().mean(dim=(-3, -2, -1))
                    grid_psnr[index, start:end] = (
                        psnr_from_mse(mse).cpu().numpy()
                    )

                baseline_metric = metric_row(name, structural, target)
                learned_metric = metric_row(name, learned, target)
                baseline_row = baseline["rows"].get(name)
                registered_row = registered["rows"].get(name)
                if baseline_row is None or registered_row is None:
                    raise RuntimeError(f"{name}: missing registered metric row")
                baseline_error = max(
                    abs(
                        baseline_metric[metric]
                        - float(baseline_row[metric])
                    )
                    for metric in ("psnr", "ssim")
                )
                registered_error = max(
                    abs(
                        learned_metric[metric]
                        - float(registered_row[metric])
                    )
                    for metric in ("psnr", "ssim")
                )
                maximum_baseline_metric_error = max(
                    maximum_baseline_metric_error,
                    baseline_error,
                )
                maximum_registered_metric_error = max(
                    maximum_registered_metric_error,
                    registered_error,
                )
                if (
                    max(baseline_error, registered_error)
                    > METRIC_REPRODUCTION_TOLERANCE
                ):
                    raise RuntimeError(
                        f"{name}: registered metric reproduction failed"
                    )
                learned_rows.append(
                    {"name": name, "psnr": learned_metric["psnr"]}
                )
                probabilities = readout.last_router_probabilities
                if probabilities is None:
                    raise RuntimeError(f"{name}: missing router probabilities")
                records.append(
                    {
                        "name": name,
                        "input_path": str(batch["input_path"][0]),
                        "category": labels[index],
                        "learned_router_probabilities": {
                            expert: float(probabilities[0, position])
                            for position, expert in enumerate(expert_names)
                        },
                    }
                )
    finally:
        for handle in handles:
            handle.remove()

    if len(records) != V27_EXPECTED_DEV_PAIRS:
        raise RuntimeError("Audit did not process the complete dev set")
    checkpoint_sha_after = sha256_file(checkpoint_path)
    model_state_after = tensor_state_sha256(model)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise RuntimeError("Checkpoint changed during read-only audit")
    if model_state_before != model_state_after:
        raise RuntimeError("Model state changed during read-only audit")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Read-only audit produced parameter gradients")

    names = [record["name"] for record in records]
    baseline_psnr = np.asarray(
        [float(baseline["rows"][name]["psnr"]) for name in names],
        dtype=np.float64,
    )
    delta = grid_psnr - baseline_psnr[:, None]
    category_masks = {
        category: np.asarray(
            [label == category for label in labels],
            dtype=np.bool_,
        )
        for category in CATEGORIES
    }
    category_means = {
        category: delta[mask].mean(axis=0)
        for category, mask in category_masks.items()
    }
    macro_scores = sum(
        DEFAULT_CATEGORY_WEIGHTS[category] * category_means[category]
        for category in CATEGORIES
    )
    global_index = choose_index(macro_scores, coefficients)
    category_indices = {
        category: choose_index(values, coefficients)
        for category, values in category_means.items()
    }
    per_image_indices = np.asarray(
        [
            choose_index(grid_psnr[index], coefficients)
            for index in range(len(records))
        ],
        dtype=np.int64,
    )
    global_indices = np.full(
        len(records),
        global_index,
        dtype=np.int64,
    )
    category_policy_indices = np.asarray(
        [category_indices[label] for label in labels],
        dtype=np.int64,
    )

    def selected_rows(indices: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "psnr": float(grid_psnr[index, indices[index]]),
            }
            for index, name in enumerate(names)
        ]

    learned_report, learned_payload = analyze_rows(
        baseline,
        baseline_schema,
        learned_rows,
        baseline_protocol,
        variant="learned_soft_router",
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    selections = {
        "global_fixed": global_indices,
        "category_oracle": category_policy_indices,
        "per_image_oracle": per_image_indices,
    }
    analyses = {}
    payloads = {}
    gains = {}
    for key, indices in selections.items():
        analysis, payload = analyze_rows(
            baseline,
            baseline_schema,
            selected_rows(indices),
            baseline_protocol,
            variant=f"oracle_signed_{key}",
            name_map=name_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        analyses[key] = analysis
        payloads[key] = payload
        gains[key] = compare_payloads(
            learned_payload,
            payload,
            name_map=name_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )

    for index, record in enumerate(records):
        selected_index = int(per_image_indices[index])
        record["per_image_oracle_coefficients"] = coefficient_list(
            coefficients[selected_index]
        )
        record["per_image_oracle_psnr"] = float(
            grid_psnr[index, selected_index]
        )
        record["per_image_oracle_gain_vs_f0"] = float(
            grid_psnr[index, selected_index] - baseline_psnr[index]
        )
        record["learned_psnr"] = float(learned_rows[index]["psnr"])
        record["per_image_oracle_gain_vs_learned"] = float(
            grid_psnr[index, selected_index] - learned_rows[index]["psnr"]
        )

    per_image_counts = Counter(
        tuple(record["per_image_oracle_coefficients"])
        for record in records
    )
    evaluations = {
        "learned_soft_router": {
            "analysis": learned_report,
        },
        "global_fixed": {
            "coefficients": coefficient_list(coefficients[global_index]),
            "analysis": analyses["global_fixed"],
            "gain_vs_learned": gains["global_fixed"],
        },
        "category_oracle": {
            "target_leaking": True,
            "coefficients": {
                category: coefficient_list(
                    coefficients[category_indices[category]]
                )
                for category in CATEGORIES
            },
            "analysis": analyses["category_oracle"],
            "gain_vs_learned": gains["category_oracle"],
        },
        "per_image_oracle": {
            "severe_target_leakage": True,
            "coefficients": "per-image",
            "analysis": analyses["per_image_oracle"],
            "gain_vs_learned": gains["per_image_oracle"],
            "unique_coefficient_vectors": len(per_image_counts),
            "most_common_coefficients": [
                {
                    "coefficients": list(values),
                    "count": count,
                }
                for values, count in per_image_counts.most_common(20)
            ],
        },
    }
    recommendation = determine_recommendation(
        analyses["category_oracle"],
        gains["category_oracle"],
        gains["per_image_oracle"],
    )
    summary = {
        "schema_version": 1,
        "protocol": "v27_c0doubleprime_signed_coefficient_oracle",
        "status": "completed",
        "read_only": True,
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": 3,
            "zero_based_epoch_index": int(raw_checkpoint["epoch"]),
            "sha256": checkpoint_sha_before,
            "ema_weights_used": True,
            "resolved_config_sha256": resolved_config_sha256(config),
        },
        "registered_evaluation": {
            "path": str(registered_path),
            "sha256": sha256_file(registered_path),
        },
        "baseline": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "dataset": {
            "images": len(records),
            "input_dir": str(args.input_dir.expanduser().resolve()),
            "target_dir": str(args.target_dir.expanduser().resolve()),
            "pairs_file": str(pairs_file),
            "pairs_file_sha256": sha256_file(pairs_file),
            "category_counts": dict(Counter(labels)),
            "official_test_accessed": False,
        },
        "grid": {
            "step": COEFFICIENT_STEP,
            "l1_radius": L1_RADIUS,
            "size": len(coefficients),
            "constraint": (
                "coefficients are multiples of 0.25 with sum(abs(c)) <= 1"
            ),
            "contains_zero": bool(
                np.any(np.all(coefficients == 0.0, axis=1))
            ),
            "contains_negative_coefficients": bool(
                np.any(coefficients < 0.0)
            ),
        },
        "evaluations": evaluations,
        "recommendation": recommendation,
        "integrity": {
            "checkpoint_sha256_unchanged": True,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "model_state_unchanged": True,
            "parameter_gradients_absent": True,
            "receiver_forward_passes_per_image": 1,
            "maximum_baseline_metric_error": (
                maximum_baseline_metric_error
            ),
            "maximum_registered_metric_error": (
                maximum_registered_metric_error
            ),
            "extra_training_steps": 0,
        },
        "runtime": {
            "device": str(device),
            "seconds": time.perf_counter() - started,
            "candidates_per_image": len(coefficients),
            "mixture_chunk_size": args.mixture_chunk_size,
        },
        "implementation": {
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "core": implementation_fingerprint(),
        },
        "limitations": [
            "All coefficient selection uses current dev targets.",
            "The bounded grid is an upper-bound diagnostic, not a deployable model.",
            "The grid does not prove that coefficients are predictable from inputs.",
            "Image bootstrap excludes training-seed variance.",
            "RainDrop contains only 55 dev images.",
            "This deterministic audit provides no evidence about BFN.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "signed_oracle_per_image.jsonl"
    per_image_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(summary), encoding="utf-8")

    print(
        f"Signed L1 grid: size={len(coefficients)} "
        f"step={COEFFICIENT_STEP} radius={L1_RADIUS}"
    )
    print(
        "Global signed coefficients: "
        f"{coefficient_list(coefficients[global_index])} "
        f"gain_vs_learned={macro_delta(gains['global_fixed']):+.6f}"
    )
    print(
        "Category signed coefficients: "
        + json.dumps(
            evaluations["category_oracle"]["coefficients"],
            sort_keys=True,
        )
    )
    print(
        "Category oracle: "
        f"gain_vs_learned={macro_delta(gains['category_oracle']):+.6f} "
        f"f0_safety={category_safety(analyses['category_oracle'])}"
    )
    print(
        "Per-image oracle: "
        f"gain_vs_learned={macro_delta(gains['per_image_oracle']):+.6f}"
    )
    print(
        f"Recommendation={recommendation['code']} "
        f"{recommendation['conclusion']}"
    )
    print(f"Wrote {per_image_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
