#!/usr/bin/env python3
"""Read-only audit of the fixed epoch-3 C0-accept gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_dev_by_category import (
    DEFAULT_CATEGORY_PATTERNS,
    load_evaluation,
)
from analyze_v27_c0prime import CATEGORIES, validate_candidate
from probe_v27_c0prime_routing import metric_row, tensor_state_sha256
from sgbformer.config import load_config
from sgbformer.data import PairedImageDataset, classify_paths
from sgbformer.inference import load_model_for_inference
from sgbformer.models import CategoryRoutedReadout
from sgbformer.protocol import implementation_fingerprint, sha256_file
from sgbformer.utils import parameter_count, resolve_device, seed_everything
from sgbformer.v27 import (
    V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS,
    V27_EXPECTED_DEV_PAIRS,
    resolved_config_sha256,
    validate_v27_c0accept_config,
)


METRIC_REPRODUCTION_TOLERANCE = 1e-4
RECONSTRUCTION_TOLERANCE = 1e-6
BOUNDARY_EPSILON = 1e-6
NEAR_BOUNDARY = 0.05
SATURATION_FRACTION = 0.90
NEAR_CONSTANT_STD = 0.02
NEAR_CONSTANT_RANGE = 0.05
USEFUL_AUC = 0.60
CATEGORY_REJECTION_GAP = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--source-evaluation", required=True, type=Path)
    parser.add_argument("--candidate-evaluation", required=True, type=Path)
    parser.add_argument("--decision-summary", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--pairs-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def quantile_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "max": None,
            "exact_zero_fraction": None,
            "exact_one_fraction": None,
            "near_zero_fraction": None,
            "near_one_fraction": None,
        }
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "max": float(array.max()),
        "exact_zero_fraction": float(
            np.mean(array <= BOUNDARY_EPSILON)
        ),
        "exact_one_fraction": float(
            np.mean(array >= 1.0 - BOUNDARY_EPSILON)
        ),
        "near_zero_fraction": float(np.mean(array <= NEAR_BOUNDARY)),
        "near_one_fraction": float(
            np.mean(array >= 1.0 - NEAR_BOUNDARY)
        ),
    }


def average_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def binary_auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = float(
        ranks[np.asarray(labels, dtype=np.bool_)].sum()
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def correlation(left: list[float], right: list[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        len(left_array) < 2
        or float(left_array.std()) <= 1e-12
        or float(right_array.std()) <= 1e-12
    ):
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def utility_summary(
    records: list[dict[str, Any]],
    utility_key: str,
) -> dict[str, Any]:
    gates = [float(record["acceptance"]) for record in records]
    utilities = [float(record[utility_key]) for record in records]
    labels = [utility > 0.0 for utility in utilities]
    beneficial = [
        gate for gate, label in zip(gates, labels) if label
    ]
    harmful = [
        gate for gate, label in zip(gates, labels) if not label
    ]
    ranks = average_ranks(utilities).tolist()
    return {
        "utility": utility_key,
        "beneficial_images": int(sum(labels)),
        "harmful_or_neutral_images": int(len(labels) - sum(labels)),
        "beneficial_fraction": float(np.mean(labels)),
        "mean_acceptance_beneficial": (
            float(np.mean(beneficial)) if beneficial else None
        ),
        "mean_acceptance_harmful_or_neutral": (
            float(np.mean(harmful)) if harmful else None
        ),
        "acceptance_mean_gap": (
            float(np.mean(beneficial) - np.mean(harmful))
            if beneficial and harmful
            else None
        ),
        "auc_for_accepting_beneficial_correction": binary_auc(
            gates,
            labels,
        ),
        "pearson_acceptance_vs_utility": correlation(gates, utilities),
        "spearman_acceptance_vs_utility": correlation(
            average_ranks(gates).tolist(),
            ranks,
        ),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(group),
            "acceptance": quantile_summary(
                [float(record["acceptance"]) for record in group]
            ),
            "mean_source_psnr_gain_vs_f0": float(
                np.mean(
                    [
                        float(record["source_psnr_gain_vs_f0"])
                        for record in group
                    ]
                )
            ),
            "mean_candidate_psnr_gain_vs_f0": float(
                np.mean(
                    [
                        float(record["candidate_psnr_gain_vs_f0"])
                        for record in group
                    ]
                )
            ),
            "mean_candidate_psnr_gain_vs_source": float(
                np.mean(
                    [
                        float(record["candidate_psnr_gain_vs_source"])
                        for record in group
                    ]
                )
            ),
        }

    return {
        "overall": group_summary(records),
        "by_category": {
            category: group_summary(
                [
                    record
                    for record in records
                    if record["category"] == category
                ]
            )
            for category in CATEGORIES
        },
        "utility_discrimination": {
            "training_rgb_mse": utility_summary(
                records,
                "source_rgb_mse_improvement",
            ),
            "formal_y_psnr": utility_summary(
                records,
                "source_psnr_gain_vs_f0",
            ),
        },
    }


def diagnose_gate(
    summary: dict[str, Any],
    *,
    formal_status: str,
) -> dict[str, Any]:
    overall = summary["overall"]["acceptance"]
    categories = summary["by_category"]
    rgb = summary["utility_discrimination"]["training_rgb_mse"]
    y_psnr = summary["utility_discrimination"]["formal_y_psnr"]
    flags = {
        "formal_decision_is_stop": formal_status == "STOP",
        "near_always_accept": (
            overall["near_one_fraction"] >= SATURATION_FRACTION
            or overall["q05"] >= 1.0 - NEAR_BOUNDARY
        ),
        "near_always_reject": (
            overall["near_zero_fraction"] >= SATURATION_FRACTION
            or overall["q95"] <= NEAR_BOUNDARY
        ),
        "near_constant": (
            overall["std"] <= NEAR_CONSTANT_STD
            or overall["q95"] - overall["q05"] <= NEAR_CONSTANT_RANGE
        ),
        "weak_training_utility_ranking": (
            rgb["auc_for_accepting_beneficial_correction"] is None
            or rgb["auc_for_accepting_beneficial_correction"] < USEFUL_AUC
        ),
        "weak_metric_utility_ranking": (
            y_psnr["auc_for_accepting_beneficial_correction"] is None
            or y_psnr["auc_for_accepting_beneficial_correction"] < USEFUL_AUC
        ),
        "insufficient_category_rejection": (
            categories["snow"]["acceptance"]["mean"]
            - max(
                categories["outdoor_rain"]["acceptance"]["mean"],
                categories["raindrop"]["acceptance"]["mean"],
            )
            < CATEGORY_REJECTION_GAP
        ),
    }
    if flags["near_always_accept"]:
        code = "G1"
        conclusion = "Near-always-accept saturation"
    elif flags["near_always_reject"]:
        code = "G2"
        conclusion = "Near-always-reject saturation"
    elif flags["near_constant"]:
        code = "G3"
        conclusion = "Gate is nearly constant across images"
    elif flags["weak_training_utility_ranking"]:
        code = "G4"
        conclusion = "Gate does not rank correction utility under its training loss"
    elif flags["weak_metric_utility_ranking"]:
        code = "G5"
        conclusion = "Gate ranking does not transfer to formal Y-PSNR"
    elif flags["insufficient_category_rejection"]:
        code = "G6"
        conclusion = "Gate lacks the category rejection gap required by E2"
    else:
        code = "G7"
        conclusion = "Gate has signal, but its learned calibration is insufficient"
    return {
        "code": code,
        "conclusion": conclusion,
        "post_hoc": True,
        "flags": flags,
        "thresholds": {
            "near_boundary": NEAR_BOUNDARY,
            "saturation_fraction": SATURATION_FRACTION,
            "near_constant_std": NEAR_CONSTANT_STD,
            "near_constant_q05_q95_range": NEAR_CONSTANT_RANGE,
            "useful_auc": USEFUL_AUC,
            "category_rejection_gap": CATEGORY_REJECTION_GAP,
        },
        "next_action": (
            "Stop C0-accept tuning. Use this audit only to choose the failure "
            "mechanism to address in a new residual basis/readout design."
        ),
        "forbidden": [
            "Do not train or modify C0-accept.",
            "Do not evaluate another checkpoint.",
            "Do not access official test sets.",
            "Do not start D4/B4.",
            "Do not present this post-hoc audit as preregistered evidence.",
        ],
    }


def load_training_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        train = row.get("train", {})
        optimization = row.get("optimization", {})
        result.append(
            {
                "epoch": int(row["epoch"]),
                "mean_acceptance": (
                    float(train["adapter"])
                    if train.get("adapter") is not None
                    else None
                ),
                "last_gate_gradient_norm": (
                    float(optimization["receiver_gradient_norm"])
                    if optimization.get("receiver_gradient_norm") is not None
                    else None
                ),
            }
        )
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    gate = summary["gate"]
    diagnosis = summary["diagnosis"]
    formal = summary["formal_decision"]
    lines = [
        "# v27 C0-accept Epoch-3 Gate Audit",
        "",
        "## Status",
        "",
        "- Mode: read-only, EMA epoch 3 only",
        f"- Images: {summary['dataset']['images']}",
        f"- Formal preregistered decision: **{formal['status']}**",
        f"- Post-hoc gate diagnosis: **{diagnosis['code']}. "
        f"{diagnosis['conclusion']}**",
        f"- Checkpoint SHA256: `{summary['checkpoint']['sha256']}`",
        "",
        "## Gate Distribution",
        "",
        "| Group | N | Mean | Std | Q05 | Median | Q95 | Near 0 | Near 1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in (
        ("overall", gate["overall"]),
        *(
            (category, gate["by_category"][category])
            for category in CATEGORIES
        ),
    ):
        acceptance = values["acceptance"]
        lines.append(
            f"| {name} | {values['n']} | {acceptance['mean']:.6f} | "
            f"{acceptance['std']:.6f} | {acceptance['q05']:.6f} | "
            f"{acceptance['median']:.6f} | {acceptance['q95']:.6f} | "
            f"{acceptance['near_zero_fraction']:.3f} | "
            f"{acceptance['near_one_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Utility Discrimination",
            "",
            "| Utility definition | Beneficial | Harmful/neutral | Gate gap | "
            "AUC | Pearson | Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, values in gate["utility_discrimination"].items():
        fields = [
            values["acceptance_mean_gap"],
            values["auc_for_accepting_beneficial_correction"],
            values["pearson_acceptance_vs_utility"],
            values["spearman_acceptance_vs_utility"],
        ]
        formatted = [
            "n/a" if value is None else f"{value:+.6f}"
            for value in fields
        ]
        lines.append(
            f"| {name} | {values['beneficial_images']} | "
            f"{values['harmful_or_neutral_images']} | {formatted[0]} | "
            f"{formatted[1]} | {formatted[2]} | {formatted[3]} |"
        )
    lines.extend(
        [
            "",
            "## Category Effect",
            "",
            "| Category | Source vs F0 | Candidate vs F0 | "
            "Candidate vs source |",
            "|---|---:|---:|---:|",
        ]
    )
    for category in CATEGORIES:
        values = gate["by_category"][category]
        lines.append(
            f"| {category} | "
            f"{values['mean_source_psnr_gain_vs_f0']:+.6f} | "
            f"{values['mean_candidate_psnr_gain_vs_f0']:+.6f} | "
            f"{values['mean_candidate_psnr_gain_vs_source']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This diagnosis is post-hoc and does not change the formal STOP.",
            "",
        ]
    )
    lines.extend(
        f"- {name}: `{value}`"
        for name, value in diagnosis["flags"].items()
    )
    lines.extend(
        [
            "",
            f"Next action: {diagnosis['next_action']}",
            "",
            "Forbidden:",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in diagnosis["forbidden"])
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
            f"- Maximum gate reconstruction error: "
            f"`{summary['integrity']['maximum_gate_reconstruction_error']:.3e}`",
            f"- Maximum baseline metric reproduction error: "
            f"`{summary['integrity']['maximum_baseline_metric_error']:.3e}`",
            f"- Maximum source metric reproduction error: "
            f"`{summary['integrity']['maximum_source_metric_error']:.3e}`",
            f"- Maximum candidate metric reproduction error: "
            f"`{summary['integrity']['maximum_candidate_metric_error']:.3e}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("C0-accept gate audit requires seed 42")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    source_path = args.source_evaluation.expanduser().resolve()
    candidate_path = args.candidate_evaluation.expanduser().resolve()
    decision_path = args.decision_summary.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = (
        args.report
        if args.report is not None
        else PROJECT_ROOT / "docs" / "v27_c0accept_gate_audit.md"
    ).expanduser().resolve()
    for path in (
        checkpoint_path,
        baseline_path,
        source_path,
        candidate_path,
        decision_path,
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
        raise ValueError("C0-accept gate audit requires EMA weights")
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
    reference_c0doubleprime = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0doubleprime_bottleneck_routed.yaml"
    )
    validate_v27_c0accept_config(
        config,
        reference_c0,
        reference_c0prime,
        reference_c0doubleprime,
    )
    expected_val = config["data"]["val"]
    for actual, expected, label in (
        (args.input_dir, expected_val["input_dir"], "input_dir"),
        (args.target_dir, expected_val["target_dir"], "target_dir"),
        (pairs_file, expected_val["pairs_file"], "pairs_file"),
    ):
        if actual.expanduser().resolve() != Path(expected).expanduser().resolve():
            raise ValueError(
                f"Gate audit {label} differs from checkpoint config"
            )

    validate_candidate(
        source_path,
        protocol="residual_bfn_v27_exploratory",
        arm="C0-double-prime",
    )
    validate_candidate(
        candidate_path,
        protocol="residual_bfn_v27_exploratory",
        arm="C0-accept",
    )
    formal = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        formal.get("protocol") != "residual_bfn_v27_exploratory"
        or formal.get("arm") != "C0-accept"
        or formal.get("primary_endpoint") != "fixed_epoch_3"
        or formal.get("decision", {}).get("status") != "STOP"
    ):
        raise ValueError(
            "Gate audit requires the fixed epoch-3 C0-accept STOP summary"
        )

    baseline, _ = load_evaluation(baseline_path)
    source, _ = load_evaluation(source_path)
    candidate, _ = load_evaluation(candidate_path)
    expected_names = set(baseline["rows"])
    if (
        set(source["rows"]) != expected_names
        or set(candidate["rows"]) != expected_names
    ):
        raise ValueError("Baseline/source/candidate image sets differ")

    device = resolve_device(args.device)
    seed_everything(args.seed)
    model, loaded_config = load_model_for_inference(
        checkpoint_path,
        device,
    )
    if resolved_config_sha256(loaded_config) != resolved_config_sha256(config):
        raise RuntimeError("Loaded checkpoint config changed")
    if parameter_count(model) != V27_C0ACCEPT_EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("C0-accept trainable parameter count changed")
    receiver = model.receiver
    if receiver is None or not receiver.routed_readout_enabled:
        raise RuntimeError("Checkpoint has no routed receiver")
    readout = receiver.output
    if not isinstance(readout, CategoryRoutedReadout):
        raise RuntimeError("Unexpected routed readout implementation")
    if not readout.acceptance_gate_enabled or readout.acceptance_gate is None:
        raise RuntimeError("Checkpoint has no acceptance gate")

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
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    checkpoint_sha_before = sha256_file(checkpoint_path)
    model_state_before = tensor_state_sha256(model)
    records: list[dict[str, Any]] = []
    maximum_gate_reconstruction_error = 0.0
    maximum_baseline_metric_error = 0.0
    maximum_source_metric_error = 0.0
    maximum_candidate_metric_error = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            degraded = batch["degraded"].to(device)
            target = batch["target"].to(device)
            name = str(batch["name"][0])
            category = labels[index]
            details = model.restore(
                degraded,
                deterministic=True,
                return_details=True,
            )
            acceptance_tensor = readout.last_acceptance
            ungated_tensor = readout.last_ungated_mixture
            if acceptance_tensor is None or acceptance_tensor.shape != (1, 1):
                raise RuntimeError(f"{name}: invalid acceptance output")
            if ungated_tensor is None:
                raise RuntimeError(f"{name}: missing ungated correction")
            acceptance = float(acceptance_tensor[0, 0])
            if not 0.0 <= acceptance <= 1.0:
                raise RuntimeError(f"{name}: acceptance left [0,1]")

            height, width = target.shape[-2:]
            structural = details["structural"].float()
            structural_raw = details["structural_raw"].float()
            candidate_prediction = details["restored"].float()
            gated_correction = details["residual"].float()
            ungated_correction = (
                ungated_tensor[..., :height, :width].float()
                * model.residual_scale
            )
            reconstructed_correction = (
                acceptance_tensor[:, :, None, None].float()
                * ungated_correction
            )
            reconstruction_error = float(
                (gated_correction - reconstructed_correction).abs().max()
            )
            maximum_gate_reconstruction_error = max(
                maximum_gate_reconstruction_error,
                reconstruction_error,
            )
            if reconstruction_error > RECONSTRUCTION_TOLERANCE:
                raise RuntimeError(
                    f"{name}: gate reconstruction error={reconstruction_error}"
                )

            source_prediction = (
                structural + ungated_correction
            ).clamp(0, 1)
            baseline_metric = metric_row(name, structural, target)
            source_metric = metric_row(name, source_prediction, target)
            candidate_metric = metric_row(
                name,
                candidate_prediction,
                target,
            )
            registered_rows = (
                baseline["rows"].get(name),
                source["rows"].get(name),
                candidate["rows"].get(name),
            )
            if any(row is None for row in registered_rows):
                raise RuntimeError(f"{name}: missing registered metric row")
            metric_errors = [
                max(
                    abs(computed[key] - float(registered[key]))
                    for key in ("psnr", "ssim")
                )
                for computed, registered in zip(
                    (baseline_metric, source_metric, candidate_metric),
                    registered_rows,
                )
            ]
            maximum_baseline_metric_error = max(
                maximum_baseline_metric_error,
                metric_errors[0],
            )
            maximum_source_metric_error = max(
                maximum_source_metric_error,
                metric_errors[1],
            )
            maximum_candidate_metric_error = max(
                maximum_candidate_metric_error,
                metric_errors[2],
            )
            if max(metric_errors) > METRIC_REPRODUCTION_TOLERANCE:
                raise RuntimeError(
                    f"{name}: registered metric reproduction error="
                    f"{max(metric_errors)}"
                )

            source_raw = structural_raw + ungated_correction
            source_rgb_mse = float(F.mse_loss(source_raw, target.float()))
            baseline_rgb_mse = float(F.mse_loss(structural_raw, target.float()))
            records.append(
                {
                    "name": name,
                    "input_path": str(batch["input_path"][0]),
                    "category": category,
                    "acceptance": acceptance,
                    "source_rgb_mse_improvement": (
                        baseline_rgb_mse - source_rgb_mse
                    ),
                    "source_psnr_gain_vs_f0": (
                        source_metric["psnr"] - baseline_metric["psnr"]
                    ),
                    "candidate_psnr_gain_vs_f0": (
                        candidate_metric["psnr"] - baseline_metric["psnr"]
                    ),
                    "candidate_psnr_gain_vs_source": (
                        candidate_metric["psnr"] - source_metric["psnr"]
                    ),
                }
            )

    if len(records) != V27_EXPECTED_DEV_PAIRS:
        raise RuntimeError("Gate audit did not process the complete dev set")
    checkpoint_sha_after = sha256_file(checkpoint_path)
    model_state_after = tensor_state_sha256(model)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise RuntimeError("Checkpoint changed during read-only gate audit")
    if model_state_before != model_state_after:
        raise RuntimeError("Model state changed during read-only gate audit")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Read-only gate audit produced parameter gradients")

    gate_summary = summarize_records(records)
    diagnosis = diagnose_gate(
        gate_summary,
        formal_status=formal["decision"]["status"],
    )
    metrics_path = (
        args.metrics_jsonl.expanduser().resolve()
        if args.metrics_jsonl is not None
        else (
            PROJECT_ROOT / config["training"]["output_dir"] / "metrics.jsonl"
        ).resolve()
    )
    gate_parameter = readout.acceptance_gate
    summary = {
        "schema_version": 1,
        "protocol": "v27_c0accept_epoch3_gate_audit",
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
        "inputs": {
            "baseline": {
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
            },
            "source_evaluation": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "candidate_evaluation": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "decision_summary": {
                "path": str(decision_path),
                "sha256": sha256_file(decision_path),
            },
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
        "formal_decision": {
            "status": formal["decision"]["status"],
            "failed_checks": formal["decision"]["failed_checks"],
            "macro_delta_psnr_vs_f0": float(
                formal["vs_f0"]["macro_weighted"]["psnr"]["delta"]
            ),
        },
        "gate_parameters": {
            "trainable_parameters": parameter_count(model),
            "weight_norm": float(
                gate_parameter.weight.detach().float().norm()
            ),
            "weight_min": float(gate_parameter.weight.detach().float().min()),
            "weight_max": float(gate_parameter.weight.detach().float().max()),
            "bias": float(gate_parameter.bias.detach().float()[0]),
        },
        "training_history": {
            "path": str(metrics_path),
            "present": metrics_path.is_file(),
            "epochs": load_training_history(metrics_path),
            "note": (
                "train.adapter is detached mean acceptance. "
                "receiver_output_weight_norm is the frozen expert output "
                "weight norm, not the gate norm."
            ),
        },
        "gate": gate_summary,
        "diagnosis": diagnosis,
        "integrity": {
            "checkpoint_sha256_unchanged": True,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "model_state_unchanged": True,
            "parameter_gradients_absent": True,
            "receiver_forward_passes_per_image": 1,
            "maximum_gate_reconstruction_error": (
                maximum_gate_reconstruction_error
            ),
            "maximum_baseline_metric_error": (
                maximum_baseline_metric_error
            ),
            "maximum_source_metric_error": maximum_source_metric_error,
            "maximum_candidate_metric_error": (
                maximum_candidate_metric_error
            ),
            "extra_training_steps": 0,
        },
        "runtime": {
            "device": str(device),
            "seconds": time.perf_counter() - started,
        },
        "implementation": {
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "core": implementation_fingerprint(),
        },
        "limitations": [
            "The gate diagnosis is post-hoc and does not alter the formal STOP.",
            "Utility labels use the same dev targets and are diagnostic only.",
            "Image-level statistics exclude training-seed variance.",
            "RainDrop contains only 55 dev images.",
            "This deterministic gate provides no evidence about BFN.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "gate_per_image.jsonl"
    per_image_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    summary_path = output_dir / "gate_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(summary), encoding="utf-8")

    overall = gate_summary["overall"]["acceptance"]
    rgb = gate_summary["utility_discrimination"]["training_rgb_mse"]
    y_psnr = gate_summary["utility_discrimination"]["formal_y_psnr"]
    print(f"Formal decision={formal['decision']['status']}")
    print(
        f"Gate mean={overall['mean']:.6f} std={overall['std']:.6f} "
        f"q05={overall['q05']:.6f} q95={overall['q95']:.6f}"
    )
    print(
        f"Gate near_zero={overall['near_zero_fraction']:.6f} "
        f"near_one={overall['near_one_fraction']:.6f}"
    )
    print(
        "Category gate means: "
        + " ".join(
            f"{category}="
            f"{gate_summary['by_category'][category]['acceptance']['mean']:.6f}"
            for category in CATEGORIES
        )
    )
    print(
        "Utility AUC: "
        f"training_rgb={rgb['auc_for_accepting_beneficial_correction']} "
        f"formal_y_psnr={y_psnr['auc_for_accepting_beneficial_correction']}"
    )
    print(f"Diagnosis={diagnosis['code']} {diagnosis['conclusion']}")
    print(f"Wrote {per_image_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
