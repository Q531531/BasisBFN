#!/usr/bin/env python3
"""Read-only epoch-3 router and expert probe for routed v27 C0 variants."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
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
from sgbformer.config import load_config
from sgbformer.data import PairedImageDataset, classify_paths
from sgbformer.inference import load_model_for_inference
from sgbformer.metrics import psnr, ssim
from sgbformer.models import CategoryRoutedReadout
from sgbformer.protocol import implementation_fingerprint, sha256_file
from sgbformer.utils import resolve_device, seed_everything
from sgbformer.v27 import (
    V27_C0DOUBLEPRIME_ARM,
    V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS,
    V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS,
    V27_EXPECTED_DEV_PAIRS,
    V27_EXPECTED_TRAINABLE_PARAMETERS,
    resolved_config_sha256,
    validate_v27_c0doubleprime_config,
    validate_v27_c0prime_config,
)


CATEGORIES = ("snow", "outdoor_rain", "raindrop")
PAIR_NAMES = (
    ("snow", "outdoor_rain"),
    ("snow", "raindrop"),
    ("outdoor_rain", "raindrop"),
)
LN3 = math.log(3.0)
UNIFORM_ENTROPY_TOLERANCE = 1e-3
UNIFORM_PROBABILITY_TOLERANCE = 0.02
EXPERT_COSINE_COLLAPSE_THRESHOLD = 0.999
EXPERT_RELATIVE_DIFFERENCE_THRESHOLD = 0.01
FORCED_MACRO_COLLAPSE_DB = 0.001
ORACLE_MAPPING_HEADROOM_DB = 0.003
PER_IMAGE_ORACLE_HEADROOM_DB = 0.005
METRIC_REPRODUCTION_TOLERANCE = 1e-4
ARM_SPECS = {
    "C0-prime": {
        "expected_parameters": V27_C0PRIME_EXPECTED_TRAINABLE_PARAMETERS,
        "protocol": "v27_c0prime_epoch3_route_probe",
        "report": "v27_c0prime_route_probe.md",
        "head_structure": "GroupNorm(16) -> SiLU -> Conv2d(16,3,3x3)",
        "parameter_formula": "(3 * 467 + 51) - 467 = 985",
        "initialization": (
            "all output convolutions zero; deterministic private GroupNorm "
            "affine offsets [-1,0,+1] * 1e-3 break exact symmetry"
        ),
    },
    V27_C0DOUBLEPRIME_ARM: {
        "expected_parameters": (
            V27_C0DOUBLEPRIME_EXPECTED_TRAINABLE_PARAMETERS
        ),
        "protocol": "v27_c0doubleprime_epoch3_route_probe",
        "report": "v27_c0doubleprime_route_probe.md",
        "head_structure": (
            "GroupNorm(16) -> Conv2d(16,16,3x3) -> SiLU -> "
            "GroupNorm(16) -> SiLU -> Conv2d(16,3,3x3)"
        ),
        "parameter_formula": "(3 * 2,819 + 51) - 467 = 8,041",
        "initialization": (
            "all output convolutions zero; each private bottleneck convolution "
            "has an independent random initialization"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--registered-evaluation", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--pairs-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
    )
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tensor_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cosine(left: Tensor, right: Tensor) -> float:
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1e-12:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def rms(value: Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def metric_row(
    name: str,
    prediction: Tensor,
    target: Tensor,
) -> dict[str, Any]:
    options = {"crop_border": 0, "test_y_channel": True}
    return {
        "name": name,
        "psnr": float(psnr(prediction, target, **options)[0]),
        "ssim": float(ssim(prediction, target, **options)[0]),
    }


def evaluation_payload(
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
        "metrics": ["psnr", "ssim"],
        "has_embedded_category": False,
        "has_protocol": True,
        "average": None,
    }
    return normalized, schema


def analyze_variant(
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
    candidate, candidate_schema = evaluation_payload(rows, protocol, variant)
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


def compare_variants(
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


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def summarize_router(
    records: list[dict[str, Any]],
    expert_names: tuple[str, ...],
) -> dict[str, Any]:
    probabilities = np.asarray(
        [record["router_probabilities_list"] for record in records],
        dtype=np.float64,
    )
    entropies = np.asarray(
        [record["router_entropy"] for record in records],
        dtype=np.float64,
    )

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        values = np.asarray(
            [record["router_probabilities_list"] for record in group],
            dtype=np.float64,
        )
        entropy_values = [
            float(record["router_entropy"]) for record in group
        ]
        top_counts = Counter(record["top1_expert"] for record in group)
        return {
            "n": len(group),
            "mean_probabilities": {
                name: float(values[:, index].mean())
                for index, name in enumerate(expert_names)
            },
            "probability_std": {
                name: float(values[:, index].std())
                for index, name in enumerate(expert_names)
            },
            "mean_entropy": _mean(entropy_values),
            "entropy_std": _std(entropy_values),
            "entropy_deficit_from_ln3": LN3 - _mean(entropy_values),
            "top1_expert_counts": {
                name: int(top_counts.get(name, 0)) for name in expert_names
            },
        }

    by_category = {
        category: group_summary(
            [record for record in records if record["true_category"] == category]
        )
        for category in CATEGORIES
    }
    confusion = {
        category: {
            expert: sum(
                record["true_category"] == category
                and record["top1_expert"] == expert
                for record in records
            )
            for expert in expert_names
        }
        for category in CATEGORIES
    }
    accuracy = sum(
        record["top1_expert"] == record["true_category"]
        for record in records
    ) / len(records)
    recalls = [
        confusion[category].get(category, 0)
        / max(sum(confusion[category].values()), 1)
        for category in CATEGORIES
    ]

    best_permuted = None
    for permutation in itertools.permutations(CATEGORIES):
        expert_to_category = dict(zip(expert_names, permutation))
        mapped = [
            expert_to_category[record["top1_expert"]]
            for record in records
        ]
        mapped_accuracy = sum(
            prediction == record["true_category"]
            for prediction, record in zip(mapped, records)
        ) / len(records)
        mapped_recalls = []
        for category in CATEGORIES:
            group = [
                index
                for index, record in enumerate(records)
                if record["true_category"] == category
            ]
            mapped_recalls.append(
                sum(mapped[index] == category for index in group)
                / max(len(group), 1)
            )
        candidate = {
            "expert_to_category": expert_to_category,
            "accuracy": mapped_accuracy,
            "balanced_accuracy": _mean(mapped_recalls),
        }
        if best_permuted is None or (
            candidate["balanced_accuracy"],
            candidate["accuracy"],
        ) > (
            best_permuted["balanced_accuracy"],
            best_permuted["accuracy"],
        ):
            best_permuted = candidate

    overall_mean = probabilities.mean(axis=0)
    entropy_mean = float(entropies.mean())
    maximum_probability_deviation = float(
        np.max(np.abs(overall_mean - 1.0 / len(expert_names)))
    )
    near_uniform = (
        LN3 - entropy_mean <= UNIFORM_ENTROPY_TOLERANCE
        and maximum_probability_deviation <= UNIFORM_PROBABILITY_TOLERANCE
    )
    return {
        "n": len(records),
        "ln3": LN3,
        "mean_probabilities": {
            name: float(overall_mean[index])
            for index, name in enumerate(expert_names)
        },
        "probability_std": {
            name: float(probabilities[:, index].std())
            for index, name in enumerate(expert_names)
        },
        "mean_entropy": entropy_mean,
        "entropy_std": float(entropies.std()),
        "entropy_deficit_from_ln3": LN3 - entropy_mean,
        "maximum_probability_deviation_from_uniform": (
            maximum_probability_deviation
        ),
        "top1_expert_counts": {
            name: int(
                Counter(
                    record["top1_expert"] for record in records
                ).get(name, 0)
            )
            for name in expert_names
        },
        "by_true_category": by_category,
        "confusion_matrix_true_by_top1_expert": confusion,
        "classification_accuracy": float(accuracy),
        "balanced_accuracy": _mean(recalls),
        "best_permutation_invariant_classification": best_permuted,
        "near_uniform": near_uniform,
        "near_uniform_thresholds": {
            "maximum_entropy_deficit": UNIFORM_ENTROPY_TOLERANCE,
            "maximum_probability_deviation": UNIFORM_PROBABILITY_TOLERANCE,
        },
        "semantic_caveat": (
            "No router CE was used. Expert names are bookkeeping labels and "
            "direct classification accuracy is permutation-dependent."
        ),
    }


def summarize_diagnostics(
    records: list[dict[str, Any]],
    expert_names: tuple[str, ...],
) -> dict[str, Any]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        experts = {
            expert: {
                "correction_rms": _mean(
                    [
                        record["experts"][expert]["correction_rms"]
                        for record in group
                    ]
                ),
                "correction_target_cosine": _mean(
                    [
                        record["experts"][expert][
                            "correction_target_cosine"
                        ]
                        for record in group
                    ]
                ),
            }
            for expert in expert_names
        }
        pairs = {}
        for left, right in PAIR_NAMES:
            key = f"{left}__{right}"
            pairs[key] = {
                "correction_cosine": _mean(
                    [record["expert_pairs"][key]["cosine"] for record in group]
                ),
                "difference_rms": _mean(
                    [
                        record["expert_pairs"][key]["difference_rms"]
                        for record in group
                    ]
                ),
            }
        return {
            "n": len(group),
            "experts": experts,
            "expert_pairs": pairs,
            "learned_soft": {
                "correction_rms": _mean(
                    [
                        record["learned_soft"]["correction_rms"]
                        for record in group
                    ]
                ),
                "correction_target_cosine": _mean(
                    [
                        record["learned_soft"][
                            "correction_target_cosine"
                        ]
                        for record in group
                    ]
                ),
            },
        }

    return {
        "overall": summarize(records),
        "by_true_category": {
            category: summarize(
                [
                    record
                    for record in records
                    if record["true_category"] == category
                ]
            )
            for category in CATEGORIES
        },
    }


def parameter_diagnostics(
    readout: CategoryRoutedReadout,
) -> dict[str, Any]:
    flattened = {
        name: torch.cat(
            [parameter.detach().float().cpu().flatten() for parameter in head.parameters()]
        )
        for name, head in readout.heads.items()
    }
    pairs = {}
    for left, right in PAIR_NAMES:
        difference = flattened[left] - flattened[right]
        pairs[f"{left}__{right}"] = {
            "parameter_cosine": cosine(flattened[left], flattened[right]),
            "parameter_difference_rms": rms(difference),
            "parameters_exactly_equal": bool(
                torch.equal(flattened[left], flattened[right])
            ),
        }
    return {
        "head_parameters": {
            name: int(sum(parameter.numel() for parameter in head.parameters()))
            for name, head in readout.heads.items()
        },
        "router_parameters": int(
            sum(parameter.numel() for parameter in readout.router.parameters())
        ),
        "pairwise": pairs,
    }


def correction_summary(
    records: list[dict[str, Any]],
    selector: dict[str, str] | None = None,
    per_image_selector: dict[str, str] | None = None,
    fixed_expert: str | None = None,
) -> dict[str, Any]:
    def expert_for(record: dict[str, Any]) -> str:
        if fixed_expert is not None:
            return fixed_expert
        if selector is not None:
            return selector[record["true_category"]]
        if per_image_selector is not None:
            return per_image_selector[record["name"]]
        raise ValueError("A correction selector is required")

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        chosen = [
            record["experts"][expert_for(record)] for record in group
        ]
        return {
            "n": len(group),
            "correction_rms": _mean(
                [item["correction_rms"] for item in chosen]
            ),
            "correction_target_cosine": _mean(
                [item["correction_target_cosine"] for item in chosen]
            ),
        }

    return {
        "overall": summarize(records),
        "by_true_category": {
            category: summarize(
                [
                    record
                    for record in records
                    if record["true_category"] == category
                ]
            )
            for category in CATEGORIES
        },
    }


def soft_correction_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(group),
            "correction_rms": _mean(
                [
                    record["learned_soft"]["correction_rms"]
                    for record in group
                ]
            ),
            "correction_target_cosine": _mean(
                [
                    record["learned_soft"]["correction_target_cosine"]
                    for record in group
                ]
            ),
        }

    return {
        "overall": summarize(records),
        "by_true_category": {
            category: summarize(
                [
                    record
                    for record in records
                    if record["true_category"] == category
                ]
            )
            for category in CATEGORIES
        },
    }


def macro_delta(report: dict[str, Any], metric: str = "psnr") -> float:
    return float(report["macro_weighted"][metric]["delta"])


def category_delta(
    report: dict[str, Any],
    category: str,
    metric: str = "psnr",
) -> float:
    return float(report["categories"][category][metric]["delta"])


def determine_attribution(
    router: dict[str, Any],
    diagnostics: dict[str, Any],
    forced_evaluations: dict[str, dict[str, Any]],
    mapping_gain_vs_learned: dict[str, Any],
    per_image_gain_vs_learned: dict[str, Any],
) -> dict[str, Any]:
    pair_values = diagnostics["overall"]["expert_pairs"].values()
    minimum_pairwise_cosine = min(
        item["correction_cosine"] for item in pair_values
    )
    maximum_pairwise_difference = max(
        item["difference_rms"] for item in pair_values
    )
    mean_expert_rms = _mean(
        [
            item["correction_rms"]
            for item in diagnostics["overall"]["experts"].values()
        ]
    )
    relative_difference = maximum_pairwise_difference / max(
        mean_expert_rms,
        1e-12,
    )
    forced_macro_values = [
        macro_delta(report) for report in forced_evaluations.values()
    ]
    forced_macro_range = max(forced_macro_values) - min(forced_macro_values)
    symmetry_collapse = (
        (
            minimum_pairwise_cosine >= EXPERT_COSINE_COLLAPSE_THRESHOLD
            and relative_difference <= EXPERT_RELATIVE_DIFFERENCE_THRESHOLD
        )
        or maximum_pairwise_difference <= 1e-7
    ) and forced_macro_range <= FORCED_MACRO_COLLAPSE_DB

    category_spreads = {
        category: max(
            category_delta(report, category)
            for report in forced_evaluations.values()
        )
        - min(
            category_delta(report, category)
            for report in forced_evaluations.values()
        )
        for category in CATEGORIES
    }
    mapping_headroom = macro_delta(mapping_gain_vs_learned)
    per_image_headroom = macro_delta(per_image_gain_vs_learned)
    experts_offer_headroom = (
        mapping_headroom >= ORACLE_MAPPING_HEADROOM_DB
        or per_image_headroom >= PER_IMAGE_ORACLE_HEADROOM_DB
    )

    if symmetry_collapse:
        code = "B"
        conclusion = "Expert symmetry collapse"
        allowed = (
            "Fix independent expert parameterization, initialization, or "
            "expert-specific learning signals before changing the router."
        )
    elif router["near_uniform"] and experts_offer_headroom:
        code = "A"
        conclusion = "Router did not learn"
        allowed = (
            "A later experiment may test router supervision or a stronger "
            "zero-preserving symmetry break as one isolated variable."
        )
    elif experts_offer_headroom:
        code = "C"
        conclusion = "Router and experts differentiated, but allocation is wrong"
        allowed = (
            "A later experiment may test supervised hard/soft routing as one "
            "isolated variable."
        )
    else:
        code = "D"
        conclusion = "Receiver/objective mismatch"
        allowed = (
            "Stop routed-receiver tuning and redesign the residual objective "
            "or readout target."
        )
    return {
        "code": code,
        "conclusion": conclusion,
        "router_near_uniform": bool(router["near_uniform"]),
        "expert_symmetry_collapse": bool(symmetry_collapse),
        "experts_offer_headroom": bool(experts_offer_headroom),
        "minimum_pairwise_expert_cosine": minimum_pairwise_cosine,
        "maximum_pairwise_difference_rms": maximum_pairwise_difference,
        "maximum_relative_pairwise_difference": relative_difference,
        "forced_expert_macro_range_db": forced_macro_range,
        "forced_category_psnr_spreads_db": category_spreads,
        "category_spread_is_headroom": False,
        "oracle_mapping_macro_gain_vs_learned_db": mapping_headroom,
        "per_image_oracle_macro_gain_vs_learned_db": per_image_headroom,
        "thresholds": {
            "expert_cosine_collapse": EXPERT_COSINE_COLLAPSE_THRESHOLD,
            "expert_relative_difference": (
                EXPERT_RELATIVE_DIFFERENCE_THRESHOLD
            ),
            "forced_macro_collapse_db": FORCED_MACRO_COLLAPSE_DB,
            "oracle_mapping_headroom_db": ORACLE_MAPPING_HEADROOM_DB,
            "per_image_oracle_headroom_db": PER_IMAGE_ORACLE_HEADROOM_DB,
        },
        "next_allowed": allowed,
        "forbidden": [
            "Do not start D4/B4.",
            "Do not scan another checkpoint.",
            "Do not access official test sets.",
            "Do not report oracle variants as deployable method results.",
            "Do not reinterpret deterministic diagnostics as BFN evidence.",
        ],
    }


def read_entropy_history(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    history = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        history.append(
            {
                "epoch": int(record["epoch"]),
                "router_entropy": float(record["train"]["adapter"]),
                "entropy_deficit_from_ln3": (
                    LN3 - float(record["train"]["adapter"])
                ),
            }
        )
    return history


def render_markdown(summary: dict[str, Any]) -> str:
    router = summary["router"]
    attribution = summary["attribution"]
    evaluations = summary["evaluations"]
    implementation = summary["implementation"]
    arm = summary["arm"]
    lines = [
        f"# v27 {arm} Epoch-3 Route Probe",
        "",
        "## Status",
        "",
        "- Probe mode: read-only, EMA epoch 3 only",
        f"- Images: {summary['dataset']['images']}",
        f"- Checkpoint SHA256: `{summary['checkpoint']['sha256']}`",
        f"- Main attribution: **{attribution['code']}. "
        f"{attribution['conclusion']}**",
        "",
        "## Implementation Audit",
        "",
        "- Structure: shared receiver trunk plus three private output heads.",
        f"- Each head: {implementation['head_structure']}.",
        "- Router: image-level GAP -> Linear(16,3) -> softmax over experts.",
        "- Router CE/category supervision: none.",
        "- C0 parameters: 614,455.",
        f"- {arm} parameters: "
        f"{implementation['candidate_trainable_parameters']:,}.",
        "- Delta versus C0: "
        f"+{implementation['parameter_delta']:,} = "
        f"{implementation['parameter_formula']}.",
        f"- Initialization: {implementation['head_initialization']}.",
        "- `train.adapter` is detached router entropy; its loss weight is zero.",
        "",
        "Source references:",
        "",
    ]
    lines.extend(
        f"- `{reference}`"
        for reference in implementation["source_references"]
    )
    lines.extend(
        [
        "",
        "## Router",
        "",
        f"- Mean entropy: `{router['mean_entropy']:.9f}`",
        f"- ln(3): `{router['ln3']:.9f}`",
        "- Entropy deficit: "
        f"`{router['entropy_deficit_from_ln3']:.9f}`",
        f"- Near uniform: `{router['near_uniform']}`",
        f"- Direct classification accuracy: "
        f"`{router['classification_accuracy']:.6f}`",
        f"- Balanced accuracy: `{router['balanced_accuracy']:.6f}`",
        "",
        "| True category | Snow p | Outdoor p | RainDrop p | Entropy |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    for category in CATEGORIES:
        values = router["by_true_category"][category]
        probabilities = values["mean_probabilities"]
        lines.append(
            f"| {category} | {probabilities['snow']:.6f} | "
            f"{probabilities['outdoor_rain']:.6f} | "
            f"{probabilities['raindrop']:.6f} | "
            f"{values['mean_entropy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Forced Experts",
            "",
            "| Variant | Natural dPSNR | Macro dPSNR | Snow | Outdoor | RainDrop |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for expert, report in evaluations["forced_experts"].items():
        lines.append(
            f"| {expert} | {report['natural_mean']['psnr']['delta']:+.6f} | "
            f"{report['macro_weighted']['psnr']['delta']:+.6f} | "
            f"{report['categories']['snow']['psnr']['delta']:+.6f} | "
            f"{report['categories']['outdoor_rain']['psnr']['delta']:+.6f} | "
            f"{report['categories']['raindrop']['psnr']['delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Expert Differentiation",
            "",
            "| Pair | Mean correction cosine | Mean difference RMS |",
            "|---|---:|---:|",
        ]
    )
    for pair, values in summary["experts"]["overall"][
        "expert_pairs"
    ].items():
        lines.append(
            f"| {pair} | {values['correction_cosine']:.6f} | "
            f"{values['difference_rms']:.8f} |"
        )
    lines.extend(
        [
            "",
            "| Expert | Correction RMS | Correction-target cosine |",
            "|---|---:|---:|",
        ]
    )
    for expert, values in summary["experts"]["overall"]["experts"].items():
        lines.append(
            f"| {expert} | {values['correction_rms']:.8f} | "
            f"{values['correction_target_cosine']:.6f} |"
        )
    selected = evaluations["oracle_category_mapping"]["selected"]
    per_image = evaluations["per_image_oracle"]
    lines.extend(
        [
            "",
            "## Oracle Diagnostics",
            "",
            "- Selected category mapping (maximum current-dev macro PSNR): "
            f"`{json.dumps(selected['category_to_expert'], sort_keys=True)}`",
            "- Category-mapping macro gain versus learned routing: "
            f"`{attribution['oracle_mapping_macro_gain_vs_learned_db']:+.6f} dB`",
            "- Per-image oracle macro gain versus learned routing: "
            f"`{attribution['per_image_oracle_macro_gain_vs_learned_db']:+.6f} dB`",
            "- Per-image oracle is target-leaking and is only a diagnostic "
            "upper bound.",
            "",
            "## Attribution",
            "",
            f"**{attribution['code']}. {attribution['conclusion']}**",
            "",
            f"Allowed next action: {attribution['next_allowed']}",
            "",
            "Forbidden:",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in attribution["forbidden"])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Category mapping and per-image oracle use dev labels/targets.",
            "- Bootstrap conditions on fixed checkpoints and excludes seed variance.",
            "- RainDrop has only 55 dev images.",
            "- No official test data was accessed.",
            "",
        ]
    )
    del per_image
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples != 10000 or args.seed != 42:
        raise ValueError("Routed probe requires bootstrap=10000 and seed=42")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    registered_path = args.registered_evaluation.expanduser().resolve()
    pairs_file = args.pairs_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if checkpoint_path.name != "epoch_0003.pt":
        raise ValueError("Only epoch_0003.pt is allowed")
    for path in (checkpoint_path, baseline_path, registered_path, pairs_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if int(raw_checkpoint.get("epoch", -1)) != 2:
        raise ValueError(
            f"Checkpoint metadata epoch index must be 2 for epoch_0003.pt, got "
            f"{raw_checkpoint.get('epoch')!r}"
        )
    embedded_config = raw_checkpoint.get("config")
    if not isinstance(embedded_config, dict):
        raise ValueError("Checkpoint has no embedded config")
    arm = str(embedded_config.get("protocol", {}).get("arm", ""))
    spec = ARM_SPECS.get(arm)
    if spec is None:
        raise ValueError(
            f"Unsupported routed checkpoint arm: {arm!r}; "
            f"expected one of {tuple(ARM_SPECS)}"
        )
    report_path = (
        args.report
        if args.report is not None
        else PROJECT_ROOT / "docs" / str(spec["report"])
    ).expanduser().resolve()
    if "ema" not in raw_checkpoint:
        raise ValueError(f"{arm} probe requires the registered EMA checkpoint")
    reference_c0 = load_config(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "ssr_residual_v27_c0_proxy.yaml"
    )
    if arm == "C0-prime":
        validate_v27_c0prime_config(embedded_config, reference_c0)
    else:
        reference_c0prime = load_config(
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / "ssr_residual_v27_c0prime_routed.yaml"
        )
        validate_v27_c0doubleprime_config(
            embedded_config,
            reference_c0,
            reference_c0prime,
        )
    expected_val = embedded_config["data"]["val"]
    for actual, expected, label in (
        (args.input_dir, expected_val["input_dir"], "input_dir"),
        (args.target_dir, expected_val["target_dir"], "target_dir"),
        (pairs_file, expected_val["pairs_file"], "pairs_file"),
    ):
        if actual.expanduser().resolve() != Path(expected).expanduser().resolve():
            raise ValueError(f"Probe {label} differs from checkpoint config")

    device = resolve_device(args.device)
    seed_everything(args.seed)
    model, config = load_model_for_inference(checkpoint_path, device)
    receiver = model.receiver
    if receiver is None or not receiver.routed_readout_enabled:
        raise RuntimeError("Checkpoint has no routed residual receiver")
    readout = receiver.output
    if not isinstance(readout, CategoryRoutedReadout):
        raise RuntimeError("Unexpected routed readout implementation")
    expert_names = tuple(readout.categories)
    if expert_names != CATEGORIES:
        raise RuntimeError(
            f"Unexpected expert order: {expert_names!r} != {CATEGORIES!r}"
        )
    if sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) != spec["expected_parameters"]:
        raise RuntimeError(f"{arm} parameter count changed")
    if embedded_config["loss"]["adapter_weight"] != 0.0:
        raise RuntimeError("Router entropy must not enter the loss")

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
    baseline, baseline_schema = load_evaluation(baseline_path)
    registered, _ = load_evaluation(registered_path)
    if (
        registered["payload"].get("protocol", {}).get("experiment_arm")
        != arm
    ):
        raise ValueError(f"Registered evaluation is not {arm}")
    name_map = read_name_map(pairs_file)
    baseline_protocol = baseline["payload"].get("protocol")
    if not isinstance(baseline_protocol, dict):
        raise ValueError("Foundation baseline lacks protocol metadata")
    if (
        baseline_protocol.get("test_y_channel") is not True
        or baseline_protocol.get("crop_border") != 0
        or baseline_protocol.get("stochastic") is not False
        or baseline_protocol.get("tile") is not None
    ):
        raise ValueError("Probe requires deterministic full-image Y/crop-0 protocol")

    captured: dict[str, Tensor] = {}

    def capture(name: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Tensor, ...],
            output: Tensor,
        ) -> None:
            captured[name] = output.detach()

        return hook

    handles = [readout.router.register_forward_hook(capture("router_logits"))]
    handles.extend(
        head.register_forward_hook(capture(f"head:{name}"))
        for name, head in readout.heads.items()
    )
    checkpoint_sha_before = sha256_file(checkpoint_path)
    model_state_before = tensor_state_sha256(model)
    parameter_report = parameter_diagnostics(readout)
    records: list[dict[str, Any]] = []
    variant_rows: dict[str, list[dict[str, Any]]] = {
        "learned_soft_router": [],
        **{f"force_{name}": [] for name in expert_names},
        "per_image_oracle": [],
    }
    per_image_oracle_expert: dict[str, str] = {}
    maximum_soft_reconstruction_error = 0.0
    maximum_foundation_metric_error = 0.0
    maximum_registered_metric_error = 0.0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                captured.clear()
                degraded = batch["degraded"].to(device)
                target = batch["target"].to(device)
                name = str(batch["name"][0])
                category = labels[index]
                details = model.restore(
                    degraded,
                    deterministic=True,
                    return_details=True,
                )
                logits = captured.get("router_logits")
                if logits is None or logits.shape != (1, len(expert_names)):
                    raise RuntimeError(f"{name}: router hook did not capture logits")
                probabilities = F.softmax(logits.float(), dim=1)[0]
                entropy = float(
                    -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                )
                height, width = target.shape[-2:]
                structural = details["structural"].float()
                learned_prediction = details["restored"].float()
                learned_correction = details["residual"].float()
                target_residual = target.float() - structural
                expert_corrections: dict[str, Tensor] = {}
                expert_predictions: dict[str, Tensor] = {}
                expert_rows: dict[str, dict[str, Any]] = {}
                expert_details: dict[str, dict[str, float]] = {}
                for expert in expert_names:
                    output = captured.get(f"head:{expert}")
                    if output is None:
                        raise RuntimeError(
                            f"{name}: head hook did not capture {expert}"
                        )
                    correction = (
                        output[..., :height, :width].float()
                        * model.residual_scale
                    )
                    prediction = (structural + correction).clamp(0, 1)
                    expert_corrections[expert] = correction
                    expert_predictions[expert] = prediction
                    row = metric_row(name, prediction, target)
                    expert_rows[expert] = row
                    expert_details[expert] = {
                        "correction_rms": rms(correction),
                        "correction_target_cosine": cosine(
                            correction,
                            target_residual,
                        ),
                    }
                    variant_rows[f"force_{expert}"].append(row)

                reconstructed = sum(
                    probabilities[position]
                    * expert_corrections[expert]
                    for position, expert in enumerate(expert_names)
                )
                reconstruction_error = float(
                    (reconstructed - learned_correction).abs().max()
                )
                maximum_soft_reconstruction_error = max(
                    maximum_soft_reconstruction_error,
                    reconstruction_error,
                )
                if reconstruction_error > 1e-6:
                    raise RuntimeError(
                        f"{name}: expert mixture does not reproduce learned "
                        f"correction, max_abs={reconstruction_error}"
                    )
                learned_row = metric_row(
                    name,
                    learned_prediction,
                    target,
                )
                variant_rows["learned_soft_router"].append(learned_row)
                structural_row = metric_row(
                    name,
                    structural,
                    target,
                )
                baseline_row = baseline["rows"].get(name)
                registered_row = registered["rows"].get(name)
                if baseline_row is None or registered_row is None:
                    raise RuntimeError(f"{name}: missing registered metric row")
                foundation_error = max(
                    abs(structural_row[metric] - float(baseline_row[metric]))
                    for metric in ("psnr", "ssim")
                )
                registered_error = max(
                    abs(learned_row[metric] - float(registered_row[metric]))
                    for metric in ("psnr", "ssim")
                )
                maximum_foundation_metric_error = max(
                    maximum_foundation_metric_error,
                    foundation_error,
                )
                maximum_registered_metric_error = max(
                    maximum_registered_metric_error,
                    registered_error,
                )
                if foundation_error > METRIC_REPRODUCTION_TOLERANCE:
                    raise RuntimeError(
                        f"{name}: foundation metric mismatch={foundation_error}"
                    )
                if registered_error > METRIC_REPRODUCTION_TOLERANCE:
                    raise RuntimeError(
                        f"{name}: learned metric mismatch={registered_error}"
                    )

                best_expert = max(
                    expert_names,
                    key=lambda expert: (
                        expert_rows[expert]["psnr"],
                        -expert_names.index(expert),
                    ),
                )
                per_image_oracle_expert[name] = best_expert
                variant_rows["per_image_oracle"].append(
                    expert_rows[best_expert]
                )
                pair_details = {}
                for left, right in PAIR_NAMES:
                    difference = (
                        expert_corrections[left] - expert_corrections[right]
                    )
                    pair_details[f"{left}__{right}"] = {
                        "cosine": cosine(
                            expert_corrections[left],
                            expert_corrections[right],
                        ),
                        "difference_rms": rms(difference),
                    }
                top_index = int(probabilities.argmax())
                top_expert = expert_names[top_index]
                records.append(
                    {
                        "name": name,
                        "input_path": str(batch["input_path"][0]),
                        "true_category": category,
                        "router_probabilities": {
                            expert: float(probabilities[position])
                            for position, expert in enumerate(expert_names)
                        },
                        "router_probabilities_list": [
                            float(value) for value in probabilities
                        ],
                        "router_entropy": entropy,
                        "entropy_deficit_from_ln3": LN3 - entropy,
                        "top1_expert": top_expert,
                        "top1_expert_index": top_index,
                        "top1_confidence": float(probabilities[top_index]),
                        "experts": expert_details,
                        "expert_pairs": pair_details,
                        "learned_soft": {
                            "correction_rms": rms(learned_correction),
                            "correction_target_cosine": cosine(
                                learned_correction,
                                target_residual,
                            ),
                            "psnr": learned_row["psnr"],
                            "ssim": learned_row["ssim"],
                        },
                        "per_image_oracle_expert": best_expert,
                    }
                )
    finally:
        for handle in handles:
            handle.remove()

    if len(records) != V27_EXPECTED_DEV_PAIRS:
        raise RuntimeError("Probe did not process the complete dev set")
    checkpoint_sha_after = sha256_file(checkpoint_path)
    model_state_after = tensor_state_sha256(model)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise RuntimeError("Checkpoint changed during read-only probe")
    if model_state_before != model_state_after:
        raise RuntimeError("Model state changed during read-only probe")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Read-only probe produced parameter gradients")

    router_summary = summarize_router(records, expert_names)
    expert_summary = summarize_diagnostics(records, expert_names)
    evaluations: dict[str, Any] = {"forced_experts": {}}
    learned_report, learned_normalized = analyze_variant(
        baseline,
        baseline_schema,
        variant_rows["learned_soft_router"],
        baseline_protocol,
        variant="learned_soft_router",
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    evaluations["learned_soft_router"] = learned_report
    for expert in expert_names:
        report, normalized = analyze_variant(
            baseline,
            baseline_schema,
            variant_rows[f"force_{expert}"],
            baseline_protocol,
            variant=f"force_{expert}",
            name_map=name_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        evaluations["forced_experts"][expert] = report
        del normalized

    mapping_results = []
    mapping_normalized = {}
    forced_rows_by_name = {
        expert: {
            row["name"]: row
            for row in variant_rows[f"force_{expert}"]
        }
        for expert in expert_names
    }
    for permutation in itertools.permutations(expert_names):
        mapping = dict(zip(CATEGORIES, permutation))
        rows = [
            forced_rows_by_name[mapping[record["true_category"]]][
                record["name"]
            ]
            for record in records
        ]
        variant = "oracle_category_mapping_" + "_".join(permutation)
        report, normalized = analyze_variant(
            baseline,
            baseline_schema,
            rows,
            baseline_protocol,
            variant=variant,
            name_map=name_map,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        mapping_results.append(
            {
                "category_to_expert": mapping,
                "selection_metric": "current_dev_macro_psnr_delta_vs_f0",
                "selection_value": macro_delta(report),
                "analysis": report,
            }
        )
        mapping_normalized[tuple(permutation)] = normalized
    selected_mapping = max(
        mapping_results,
        key=lambda item: (
            item["selection_value"],
            tuple(item["category_to_expert"][name] for name in CATEGORIES),
        ),
    )
    selected_tuple = tuple(
        selected_mapping["category_to_expert"][name] for name in CATEGORIES
    )
    mapping_gain = compare_variants(
        learned_normalized,
        mapping_normalized[selected_tuple],
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    per_image_report, per_image_normalized = analyze_variant(
        baseline,
        baseline_schema,
        variant_rows["per_image_oracle"],
        baseline_protocol,
        variant="oracle_per_image_best_psnr",
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    per_image_gain = compare_variants(
        learned_normalized,
        per_image_normalized,
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    evaluations["oracle_category_mapping"] = {
        "selection_is_dev_leaking": True,
        "selection_rule": "maximize current-dev macro PSNR delta versus F0",
        "all_six_mappings": mapping_results,
        "selected": selected_mapping,
        "selected_gain_vs_learned": mapping_gain,
        "correction": correction_summary(
            records,
            selector=selected_mapping["category_to_expert"],
        ),
    }
    evaluations["per_image_oracle"] = {
        "severe_target_leakage": True,
        "selection_rule": "choose the expert with maximum target PSNR per image",
        "analysis": per_image_report,
        "gain_vs_learned": per_image_gain,
        "expert_counts": dict(Counter(per_image_oracle_expert.values())),
        "correction": correction_summary(
            records,
            per_image_selector=per_image_oracle_expert,
        ),
    }
    evaluations["learned_soft_router_correction"] = soft_correction_summary(
        records
    )
    evaluations["forced_expert_corrections"] = {
        expert: correction_summary(records, fixed_expert=expert)
        for expert in expert_names
    }

    attribution = determine_attribution(
        router_summary,
        expert_summary,
        evaluations["forced_experts"],
        mapping_gain,
        per_image_gain,
    )
    metrics_path = (
        args.metrics_jsonl.expanduser().resolve()
        if args.metrics_jsonl is not None
        else (
            PROJECT_ROOT
            / embedded_config["training"]["output_dir"]
            / "metrics.jsonl"
        ).resolve()
    )
    summary = {
        "schema_version": 1,
        "protocol": spec["protocol"],
        "arm": arm,
        "status": "completed",
        "read_only": True,
        "checkpoint": {
            "path": str(checkpoint_path),
            "name": checkpoint_path.name,
            "epoch": 3,
            "zero_based_epoch_index": int(raw_checkpoint["epoch"]),
            "sha256": checkpoint_sha_before,
            "ema_weights_used": "ema" in raw_checkpoint,
            "resolved_config_sha256": resolved_config_sha256(config),
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
        "implementation": {
            "fingerprint": implementation_fingerprint(),
            "structure": (
                "shared residual receiver trunk plus three private output heads "
                "and one image-level soft router"
            ),
            "c0_trainable_parameters": V27_EXPECTED_TRAINABLE_PARAMETERS,
            "candidate_trainable_parameters": spec["expected_parameters"],
            "parameter_delta": (
                spec["expected_parameters"]
                - V27_EXPECTED_TRAINABLE_PARAMETERS
            ),
            "parameter_formula": spec["parameter_formula"],
            "head_structure": spec["head_structure"],
            "router_input": "GAP of shared [B,16,H,W] receiver feature",
            "router_output": "one softmax probability vector [B,3] per image",
            "router_supervision": "none",
            "router_ce_weight": 0.0,
            "head_initialization": spec["initialization"],
            "train_adapter_semantics": (
                "detached mean router entropy; adapter_weight=0, so diagnostic only"
            ),
            "source_references": [
                "sgbformer/models/receiver.py:131-288",
                "sgbformer/models/receiver.py:389-435",
                "sgbformer/models/model.py:204-224",
                "sgbformer/models/model.py:533-539",
                "sgbformer/losses.py:155-191",
                "train.py:732-742,987-1012",
            ],
        },
        "parameter_diagnostics": parameter_report,
        "training_router_entropy_history": read_entropy_history(metrics_path),
        "router": router_summary,
        "experts": expert_summary,
        "evaluations": evaluations,
        "attribution": attribution,
        "integrity": {
            "checkpoint_sha256_unchanged": True,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "model_state_unchanged": True,
            "parameter_gradients_absent": True,
            "maximum_soft_reconstruction_error": (
                maximum_soft_reconstruction_error
            ),
            "maximum_foundation_metric_reproduction_error": (
                maximum_foundation_metric_error
            ),
            "maximum_registered_metric_reproduction_error": (
                maximum_registered_metric_error
            ),
        },
        "runtime": {
            "device": str(device),
            "seconds": time.perf_counter() - started,
            "receiver_trunk_calls_per_image": 1,
            "extra_model_training_steps": 0,
        },
        "limitations": [
            "Category-mapping oracle selects on the same dev set it reports.",
            "Per-image oracle directly uses target PSNR and is severely leaking.",
            "Expert names are not semantically identified without router CE.",
            "Image bootstrap excludes training-seed variance.",
            "RainDrop contains only 55 dev images.",
            "This deterministic arm provides no evidence about BFN.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "router_per_image.jsonl"
    per_image_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    router_path = output_dir / "router_summary.json"
    router_path.write_text(
        json.dumps(router_summary, indent=2, sort_keys=True) + "\n",
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
        f"Router entropy={router_summary['mean_entropy']:.9f} "
        f"ln3_deficit={router_summary['entropy_deficit_from_ln3']:.9f} "
        f"near_uniform={router_summary['near_uniform']}"
    )
    print(
        f"Attribution={attribution['code']} "
        f"{attribution['conclusion']}"
    )
    for expert, report in evaluations["forced_experts"].items():
        print(
            f"force_{expert}: "
            f"macro_dPSNR={macro_delta(report):+.6f}"
        )
    print(
        "oracle_category_mapping: "
        f"macro_gain_vs_learned={macro_delta(mapping_gain):+.6f}"
    )
    print(
        "per_image_oracle: "
        f"macro_gain_vs_learned={macro_delta(per_image_gain):+.6f}"
    )
    print(f"Wrote {per_image_path}")
    print(f"Wrote {router_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
