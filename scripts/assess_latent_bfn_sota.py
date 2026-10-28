#!/usr/bin/env python3
"""Assess locked official-split results against manuscript Table 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


# Published All-Weather PSNR targets.
#
# PROVENANCE (PSNR column, verified 2026-09-20 against the original papers):
#   MODEM, NeurIPS 2025, arXiv:2505.17581, Table 1
#     Snow100K-S 38.08 | Snow100K-L 32.52 | Outdoor-Rain 33.10 | RainDrop 33.01
#     four-split macro average 34.18  <- current macro-average SOTA
#   SSR (our frozen foundation), ECCV 2026, arXiv:2608.01760, Table 1
#     Snow100K-S 37.87 | Snow100K-L 32.70 | Outdoor-Rain 33.22 | RainDrop 32.82
#     four-split macro average 34.15
#   Histoformer, ECCV 2024, Table 1
#     Snow100K-S 37.41 | Snow100K-L 32.16 | Outdoor-Rain 32.08 | RainDrop 33.06
#     four-split macro average 33.68
#
# The per-split gate below uses the MODEM row because MODEM is the single
# coherent macro-average leader.  Note that it is NOT the per-split best
# published value: SSR leads on Snow100K-L (32.70) and Outdoor-Rain (33.22),
# and Histoformer leads on RainDrop (33.06).  The headline claim must therefore
# be read off ``macro_average`` in the report, not off the per-split flags.
#
# Superseded values: this table previously held
#   35.05 / 31.40 / 31.25 / 32.00 (macro average 32.43).
# That row does not correspond to any published method -- it is 1.7 dB below
# the SSR checkpoint we already start from, so it would have declared a false
# "beat SOTA" result.  Do not restore it.
#
# CAVEAT: only the PSNR column is verified.  The SSIM and LPIPS entries are
# inherited placeholders and are used solely as non-inferiority guards, not as
# SOTA claims.  The published tables also do not state whether PSNR is computed
# on the Y channel with zero border crop; that is a community convention
# realised in the reference ``compute_psnr.py``, not a claim in the papers.
PUBLISHED_MACRO_AVERAGES = {
    "modem_neurips_2025": 34.18,
    "ssr_eccv_2026": 34.15,
    "histoformer_eccv_2024": 33.68,
}
MANUSCRIPT_MACRO_AVERAGE_TARGET = 34.28

# Our own reproduction of the untouched SSR foundation under this repository's
# full-image four-set Y-channel protocol. The underlying adapter-only route was
# archived during the v26d-only cleanup.
#
#     macro PSNR 34.239638 dB, macro LPIPS 0.062147
#
# This is the bar that actually matters.  It is +0.09 dB above SSR's own
# published 34.15 and +0.06 dB above MODEM's 34.18, which means every published
# row is already cleared by the frozen baseline -- beating them is not evidence
# that our contribution works.  A reviewer will ask whether the method beats the
# reproduced foundation, so `beats_reproduced_foundation` is the headline check.
#
# The +0.09 dB offset against SSR's published number also means our evaluation
# pipeline is not demonstrably identical to theirs.  Always report the
# reproduced foundation next to any published row rather than claiming a direct
# win over a number produced by a different pipeline.
REPRODUCED_FOUNDATION_MACRO_PSNR = 34.239638
REPRODUCED_FOUNDATION_MACRO_LPIPS = 0.062147

TARGETS = {
    "snow100k_s": {"psnr": 38.08, "ssim": 0.96, "lpips": 0.049},
    "snow100k_l": {"psnr": 32.52, "ssim": 0.94, "lpips": 0.070},
    "outdoor_rain": {"psnr": 33.10, "ssim": 0.94, "lpips": 0.081},
    "raindrop": {"psnr": 33.01, "ssim": 0.95, "lpips": 0.084},
}


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    generator: np.random.Generator,
) -> tuple[float, float]:
    if values.size < 1:
        raise ValueError("Cannot bootstrap an empty split")
    chunk = max(1, min(samples, 256))
    means = []
    remaining = samples
    while remaining:
        count = min(chunk, remaining)
        indices = generator.integers(
            0,
            values.size,
            size=(count, values.size),
        )
        means.append(values[indices].mean(axis=1))
        remaining -= count
    distribution = np.concatenate(means)
    low, high = np.quantile(distribution, (0.025, 0.975))
    return float(low), float(high)


def assess(
    candidate_dir: Path,
    ssr_dir: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    ssim_noninferiority: float = 0.001,
    lpips_noninferiority: float = 0.001,
    max_cost_ratio: float = 1.5,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    split_reports = {}
    for split, targets in TARGETS.items():
        candidate = json.loads(
            (candidate_dir / f"{split}.json").read_text(encoding="utf-8")
        )
        ssr = json.loads(
            (ssr_dir / f"{split}.json").read_text(encoding="utf-8")
        )
        rows = candidate["images"]
        metrics = {}
        for metric in ("psnr", "ssim", "lpips"):
            values = np.asarray(
                [float(row[metric]) for row in rows],
                dtype=np.float64,
            )
            metrics[metric] = {
                "mean": float(values.mean()),
                "bootstrap_ci95": list(
                    _bootstrap_mean_ci(
                        values,
                        samples=bootstrap_samples,
                        generator=generator,
                    )
                ),
                "published_sota": targets[metric],
            }
        candidate_latency = float(
            candidate["runtime"]["average_inference_seconds_per_image"]
        )
        ssr_latency = float(
            ssr["runtime"]["average_inference_seconds_per_image"]
        )
        cost_ratio = candidate_latency / ssr_latency
        psnr_ci_wins = (
            metrics["psnr"]["bootstrap_ci95"][0] > targets["psnr"]
        )
        ssim_pass = (
            metrics["ssim"]["bootstrap_ci95"][0]
            >= targets["ssim"] - ssim_noninferiority
        )
        lpips_pass = (
            metrics["lpips"]["bootstrap_ci95"][1]
            <= targets["lpips"] + lpips_noninferiority
        )
        split_reports[split] = {
            "images": len(rows),
            "metrics": metrics,
            "cost_ratio_vs_ssr": cost_ratio,
            "checks": {
                "psnr_ci_low_above_published_sota": psnr_ci_wins,
                "ssim_ci_noninferior": ssim_pass,
                "lpips_ci_noninferior": lpips_pass,
                "cost_within_budget": cost_ratio <= max_cost_ratio,
            },
            "pass": (
                psnr_ci_wins
                and ssim_pass
                and lpips_pass
                and cost_ratio <= max_cost_ratio
            ),
        }
    macro_psnr = float(
        np.mean(
            [
                report["metrics"]["psnr"]["mean"]
                for report in split_reports.values()
            ]
        )
    )
    macro_average = {
        "candidate_psnr": macro_psnr,
        "reproduced_foundation_psnr": REPRODUCED_FOUNDATION_MACRO_PSNR,
        "delta_vs_reproduced_foundation": (
            macro_psnr - REPRODUCED_FOUNDATION_MACRO_PSNR
        ),
        "beats_reproduced_foundation": (
            macro_psnr > REPRODUCED_FOUNDATION_MACRO_PSNR
        ),
        "published": dict(PUBLISHED_MACRO_AVERAGES),
        "manuscript_target": MANUSCRIPT_MACRO_AVERAGE_TARGET,
        "delta_vs_published": {
            name: macro_psnr - value
            for name, value in PUBLISHED_MACRO_AVERAGES.items()
        },
        "beats_published": {
            name: macro_psnr > value
            for name, value in PUBLISHED_MACRO_AVERAGES.items()
        },
        "meets_manuscript_target": (
            macro_psnr >= MANUSCRIPT_MACRO_AVERAGE_TARGET
        ),
        "splits_used": sorted(split_reports),
        "note": (
            "Unweighted four-split macro average, the quantity the published "
            "tables compare on. This is a point estimate without a CI because "
            "per-image outputs for the published methods are unavailable. "
            "`beats_reproduced_foundation` is the headline check: the frozen "
            "foundation already clears every published row, so clearing them "
            "is not evidence that the contribution works."
        ),
    }
    return {
        "success": all(report["pass"] for report in split_reports.values()),
        "macro_average": macro_average,
        "splits": split_reports,
        "published_source": (
            "MODEM (NeurIPS 2025, arXiv:2505.17581) Table 1 for the per-split "
            "PSNR gate; see the TARGETS comment for the full provenance and "
            "for the SSIM/LPIPS caveat"
        ),
        "ci_scope": (
            "Image-bootstrap CI for the candidate mean compared with a fixed "
            "published scalar. This is not a paired CI because per-image "
            "outputs for the published methods are unavailable."
        ),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "ssim_noninferiority": ssim_noninferiority,
        "lpips_noninferiority": lpips_noninferiority,
        "max_cost_ratio": max_cost_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--ssr-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssim-noninferiority", type=float, default=0.001)
    parser.add_argument("--lpips-noninferiority", type=float, default=0.001)
    parser.add_argument("--max-cost-ratio", type=float, default=1.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = assess(
        args.candidate_dir,
        args.ssr_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        ssim_noninferiority=args.ssim_noninferiority,
        lpips_noninferiority=args.lpips_noninferiority,
        max_cost_ratio=args.max_cost_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
