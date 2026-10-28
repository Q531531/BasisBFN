#!/usr/bin/env python3
r"""Compare matched dev evaluations by weather category without GPU work.

Reports pooled and category-weighted deltas, with paired image bootstrap
within each category. Snow, Outdoor-Rain and RainDrop use weights 0.50,
0.25 and 0.25. These are evaluation weights; training uses natural shuffle.

Pass --pairs-file to recover exact filenames from extension-less image
names. Without a manifest, suffix recovery tries .png, .jpg and .jpeg; a
foreign stem may consequently be classified as snow. Ambiguous matches and
disagreements with recorded per-image categories raise errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgbformer.comparison import paired_image_comparison

# All-Weather filename patterns, also used by evaluate.py.
DEFAULT_CATEGORY_PATTERNS: dict[str, str] = {
    "snow": r".*\.jpe?g",
    "outdoor_rain": r"im_\d+_s\d+_a0[456]\.png",
    "raindrop": r"\d+_rain\.png",
}
# Official four-set macro collapsed onto three dev weather categories.
DEFAULT_CATEGORY_WEIGHTS: dict[str, float] = {
    "snow": 0.50,
    "outdoor_rain": 0.25,
    "raindrop": 0.25,
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
SUFFIX_RECOVERY_ORDER = (".png", ".jpg", ".jpeg")
NAME_KEYS = ("name", "image", "path", "input_path", "file", "filename")
SUPPORTED_METRICS = ("psnr", "ssim", "lpips")
SMALL_CATEGORY_THRESHOLD = 60


class AnalysisError(ValueError):
    """Raised for unusable inputs, so the CLI can fail loudly and precisely."""


def parse_key_value_options(values: list[str], *, option: str) -> dict[str, str]:
    """Parse repeated ``NAME=VALUE`` options (same shape as evaluate.py:28-39)."""

    parsed: dict[str, str] = {}
    for value in values or []:
        name, separator, payload = value.partition("=")
        if not separator or not name or not payload:
            raise AnalysisError(f"{option} must have NAME=VALUE form, got {value!r}")
        if name in parsed:
            raise AnalysisError(f"Duplicate {option} entry: {name}")
        parsed[name] = payload
    return parsed


def compile_patterns(patterns: dict[str, str]) -> dict[str, re.Pattern[str]]:
    if not patterns:
        raise AnalysisError("At least one category pattern is required")
    return {
        name: re.compile(pattern, flags=re.IGNORECASE)
        for name, pattern in patterns.items()
    }


def basename_of(name: str) -> str:
    return PurePosixPath(str(name).replace("\\", "/")).name


def read_name_map(pairs_file: Path) -> dict[str, str]:
    """Map the JSON image key back to the true input filename.

    The manifest is parsed exactly like ``sgbformer/data/paired.py:96-126``:
    tab-separated, ``#`` comments and blank lines skipped, first column is the
    degraded input path relative to ``data.*.input_dir``. The JSON key is
    ``str(Path(column0).with_suffix(""))`` (``paired.py:124``), so that stem is
    used as the lookup key and the value keeps the real extension.
    """

    path = Path(pairs_file).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise AnalysisError(f"{path}: pairs file does not exist") from error
    mapping: dict[str, str] = {}
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split("\t")[0]
        stem = str(PurePosixPath(first.replace("\\", "/")).with_suffix(""))
        filename = PurePosixPath(first.replace("\\", "/")).name
        if stem in mapping and mapping[stem] != filename:
            raise AnalysisError(
                f"{path}:{number}: stem {stem!r} maps to both "
                f"{mapping[stem]!r} and {filename!r}"
            )
        mapping[stem] = filename
    if not mapping:
        raise AnalysisError(f"{path}: contains no usable pair lines")
    return mapping


def classify_basename(
    basename: str,
    patterns: dict[str, re.Pattern[str]],
    *,
    suffix_recovery: bool = True,
) -> tuple[str | None, str, list[str]]:
    """Return ``(category, matched_string, ambiguous_matches)``.

    ``category`` is ``None`` when nothing matched. ``ambiguous_matches`` is
    non-empty when one attempt matched more than one category.
    """

    attempts = [basename]
    has_suffix = PurePosixPath(basename).suffix.lower() in IMAGE_SUFFIXES
    if suffix_recovery and not has_suffix:
        attempts.extend(basename + suffix for suffix in SUFFIX_RECOVERY_ORDER)
    for attempt in attempts:
        matches = [
            category
            for category, pattern in patterns.items()
            if pattern.fullmatch(attempt)
        ]
        if len(matches) == 1:
            return matches[0], attempt, []
        if len(matches) > 1:
            return None, attempt, sorted(matches)
    return None, basename, []


def classify_names(
    names: list[str],
    patterns: dict[str, re.Pattern[str]],
    *,
    name_map: dict[str, str] | None = None,
    suffix_recovery: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Classify every image name, collecting (never swallowing) failures."""

    assignment: dict[str, str] = {}
    unmatched: list[str] = []
    ambiguous: list[dict[str, Any]] = []
    unresolved: list[str] = []
    recovered = 0
    catch_all_recovered = 0
    for name in names:
        if name_map is not None:
            key = str(name)
            resolved = name_map.get(key) or name_map.get(basename_of(key))
            if resolved is None:
                unresolved.append(name)
                continue
            basename = resolved
            attempts_allowed = False
        else:
            basename = basename_of(name)
            attempts_allowed = suffix_recovery
        category, attempt, conflicts = classify_basename(
            basename, patterns, suffix_recovery=attempts_allowed
        )
        if conflicts:
            ambiguous.append(
                {"name": name, "matched_string": attempt, "categories": conflicts}
            )
            continue
        if category is None:
            unmatched.append(name)
            continue
        if attempt != basename:
            recovered += 1
            if not attempt.lower().endswith(SUFFIX_RECOVERY_ORDER[0]):
                catch_all_recovered += 1
        assignment[name] = category
    diagnostics = {
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "unresolved": unresolved,
        "extension_recovered": recovered,
        "catch_all_recovered": catch_all_recovered,
        "suffix_recovery_order": (
            [] if name_map is not None else list(SUFFIX_RECOVERY_ORDER)
        ),
    }
    return assignment, diagnostics


def _preview(items: list[Any], limit: int = 20) -> str:
    head = ", ".join(json.dumps(item, ensure_ascii=True) for item in items[:limit])
    if len(items) > limit:
        head += f", ... (+{len(items) - limit} more)"
    return head


def load_evaluation(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one evaluation JSON and report the schema variant that was found.

    Accepts the schema written by ``evaluate.py:630-710`` and by
    ``evaluate_foundation.py:148-177``; both use
    ``{"average": ..., "images": [{"name", "psnr", "ssim", ...}], ...}``.
    Anything else raises ``AnalysisError`` rather than producing empty output.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AnalysisError(f"{path}: file does not exist") from error
    except json.JSONDecodeError as error:
        raise AnalysisError(f"{path}: not valid JSON ({error})") from error
    if not isinstance(payload, dict):
        raise AnalysisError(f"{path}: JSON root must be an object")
    rows = payload.get("images")
    if not isinstance(rows, list) or not rows:
        raise AnalysisError(
            f"{path}: missing a non-empty 'images' array; expected the schema of "
            "evaluate.py:630-710 or evaluate_foundation.py:148-177, found keys "
            f"{sorted(payload)}"
        )
    name_key = next(
        (key for key in NAME_KEYS if isinstance(rows[0], dict) and key in rows[0]),
        None,
    )
    if name_key is None:
        raise AnalysisError(
            f"{path}: images[0] has no recognised image identifier; tried "
            f"{list(NAME_KEYS)}, found {sorted(rows[0]) if isinstance(rows[0], dict) else type(rows[0])}"
        )
    metrics = [
        metric
        for metric in SUPPORTED_METRICS
        if isinstance(rows[0], dict) and metric in rows[0]
    ]
    if "psnr" not in metrics:
        raise AnalysisError(f"{path}: images[0] carries no 'psnr' value")
    normalized: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AnalysisError(f"{path}: images[{index}] is not an object")
        if name_key not in row:
            raise AnalysisError(f"{path}: images[{index}] has no {name_key!r}")
        key = str(row[name_key])
        if key in normalized:
            raise AnalysisError(f"{path}: duplicate image name {key!r}")
        record: dict[str, Any] = {"name": key}
        for metric in metrics:
            if metric not in row:
                raise AnalysisError(
                    f"{path}: images[{index}] ({key}) has no {metric!r} while "
                    "images[0] does"
                )
            value = float(row[metric])
            if not np.isfinite(value):
                raise AnalysisError(f"{path}: non-finite {metric} for {key!r}")
            record[metric] = value
        if "category" in row:
            record["category"] = str(row["category"])
        normalized[key] = record
    schema = {
        "path": str(path),
        "images": len(normalized),
        "name_key": name_key,
        "metrics": metrics,
        "has_embedded_category": all("category" in row for row in normalized.values()),
        "has_protocol": isinstance(payload.get("protocol"), dict),
        "average": payload.get("average"),
    }
    return {"payload": payload, "rows": normalized}, schema


def _bootstrap_means(
    delta: np.ndarray,
    *,
    samples: int,
    generator: np.random.Generator,
    chunk_elements: int = 2_000_000,
) -> np.ndarray:
    """Bootstrap the mean of a paired delta vector in bounded-memory chunks."""

    count = len(delta)
    if count == 0:
        raise AnalysisError("Cannot bootstrap an empty group")
    out = np.empty(samples, dtype=np.float64)
    chunk = max(1, min(samples, chunk_elements // max(count, 1)))
    for start in range(0, samples, chunk):
        width = min(chunk, samples - start)
        indices = generator.integers(0, count, size=(width, count))
        out[start : start + width] = delta[indices].mean(axis=1)
    return out


def _group_summary(
    names: list[str],
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    metrics: list[str],
    *,
    bootstrap_samples: int,
    generator: np.random.Generator,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(names)}
    for metric in metrics:
        left = np.array([baseline[name][metric] for name in names], dtype=np.float64)
        right = np.array([candidate[name][metric] for name in names], dtype=np.float64)
        delta = right - left
        boot = _bootstrap_means(
            delta, samples=bootstrap_samples, generator=generator
        )
        low, high = np.quantile(boot, [0.025, 0.975])
        direction = -1 if metric == "lpips" else 1
        summary[metric] = {
            "baseline_mean": float(left.mean()),
            "candidate_mean": float(right.mean()),
            "delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "paired_image_bootstrap_ci95": [float(low), float(high)],
            "improved": int((direction * delta > 0).sum()),
            "improved_fraction": float((direction * delta > 0).mean()),
        }
    return summary


def _macro_weighted(
    grouped: dict[str, list[str]],
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    metrics: list[str],
    weights: dict[str, float],
    *,
    bootstrap_samples: int,
    generator: np.random.Generator,
) -> dict[str, Any]:
    """Category-weighted delta with *stratified* (within-category) resampling."""

    missing = [
        category
        for category, weight in weights.items()
        if weight > 0 and not grouped.get(category)
    ]
    if missing:
        raise AnalysisError(
            "Macro weighting needs at least one image per weighted category; "
            f"empty: {sorted(missing)}"
        )
    unknown = sorted(set(weights) - set(grouped))
    if unknown:
        raise AnalysisError(f"Macro weights name unknown categories: {unknown}")
    total = float(sum(weights.values()))
    if total <= 0:
        raise AnalysisError("Macro weights must sum to a positive number")
    normalized = {category: weight / total for category, weight in weights.items()}
    result: dict[str, Any] = {
        "weights": dict(weights),
        "normalized_weights": normalized,
        "bootstrap": "stratified: resample within each category, then reweight",
        "category_sizes": {
            category: len(names)
            for category, names in grouped.items()
            if category in weights
        },
    }
    for metric in metrics:
        aggregate = np.zeros(bootstrap_samples, dtype=np.float64)
        point = 0.0
        for category in sorted(normalized):
            names = grouped[category]
            left = np.array(
                [baseline[name][metric] for name in names], dtype=np.float64
            )
            right = np.array(
                [candidate[name][metric] for name in names], dtype=np.float64
            )
            delta = right - left
            point += normalized[category] * float(delta.mean())
            aggregate += normalized[category] * _bootstrap_means(
                delta, samples=bootstrap_samples, generator=generator
            )
        low, high = np.quantile(aggregate, [0.025, 0.975])
        result[metric] = {
            "delta": float(point),
            "paired_image_bootstrap_ci95": [float(low), float(high)],
        }
    return result


def analyze_dev_by_category(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    *,
    baseline_schema: dict[str, Any] | None = None,
    candidate_schema: dict[str, Any] | None = None,
    category_patterns: dict[str, str] | None = None,
    category_weights: dict[str, float] | None = None,
    name_map: dict[str, str] | None = None,
    bootstrap_samples: int = 10000,
    seed: int = 42,
    small_category_threshold: int = SMALL_CATEGORY_THRESHOLD,
    reported_delta_psnr: float | None = None,
    reported_delta_tolerance: float = 1e-4,
    allow_protocol_mismatch: bool = False,
    suffix_recovery: bool = True,
) -> dict[str, Any]:
    """Per-category, natural-mean and macro-weighted comparison of two JSONs."""

    if bootstrap_samples < 100:
        raise AnalysisError("Use at least 100 bootstrap samples")
    baseline_rows = baseline_payload["rows"]
    candidate_rows = candidate_payload["rows"]
    if baseline_rows.keys() != candidate_rows.keys():
        only_baseline = sorted(set(baseline_rows) - set(candidate_rows))
        only_candidate = sorted(set(candidate_rows) - set(baseline_rows))
        raise AnalysisError(
            "Baseline and candidate image sets differ; "
            f"baseline-only={_preview(only_baseline)} "
            f"candidate-only={_preview(only_candidate)}"
        )
    metrics = [
        metric
        for metric in SUPPORTED_METRICS
        if all(
            metric in rows[name]
            for rows in (baseline_rows, candidate_rows)
            for name in rows
        )
    ]
    if "psnr" not in metrics:
        raise AnalysisError("Both evaluations must provide a per-image psnr")
    names = sorted(baseline_rows)
    warnings: list[str] = []

    patterns = dict(category_patterns or DEFAULT_CATEGORY_PATTERNS)
    compiled = compile_patterns(patterns)
    assignment, diagnostics = classify_names(
        names, compiled, name_map=name_map, suffix_recovery=suffix_recovery
    )
    classification_source = (
        "manifest_filenames"
        if name_map is not None
        else (
            "filename_regex_with_extension_recovery"
            if suffix_recovery
            else "filename_regex_verbatim"
        )
    )
    embedded_complete = all(
        "category" in baseline_rows[name] and "category" in candidate_rows[name]
        for name in names
    )
    if embedded_complete:
        conflicts = [
            name
            for name in names
            if baseline_rows[name]["category"] != candidate_rows[name]["category"]
        ]
        if conflicts:
            raise AnalysisError(
                "Embedded per-image categories disagree between baseline and "
                f"candidate: {_preview(conflicts)}"
            )
    if (
        diagnostics["unmatched"]
        or diagnostics["ambiguous"]
        or diagnostics["unresolved"]
    ):
        if embedded_complete and not diagnostics["unresolved"]:
            classification_source = "embedded_category"
            assignment = {
                name: str(baseline_rows[name]["category"]) for name in names
            }
            warnings.append(
                "Filename regexes failed on "
                f"{len(diagnostics['unmatched'])} unmatched and "
                f"{len(diagnostics['ambiguous'])} ambiguous names; fell back to "
                "the per-image 'category' field written by evaluate.py"
            )
        else:
            details = []
            if diagnostics["unmatched"]:
                details.append(
                    f"{len(diagnostics['unmatched'])} matched no category: "
                    f"{_preview(diagnostics['unmatched'])}"
                )
            if diagnostics["ambiguous"]:
                details.append(
                    f"{len(diagnostics['ambiguous'])} matched several categories: "
                    f"{_preview(diagnostics['ambiguous'])}"
                )
            if diagnostics["unresolved"]:
                details.append(
                    f"{len(diagnostics['unresolved'])} absent from the pairs file: "
                    f"{_preview(diagnostics['unresolved'])}"
                )
            raise AnalysisError(
                "Every image must map to exactly one category (patterns: "
                f"{json.dumps(patterns, ensure_ascii=True)}); " + "; ".join(details)
            )
    elif embedded_complete:
        disagreements = [
            {
                "name": name,
                "regex": assignment[name],
                "embedded": baseline_rows[name]["category"],
            }
            for name in names
            if assignment[name] != baseline_rows[name]["category"]
        ]
        if disagreements:
            raise AnalysisError(
                "Filename regex classification disagrees with the embedded "
                f"'category' field: {_preview(disagreements)}"
            )
        classification_source = "filename_regex_confirmed_by_embedded_category"

    grouped: dict[str, list[str]] = {category: [] for category in patterns}
    for name in names:
        grouped.setdefault(assignment[name], []).append(name)

    if diagnostics["extension_recovered"]:
        warnings.append(
            f"{diagnostics['extension_recovered']}/{len(names)} names carried no "
            "image extension (paired.py:124 stores extension-less stems); the "
            f"suffix recovery order {list(SUFFIX_RECOVERY_ORDER)} was used"
        )
    if diagnostics["catch_all_recovered"]:
        warnings.append(
            f"{diagnostics['catch_all_recovered']}/{len(names)} names were only "
            "classified because the restored suffix made a catch-all pattern "
            "match (with the default set that is snow's '.*\\.jpe?g'), so a "
            "genuinely foreign filename could not be detected; pass "
            "--pairs-file ${SGB_DATA_ROOT}/val_pairs.tsv for an exact split"
        )

    category_generator = np.random.default_rng([seed, 1])
    categories: dict[str, Any] = {}
    for category in sorted(grouped):
        group = grouped[category]
        if not group:
            warnings.append(f"Category {category!r} has no image in this dev set")
            categories[category] = {"n": 0}
            continue
        categories[category] = _group_summary(
            group,
            baseline_rows,
            candidate_rows,
            metrics,
            bootstrap_samples=bootstrap_samples,
            generator=category_generator,
        )
        if len(group) < small_category_threshold:
            warnings.append(
                f"High variance: category {category!r} has only {len(group)} "
                f"images (< {small_category_threshold}); its delta and CI95 are "
                "weakly determined and it dominates the macro-weighted CI width"
            )

    natural_source = "sgbformer.comparison.paired_image_comparison"
    try:
        canonical = paired_image_comparison(
            baseline_payload["payload"],
            candidate_payload["payload"],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        overall = canonical["overall"]
        natural: dict[str, Any] = {"n": overall["images"]}
        for metric, values in overall["metrics"].items():
            natural[metric] = {
                "baseline_mean": values["baseline"],
                "candidate_mean": values["candidate"],
                "delta": values["delta"],
                "median_delta": values["median_delta"],
                "paired_image_bootstrap_ci95": values["paired_image_bootstrap_ci95"],
                "improved": values["improved_images"],
                "improved_fraction": values["improved_image_fraction"],
            }
    except (ValueError, KeyError) as error:
        if not allow_protocol_mismatch:
            raise AnalysisError(
                "paired_image_comparison rejected the pair "
                f"({error}); rerun with --allow-protocol-mismatch to fall back to "
                "a local pooled bootstrap, but then the natural-mean statistic is "
                "no longer protocol-validated"
            ) from error
        natural_source = "local pooled bootstrap (protocol validation skipped)"
        natural = _group_summary(
            names,
            baseline_rows,
            candidate_rows,
            metrics,
            bootstrap_samples=bootstrap_samples,
            generator=np.random.default_rng([seed, 3]),
        )
        warnings.append(
            "Protocol validation was skipped: "
            f"{error}. natural_mean no longer matches the compare_* convention"
        )
    natural["source"] = natural_source
    if reported_delta_psnr is not None:
        observed = float(natural["psnr"]["delta"])
        difference = abs(observed - float(reported_delta_psnr))
        natural["reproduces_reported_delta"] = {
            "metric": "psnr",
            "reported": float(reported_delta_psnr),
            "observed": observed,
            "absolute_difference": difference,
            "tolerance": float(reported_delta_tolerance),
            "matches": bool(difference <= float(reported_delta_tolerance)),
        }
        if not natural["reproduces_reported_delta"]["matches"]:
            warnings.append(
                "natural_mean ΔPSNR "
                f"{observed:+.6f} differs from the reported "
                f"{float(reported_delta_psnr):+.6f} by {difference:.6f} > "
                f"{float(reported_delta_tolerance):g}: either these are not the "
                "expected dev JSONs, or the artefacts changed"
            )

    weights = dict(category_weights or DEFAULT_CATEGORY_WEIGHTS)
    macro = _macro_weighted(
        grouped,
        baseline_rows,
        candidate_rows,
        metrics,
        weights,
        bootstrap_samples=bootstrap_samples,
        generator=np.random.default_rng([seed, 2]),
    )
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        warnings.append(
            f"Macro weights sum to {sum(weights.values()):g}, they were "
            "normalized to 1.0"
        )
    unweighted = sorted(set(grouped) - set(weights))
    if unweighted:
        warnings.append(
            f"Categories excluded from the macro weighting: {unweighted}"
        )

    return {
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        "metrics": metrics,
        "images": len(names),
        "schema": {
            "baseline": baseline_schema or {},
            "candidate": candidate_schema or {},
        },
        "classification": {
            "source": classification_source,
            "patterns": patterns,
            "counts": {
                category: len(group) for category, group in sorted(grouped.items())
            },
            "extension_recovered": diagnostics["extension_recovered"],
            "catch_all_recovered": diagnostics["catch_all_recovered"],
            "suffix_recovery_order": diagnostics["suffix_recovery_order"],
        },
        "categories": categories,
        "natural_mean": natural,
        "macro_weighted": macro,
        "warnings": warnings,
        "limitation": (
            "Image bootstrap conditions on these two checkpoints; it does not "
            "measure training-seed variance. The macro weighting reweights a "
            "natural-distribution dev subset and cannot create images that are "
            "not there, so a small category keeps its sampling noise."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--category-pattern",
        action="append",
        default=[],
        metavar="NAME=REGEX",
        help=(
            "Repeatable override of the default classifier "
            f"({json.dumps(DEFAULT_CATEGORY_PATTERNS)}). Any use replaces the "
            "whole default set."
        ),
    )
    parser.add_argument(
        "--category-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help=(
            "Repeatable override of the macro weights "
            f"({json.dumps(DEFAULT_CATEGORY_WEIGHTS)})."
        ),
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        help=(
            "Optional manifest (e.g. ${SGB_DATA_ROOT}/val_pairs.tsv) used to "
            "restore the true input filenames, because paired.py:124 stores "
            "extension-less stems in the JSON. Strongly recommended: it makes "
            "the category split exact instead of suffix-recovery based."
        ),
    )
    parser.add_argument(
        "--small-category-threshold",
        type=int,
        default=SMALL_CATEGORY_THRESHOLD,
        help="Emit a high-variance warning below this image count (default 60).",
    )
    parser.add_argument(
        "--reported-delta-psnr",
        type=float,
        default=None,
        help=(
            "Natural-mean ΔPSNR that the script should reproduce as a self-check "
            "(disabled unless an expected delta is explicitly provided)."
        ),
    )
    parser.add_argument(
        "--reported-delta-tolerance", type=float, default=1e-4
    )
    parser.add_argument(
        "--require-reported-delta",
        action="store_true",
        help="Exit non-zero when the natural-mean self-check does not reproduce.",
    )
    parser.add_argument(
        "--no-reported-delta-check",
        action="store_true",
        help="Skip the natural-mean self-check entirely.",
    )
    parser.add_argument(
        "--allow-protocol-mismatch",
        action="store_true",
        help=(
            "Fall back to a local pooled bootstrap when paired_image_comparison "
            "rejects the protocol pair."
        ),
    )
    parser.add_argument(
        "--no-suffix-recovery",
        action="store_true",
        help=(
            "Match the stored names verbatim. Use only when the JSON keeps file "
            "extensions; paired.py:124 normally strips them."
        ),
    )
    return parser.parse_args(argv)


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"images={report['images']} metrics={','.join(report['metrics'])} "
        f"bootstrap={report['bootstrap_samples']} seed={report['seed']} "
        f"classification={report['classification']['source']}",
        "category            n   baseline    candidate     dPSNR   "
        "CI95(dPSNR)                improved",
    ]
    for category, values in report["categories"].items():
        if not values.get("n"):
            lines.append(f"{category:<18} 0   (absent)")
            continue
        psnr = values["psnr"]
        low, high = psnr["paired_image_bootstrap_ci95"]
        lines.append(
            f"{category:<18}{values['n']:>4} {psnr['baseline_mean']:>10.5f} "
            f"{psnr['candidate_mean']:>12.5f} {psnr['delta']:>+9.5f}  "
            f"[{low:+.5f}, {high:+.5f}]  "
            f"{psnr['improved']}/{values['n']}"
        )
    natural = report["natural_mean"]["psnr"]
    low, high = natural["paired_image_bootstrap_ci95"]
    lines.append(
        f"{'natural_mean':<18}{report['natural_mean']['n']:>4} "
        f"{natural['baseline_mean']:>10.5f} {natural['candidate_mean']:>12.5f} "
        f"{natural['delta']:>+9.5f}  [{low:+.5f}, {high:+.5f}]  "
        f"{natural['improved']}/{report['natural_mean']['n']}"
    )
    macro = report["macro_weighted"]["psnr"]
    low, high = macro["paired_image_bootstrap_ci95"]
    weights = ", ".join(
        f"{key} {value:g}"
        for key, value in report["macro_weighted"]["weights"].items()
    )
    lines.append(
        f"{'macro_weighted':<18}   - {'-':>10} {'-':>12} {macro['delta']:>+9.5f}  "
        f"[{low:+.5f}, {high:+.5f}]  ({weights})"
    )
    check = report["natural_mean"].get("reproduces_reported_delta")
    if check is not None:
        lines.append(
            f"self-check reproduces_reported_delta={check['matches']} "
            f"(reported {check['reported']:+.6f}, observed {check['observed']:+.6f}, "
            f"|diff|={check['absolute_difference']:.6f}, tol={check['tolerance']:g})"
        )
    for warning in report["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.append(report["limitation"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    patterns = parse_key_value_options(
        args.category_pattern, option="--category-pattern"
    ) or dict(DEFAULT_CATEGORY_PATTERNS)
    raw_weights = parse_key_value_options(
        args.category_weight, option="--category-weight"
    )
    weights = (
        {name: float(value) for name, value in raw_weights.items()}
        if raw_weights
        else dict(DEFAULT_CATEGORY_WEIGHTS)
    )
    baseline, baseline_schema = load_evaluation(args.baseline)
    candidate, candidate_schema = load_evaluation(args.candidate)
    name_map = read_name_map(args.pairs_file) if args.pairs_file else None
    report = analyze_dev_by_category(
        baseline,
        candidate,
        baseline_schema=baseline_schema,
        candidate_schema=candidate_schema,
        category_patterns=patterns,
        category_weights=weights,
        name_map=name_map,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        small_category_threshold=args.small_category_threshold,
        reported_delta_psnr=(
            None if args.no_reported_delta_check else args.reported_delta_psnr
        ),
        reported_delta_tolerance=args.reported_delta_tolerance,
        allow_protocol_mismatch=args.allow_protocol_mismatch,
        suffix_recovery=not args.no_suffix_recovery,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(format_report(report))
    check = report["natural_mean"].get("reproduces_reported_delta")
    if args.require_reported_delta and (check is None or not check["matches"]):
        print(
            "FAIL: --require-reported-delta requires an explicit expected delta "
            "and a matching self-check",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
