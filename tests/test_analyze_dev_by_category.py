"""Tests for scripts/analyze_dev_by_category.py.

Everything here runs on tiny synthetic JSONs: the script is a pure
post-processing tool over evaluation artefacts (zero GPU cost, no dataset).
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_dev_by_category.py"
)
SPEC = spec_from_file_location("analyze_dev_by_category", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

AnalysisError = MODULE.AnalysisError
analyze_dev_by_category = MODULE.analyze_dev_by_category
load_evaluation = MODULE.load_evaluation

PROTOCOL = {
    "test_y_channel": True,
    "crop_border": 0,
    "lpips_network": None,
    "tile": None,
    "overlap": None,
    "stochastic": False,
    "output_quantization": "float32_clamp_0_1",
    "foundation_signature": "same-ssr",
}

# Extension-less relative stems, exactly as sgbformer/data/paired.py:124 writes
# them into every evaluation JSON.
SNOW = [f"snow_{index:05d}" for index in range(4)]
OUTDOOR = [f"im_{index:04d}_s80_a04" for index in range(3)]
RAINDROP = [f"{index}_rain" for index in range(2)]


def evaluation(values: dict[str, tuple[float, float]]) -> dict:
    return {
        "average": {
            "psnr": sum(psnr for psnr, _ in values.values()) / len(values),
            "ssim": sum(ssim for _, ssim in values.values()) / len(values),
        },
        "images": [
            {"name": name, "psnr": psnr, "ssim": ssim}
            for name, (psnr, ssim) in values.items()
        ],
        "runtime": {"images": len(values)},
        "protocol": dict(PROTOCOL),
    }


def make_pair(
    *,
    snow_delta: float = 0.2,
    outdoor_delta: float = -0.4,
    raindrop_delta: float = 1.0,
    extra: dict[str, tuple[float, float]] | None = None,
) -> tuple[dict, dict]:
    baseline: dict[str, tuple[float, float]] = {}
    candidate: dict[str, tuple[float, float]] = {}
    groups = (
        (SNOW, 30.0, snow_delta),
        (OUTDOOR, 28.0, outdoor_delta),
        (RAINDROP, 26.0, raindrop_delta),
    )
    for names, base_psnr, delta in groups:
        for offset, name in enumerate(names):
            psnr = base_psnr + 0.1 * offset
            baseline[name] = (psnr, 0.90)
            candidate[name] = (psnr + delta, 0.90 + delta / 100.0)
    for name, (psnr, ssim) in (extra or {}).items():
        baseline[name] = (psnr, ssim)
        candidate[name] = (psnr, ssim)
    return evaluation(baseline), evaluation(candidate)


def run(baseline: dict, candidate: dict, **kwargs) -> dict:
    options = {
        "bootstrap_samples": 200,
        "seed": 42,
        "reported_delta_psnr": None,
    }
    options.update(kwargs)
    return analyze_dev_by_category(
        {"payload": baseline, "rows": _rows(baseline)},
        {"payload": candidate, "rows": _rows(candidate)},
        **options,
    )


def _rows(payload: dict) -> dict[str, dict]:
    return {row["name"]: dict(row) for row in payload["images"]}


def test_extension_less_stems_are_classified_into_the_three_categories() -> None:
    report = run(*make_pair())
    assert report["classification"]["counts"] == {
        "outdoor_rain": 3,
        "raindrop": 2,
        "snow": 4,
    }
    # paired.py:124 strips the suffix, so every name needed recovery.
    assert report["classification"]["extension_recovered"] == 9
    assert report["classification"]["source"] == (
        "filename_regex_with_extension_recovery"
    )
    # Only the snow stems resolve through the catch-all '.*\.jpe?g'.
    assert report["classification"]["catch_all_recovered"] == 4
    assert any("catch-all" in warning for warning in report["warnings"])


def test_pairs_file_gives_an_exact_split_without_recovery(tmp_path: Path) -> None:
    manifest = tmp_path / "val_pairs.tsv"
    lines = ["# comment", ""]
    lines += [f"{name}.jpg\t{name}.jpg" for name in SNOW]
    lines += [f"{name}.png\t{name}.png" for name in OUTDOOR]
    lines += [f"{name}.png\tclean/{name}.png" for name in RAINDROP]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    name_map = MODULE.read_name_map(manifest)
    assert name_map[SNOW[0]] == f"{SNOW[0]}.jpg"
    report = run(*make_pair(), name_map=name_map)
    assert report["classification"]["source"] == "manifest_filenames"
    assert report["classification"]["counts"] == {
        "outdoor_rain": 3,
        "raindrop": 2,
        "snow": 4,
    }
    assert report["classification"]["extension_recovered"] == 0
    assert not [w for w in report["warnings"] if "catch-all" in w]


def test_per_category_deltas_and_counts() -> None:
    report = run(*make_pair())
    snow = report["categories"]["snow"]
    outdoor = report["categories"]["outdoor_rain"]
    raindrop = report["categories"]["raindrop"]
    assert snow["n"] == 4 and outdoor["n"] == 3 and raindrop["n"] == 2
    assert snow["psnr"]["delta"] == pytest.approx(0.2)
    assert outdoor["psnr"]["delta"] == pytest.approx(-0.4)
    assert raindrop["psnr"]["delta"] == pytest.approx(1.0)
    assert snow["psnr"]["improved"] == 4
    assert outdoor["psnr"]["improved"] == 0
    assert raindrop["psnr"]["improved"] == 2
    assert snow["psnr"]["baseline_mean"] == pytest.approx(30.15)
    assert snow["psnr"]["candidate_mean"] == pytest.approx(30.35)
    low, high = snow["psnr"]["paired_image_bootstrap_ci95"]
    assert low == pytest.approx(0.2) and high == pytest.approx(0.2)


def test_macro_weighting_uses_category_weights_not_image_counts() -> None:
    report = run(*make_pair())
    # 0.5 * 0.2 + 0.25 * (-0.4) + 0.25 * 1.0 = 0.25
    assert report["macro_weighted"]["psnr"]["delta"] == pytest.approx(0.25)
    assert report["macro_weighted"]["weights"] == {
        "snow": 0.50,
        "outdoor_rain": 0.25,
        "raindrop": 0.25,
    }
    assert report["macro_weighted"]["category_sizes"] == {
        "snow": 4,
        "outdoor_rain": 3,
        "raindrop": 2,
    }
    assert "stratified" in report["macro_weighted"]["bootstrap"]
    # The natural mean is image-weighted and therefore different.
    natural = report["natural_mean"]["psnr"]["delta"]
    assert natural == pytest.approx((4 * 0.2 + 3 * -0.4 + 2 * 1.0) / 9)
    assert natural != pytest.approx(0.25)


def test_macro_weighting_is_stratified_so_a_tiny_class_cannot_dominate() -> None:
    report = run(*make_pair(raindrop_delta=1.0))
    low, high = report["macro_weighted"]["psnr"]["paired_image_bootstrap_ci95"]
    # All within-category deltas are constant here, so stratified resampling
    # must collapse the interval onto the point estimate.
    assert low == pytest.approx(0.25) and high == pytest.approx(0.25)


def test_custom_weights_are_normalized_with_a_warning() -> None:
    report = run(
        *make_pair(),
        category_weights={"snow": 1.0, "outdoor_rain": 1.0, "raindrop": 1.0},
    )
    assert report["macro_weighted"]["psnr"]["delta"] == pytest.approx(
        (0.2 - 0.4 + 1.0) / 3
    )
    assert any("normalized" in warning for warning in report["warnings"])


def test_small_category_triggers_a_high_variance_warning() -> None:
    report = run(*make_pair())
    assert any(
        "raindrop" in warning and "High variance" in warning
        for warning in report["warnings"]
    )
    # The threshold is configurable; below it every category warns.
    relaxed = run(*make_pair(), small_category_threshold=1)
    assert not [
        warning for warning in relaxed["warnings"] if "High variance" in warning
    ]


def _manifest(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    manifest = tmp_path / "val_pairs.tsv"
    lines = [f"{name}.jpg\t{name}.jpg" for name in SNOW]
    lines += [f"{name}.png\t{name}.png" for name in OUTDOOR]
    lines += [f"{name}.png\t{name}.png" for name in RAINDROP]
    lines += [
        f"{stem}{suffix}\t{stem}{suffix}" for stem, suffix in (extra or {}).items()
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_illegal_filename_raises_and_is_listed(tmp_path: Path) -> None:
    # An extension the three regexes cannot match at all. This is only
    # detectable with the true filenames, because under suffix recovery snow's
    # '.*\.jpe?g' is a catch-all (documented in the script docstring).
    baseline, candidate = make_pair(extra={"mystery_file": (31.0, 0.9)})
    name_map = MODULE.read_name_map(_manifest(tmp_path, {"mystery_file": ".bmp"}))
    with pytest.raises(AnalysisError) as error:
        run(baseline, candidate, name_map=name_map)
    message = str(error.value)
    assert "matched no category" in message
    assert "mystery_file" in message


def test_name_absent_from_the_pairs_file_raises(tmp_path: Path) -> None:
    baseline, candidate = make_pair(extra={"not_in_manifest": (31.0, 0.9)})
    name_map = MODULE.read_name_map(_manifest(tmp_path))
    with pytest.raises(AnalysisError) as error:
        run(baseline, candidate, name_map=name_map)
    assert "absent from the pairs file" in str(error.value)
    assert "not_in_manifest" in str(error.value)


def test_verbatim_mode_reports_the_extension_less_stems_as_unmatched() -> None:
    with pytest.raises(AnalysisError) as error:
        run(*make_pair(), suffix_recovery=False)
    assert "matched no category" in str(error.value)


def test_ambiguous_filename_raises() -> None:
    baseline, candidate = make_pair()
    with pytest.raises(AnalysisError) as error:
        run(
            baseline,
            candidate,
            category_patterns={"a": r".*", "b": r"snow.*"},
        )
    assert "matched several categories" in str(error.value)


def test_reported_delta_self_check_reports_match_and_mismatch() -> None:
    baseline, candidate = make_pair()
    observed = (4 * 0.2 + 3 * -0.4 + 2 * 1.0) / 9
    matched = run(baseline, candidate, reported_delta_psnr=observed)
    check = matched["natural_mean"]["reproduces_reported_delta"]
    assert check["matches"] is True
    assert check["observed"] == pytest.approx(observed)
    mismatched = run(baseline, candidate, reported_delta_psnr=-0.083995)
    assert mismatched["natural_mean"]["reproduces_reported_delta"]["matches"] is False
    assert any(
        "differs from the reported" in warning
        for warning in mismatched["warnings"]
    )


def test_natural_mean_delegates_to_the_canonical_paired_comparison() -> None:
    report = run(*make_pair())
    assert report["natural_mean"]["source"].endswith("paired_image_comparison")
    assert report["natural_mean"]["n"] == 9
    assert report["natural_mean"]["psnr"]["improved"] == 6


def test_mismatched_image_sets_and_broken_schema_fail_loudly() -> None:
    baseline, candidate = make_pair()
    candidate["images"] = candidate["images"][:-1]
    with pytest.raises(AnalysisError) as error:
        run(baseline, candidate)
    assert "image sets differ" in str(error.value)


def test_protocol_mismatch_requires_an_explicit_opt_in() -> None:
    baseline, candidate = make_pair()
    candidate["protocol"]["test_y_channel"] = False
    with pytest.raises(AnalysisError) as error:
        run(baseline, candidate)
    assert "--allow-protocol-mismatch" in str(error.value)
    report = run(baseline, candidate, allow_protocol_mismatch=True)
    assert "pooled" in report["natural_mean"]["source"]
    assert any("Protocol validation" in w for w in report["warnings"])


def test_embedded_categories_are_cross_checked(tmp_path: Path) -> None:
    baseline, candidate = make_pair()
    for payload in (baseline, candidate):
        for row in payload["images"]:
            row["category"] = "snow"
    with pytest.raises(AnalysisError) as error:
        run(baseline, candidate)
    assert "disagrees with the embedded" in str(error.value)


def test_load_evaluation_accepts_both_repository_schemas(tmp_path: Path) -> None:
    baseline, _ = make_pair()
    path = tmp_path / "foundation.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    payload, schema = load_evaluation(path)
    assert schema["images"] == 9
    assert schema["name_key"] == "name"
    assert schema["metrics"] == ["psnr", "ssim"]
    assert schema["has_protocol"] is True
    assert payload["rows"][SNOW[0]]["psnr"] == pytest.approx(30.0)

    missing = tmp_path / "bad.json"
    missing.write_text(json.dumps({"average": {"psnr": 1.0}}), encoding="utf-8")
    with pytest.raises(AnalysisError) as error:
        load_evaluation(missing)
    assert "non-empty 'images'" in str(error.value)

    nameless = tmp_path / "nameless.json"
    nameless.write_text(
        json.dumps({"images": [{"psnr": 1.0, "ssim": 0.5}]}), encoding="utf-8"
    )
    with pytest.raises(AnalysisError) as error:
        load_evaluation(nameless)
    assert "no recognised image identifier" in str(error.value)


def test_cli_writes_a_json_artifact(tmp_path: Path) -> None:
    baseline, candidate = make_pair()
    baseline_path = tmp_path / "foundation.json"
    candidate_path = tmp_path / "pilot.json"
    output_path = tmp_path / "out" / "by_category.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    code = MODULE.main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(output_path),
            "--bootstrap-samples",
            "200",
            "--seed",
            "42",
            "--no-reported-delta-check",
        ]
    )
    assert code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["images"] == 9
    assert report["macro_weighted"]["psnr"]["delta"] == pytest.approx(0.25)
    assert "reproduces_reported_delta" not in report["natural_mean"]
    assert MODULE.format_report(report).startswith("images=9")


def test_cli_can_fail_on_a_broken_self_check(tmp_path: Path) -> None:
    baseline, candidate = make_pair()
    baseline_path = tmp_path / "foundation.json"
    candidate_path = tmp_path / "pilot.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    code = MODULE.main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(tmp_path / "out.json"),
            "--bootstrap-samples",
            "200",
            "--require-reported-delta",
        ]
    )
    assert code == 2
