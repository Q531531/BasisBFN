import copy

import pytest

from sgbformer.comparison import (
    foundation_signature,
    paired_comparison,
    paired_image_comparison,
)


def results(delta=0.0, count=4):
    return {
        "protocol": {
            "test_y_channel": True, "crop_border": 0, "lpips_network": "alex",
            "tile": None, "stochastic": False,
        },
        "images": [
            {"name": str(i), "psnr": 30 + delta, "ssim": 0.9 + delta / 100,
             "lpips": 0.1 - delta / 100}
            for i in range(count)
        ],
    }


def test_comparison_uses_macro_not_image_weighted_average():
    baseline = {"a": results(count=4), "b": results(count=20)}
    candidate = {"a": results(0.2, count=4), "b": results(0.4, count=20)}
    result = paired_comparison(baseline, candidate, bootstrap_samples=100)
    assert result["macro"]["psnr"]["delta"] == pytest.approx(0.3)
    assert result["macro"]["psnr"]["paired_image_bootstrap_ci95"] == pytest.approx([0.3, 0.3])
    assert result["datasets"]["a"]["lpips"]["improved_image_fraction"] == 1


@pytest.mark.parametrize("change", ["protocol", "names", "duplicate", "metric"])
def test_comparison_rejects_invalid_inputs(change):
    baseline = {"a": results()}
    candidate = copy.deepcopy(baseline)
    if change == "protocol":
        candidate["a"]["protocol"]["crop_border"] = 4
    elif change == "names":
        candidate["a"]["images"][0]["name"] = "other"
    elif change == "duplicate":
        candidate["a"]["images"].append(candidate["a"]["images"][0])
    else:
        del candidate["a"]["images"][0]["lpips"]
    with pytest.raises(ValueError):
        paired_comparison(baseline, candidate, bootstrap_samples=100)


def test_tiled_comparison_requires_matching_overlap():
    baseline, candidate = {"a": results()}, {"a": results(0.1)}
    for result in (baseline, candidate):
        result["a"]["protocol"].update(tile=512, overlap=32)
    paired_comparison(baseline, candidate, bootstrap_samples=100)
    candidate["a"]["protocol"]["overlap"] = 64
    with pytest.raises(ValueError, match="overlap"):
        paired_comparison(baseline, candidate, bootstrap_samples=100)
    del candidate["a"]["protocol"]["overlap"]
    with pytest.raises(ValueError, match="overlap"):
        paired_comparison(baseline, candidate, bootstrap_samples=100)


def test_comparison_rejects_mismatched_or_missing_quantization():
    baseline, candidate = {"a": results()}, {"a": results(0.1)}
    baseline["a"]["protocol"]["output_quantization"] = "float32_clamp_0_1"
    with pytest.raises(ValueError, match="output_quantization"):
        paired_comparison(baseline, candidate, bootstrap_samples=100)
    candidate["a"]["protocol"]["output_quantization"] = "uint8_round"
    with pytest.raises(ValueError, match="output_quantization"):
        paired_comparison(baseline, candidate, bootstrap_samples=100)
    candidate["a"]["protocol"]["output_quantization"] = "float32_clamp_0_1"
    paired_comparison(baseline, candidate, bootstrap_samples=100)


def test_single_evaluation_comparison_reports_bootstrap_and_categories():
    baseline = results(count=4)
    candidate = results(delta=0.2, count=4)
    baseline["protocol"]["foundation_signature"] = "same-foundation"
    candidate["protocol"]["foundation_signature"] = "same-foundation"
    for index, row in enumerate(baseline["images"]):
        row["category"] = "snow" if index < 2 else "raindrop"
    for index, row in enumerate(candidate["images"]):
        row["category"] = "snow" if index < 2 else "raindrop"

    result = paired_image_comparison(
        baseline,
        candidate,
        bootstrap_samples=100,
    )

    psnr = result["overall"]["metrics"]["psnr"]
    assert psnr["delta"] == pytest.approx(0.2)
    assert psnr["median_delta"] == pytest.approx(0.2)
    assert psnr["improved_images"] == 4
    assert psnr["paired_image_bootstrap_ci95"] == pytest.approx([0.2, 0.2])
    assert result["categories"]["snow"]["images"] == 2
    assert result["categories"]["raindrop"]["metrics"]["lpips"][
        "improved_images"
    ] == 2


def test_single_evaluation_comparison_rejects_foundation_mismatch():
    baseline = results()
    candidate = results(delta=0.1)
    baseline["protocol"]["foundation_signature"] = "foundation-a"
    candidate["protocol"]["foundation_signature"] = "foundation-b"
    with pytest.raises(ValueError, match="frozen foundation"):
        paired_image_comparison(baseline, candidate, bootstrap_samples=100)


def test_foundation_signature_ignores_training_policy_and_adapters():
    base = {
        "model": {
            "foundation": {
                "type": "external_ssr",
                "repo_path": "/srv/SSR",
                "checkpoint": "/srv/net_g_best.pth",
                "required_multiple": 8,
                "architecture": {"dim": 48, "num_blocks": [3, 3, 6]},
                "freeze": True,
                "frozen_modules_eval": True,
                "trainable_modules": [],
            }
        }
    }
    fine_tuned = copy.deepcopy(base)
    fine_tuned["model"]["foundation"].update({
        "trainable_modules": ["decoder_level1", "output"],
        "feature_adapters": {
            "decoder_level3": {
                "type": "state_spectral",
                "channels": 192,
            }
        },
    })
    signature = foundation_signature(base)
    assert signature == foundation_signature(fine_tuned)

    frozen_result = results()
    fine_tuned_result = results(delta=0.1)
    frozen_result["protocol"]["foundation_signature"] = signature
    fine_tuned_result["protocol"]["foundation_signature"] = (
        foundation_signature(fine_tuned)
    )
    comparison = paired_image_comparison(
        frozen_result,
        fine_tuned_result,
        bootstrap_samples=100,
    )
    assert comparison["overall"]["metrics"]["psnr"]["delta"] == pytest.approx(
        0.1
    )
