from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "assess_latent_bfn_sota.py"
)
SPEC = spec_from_file_location("assess_latent_bfn_sota", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TARGETS = MODULE.TARGETS
assess = MODULE.assess


def _write_result(
    path: Path,
    *,
    psnr: float,
    ssim: float,
    lpips: float,
    latency: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "images": [
                {
                    "name": str(index),
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                }
                for index in range(8)
            ],
            "runtime": {
                "average_inference_seconds_per_image": latency,
            },
        }),
        encoding="utf-8",
    )


def test_sota_assessment_requires_all_splits_and_cost_gate(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    ssr = tmp_path / "ssr"
    for split, target in TARGETS.items():
        _write_result(
            candidate / f"{split}.json",
            psnr=target["psnr"] + 0.1,
            ssim=target["ssim"],
            lpips=target["lpips"],
            latency=1.4,
        )
        _write_result(
            ssr / f"{split}.json",
            psnr=target["psnr"],
            ssim=target["ssim"],
            lpips=target["lpips"],
            latency=1.0,
        )
    report = assess(
        candidate,
        ssr,
        bootstrap_samples=100,
    )
    assert report["success"] is True
    assert all(row["pass"] for row in report["splits"].values())

    _write_result(
        candidate / "raindrop.json",
        psnr=TARGETS["raindrop"]["psnr"] + 0.1,
        ssim=TARGETS["raindrop"]["ssim"],
        lpips=TARGETS["raindrop"]["lpips"],
        latency=1.6,
    )
    failed = assess(
        candidate,
        ssr,
        bootstrap_samples=100,
    )
    assert failed["success"] is False
    assert failed["splits"]["raindrop"]["checks"][
        "cost_within_budget"
    ] is False
