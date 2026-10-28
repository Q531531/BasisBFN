from pathlib import Path

import torch

from scripts.diagnose_arm_divergence import analyze


def _checkpoint(path: Path, arm: str, offset: float) -> None:
    names = {
        "latent_flow.global_observer.0.weight": torch.ones(2, 2) + offset,
        (
            "backbone.adapters.decoder_level3."
            "gamma_projection.weight"
        ): torch.full((2, 2), 2e-4 + offset),
        "backbone.adapters.decoder_level3.fusion_gain": torch.tensor(1.0),
        "backbone.model.decoder_level1.weight": torch.ones(2, 2),
    }
    torch.save(
        {
            "format_version": 2,
            "model": names,
            "ema": {
                "updates": 10,
                "shadow": {
                    name: value.clone()
                    for name, value in names.items()
                },
            },
            "epoch": 3,
            "global_step": 4520,
            "best_psnr": 34.7,
            "config": {
                "seed": 42,
                "protocol": {"arm": arm},
                "model": {
                    "latent_bfn": {
                        "update_mode": (
                            "deterministic" if arm == "D" else "bayesian"
                        )
                    }
                },
            },
        },
        path,
    )


def test_diagnosis_uses_trainable_raw_and_ema_states(tmp_path: Path) -> None:
    d = tmp_path / "d.pt"
    b = tmp_path / "b.pt"
    _checkpoint(d, "D", 0.0)
    _checkpoint(b, "B", 0.01)
    report = analyze(
        d,
        b,
        atol=1e-12,
        rtol=1e-10,
        identity_rms_threshold=1e-6,
    )
    assert report["arms_numerically_identical"] is False
    assert report["stop_condition"] == "continue_locked_training"
    assert report["comparison"]["raw"]["trainable_parameters"] == 13
    assert report["comparison"]["ema"]["groups"]["latent_flow"][
        "numerically_equal"
    ] is False
    assert report["adapter_identity"]["raw"]["D"][
        "left_identity_initialisation"
    ] is True


def test_diagnosis_stops_for_identical_trainable_ema(tmp_path: Path) -> None:
    d = tmp_path / "d.pt"
    b = tmp_path / "b.pt"
    _checkpoint(d, "D", 0.0)
    _checkpoint(b, "B", 0.0)
    report = analyze(
        d,
        b,
        atol=1e-12,
        rtol=1e-10,
        identity_rms_threshold=1e-6,
    )
    assert report["arms_numerically_identical"] is True
    assert report["stop_condition"] == "stop_and_debug"
