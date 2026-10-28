import copy

import pytest
import torch
from torch import nn

from sgbformer.training import (
    build_scheduler,
    clip_training_gradients,
    cosine_warmup_factor,
    resolve_training_stages,
    stage_for_epoch,
    stage_loss_config,
    validate_resume_config,
)
from train import build_optimizer, validate


def test_staged_schedule_boundaries_and_loss_override() -> None:
    stages = resolve_training_stages(
        {
            "epochs": 10,
            "stages": [
                {"name": "structure", "epochs": 5},
                {
                    "name": "bfn",
                    "epochs": 3,
                    "loss": {"structural_weight": 0, "bfn_weight": 0.01},
                },
                {"name": "joint", "epochs": 2},
            ],
        }
    )
    assert stage_for_epoch(stages, 0).name == "structure"
    assert stage_for_epoch(stages, 5).name == "bfn"
    assert stage_for_epoch(stages, 9).name == "joint"
    loss = stage_loss_config({"structural_weight": 1.0}, stages[1])
    assert loss == {"structural_weight": 0.0, "bfn_weight": 0.01}


def test_stage_epoch_total_is_validated() -> None:
    with pytest.raises(ValueError, match="sum"):
        resolve_training_stages(
            {
                "epochs": 4,
                "stages": [{"name": "structure", "epochs": 3}],
            }
        )


def test_cosine_warmup_reaches_minimum() -> None:
    assert cosine_warmup_factor(
        0, total_epochs=10, warmup_epochs=2, minimum_ratio=0.01
    ) == 0.5
    assert cosine_warmup_factor(
        10, total_epochs=10, warmup_epochs=2, minimum_ratio=0.01
    ) == pytest.approx(0.01)


def test_optimizer_uses_longest_matching_parameter_prefix() -> None:
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.model = nn.Linear(2, 2)
    model.backbone.adapters = nn.ModuleDict({"a": nn.Linear(2, 2)})
    optimizer = build_optimizer(
        model,
        {
            "learning_rate": 1e-3,
            "parameter_lrs": {
                "backbone": 1e-4,
                "backbone.model": 1e-5,
            },
        },
    )
    assert sorted(group["lr"] for group in optimizer.param_groups) == [
        1e-5,
        1e-4,
    ]
    assert sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    ) == sum(parameter.numel() for parameter in model.parameters())


def test_per_group_schedule_matches_ssr_control_for_every_epoch() -> None:
    config = {
        "epochs": 6, "warmup_epochs": 1,
        "minimum_learning_rate": 1e-6, "schedule_minimum_mode": "per_group",
    }
    control = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=1e-5)
    proposed = torch.optim.AdamW([
        {"params": [nn.Parameter(torch.ones(1))], "lr": 1e-5},
        {"params": [nn.Parameter(torch.ones(1))], "lr": 1e-4},
    ])
    control_schedule = build_scheduler(control, {**config, "learning_rate": 1e-5})
    proposed_schedule = build_scheduler(proposed, {**config, "learning_rate": 1e-4})
    for _ in range(6):
        assert control.param_groups[0]["lr"] == proposed.param_groups[0]["lr"]
        control.step()
        proposed.step()
        control_schedule.step()
        proposed_schedule.step()
    assert [group["lr"] for group in proposed.param_groups] == pytest.approx([1e-6, 1e-6])


def test_per_group_ratio_decays_each_group_to_its_own_tenth() -> None:
    optimizer = torch.optim.AdamW([
        {"params": [nn.Parameter(torch.ones(1))], "lr": 2e-5},
        {"params": [nn.Parameter(torch.ones(1))], "lr": 2e-4},
    ])
    schedule = build_scheduler(
        optimizer,
        {
            "epochs": 6,
            "warmup_epochs": 1,
            "minimum_learning_rate_ratio": 0.1,
            "schedule_minimum_mode": "per_group",
        },
    )
    for _ in range(6):
        optimizer.step()
        schedule.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [2e-6, 2e-5]
    )


def test_optimizer_prefix_does_not_match_sibling_name() -> None:
    model = nn.Module()
    model.head = nn.Linear(2, 2)
    model.head_extra = nn.Linear(2, 2)
    optimizer = build_optimizer(
        model, {"learning_rate": 1e-3, "parameter_lrs": {"head": 1e-5}}
    )
    rates = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
    assert rates[id(model.head.weight)] == 1e-5
    assert rates[id(model.head_extra.weight)] == 1e-3


def test_risk_gradient_clipping_does_not_scale_restoration_gradients() -> None:
    model = nn.Module()
    model.mean_head = nn.Linear(1, 1, bias=False)
    model.risk_head = nn.Linear(1, 1, bias=False)
    model.mean_head.weight.grad = torch.full_like(model.mean_head.weight, 0.2)
    model.risk_head.weight.grad = torch.full_like(model.risk_head.weight, 100)
    clip_training_gradients(model, 1, separate_risk=True)
    assert model.mean_head.weight.grad.item() == pytest.approx(0.2)
    assert model.risk_head.weight.grad.item() == pytest.approx(1, abs=1e-6)


def test_resume_allows_new_output_but_rejects_changed_experiment() -> None:
    saved = {
        "seed": 42, "model": {"mode": "clamp"},
        "training": {"output_dir": "old", "epochs": 6, "learning_rate": 1e-4},
    }
    current = copy.deepcopy(saved)
    current["training"].update(output_dir="new", log_interval=5)
    validate_resume_config(saved, current)
    for section, key, value in (
        ("model", "mode", "smooth"), ("training", "epochs", 12),
        ("training", "learning_rate", 1e-5),
    ):
        changed = copy.deepcopy(current)
        changed[section][key] = value
        with pytest.raises(ValueError, match=f"Resume changes {section}"):
            validate_resume_config(saved, changed)


def test_validation_reports_mean_correction_diagnostics() -> None:
    class PerfectMeanModel(nn.Module):
        def restore(self, degraded, **kwargs):
            del kwargs
            foundation = torch.zeros_like(degraded)
            return {
                "restored": degraded,
                "structural": degraded,
                "foundation": foundation,
            }

    degraded = torch.full((1, 3, 8, 8), 0.4)
    target = torch.full((1, 3, 8, 8), 0.5)
    metrics = validate(
        PerfectMeanModel(),
        [{"degraded": degraded, "target": target}],
        torch.device("cpu"),
        steps=1,
    )
    assert metrics["mean_correction_cosine"] == pytest.approx(1)
    assert metrics["mean_correction_rms_ratio"] == pytest.approx(0.8)
    assert metrics["mean_improved_fraction"] == pytest.approx(1)
    assert metrics["structural_aggregate_mse"] == pytest.approx(0.01)
    assert metrics["foundation_aggregate_mse"] == pytest.approx(0.25)

    class PerfectBfnModel(nn.Module):
        def restore(self, degraded, **kwargs):
            del kwargs
            structural = torch.zeros_like(degraded)
            return {
                "restored": degraded,
                "structural": structural,
                "foundation": structural,
            }

    bfn_metrics = validate(
        PerfectBfnModel(),
        [{"degraded": degraded, "target": target}],
        torch.device("cpu"),
        steps=1,
    )
    assert bfn_metrics["bfn_correction_cosine"] == pytest.approx(1)
    assert bfn_metrics["bfn_correction_rms_ratio"] == pytest.approx(0.8)
    assert bfn_metrics["bfn_improved_fraction"] == pytest.approx(1)
    assert bfn_metrics["output_aggregate_mse"] == pytest.approx(0.01)
