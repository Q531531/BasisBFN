"""Training schedule helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TrainingStage:
    name: str
    start_epoch: int
    end_epoch: int
    loss: dict[str, float | str]


def resolve_training_stages(config: dict[str, Any]) -> list[TrainingStage]:
    total_epochs = int(config.get("epochs", 1))
    raw_stages = config.get("stages")
    if not raw_stages:
        return [TrainingStage("joint", 0, total_epochs, {})]

    stages = []
    start = 0
    for raw in raw_stages:
        name = str(raw["name"]).lower()
        if name not in {"structure", "mean", "residual", "bfn", "joint"}:
            raise ValueError(f"Unsupported training stage: {name}")
        duration = int(raw["epochs"])
        if duration < 1:
            raise ValueError(f"Stage {name} must contain at least one epoch")
        stages.append(
            TrainingStage(
                name=name,
                start_epoch=start,
                end_epoch=start + duration,
                loss={
                    key: str(value)
                    if key in {"structural_loss_type", "final_loss_type"}
                    else float(value)
                    for key, value in raw.get("loss", {}).items()
                },
            )
        )
        start += duration
    if start != total_epochs:
        raise ValueError(
            f"Stage epochs sum to {start}, but training.epochs is {total_epochs}"
        )
    return stages


def stage_for_epoch(stages: list[TrainingStage], epoch: int) -> TrainingStage:
    for stage in stages:
        if stage.start_epoch <= epoch < stage.end_epoch:
            return stage
    raise IndexError(f"Epoch {epoch} is outside the configured stages")


def stage_loss_config(
    base: dict[str, Any], stage: TrainingStage
) -> dict[str, Any]:
    merged = dict(base)
    merged.update(stage.loss)
    return merged


def validate_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    for section in ("seed", "model", "data", "loss", "validation", "training"):
        default = 42 if section == "seed" else {}
        previous = saved.get(section, default)
        requested = current.get(section, default)
        if section == "training":
            administrative = {"output_dir", "log_interval", "save_interval"}
            previous = {k: v for k, v in previous.items() if k not in administrative}
            requested = {k: v for k, v in requested.items() if k not in administrative}
        if previous != requested:
            raise ValueError(
                f"Resume changes {section}; use the original experiment config. "
                "Changed models or budgets require a new run, not --resume."
            )


def cosine_warmup_factor(
    epoch: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    minimum_ratio: float,
) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return max((epoch + 1) / warmup_epochs, 1e-8)
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    progress = min(max((epoch - warmup_epochs) / cosine_epochs, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> torch.optim.lr_scheduler.LambdaLR:
    minimum_lr = float(config.get("minimum_learning_rate", 1e-6))
    minimum_ratio = config.get("minimum_learning_rate_ratio")
    if minimum_ratio is not None:
        minimum_ratio = float(minimum_ratio)
        if not 0 <= minimum_ratio <= 1:
            raise ValueError(
                "minimum_learning_rate_ratio must lie in [0, 1]"
            )
    base_lr = float(config.get("learning_rate", 2e-4))
    mode = str(config.get("schedule_minimum_mode", "legacy_ratio"))
    if mode not in {"legacy_ratio", "per_group"}:
        raise ValueError("schedule_minimum_mode must be legacy_ratio or per_group")
    schedules = []
    for group in optimizer.param_groups:
        initial_lr = float(group["lr"]) if mode == "per_group" else base_lr
        if initial_lr <= 0:
            raise ValueError("Initial learning rate must be positive")
        if minimum_ratio is None:
            if not 0 <= minimum_lr <= initial_lr:
                raise ValueError(
                    "Learning rate floor must be between zero and initial LR"
                )
            ratio = minimum_lr / initial_lr
        else:
            ratio = minimum_ratio
        schedules.append(
            lambda epoch, ratio=ratio: cosine_warmup_factor(
                epoch,
                total_epochs=int(config.get("epochs", 300)),
                warmup_epochs=int(config.get("warmup_epochs", 0)),
                minimum_ratio=ratio,
            )
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)


def clip_training_gradients(
    model: torch.nn.Module,
    max_norm: float,
    *,
    separate_risk: bool = False,
    error_if_nonfinite: bool = True,
) -> None:
    groups: dict[bool, list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            is_risk = separate_risk and "risk_head" in name.split(".")
            groups.setdefault(is_risk, []).append(parameter)
    for parameters in groups.values():
        torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=error_if_nonfinite
        )
