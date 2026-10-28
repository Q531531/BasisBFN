#!/usr/bin/env python3
"""Run and archive the four frozen-main end-to-end preflight checks."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sgbformer.config import load_config
from sgbformer.models import build_model
from sgbformer.models.adapters import LatentConditionedFiLMSpectralAdapter
from sgbformer.models.bfn import DegradationLatentBFN, degradation_latent_targets
from sgbformer.models.foundation import ExternalSSRFoundation
from sgbformer.models.model import SGBFormer
from sgbformer.protocol import (
    foundation_artifact_fingerprint,
    implementation_fingerprint,
    sha256_file,
    validate_frozen_main_contract,
)
from sgbformer.utils import parameter_count, seed_everything
from train import build_dataset, build_optimizer, latent_category_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-d", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalize_arm_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["protocol"].pop("arm")
    normalized["protocol"].pop("arm_description")
    normalized["model"]["latent_bfn"].pop("update_mode")
    normalized["training"].pop("output_dir")
    normalized["training"].pop("budget_arm")
    return normalized


def grad_norm(module: torch.nn.Module) -> float:
    energy = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return energy**0.5


def projection_grad_norm(model: SGBFormer) -> float:
    energy = 0.0
    for adapter in model.backbone.adapters.values():
        if not isinstance(adapter, LatentConditionedFiLMSpectralAdapter):
            continue
        for projection in (
            adapter.gamma_projection,
            adapter.beta_projection,
            adapter.spectral_projection,
        ):
            for parameter in projection.parameters():
                if parameter.grad is not None:
                    energy += float(
                        parameter.grad.detach().float().square().sum()
                    )
    return energy**0.5


def source_site(owner: type[Any], method_name: str) -> dict[str, Any]:
    method = getattr(owner, method_name)
    lines, first = inspect.getsourcelines(method)
    source = "".join(lines)
    return {
        "symbol": f"{owner.__name__}.{method_name}",
        "file": str(Path(inspect.getsourcefile(method) or "").resolve()),
        "start_line": first,
        "end_line": first + len(lines) - 1,
        "detach_occurrences": source.count(".detach("),
        "no_grad_occurrences": source.count("no_grad"),
        "requires_grad_occurrences": source.count("requires_grad"),
    }


def assert_initialization(model: SGBFormer) -> dict[str, Any]:
    allowed_suffixes = ("decoder_level1", "decoder_level2", "output")
    trainable_ssr = [
        name
        for name, parameter in model.backbone.model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_ssr or any(
        not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in allowed_suffixes
        )
        for name in trainable_ssr
    ):
        raise AssertionError("Trainable SSR parameters escape the frozen suffix")
    for prefix in allowed_suffixes:
        parameters = list(
            model.backbone.model.get_submodule(prefix).parameters()
        )
        if not parameters or not all(
            parameter.requires_grad for parameter in parameters
        ):
            raise AssertionError(f"SSR suffix module is not trainable: {prefix}")
    if not model.backbone.foundation_trainable:
        raise AssertionError("Trainable SSR suffix must keep the tensor path live")

    latent_adapters = []
    paper_gates = []
    for module_name, key in model.backbone._adapter_paths.items():
        adapter = model.backbone.adapters[key]
        if isinstance(adapter, LatentConditionedFiLMSpectralAdapter):
            if adapter.fusion_gain.item() != 1.0:
                raise AssertionError("Latent fusion gain must initialize to one")
            zero_counts = {}
            for name in (
                "gamma_projection",
                "beta_projection",
                "spectral_projection",
            ):
                projection = getattr(adapter, name)
                nonzero = int(torch.count_nonzero(projection.weight))
                nonzero += int(torch.count_nonzero(projection.bias))
                if nonzero:
                    raise AssertionError(f"{module_name}.{name} is not zero initialized")
                zero_counts[name] = nonzero
            latent_adapters.append(
                {
                    "module": module_name,
                    "fusion_gain": adapter.fusion_gain.item(),
                    "projection_nonzero_counts": zero_counts,
                }
            )
        if getattr(adapter, "is_paper_spectral_gate", False):
            paper_gates.append(module_name)
    if paper_gates != ["encoder_level3"]:
        raise AssertionError("Expected one paper gate at encoder_level3")
    if sorted(item["module"] for item in latent_adapters) != [
        "decoder_level2",
        "decoder_level3",
    ]:
        raise AssertionError("Expected latent adapters at decoder levels 3 and 2")
    return {
        "ssr_trainable_modules": list(allowed_suffixes),
        "ssr_trainable_parameter_names": trainable_ssr,
        "foundation_trainable": model.backbone.foundation_trainable,
        "paper_spectral_gate_modules": paper_gates,
        "latent_adapters": latent_adapters,
    }


def check_parameter_contract(
    models: dict[str, SGBFormer],
) -> dict[str, Any]:
    signatures = {}
    for arm, model in models.items():
        signatures[arm] = {
            name: {
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in model.named_parameters()
        }
    if signatures["D"] != signatures["B"]:
        raise AssertionError("D/B named parameter signatures differ")
    for name, parameter_d in models["D"].named_parameters():
        parameter_b = dict(models["B"].named_parameters())[name]
        if not torch.equal(parameter_d, parameter_b):
            raise AssertionError(f"D/B initialization differs at {name}")

    counts = {
        arm: parameter_count(model)
        for arm, model in models.items()
    }
    difference_fraction = abs(counts["B"] - counts["D"]) / max(counts["D"], 1)
    if difference_fraction >= 0.05:
        raise AssertionError("D/B trainable parameter difference is not below 5%")
    return {
        "trainable_parameters": counts,
        "difference_fraction": difference_fraction,
        "named_parameter_signature_equal": True,
        "initial_parameter_values_equal": True,
    }


def check_gradient_path(
    model: SGBFormer,
    config: dict[str, Any],
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.to(device).train()
    optimizer = build_optimizer(model, config["training"])
    optimizer_groups = {
        float(group["lr"]): sum(
            parameter.numel()
            for parameter in group["params"]
        )
        for group in optimizer.param_groups
    }
    if set(optimizer_groups) != {2e-5, 2e-4}:
        raise AssertionError("Frozen-main optimizer LR groups changed")
    expected_suffix_numel = sum(
        parameter.numel()
        for module_name in ("decoder_level1", "decoder_level2", "output")
        for parameter in model.backbone.model.get_submodule(
            module_name
        ).parameters()
    )
    if optimizer_groups[2e-5] != expected_suffix_numel:
        raise AssertionError(
            "The 2e-5 optimizer group does not exactly match the SSR suffix"
        )
    degraded = batch["degraded"].to(device)
    target = batch["target"].to(device)
    category = latent_category_tensor(
        batch["input_path"],
        config["model"]["latent_bfn"]["category_patterns"],
        device,
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        first_outputs = model(
            degraded,
            target,
            training_stage="joint",
            category=category,
        )
        first_image_loss = F.mse_loss(
            first_outputs["structural"].float(),
            target.float(),
        )
    first_image_loss.backward()
    first_projection = projection_grad_norm(model)
    first_latent = grad_norm(model.latent_flow)
    if first_projection <= 0:
        raise AssertionError("Image loss did not reach zero-initialized projections")
    if first_latent != 0:
        raise AssertionError(
            "Zero output projections should block the first latent gradient"
        )
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        second_outputs = model(
            degraded,
            target,
            training_stage="joint",
            category=category,
        )
        second_image_loss = F.mse_loss(
            second_outputs["structural"].float(),
            target.float(),
        )
    second_image_loss.backward()
    second_latent = grad_norm(model.latent_flow)
    second_encoder = grad_norm(model.latent_flow.degradation_encoder)
    if second_latent <= 0 or second_encoder <= 0:
        raise AssertionError(
            "Image loss did not reach the upstream latent branch after one step"
        )
    if int(second_outputs["decoder_calls"]) != 1:
        raise AssertionError("Frozen-main training path must decode exactly once")
    optimizer.zero_grad(set_to_none=True)
    return {
        "auxiliary_loss_weights": {
            "bfn_weight": 0.0,
            "latent_category_weight": 0.0,
        },
        "optimizer_parameter_groups": optimizer_groups,
        "expected_ssr_suffix_parameters_at_2e-5": expected_suffix_numel,
        "first_backward": {
            "image_loss": float(first_image_loss.detach()),
            "adapter_output_projection_grad_norm": first_projection,
            "latent_flow_grad_norm": first_latent,
            "expected_zero_latent_reason": (
                "zero-initialized adapter output projections block upstream "
                "image gradients before the first optimizer step"
            ),
        },
        "after_one_optimizer_step": {
            "image_loss": float(second_image_loss.detach()),
            "latent_flow_grad_norm": second_latent,
            "degradation_encoder_grad_norm": second_encoder,
            "decoder_calls": 1,
        },
    }


@torch.no_grad()
def check_alpha_and_determinism(
    model: SGBFormer,
    degraded: torch.Tensor,
) -> dict[str, Any]:
    cache = model.backbone.cache_decoder_prefix(degraded)
    feature = cache.decoder_level3
    flow = model.latent_flow
    flow.train()
    training = flow(feature, steps=model.latent_steps, return_trajectory=True)
    flow.eval()
    inference = flow(feature, steps=model.latent_steps, return_trajectory=True)
    exact_fields = (
        "global_mean",
        "global_precision",
        "spatial_mean",
        "spatial_precision",
    )
    train_eval_equal = all(
        torch.equal(training[name], inference[name])
        for name in exact_fields
    ) and all(
        torch.equal(train_step, inference_step)
        for train_step, inference_step in zip(
            training["trajectory"],
            inference["trajectory"],
        )
    )
    if not train_eval_equal:
        raise AssertionError(
            f"{flow.update_mode} train/eval state updates differ"
        )
    if model.latent_adapter_precision_mode != "belief":
        raise AssertionError("Training and deployment must use belief precision")

    global_alpha = torch.stack(inference["global_accuracies"]).sum(dim=0)
    spatial_alpha = torch.stack(inference["spatial_accuracies"]).sum(dim=0)
    global_weight, spatial_weight = flow.latent_loss_weights(inference)
    consistency = (
        torch.allclose(
            inference["global_precision"] - 1.0,
            global_alpha,
            rtol=1e-6,
            atol=1e-6,
        )
        and torch.allclose(
            inference["spatial_precision"] - 1.0,
            spatial_alpha,
            rtol=1e-6,
            atol=1e-6,
        )
        and torch.allclose(
            global_weight,
            global_alpha,
            rtol=1e-6,
            atol=1e-6,
        )
        and torch.allclose(
            spatial_weight,
            spatial_alpha,
            rtol=1e-6,
            atol=1e-6,
        )
    )
    if not consistency:
        raise AssertionError(
            "State precision, adaptive loss weighting, and inference alpha differ"
        )
    return {
        "update_mode": flow.update_mode,
        "accuracy_schedule": flow.flow.accuracy_schedule,
        "loss_weighting": flow.loss_weighting,
        "adapter_precision_mode": model.latent_adapter_precision_mode,
        "precision_equals_one_plus_sum_adaptive_alpha": True,
        "loss_weight_equals_sum_adaptive_alpha": True,
        "train_eval_elementwise_equal": train_eval_equal,
    }


def gradient_boundary_inventory(model: SGBFormer) -> dict[str, Any]:
    trainable = []
    frozen = []
    for name, parameter in model.named_parameters():
        record = {
            "name": name,
            "numel": parameter.numel(),
            "intent": (
                "optimized frozen-main path"
                if parameter.requires_grad
                else "frozen pretrained/non-selected parameter; tensor path is not detached"
            ),
        }
        (trainable if parameter.requires_grad else frozen).append(record)
    sites = [
        source_site(SGBFormer, "restore"),
        source_site(SGBFormer, "predict_latent_state"),
        source_site(SGBFormer, "latent_restoration_forward"),
        source_site(ExternalSSRFoundation, "cache_decoder_prefix"),
        source_site(ExternalSSRFoundation, "decode_from_cached_decoder3"),
        source_site(DegradationLatentBFN, "forward"),
        source_site(DegradationLatentBFN, "_weighted_mse"),
    ]
    return {
        "parameter_freezing": {
            "trainable": trainable,
            "requires_grad_false": frozen,
            "meaning": (
                "requires_grad=False freezes parameter storage but does not "
                "detach activations needed by trainable downstream/upstream modules"
            ),
        },
        "autograd_sites": sites,
        "explicit_boundaries": [
            {
                "site": "SGBFormer.restore",
                "kind": "torch.no_grad",
                "intent": "outer deployment inference boundary only",
            },
            {
                "site": "ExternalSSRFoundation.cache_decoder_prefix",
                "kind": "conditional torch.no_grad",
                "intent": (
                    "inactive in D/B because the configured SSR suffix is "
                    "trainable and foundation_trainable is true"
                ),
            },
            {
                "site": "degradation_latent_targets",
                "kind": "detach",
                "intent": (
                    "GT-derived auxiliary targets only; function is absent "
                    "from deployment"
                ),
            },
            {
                "site": "DegradationLatentBFN._weighted_mse",
                "kind": "detach",
                "intent": (
                    "prevents gaming adaptive loss weights; does not truncate "
                    "the deployed state recurrence"
                ),
            },
        ],
        "rollout_boundary": {
            "training": (
                "No truncation: gradients traverse all K=4 latent head calls "
                "and state updates."
            ),
            "deployment": (
                "The only global boundary is SGBFormer.restore @torch.no_grad."
            ),
        },
    }


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen Main Preflight",
        "",
        f"- Status: `{report['status']}`",
        f"- Protocol: `{report['protocol']}`",
        f"- D trainable parameters: `{report['parameter_contract']['trainable_parameters']['D']}`",
        f"- B trainable parameters: `{report['parameter_contract']['trainable_parameters']['B']}`",
        "- D/B parameter difference: `0.0%`",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"- `{check['id']}`: `{check['status']}` - {check['summary']}")
    lines.extend(
        [
            "",
            "The first image-only backward is expected to stop at the zero-initialized",
            "adapter output projections. After exactly one optimizer step, the second",
            "image-only backward must produce nonzero latent-flow and evidence-encoder",
            "gradients. No auxiliary latent or category loss is used in either backward.",
            "",
            "The JSON archive contains the complete parameter-freezing inventory and",
            "source-level detach/no-grad inventory.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = {
        "D": Path(args.config_d).expanduser().resolve(),
        "B": Path(args.config_b).expanduser().resolve(),
    }
    configs = {arm: load_config(path) for arm, path in paths.items()}
    for arm, config in configs.items():
        validate_frozen_main_contract(config, expected_arm=arm)
    if normalize_arm_config(configs["D"]) != normalize_arm_config(configs["B"]):
        raise AssertionError("D/B configs differ outside preregistered arm fields")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Frozen-main preflight requires the training CUDA GPU")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    models = {}
    for arm in ("D", "B"):
        seed_everything(int(configs[arm]["seed"]))
        models[arm] = build_model(configs[arm])
    parameter_contract = check_parameter_contract(models)
    initialization = {
        arm: assert_initialization(model)
        for arm, model in models.items()
    }

    dataset = build_dataset(
        configs["D"]["data"]["train"],
        training=True,
        sample_seed=int(configs["D"]["seed"]),
    )
    if len(dataset) != 18069:
        raise RuntimeError(f"Expected 18069 training pairs, got {len(dataset)}")
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    gradient_checks = {}
    alpha_checks = {}
    for arm in ("D", "B"):
        gradient_checks[arm] = check_gradient_path(
            models[arm],
            configs[arm],
            batch,
            device,
        )
        degraded = batch["degraded"].to(device)
        alpha_checks[arm] = check_alpha_and_determinism(
            models[arm],
            degraded,
        )
        models[arm].cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    boundary_inventory = gradient_boundary_inventory(models["D"])
    checks = [
        {
            "id": 1,
            "status": "passed",
            "summary": (
                "With both auxiliary weights zero, image loss reaches "
                "latent_flow after the one-step zero-projection bootstrap."
            ),
        },
        {
            "id": 2,
            "status": "passed",
            "summary": (
                "Parameter freezing and every detach/no-grad site on the "
                "deployment route are inventoried with intent and boundary."
            ),
        },
        {
            "id": 3,
            "status": "passed",
            "summary": (
                "State precision and detached loss weights equal the same "
                "summed adaptive alpha; D train/eval updates are elementwise equal."
            ),
        },
        {
            "id": 4,
            "status": "passed",
            "summary": (
                "After one optimizer step, both arms have nonzero image-only "
                "gradients in latent_flow and degradation_encoder."
            ),
        },
    ]
    report = {
        "schema_version": 1,
        "protocol": "frozen_main_db",
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "source_config": {arm: str(path) for arm, path in paths.items()},
        "source_config_sha256": {
            arm: sha256_file(path)
            for arm, path in paths.items()
        },
        "implementation": implementation_fingerprint(),
        "foundation_artifacts": foundation_artifact_fingerprint(configs["D"]),
        "parameter_contract": parameter_contract,
        "initialization": initialization,
        "gradient_checks": gradient_checks,
        "alpha_checks": alpha_checks,
        "gradient_boundary_inventory": boundary_inventory,
        "checks": checks,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "preflight.json"
    markdown_path = output_dir / "preflight.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown_summary(report), encoding="utf-8")
    print(json.dumps({"status": "passed", "archive": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
