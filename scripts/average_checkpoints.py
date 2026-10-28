#!/usr/bin/env python3
"""Average compatible checkpoints for inference.

EMA averaging overlays trainable EMA weights on the complete model state.
The output records its inputs and drops optimizer, EMA and RNG state, so it
cannot resume training. Averaged checkpoints are not eligible for the fixed
epoch-6 BasisBFN endpoint; use analyze_basisbfn.py for that protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sgbformer.checkpoint import atomic_torch_save  # noqa: E402
from sgbformer.inference import validate_inference_config  # noqa: E402

# Resume-only state that becomes meaningless after weight averaging.
DROPPED_KEYS = (
    "optimizer",
    "scheduler",
    "scaler",
    "rng_states",
    "ema",
    "ema_parameter_names",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[],
        metavar="CKPT",
        help="Explicit checkpoint paths to average.",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory scanned with --pattern when --inputs is not given.",
    )
    parser.add_argument(
        "--pattern",
        default="epoch_*.pt",
        help="Glob used with --input-dir (default: epoch_*.pt).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--weights",
        choices=("ema", "raw"),
        default="ema",
        help=(
            "ema averages the EMA view (model overwritten by ema.shadow, the "
            "same view evaluate.py loads); raw averages the online weights."
        ),
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help=(
            "Skip the model-config equality check. Only use this for "
            "checkpoints you know share an identical architecture."
        ),
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.inputs and args.input_dir:
        raise ValueError("Use either --inputs or --input-dir, not both")
    if args.inputs:
        paths = [Path(value) for value in args.inputs]
    else:
        if not args.input_dir:
            raise ValueError("Provide --inputs or --input-dir")
        directory = Path(args.input_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Not a directory: {directory}")
        paths = sorted(directory.glob(args.pattern))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    if len(paths) < 2:
        raise ValueError(
            f"Averaging requires at least two checkpoints, got {len(paths)}"
        )
    return paths


def weight_view(
    checkpoint: dict[str, Any],
    path: Path,
    mode: str,
) -> dict[str, torch.Tensor]:
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"{path} has no 'model' state dict")
    state = dict(model_state)
    if mode == "raw":
        return state
    ema = checkpoint.get("ema")
    if not isinstance(ema, dict) or not isinstance(ema.get("shadow"), dict):
        raise ValueError(
            f"{path} has no EMA shadow; rerun with --weights raw if the run "
            "was trained without EMA"
        )
    shadow = ema["shadow"]
    extra = sorted(set(shadow) - set(state))
    if extra:
        raise ValueError(
            f"{path} EMA shadow has keys absent from 'model': {extra}"
        )
    state.update(shadow)
    return state


def average_states(
    states: list[dict[str, torch.Tensor]],
    paths: list[Path],
) -> dict[str, torch.Tensor]:
    reference_keys = set(states[0])
    for path, state in zip(paths[1:], states[1:]):
        if set(state) != reference_keys:
            missing = sorted(reference_keys - set(state))
            unexpected = sorted(set(state) - reference_keys)
            raise ValueError(
                f"{path} parameter names differ from {paths[0]}. "
                f"Missing: {missing}; unexpected: {unexpected}"
            )
    averaged: dict[str, torch.Tensor] = {}
    for name in states[0]:
        tensors = []
        for path, state in zip(paths, states):
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{path}:{name} is not a tensor")
            if value.shape != states[0][name].shape:
                raise ValueError(
                    f"{path}:{name} shape {tuple(value.shape)} differs from "
                    f"{paths[0]}:{name} shape {tuple(states[0][name].shape)}"
                )
            if value.dtype != states[0][name].dtype:
                raise ValueError(
                    f"{path}:{name} dtype {value.dtype} differs from "
                    f"{paths[0]}:{name} dtype {states[0][name].dtype}"
                )
            tensors.append(value)
        reference = tensors[0]
        if reference.is_floating_point():
            accumulator = torch.zeros(
                reference.shape,
                dtype=torch.float64,
            )
            for tensor in tensors:
                accumulator += tensor.detach().to(torch.float64)
            averaged[name] = (accumulator / len(tensors)).to(reference.dtype)
        else:
            for path, tensor in zip(paths[1:], tensors[1:]):
                if not torch.equal(tensor, reference):
                    raise ValueError(
                        f"{path}:{name} is a non-floating tensor that differs "
                        f"from {paths[0]}:{name}; averaging is undefined"
                    )
            averaged[name] = reference.clone()
    return averaged


def main() -> None:
    args = parse_args()
    paths = resolve_inputs(args)
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in paths
    ]
    for path, checkpoint in zip(paths, checkpoints):
        if not isinstance(checkpoint, dict):
            raise TypeError(f"{path} is not a BasisBFN checkpoint payload")
    if not args.allow_config_mismatch:
        reference_config = checkpoints[0].get("config")
        for path, checkpoint in zip(paths[1:], checkpoints[1:]):
            candidate = checkpoint.get("config")
            if reference_config is None or candidate is None:
                raise ValueError(
                    f"{path} or {paths[0]} has no embedded config; pass "
                    "--allow-config-mismatch to override"
                )
            try:
                validate_inference_config(reference_config, candidate)
            except ValueError as error:
                raise ValueError(
                    f"{path} was trained with a different model than "
                    f"{paths[0]}: {error}"
                ) from error
    states = [
        weight_view(checkpoint, path, args.weights)
        for path, checkpoint in zip(paths, checkpoints)
    ]
    averaged = average_states(states, paths)

    payload: dict[str, Any] = {
        key: value
        for key, value in checkpoints[0].items()
        if key not in DROPPED_KEYS and key != "model"
    }
    payload["model"] = averaged
    payload["averaged_from"] = [str(path) for path in paths]
    payload["averaged_weights"] = args.weights
    payload["averaged_checkpoints"] = len(paths)
    payload["averaged_epochs"] = [
        checkpoint.get("epoch") for checkpoint in checkpoints
    ]
    payload["averaged_global_steps"] = [
        checkpoint.get("global_step") for checkpoint in checkpoints
    ]
    payload["resumable"] = False

    output = Path(args.output)
    atomic_torch_save(payload, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "weights": args.weights,
                "checkpoints": [str(path) for path in paths],
                "parameters": len(averaged),
                "epochs": payload["averaged_epochs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
