#!/usr/bin/env python3
"""Run a small BasisBFN readout check without SSR, datasets, or checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sgbformer.models import BasisBFNReadout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42)
    torch.set_num_threads(1)

    readout = BasisBFNReadout(
        channels=16,
        residual_channels=3,
        update_mode="bayesian",
        acceptance_gate={"enabled": True},
    ).to(device)
    feature = torch.linspace(-1, 1, 16 * 9 * 11, device=device).reshape(
        1, 16, 9, 11
    )
    target = torch.linspace(-0.25, 0.25, 3 * 9 * 11, device=device).reshape(
        1, 3, 9, 11
    )
    correction = readout(feature, deterministic=True)
    torch.testing.assert_close(
        correction, torch.zeros_like(target), atol=0, rtol=0
    )
    if readout.last_basis_evaluations != 1 or readout.last_observer_calls != 4:
        raise RuntimeError("Expected one basis evaluation and four observer calls")
    if len(readout.last_trajectory) != 4:
        raise RuntimeError("Expected four coefficient-state updates")
    precision = readout.last_posterior_precision
    gate = readout.last_acceptance
    if precision is None or not torch.all(torch.isfinite(precision) & (precision > 1)):
        raise RuntimeError("Posterior precision did not increase")
    if gate is None:
        raise RuntimeError("Acceptance gate output is missing")
    torch.testing.assert_close(gate, torch.full_like(gate, 0.5), atol=0, rtol=0)

    bfn_loss = readout.bfn_training_loss(
        target,
        step=torch.full((1,), 2, dtype=torch.long, device=device),
    )
    if not torch.isfinite(bfn_loss) or bfn_loss.item() <= 0:
        raise RuntimeError("Expected a finite, positive coefficient BFN loss")
    sender = readout.last_bfn_sender_mean
    if sender is None or not torch.isfinite(sender).all() or not torch.count_nonzero(sender):
        raise RuntimeError("BFN sender state is missing or invalid")
    (F.mse_loss(correction, target) + 4e-5 * bfn_loss).backward()
    gradient = readout.observer_output.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
        raise RuntimeError("Observer output did not receive a finite gradient")

    print(
        json.dumps(
            {
                "status": "passed",
                "scope": "readout_module_check_without_ssr_or_quality_metrics",
                "device": str(device),
                "torch_version": torch.__version__,
                "correction_shape": list(correction.shape),
                "initial_correction_max_abs": correction.detach().abs().max().item(),
                "basis_evaluations": readout.last_basis_evaluations,
                "coefficient_state_updates": len(readout.last_trajectory),
                "observer_calls_with_training_objective": readout.last_observer_calls,
                "posterior_precision_min": precision.min().item(),
                "initial_gate_mean": gate.mean().item(),
                "coefficient_bfn_loss": bfn_loss.detach().item(),
                "observer_gradient_norm": gradient.norm().item(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
