# v27 C0-double-prime Residual Geometry Audit

## Status

- Implementation: complete
- Local synthetic verification: complete
- Runtime audit: pending
- Checkpoint: EMA `epoch_0003.pt` only
- Dataset: fixed 903-image dev split only
- Training and parameter updates: prohibited
- Official test access: prohibited

`scripts/audit_v27_residual_geometry.py` replaces this file with measured
results after the server audit.

## Static Coordinate Audit

| Stage | Definition | Space and clamp |
| --- | --- | --- |
| Input and GT | PIL RGB converted to `float32 / 255` | RGB `[0,1]` |
| Baseline `y0` | frozen SSR output | clamped to `[0,1]` before residual training |
| Residual target `r` | `GT - y0` | raw RGB difference |
| Receiver output | identity activation, then multiplied by `residual_scale=0.2` | unbounded RGB correction |
| Training output | `y_hat_raw = y0 + c_raw` | not clamped |
| Training loss | final RGB MSE | augmented `256x256` patches |
| Deployed output | `clamp(y0 + c_raw, 0, 1)` | float RGB `[0,1]` |
| Dev metric | MATLAB-compatible RGB-to-Y PSNR | full image, crop border 0 |
| Endpoint weights | checkpoint EMA shadow | epoch 3 only |

Static inspection therefore identifies three possible alignment gaps:

1. RGB training MSE versus Y-channel model selection;
2. unclamped training output versus clamped evaluation output;
3. augmented patch loss versus complete-image PSNR.

It does not identify a residual scaling-coordinate bug: target and correction
are both represented in raw normalized RGB after the receiver output is
multiplied by `0.2`.

## Runtime Checks

The audit performs one receiver forward pass per image and then computes:

- exact Y-space MSE decomposition;
- the fixed 12-point correction-scale sweep;
- global, category, and per-image analytic unclamped scale coefficients;
- all 231 nonnegative three-expert mixtures on the `0.05` simplex grid;
- current-loss and alternative-loss output gradients;
- checkpoint/model hash preservation.

The `0.05` simplex grid is a finite approximation, not a proof over every
continuous mixture. Category and per-image selections use dev labels or
targets and are diagnostic only.

## Attribution Contract

The final report chooses exactly one:

- `E1`: correction scale/calibration mismatch;
- `E2`: category-dependent accept/reject mismatch;
- `E3`: loss/metric mismatch;
- `E4`: receiver representation/optimization mismatch;
- `E5`: implementation/coordinate mismatch.

Only one corresponding deterministic experiment may follow. A
parameter-matched BFN arm remains required later, after the deterministic
receiver contract is repaired.
