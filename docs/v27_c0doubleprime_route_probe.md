# v27 C0-double-prime Epoch-3 Route Probe

## Status

- Fixed endpoint decision: `STOP`
- Allowed checkpoint: `epoch_0003.pt` only
- Dataset: existing 903-image dev split
- Mode: read-only; no gradients or parameter updates
- Runtime probe: pending server execution

The fixed endpoint produced:

| Metric | Result |
| --- | ---: |
| Macro delta PSNR vs F0 | `+0.005547` |
| Macro CI95 | `[+0.002833, +0.008284]` |
| Snow vs F0 | `+0.018012` |
| Outdoor-Rain vs F0 | `-0.006517` |
| Outdoor-Rain CI95 | `[-0.011815, -0.000912]` |
| RainDrop vs F0 | `-0.007319` |
| RainDrop CI95 | `[-0.012490, -0.002141]` |
| Outdoor-Rain vs C0 | `+0.005980` |
| RainDrop vs C0 | `+0.006755` |
| Correction-target cosine | `0.04318` |

The private bottleneck improved both rain categories versus C0 and C0-prime,
but both remain significantly below F0. The quality STOP is therefore
unaffected by the loss of historical per-image C0/C0-prime JSONs.

This probe answers only:

1. whether the C0-double-prime experts remain symmetric after training;
2. whether the router remains near uniform;
3. whether forced experts, category-oracle routing, or the target-leaking
   per-image oracle expose enough unused headroom to explain the remaining
   rain deficits.

The implementation writes measured results over this document after the
server run. Oracle outputs remain diagnostics and must not be reported as
deployable method results.

## Stop Rules

- Do not evaluate another checkpoint.
- Do not train or update any parameter.
- Do not access official test data.
- Do not start D4/B4.
- Do not treat this deterministic arm as BFN evidence.
