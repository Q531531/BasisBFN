# v28-R0 Deterministic Signed-Basis Contract

## 1. Motivation

The fixed v27 C0-double-prime signed-coefficient audit rejected the old
three-expert basis:

- category signed oracle gain over learned routing: `+0.005800 dB`;
- per-image target-leaking oracle gain: `+0.041991 dB`;
- the per-image bound remained below the registered `0.05 dB` capacity floor.

R0 therefore replaces the old expert basis. It does not add a coefficient BFN,
router, gate, category label, rebalanced sampler, or new loss.

## 2. Single Behavioral Variable

Reference: `v27-C0`

Candidate: `v28-R0`

The only model behavior change is:

```text
model.bfn.signed_basis_readout
```

Unchanged:

- frozen SSR and no feature adapters;
- one direct deterministic receiver call;
- observation-rich receiver input and the same receiver trunk;
- residual scale `0.2` and clamped structural baseline;
- RGB MSE objective with no auxiliary loss;
- natural shuffle sampling;
- seed `42`, optimizer, learning rate, schedule, augmentation, batch sizes,
  accumulation, EMA, and six-epoch budget;
- train/dev manifests with `17,166` and `903` pairs;
- full-image deterministic Y-channel dev evaluation.

## 3. Readout

The final receiver feature `h` has 16 channels. R0 computes:

```text
h_norm = SiLU(GroupNorm(h))
basis = Conv2d(16, 4 * 3, 3x3)(h_norm)
coeff = tanh(Conv2d(16, 4, 3x3)(h_norm))
correction = sum_k(coeff[k] * basis[k]) / sqrt(4)
```

Each coefficient is spatial and bounded to `[-1, 1]`. The four RGB basis maps
are initialized with the first 12 orthonormal rows of an explicit DCT matrix.
This initialization is independent of the process RNG. The coefficient
projection is exactly zero initialized.

At initialization:

- coefficients and correction are exactly zero;
- restoration is the frozen SSR output;
- the first image gradient reaches only the coefficient projection;
- after one optimizer update, gradients reach the basis projection and shared
  receiver trunk.

All receiver parameters remain trainable. The SSR remains frozen.

## 4. Parameter Budget

| Component | Parameters |
| --- | ---: |
| Shared receiver trunk without v27-C0 readout | 613,988 |
| GroupNorm(16) | 32 |
| Basis `Conv2d(16,12,3x3)` | 1,740 |
| Coefficient `Conv2d(16,4,3x3)` | 580 |
| Complete R0 readout | 2,352 |
| Complete v28-R0 model | 616,340 |

R0 adds `1,885` trainable parameters over v27-C0 (`0.3068%`) and stays below
the existing `2x C0` ceiling.

## 5. Preflight

Set the same dataset and SSR environment used by v27, then run:

```bash
python3 scripts/preflight_v28_r0.py \
  --config configs/experiments/ssr_residual_v28_r0_deterministic_basis.yaml \
  --reference-c0 configs/experiments/ssr_residual_v27_c0_proxy.yaml \
  --output-dir runs/deterministic_basis_v28/preflight \
  --device cuda
```

The preflight must prove:

1. R0 differs from v27-C0 only at the registered readout and administrative
   protocol paths.
2. The trainable parameter count is exactly `616,340`.
3. Only `receiver.*` is trainable and the complete SSR state remains unchanged.
4. Basis weights are nonzero and DCT-orthogonal within `1e-5`.
5. Coefficient weights and biases are exactly zero.
6. Eight seeded dev images and a zero input preserve the F0 output within
   `1e-6`, with one receiver call per image.
7. The initial RGB-MSE is the F0 loss.
8. First-step coefficient gradient is nonzero while basis and trunk gradients
   are zero.
9. Second-step basis and trunk gradients are nonzero.
10. Foundation, config, and train/dev manifest hashes are recorded.

## 6. 200-Step Smoke Profile

Only after preflight passes:

```bash
python3 scripts/profile_v28_r0.py \
  --config configs/experiments/ssr_residual_v28_r0_deterministic_basis.yaml \
  --preflight runs/deterministic_basis_v28/preflight/preflight_r0.json \
  --output runs/deterministic_basis_v28/profiles/R0.json \
  --device cuda
```

The profile performs exactly 200 optimizer steps with the production data
loader, BF16 autocast, gradient clipping, optimizer, and EMA. It does not run
validation or inspect quality. Peak allocated memory must not exceed 2 GiB.
The config, implementation, preflight, parameter count, device, throughput,
and coefficient activity are recorded.

Do not start the six-epoch run until the preflight and profile reports have
been reviewed.

## 7. Training And Fixed Evaluation

After explicit authorization:

```bash
nohup python3 -u train.py \
  --config configs/experiments/ssr_residual_v28_r0_deterministic_basis.yaml \
  --device cuda \
  > logs/v28-R0-train.log 2>&1 &
```

Epochs 1-5 are diagnostic only. They must not be used for checkpoint
selection. Evaluate only EMA `epoch_0006.pt`:

```bash
python3 evaluate.py \
  --checkpoint runs/deterministic_basis_v28/r0/epoch_0006.pt \
  --input-dir "$SGB_DATA_ROOT/input" \
  --target-dir "$SGB_DATA_ROOT/gt" \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --device cuda \
  --test-y-channel \
  --category-pattern 'snow=.*\.jpe?g' \
  --category-pattern 'outdoor_rain=im_\d+_s\d+_a0[456]\.png' \
  --category-pattern 'raindrop=\d+_rain\.png' \
  --json runs/deterministic_basis_v28/analysis/r0_epoch6/evaluation.json
```

Then apply the fixed decision:

```bash
python3 scripts/analyze_v28_r0.py \
  --baseline runs/residual_bfn_v27_exploratory/analysis/c0doubleprime_epoch3/f0_reconstructed_from_structural.json \
  --checkpoint runs/deterministic_basis_v28/r0/epoch_0006.pt \
  --candidate-epoch6 runs/deterministic_basis_v28/analysis/r0_epoch6/evaluation.json \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --output-dir runs/deterministic_basis_v28/analysis/r0_epoch6 \
  --bootstrap-samples 10000 \
  --seed 42
```

The analyzer binds the evaluation to the checkpoint SHA-256, embedded config,
completed epoch, and EMA weights.

## 8. Decision

`GO` requires every check:

- macro Y-PSNR delta versus F0 is at least `+0.05 dB`;
- macro paired-image bootstrap CI95 lower bound is strictly positive;
- Outdoor-Rain delta is at least `-0.003 dB` and its CI95 upper bound is
  nonnegative;
- RainDrop delta is at least `-0.005 dB` and its CI95 upper bound is
  nonnegative.

Any failed check is `STOP`. No checkpoint scan is permitted. A `GO` establishes
deterministic basis capacity only; it does not authorize or validate a
coefficient-state BFN. The separately implemented v28 coefficient-state matrix
has its own preflight/profile launch gates; official test data remain locked.
