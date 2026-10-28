# BasisBFN: Bayesian Residual Refinement with Image-Dependent Bases for Adverse Weather Image Restoration

![BasisBFN architecture](assets/basisbfn_architecture.png)

## 1. Scope

BasisBFN attaches a compact Bayesian residual refiner to a frozen SSR
restoration backbone. It moves the probabilistic state from RGB residual
pixels to four signed spatial coefficient maps over image-dependent RGB bases.
The main paper model is the four-step gated Bayesian arm, B4-G.

The implementation does not access official test data and does not start
training. Every arm remains launch-locked until its own preflight and 200-step
CUDA profile pass.

## 2. Architecture

For one receiver feature map `h`, the readout computes the basis exactly once:

```text
basis = Conv2d(SiLU(GroupNorm(h)))          # [B, 4, 3, H, W]
mu_0 = 0                                   # [B, 4, H, W]
rho_0 = 1
```

At each of four coefficient-state steps, a shared observer receives:

```text
[h, mu_i, log(rho_i), time_i]
```

It predicts a bounded coefficient observation `y_i` and an adaptive accuracy.
The two matched update rules are:

```text
rho_next = rho + alpha

Bayesian:      mu_next = (rho * mu + alpha * y) / rho_next
Deterministic: mu_next = mu + (alpha / rho_next) * y
```

During training, the normalized target RGB residual is projected onto the
current basis with a ridge-stabilized minimum-norm solve. A random discrete
time is sampled, the Gaussian sender produces a noisy coefficient belief, and
the shared observer predicts the clean coefficient target. Both D and B use
the same discrete BFN objective from Eq. (15) with weight `0.00004`; this keeps the training
signal matched while isolating the state-update equation.

The final coefficient maps are `tanh(mu_4)`. The residual is:

```text
correction = gate * sum_k(basis_k * tanh(mu_4,k)) / sqrt(4)
```

The optional spatial gate is explicitly posterior-conditioned:

```text
gate = sigmoid(Conv([h, mu_4, log(rho_4)]))
```

The observer output is zero initialized. The gate projection is also zero
initialized, giving a neutral probability of `0.5`. Since all initial
coefficients are exactly zero, every arm is exactly the frozen SSR mapping at
initialization while preserving a live gradient into the observer.

## 3. Controlled Matrix

| Arm | Coefficient update | Posterior gate |
| --- | --- | --- |
| D4 | deterministic | off |
| D4-G | deterministic | on |
| B4 | Bayesian | off |
| B4-G | Bayesian | on |

All four arms instantiate the gate parameters and use the same coefficient
sender/receiver objective, including gate-off controls. They therefore have
identical parameter names and exactly `624,217` trainable parameters. D/B
changes only a parameter-free state-update equation; `-G` changes only whether
the already-present gate output is applied.

The legacy outer `model.bfn.update_mode` remains `deterministic` in every arm
because that RGB-state path is bypassed. The authoritative runtime banner is
`coefficient_bfn`, which records `steps=4`, the coefficient update mode,
discrete-BFN training objective, BFN loss weight, and gate state.

Per image:

- frozen SSR calls: `1`;
- receiver calls: `1`;
- shared basis evaluations: `1`;
- coefficient observer calls/state updates: `4`;
- training-only random-time BFN observer calls: `1`;
- terminal RGB receiver call: `0`.

## 4. Configs

```text
configs/experiments/basisbfn_d4.yaml
configs/experiments/basisbfn_d4g.yaml
configs/experiments/basisbfn_b4.yaml
configs/experiments/basisbfn_b4g.yaml
```

The common non-launchable base is:

```text
configs/experiments/_basisbfn_common.yaml
```

The manuscript reports independent runs with seeds `42`, `43`, `44`, and
`45`. Generate non-overlapping arm/seed overlays with
`scripts/make_basisbfn_config.py`.

## 5. Preflight And Profile

Run the preflight separately for every arm. It also loads and audits the full
2x2 matrix:

```bash
python3 scripts/preflight_basisbfn.py \
  --config configs/experiments/basisbfn_b4g.yaml \
  --reference-c0 configs/experiments/ssr_residual_v27_c0_proxy.yaml \
  --matrix-config-dir configs/experiments \
  --output-dir runs/basisbfn/preflight \
  --device cuda
```

The preflight checks exact F0 identity on eight seeded dev images, DCT basis
orthogonality, posterior precision growth, gate bounds, one basis/four observer
calls, an active finite coefficient BFN loss, frozen SSR state, parameter
ownership, and first/second optimizer-step gradient reachability.

After reviewing the passed preflight:

```bash
python3 scripts/profile_basisbfn.py \
  --config configs/experiments/basisbfn_b4g.yaml \
  --preflight runs/basisbfn/preflight/preflight_b4-g.json \
  --output runs/basisbfn/profiles/B4-G.json \
  --device cuda
```

The profile executes exactly 200 optimizer steps and records throughput, peak
allocated CUDA memory, coefficient activity, posterior precision, and gate
activity. It computes no validation quality metric. Peak allocated memory must
remain at or below 2 GiB.

## 6. Training And Evaluation

Only after the matching preflight and profile artifacts have been reviewed:

```bash
python3 -u train.py \
  --config configs/experiments/basisbfn_b4g.yaml \
  --device cuda
```

Epochs 1-5 are diagnostic. The formal endpoint is EMA `epoch_0006.pt`.
Evaluation is deterministic, full-image, Y-channel, and uses no self-ensemble:

```bash
python3 evaluate.py \
  --checkpoint runs/basisbfn/b4-g/epoch_0006.pt \
  --input-dir "$SGB_DATA_ROOT/input" \
  --target-dir "$SGB_DATA_ROOT/gt" \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --device cuda \
  --test-y-channel \
  --category-pattern 'snow=.*\.jpe?g' \
  --category-pattern 'outdoor_rain=im_\d+_s\d+_a0[456]\.png' \
  --category-pattern 'raindrop=\d+_rain\.png' \
  --json runs/basisbfn/analysis/b4-g_epoch6/evaluation.json
```

## 7. Fixed Matrix Analysis

After all four fixed endpoints are available:

```bash
python3 scripts/analyze_basisbfn.py \
  --baseline runs/residual_bfn_v27_exploratory/analysis/c0doubleprime_epoch3/f0_reconstructed_from_structural.json \
  --checkpoint D4=runs/basisbfn/d4/epoch_0006.pt \
  --checkpoint D4-G=runs/basisbfn/d4-g/epoch_0006.pt \
  --checkpoint B4=runs/basisbfn/b4/epoch_0006.pt \
  --checkpoint B4-G=runs/basisbfn/b4-g/epoch_0006.pt \
  --candidate D4=runs/basisbfn/analysis/d4_epoch6/evaluation.json \
  --candidate D4-G=runs/basisbfn/analysis/d4-g_epoch6/evaluation.json \
  --candidate B4=runs/basisbfn/analysis/b4_epoch6/evaluation.json \
  --candidate B4-G=runs/basisbfn/analysis/b4-g_epoch6/evaluation.json \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --output-dir runs/basisbfn/analysis/matrix_epoch6 \
  --bootstrap-samples 10000 \
  --seed 42
```

The analyzer verifies every checkpoint/evaluation hash and reports:

- every arm versus F0;
- B4 versus D4;
- B4-G versus D4-G;
- D4-G versus D4;
- B4-G versus B4;
- the macro PSNR Bayesian-by-gate interaction.

It does not scan checkpoints or replace the fixed endpoint.
