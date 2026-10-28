# Frozen D/B Main Experiment Protocol

## Status

The implementation and source configs are frozen. Long training is locked.
The budget policy is fixed at 10 GPU hours per arm, a target interval of
12--30 epochs, and a hard ceiling of 16 GPU hours per arm. The final epoch
count remains unset until both 200-optimizer-step throughput profiles and the
four preflight checks pass. No long training has been started by this
delivery.

The Phase 1.5 development choice is:

- accuracy schedule: `linear_beta`
- latent loss: `final_mean`
- latent loss weighting: `adaptive_accuracy`

This choice records better latent fitting only. Better z fitting does not
imply better final restoration.

## Frozen Arms

Both D and B use:

- the same released SSR initialization;
- trainable SSR modules `decoder_level1`, `decoder_level2`, and `output`;
- the existing paper spectral gate at `encoder_level3` with
  `initial_gate=0.999`;
- the same two latent FiLM/spectral adapters at `decoder_level3` and
  `decoder_level2`;
- four calls to the same latent global/spatial heads;
- the same latent dimensions, head parameters, exogenous feature, state
  schema, adaptive-alpha function, and adapter precision regime;
- zero-initialized gamma, beta, and spectral correction projections;
- a scalar correction fusion gain initialized to exactly 1.

Here, the frozen "frequency module" means exactly the `encoder_level3` paper
spectral gate plus the latent adapters' internal `spectral_scale` correction.
There is no independent output-side spectral head.

Latent-target MSE and category cross-entropy are common supervision in both
arms. The only D/B difference is the parameter-free `update_mode`:

- D: deterministic residual refinement
  `mean <- mean + alpha / (precision + alpha) * observation`;
- B: Bayesian posterior update
  `mean <- (precision * mean + alpha * observation) / (precision + alpha)`.

Both use the same adaptive-alpha definition and update
`precision <- precision + alpha`. Alpha is predicted from the recurrent belief
state, so its numerical value, and therefore precision, need not match between
arms after their states diverge. A K=4 measurement found an approximately
`7.7e-2` global-precision difference. The head architecture and the
information supplied to each corresponding field remain matched.

Trainable parameter counts must differ by less than 5%. The preflight requires
the named parameter signatures and counts to be exactly equal.

## Frozen Training

- Train pairs: 18,069, patch 256, augmentation enabled, 12 workers.
- Global batch: 16; micro batch: 4.
- BF16 AMP, TF32, and cuDNN benchmark enabled.
- New modules: `2e-4`.
- SSR trainable suffix: `2e-5`.
- Per-group cosine floor: each group's own initial LR times `0.1`.
- Warmup: one epoch.
- Common image objective: MSE with weight `1.0`.
- Common latent final-belief MSE weight: `0.01`.
- Common latent category CE weight: `0.01`.
- Validation every epoch; log every 250 batches.
- EMA: decay `0.999`, warmup 2,000 updates in both arms.
- Checkpoint selection: highest dev Y-PSNR on the fixed 903-image dev set.
- Seed: 42 in both arms.

No Charbonnier, frequency, MS-SSIM, added gate, expanded latent, TTA, tiling,
EMA sweep, or sampling change is permitted.

Allowed early stops are only NaN/inf, OOM, measured throughput deviation over
30%, or an explicit crash. Early validation quality is not a stop condition.

## Preflight And Budget Lock

Run the four checks:

```bash
python3 scripts/preflight_frozen_main.py \
  --config-d configs/experiments/ssr_frozen_main_d.yaml \
  --config-b configs/experiments/ssr_frozen_main_b.yaml \
  --output-dir runs/frozen_main/preflight \
  --device cuda
```

The archive explicitly records the zero-projection bootstrap: the first
image-only backward reaches the adapter output projections but cannot reach
the upstream latent flow. After exactly one optimizer step, a second
image-only backward must give nonzero gradients in `latent_flow` and its
evidence encoder. For this gradient-path probe only, both auxiliary loss
weights are explicitly zeroed in both arms; the configured training weights
remain `0.01`.

Check 3 requires elementwise train/eval equality for both deterministic and
Bayesian updates. The optimizer check also requires the complete `2e-5` group
numel to equal exactly the combined SSR `decoder_level1`, `decoder_level2`,
and `output` numel; merely observing both configured LR values is insufficient.

Run exactly 200 optimizer steps per arm, without validation or quality metrics:

```bash
python3 scripts/profile_frozen_main.py \
  --config configs/experiments/ssr_frozen_main_d.yaml \
  --output runs/frozen_main/profile_D.json --device cuda

python3 scripts/profile_frozen_main.py \
  --config configs/experiments/ssr_frozen_main_b.yaml \
  --output runs/frozen_main/profile_B.json --device cuda
```

Reuse the measured Phase 1-R frozen-SSR JSON only if it passes the freezer's
protocol checks. Otherwise regenerate it without LPIPS:

```bash
python3 evaluate_foundation.py \
  --config configs/experiments/ssr_latent_ssr_stage0.yaml \
  --input-dir "$SGB_DATA_ROOT/input" \
  --target-dir "$SGB_DATA_ROOT/gt" \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --device cuda --test-y-channel \
  --json runs/frozen_main/frozen_ssr_dev.json

export SSR_EVAL_JSON=runs/frozen_main/frozen_ssr_dev.json
```

After both profiles exist:

```bash
python3 scripts/freeze_frozen_main_budget.py \
  --config-d configs/experiments/ssr_frozen_main_d.yaml \
  --config-b configs/experiments/ssr_frozen_main_b.yaml \
  --profile-d runs/frozen_main/profile_D.json \
  --profile-b runs/frozen_main/profile_B.json \
  --preflight runs/frozen_main/preflight/preflight.json \
  --baseline "$SSR_EVAL_JSON" \
  --output-dir runs/frozen_main/locked
```

`SSR_EVAL_JSON` must be the frozen-SSR result on the same 903-image dev
manifest. The freezer verifies the image names, Y-channel setting, crop,
single-pass settings, output quantization, decoder count, and foundation
signature. It reads the exact baseline PSNR from that file and locks the
epoch-4 threshold to `baseline_y_psnr - 0.05`.

The freezer uses:

```text
steps_per_epoch = ceil(18069 / 16) = 1130
base_supported_epochs_arm = floor(
  10 * 3600
  / (seconds_per_step_arm * 1130)
)
base_common_epochs = min(base_supported_epochs_D, base_supported_epochs_B)

if base_common_epochs < 12:
  raise one common per-arm budget to the exact amount needed by the slower
  arm for 12 epochs, capped at 16 GPU hours per arm

supported_epochs_arm = floor(
  effective_gpu_hours_per_arm * 3600
  / (seconds_per_step_arm * 1130)
)
epochs_D = epochs_B = min(supported_epochs_D, supported_epochs_B, 30)
```

If 16 hours cannot support 12 epochs on the slower arm, the common epoch count
is whatever 16 hours supports and the manifest records
`budget_floor_reached=true`. The manifest also fixes A0 and A1 to reuse this
same epoch count later while keeping both ablations launch-locked now.

The freezer emits only read-only D/B configs and a SHA256 manifest. `train.py`
rejects unlocked configs, modified configs, failed/missing preflight archives,
different GPUs, and a first-200-step throughput deviation above 30%.

Only then may the two arms run to the mandatory epoch-4 pause:

```bash
python3 train.py --config runs/frozen_main/locked/frozen_D.yaml --device cuda
python3 train.py --config runs/frozen_main/locked/frozen_B.yaml --device cuda
```

Each command saves `latest.pt`, its normal epoch metrics, and a tripwire status
after epoch 4, then exits successfully regardless of the individual result.
After both have paused, make the paired decision:

```bash
python3 scripts/decide_frozen_main_tripwire.py \
  --config-d runs/frozen_main/locked/frozen_D.yaml \
  --config-b runs/frozen_main/locked/frozen_B.yaml
```

This decision uses only each arm's epoch-4 EMA dev Y-PSNR against the locked
SSR threshold. It never computes or inspects B minus D. If either arm is below
the threshold, it writes a failed decision and neither arm can resume. If both
pass, resume the unchanged full schedules:

```bash
python3 train.py \
  --config runs/frozen_main/locked/frozen_D.yaml \
  --resume runs/frozen_main/D/latest.pt --device cuda

python3 train.py \
  --config runs/frozen_main/locked/frozen_B.yaml \
  --resume runs/frozen_main/B/latest.pt --device cuda
```

Resume uses the original locked total epoch count and scheduler state, so the
epoch-4 checkpoint is an exact prefix rather than a short-run schedule.

## Dev Analysis

Report all of the following, without selecting a favorable column:

- D vs frozen SSR, B vs frozen SSR, and B vs D Y-PSNR with paired
  image-bootstrap CI95;
- SSIM and LPIPS;
- category results;
- time/image, peak memory, decoder calls, and trainable parameters.

The current 903-image dev manifest supports three verified source labels:
Snow (combined), Outdoor-Rain, and RainDrop. It does not provide a verified
Snow100K-S versus Snow100K-L split. Do not fabricate a four-way dev result.
Four separate subsets are mandatory in the official evaluation after their
protocol is locked.

Interpretation is fixed:

- D and B both significantly beat frozen SSR: both combinations provide a
  system gain; this comparison cannot attribute the shared gain.
- B significantly beats D: supports additional B value only for that metric.
- B and D are close and neither beats frozen SSR: the system target failed.
- B improves LPIPS but lowers PSNR: report the tradeoff directly.
- CI covers zero: "no clear difference detected", never "equivalent".
- If B minus D is below 0.10 dB, use at least three training seeds before a
  paper claim. Image bootstrap does not replace seed replication.
- A single best-checkpoint dev difference is not a generalization claim.

## Paper Evaluation Checklist

Registered ablation configs:

- `ssr_frozen_ablation_a0.yaml`: trainable SSR suffix only.
- `ssr_frozen_ablation_a1.yaml`: A0 plus only the `encoder_level3` paper
  spectral gate. Thus A1 minus A0 measures that gate alone.

These use the same data, optimizer, image loss, checkpoint rule, and eventual
matched epoch count. They remain budget-locked until the D/B conclusion; the
main budget manifest records their required epoch count but does not emit
launchable A0/A1 configs.

Final tables:

- official Snow100K-S, Snow100K-L, Outdoor-Rain, and RainDrop subsets;
- single-pass evaluation under one locked protocol;
- ablation: frozen SSR / A0 / A1 / D / B;
- Y-PSNR, SSIM, LPIPS, per-subset results;
- time, peak memory, decoder calls, and parameters;
- required repeated seeds.

Every SOTA number must be checked against the original paper and annotated
with split, color space, crop border, self-ensemble, and tiling. Protocol
mismatches are not directly comparable. Official subsets are checked once and
then locked; all development decisions remain on dev.

All GT-derived probes are `deployable=false`. A shared-system gain without an
independent B-over-D gain is not a BFN success. A small stable B increment is
reported at its measured magnitude.
