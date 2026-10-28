# v27 Residual Bayesian Flow Restoration Design Contract

## 0. Status

This document freezes the v27 design before implementation.

- Status: **IMPLEMENTED AND LOCALLY VERIFIED; SERVER PREFLIGHT NOT YET RUN**
- Local verification: **246 tests passed**
- Implementation fingerprint:
  `b19d63251e999fda30f0cd2d3efc419bd8f731f9968e9949d37c0dbb35f6f07e`
- Long training: **FORBIDDEN**
- Official-test evaluation: **FORBIDDEN**
- Allowed next work: server preflight, parameter/identity/gradient audit, and
  a 200-step throughput profile.
- First permitted training experiment: the Stage A0 six-epoch direct-residual
  capacity gate defined below.

Any behavioral change to this contract requires a new document revision and a
new experiment identifier. Failed results must not be followed by unregistered
loss, sampling, width, step-count, or learning-rate changes.

## 1. Evidence That Closes v26

All values below use the fixed 903-image dev manifest, MATLAB Y-channel PSNR,
crop border 0, 10,000 paired bootstrap samples, and seed 42.

### 1.1 Four-way causal decomposition

| Contribution | Macro delta PSNR | CI95 | Decision |
|---|---:|---:|---|
| `Fback-F0`: EMA SSR weight drift | +0.000937 | [-0.011499, +0.012459] | no detected aggregate gain |
| `F1-Fback`: encoder spectral gate | -0.000191 | [-0.000609, +0.000174] | no detected contribution |
| `F2-F1`: decoder latent BFN | **-0.002961** | **[-0.004903, -0.001058]** | significantly harmful |
| `F2-F0`: complete v26d | -0.002215 | [-0.015525, +0.010351] | does not beat SSR |

The point estimates telescope exactly:

```text
+0.000937 - 0.000191 - 0.002961 = -0.002215
```

This agreement is a protocol sanity check. It does not imply that confidence
interval endpoints can be added.

### 1.2 Category attribution

| Contribution | Snow | Outdoor-Rain | RainDrop |
|---|---:|---:|---:|
| EMA SSR weight drift | +0.050212 | -0.098706 | +0.002027 |
| Encoder spectral gate | -0.000229 | +0.000017 | -0.000322 |
| Decoder latent BFN | +0.002462 | -0.011660 | -0.005107 |
| Complete model, point-estimate sum | +0.052445 | -0.110349 | -0.003402 |

The Snow and Outdoor-Rain changes are individually significant for SSR weight
drift. The latent BFN significantly improves Snow by a small amount and
significantly harms Outdoor-Rain. RainDrop has only 55 dev images, so its
interval remains weakly determined.

The v26d macro average therefore masks a material category transfer. It is not
evidence of category-wise parity.

### 1.3 Fusion and Oracle evidence

Relative fusion strength against F1:

| Alpha | Macro delta PSNR | CI95 |
|---:|---:|---:|
| 0.00 | 0.000000 | [0.000000, 0.000000] |
| 0.25 | -0.000248 | [-0.000774, +0.000253] |
| 0.50 | -0.000792 | [-0.001787, +0.000179] |
| 0.75 | -0.001702 | [-0.003152, -0.000271] |
| 1.00 | -0.002961 | [-0.004903, -0.001058] |

The degradation grows approximately monotonically with fusion strength.
Decoder-level2-only fusion is significantly harmful
(-0.001737, CI95 [-0.002265, -0.001209]). Decoder-level3-only fusion has no
detected gain (-0.001048, CI95 [-0.002718, +0.000583]).

Oracle conditions against F1 are also significantly negative:

| Oracle condition | Macro delta PSNR | CI95 |
|---|---:|---:|
| GT mean + predicted precision | -0.003303 | [-0.005135, -0.001509] |
| GT mean + unit precision | -0.002267 | [-0.003476, -0.000997] |

Unit precision is an experimental condition, not a mathematical upper bound.
These results establish only that the current GT-derived degradation target
has no usable headroom through the current decoder FiLM/spectral injection.
They do not reject Bayesian flow networks in general.

### 1.4 Frozen decisions

The following paths are closed:

- v26b, v26c, and v26d;
- full SSR fine-tuning;
- the encoder-level3 paper spectral gate;
- decoder-level2/level3 latent FiLM/spectral injection;
- category supervision;
- additional fusion-gain, precision, or v26 loss-weight sweeps;
- a frozen-SSR continuation that retains the v26 latent target or adapters.

The original SSR result F0 remains the system baseline.

The statement in `FULLFT_BFN_V26.md` that freezing the foundation is itself an
upper bound is historical and falsified for the current development setting.
v26 showed that full fine-tuning creates a Snow/Outdoor-Rain tradeoff rather
than a robust aggregate gain.

### 1.5 Stage A0 result and exploratory branch B

The preregistered C0 epoch-6 endpoint is frozen as STOP:

```text
macro delta = +0.001658, CI95 [-0.004788, +0.008317]
Snow        = +0.026181, CI95 [+0.020241, +0.032153]
Outdoor     = -0.017652, CI95 [-0.028211, -0.007059]
RainDrop    = -0.028080, CI95 [-0.048963, -0.007226]
```

Epoch 3 is exploratory, not a replacement primary endpoint:

```text
macro delta = +0.005099, CI95 [+0.001063, +0.009196]
Snow        = +0.023480, CI95 [+0.020030, +0.027010]
Outdoor     = -0.012500, CI95 [-0.020220, -0.004440]
RainDrop    = -0.014070, CI95 [-0.026160, -0.001660]
```

Both rain categories still significantly regress, so the registered D1/D4/B4
arms remain blocked. The explicitly non-preregistered C0-E1 branch changes
only sampling:

```text
Snow 0.30 / Outdoor-Rain 0.45 / RainDrop 0.25
```

It retains the complete six-epoch optimizer and LR schedule. Epoch 3 is fixed
in advance as its only decision endpoint; epochs 1, 2, 4, 5, and 6 may be
logged but cannot select the decision checkpoint.

## 2. v27 Claim Boundary

The retained research constraint is BFN. The rejected component is the v26
combination of:

```text
degradation/category descriptor
    -> internal decoder FiLM/spectral modulation
    -> restored image
```

v27 instead models the restoration residual directly:

```text
frozen SSR: x -> (y0, detached final decoder feature)
                         |
                         v
residual receiver + Bayesian state over normalized RGB residual
                         |
                         v
                    correction
                         |
                         v
                 y = clamp(y0 + correction)
```

The admissible v27 claim is:

> A Bayesian flow over the SSR restoration residual improves restoration
> beyond an exactly parameter-matched deterministic residual refiner.

A gain over F0 alone is a residual-refiner system gain. It is not evidence for
BFN unless the matched Bayesian arm also beats the matched deterministic arm.

## 3. Mathematical Contract

For degraded input `x` and target `y_gt`, define:

```text
y0 = clamp(SSR(x), 0, 1)
r* = (y_gt - y0) / s
s  = 0.2
```

The residual state is image-space RGB:

```text
mu_0  = zeros_like(y0)
rho_0 = ones_like(y0[:, :1])
```

At step `k`, one shared receiver predicts an RGB observation:

```text
o_k = R(mu_{k-1}, rho_{k-1}^{-1}, t_k,
        y0, f_ssr, x, x - y0)
```

`f_ssr` is the detached final SSR decoder feature from the same frozen
foundation pass. It is conditioning only; no module is inserted into SSR.

The two registered state updates are:

```text
Deterministic:
rho_k = rho_{k-1} + alpha_k
mu_k  = mu_{k-1}
        + alpha_k / rho_k * o_k

Bayesian:
rho_k = rho_{k-1} + alpha_k
mu_k  = (rho_{k-1} * mu_{k-1} + alpha_k * o_k) / rho_k
```

The deployed correction is the posterior/state mean itself:

```text
correction = s * mu_K
y          = clamp(y0 + correction, 0, 1)
```

There is no extra terminal receiver call and no learned output confidence gate
in v27a. Precision already controls the state update and is supplied to the
receiver as uncertainty. An additional confidence gate would be a separate
behavioral variable and is deferred.

## 4. Architecture Contract

### 4.1 Frozen SSR

The foundation must use:

```yaml
foundation:
  freeze: true
  frozen_modules_eval: true
  trainable_modules: []
  feature_adapters: {}
```

Requirements:

- all SSR parameters have `requires_grad=False`;
- SSR remains in eval mode during training;
- no encoder gate, decoder FiLM, state adapter, or other hook is registered;
- `y0` and `f_ssr` are produced under `torch.no_grad()`;
- v27 initialization reproduces F0 elementwise before any optimizer step.

### 4.2 Residual receiver

Reuse `sgbformer.models.receiver.ResidualReceiver` with:

```yaml
receiver_type: unet
condition_mode: observation_rich
base_channels: 16
channel_multipliers: [1, 2, 4]
blocks_per_level: 2
attention_heads: 4
dropout: 0.0
receiver_output_mode: identity
```

For the released SSR (`decoder_channels=48`) this receiver has exactly
**614,455 trainable parameters**. `ResidualBayesianFlow` has no trainable
parameters, so all trainable v27 arms must have exactly this count and exactly
the same named parameter signature.

The deployed total is therefore:

```text
10,825,680 frozen SSR parameters
   614,455 trainable receiver parameters
------------
11,440,135 total parameters
```

No width adjustment is allowed after seeing proxy quality.

### 4.3 Identity-safe initialization

`ResidualReceiver.output.weight` and `.bias` remain exactly zero initialized.
The fixed residual normalization scale remains `s=0.2`.

Do **not** also initialize a multiplicative residual gate to zero. A zero final
projection plus a zero gate would block the image-loss gradient twice. The
zero final projection alone guarantees:

```text
correction == 0
y_v27 == y0
```

At the first image-only backward pass, the output projection must receive a
nonzero gradient while upstream receiver blocks may receive zero gradient.
After exactly one optimizer step, a second image-only backward must produce a
nonzero gradient upstream of the output projection.

### 4.4 Residual state and schedule

Stage A freezes:

```yaml
residual_scale: 0.2
mask_mode: none
accuracy_schedule: linear_beta
min_variance: 0.01
min_accuracy_scale: 1.0
deployment_stochastic: false
```

The state is the normalized three-channel RGB residual. No category one-hot
vector, severity descriptor, hand-built frequency target, coarse degradation
map, or category classifier is permitted.

The receiver is already multi-scale. An explicit FFT/wavelet residual state is
deferred until image-space residual BFN passes Stage A; adding it now would
confound target alignment with representation choice.

## 5. Registered Arms

All trainable arms share the same receiver instance definition, optimizer,
data, augmentation, initialization seed, EMA, residual scale, and final-image
loss. Step count and state-update policy are parameter-free.

| Arm | Receiver calls | State behavior | Purpose |
|---|---:|---|---|
| F0 | 0 | none | original SSR baseline |
| C0 | 1 | direct receiver output from neutral state | tests added deterministic capacity |
| D1 | 1 | one deterministic state update, output `s * mu_1` | tests the residual-state interface |
| D4 | 4 | four deterministic state updates, output `s * mu_4` | compute/recurrence-matched control |
| B4 | 4 | four Bayesian updates, output `s * mu_4` | complete v27 method |

The critical comparisons are:

```text
C0 - F0: can a small residual refiner improve SSR?
D1 - C0: does the explicit residual-state interface help?
D4 - D1: does repeated deterministic refinement help?
B4 - D4: does the Bayesian update help at matched capacity and calls?
B4 - F0: does the complete system improve restoration?
```

`B4-D1` is not an admissible BFN attribution because it changes both update
rule and receiver-call count.

## 6. Implementation Boundary

The implementation may reuse:

- `ResidualBayesianFlow`;
- `ResidualReceiver`;
- the external SSR foundation and its final decoder feature;
- existing checkpoint, EMA, data, metric, and paired-bootstrap utilities.

The implementation must add explicit residual-flow controls:

- update mode: `deterministic` or `bayesian`;
- readout mode: `direct` for C0 or `posterior_mean` for D1/D4/B4;
- registered step count;
- a common state-update helper used by training and inference;
- per-run logging of update mode, readout mode, calls, residual scale, and
  trainable parameter signature.

The old `latent_bfn` path must remain disabled. v27 must not silently reuse
`DegradationLatentBFN`, `degradation_latent_targets`, category logits, or
latent-conditioned SSR adapters.

Because this work changes `sgbformer/**`, it must use a new source release and
implementation fingerprint. Existing v26 and frozen-main checkpoints may only
be evaluated from their original archived source trees; they must not be
resumed from the v27 tree.

## 7. Loss Contract

All arms optimize the restored image, not the frozen SSR output:

```yaml
structural_weight: 0.0
final_weight: 1.0
final_loss_type: mse
frequency_weight: 0.0
correlation_weight: 0.0
adapter_weight: 0.0
uncertainty_weight: 0.0
latent_category_weight: 0.0
```

For D1, D4, and B4:

```yaml
bfn_weight: 0.01
```

C0 uses `bfn_weight: 0.0`. D4 and B4 must use the same sender, sampled step
distribution, alpha schedule, auxiliary BFN loss, and rollout-gradient policy.
Their only behavioral difference is the parameter-free state-update equation.

The normalized target is always `(y_gt - y0) / 0.2`. Loss code must never use
`x - y_gt`, weather category, or the v26 degradation descriptors.

## 8. Data And Optimization Contract

Stage A uses:

```text
train manifest: ${SGB_DATA_ROOT}/train_pairs.tsv
dev manifest:   ${SGB_DATA_ROOT}/val_pairs.tsv
train/dev:      approximately 17,166 / exactly 903
patch:          256
augmentation:   enabled
sampling:       shuffle
seed:           42
global batch:   16
micro batch:    2
accumulation:   8
workers:        12
epochs:         6
new-module LR:  2e-4
warmup:         1 epoch
cosine floor:   0.1 of the arm's initial LR
weight decay:   1e-4
gradient clip:  1.0
EMA:            0.999, warmup 2,000 updates
BF16/TF32:      enabled
validation:     every epoch
checkpoint:     every epoch
```

No category-balanced sampling is permitted in Stage A. The natural training
distribution is retained to avoid reproducing the Snow/Outdoor-Rain transfer.

D4 and B4 use:

```text
training rollout steps: 4
rollout backprop steps: 1
rollout batch fraction: 1.0
```

D1 uses one rollout step and one backpropagated step. C0 uses the residual
training path with one receiver call.

## 9. Mandatory Preflight

No proxy may start until the common preflight artifact and that arm's profile
artifact record all checks below.

### 9.1 Structural checks

- F0 contains no trainable parameters.
- C0/D1/D4/B4 each contain exactly 614,455 trainable parameters.
- Their sorted trainable parameter names and shapes are identical.
- No trainable or registered module exists under `backbone`.
- `feature_adapters` is empty.
- `latent_flow` is absent or disabled.
- No category classifier or category target is present.

### 9.2 Identity checks

On at least three image shapes, including one not divisible by the SSR required
multiple:

- receiver raw output is exactly zero;
- correction is exactly zero;
- v27 output equals F0 output with `rtol=0`, `atol=0`;
- saved per-image evaluation metrics reproduce F0 within `1e-8`;
- padding and cropping preserve the original image shape.

### 9.3 Gradient checks

Using image MSE only:

1. first backward: SSR gradients absent, output projection gradient nonzero;
2. one optimizer step;
3. second backward: at least one upstream receiver parameter gradient nonzero;
4. SSR parameters and buffers remain unchanged;
5. D4 and B4 both reach every intended trainable parameter under their common
   BFN loss.

### 9.4 Causal checks

- D4 and B4 configs differ only in `update_mode` and output directory.
- Equal inputs and equal mocked observations produce the analytically expected
  deterministic and Bayesian states.
- K=1 deterministic and Bayesian updates are equal from the neutral state.
- K>1 deterministic and Bayesian states diverge on a fixed nontrivial
  observation sequence.
- Training and deterministic evaluation implement the same update equations.

### 9.5 Runtime checks

- 200 optimizer steps per trainable arm;
- no validation during profiling;
- peak allocated and incremental GPU memory;
- seconds per optimizer step;
- receiver calls per image;
- no autograd graph during evaluation;
- repeated deterministic evaluation is elementwise stable.

An OOM, nonfinite loss, parameter mismatch, F0 mismatch, or missing gradient is
a preflight failure. It is not permission to silently reduce width or change
the batch recipe.

## 10. Stage A Execution And Gates

### 10.1 Stage A0: capacity gate

Run C0 for exactly six epochs first. Evaluate all six EMA checkpoints against
the same F0 JSON with:

- explicit `--pairs-file`;
- Y channel;
- crop border 0;
- 10,000 paired bootstrap samples;
- seed 42;
- per-category, natural-mean, and macro-weighted results.

Primary result: the preregistered epoch-6 comparison.

Secondary result: the best macro checkpoint among epochs 1--6, explicitly
labelled as selected on the same dev set. Its image-bootstrap CI is conditional
on that selected checkpoint and does not correct selection bias.

Capacity-gate decision:

- epoch-6 macro delta > 0 and CI95 lower bound > 0: **GO**;
- epoch-6 macro delta > 0 and CI95 covers 0: **CONDITIONAL GO** to D1/D4/B4;
- epoch-6 macro delta <= 0, or CI95 upper bound <= 0: **STOP** the v27 residual
  family;
- any category with delta <= -0.02 dB and CI95 upper bound < 0: **STOP** even
  if macro is positive.

No LR, width, loss, or sampling adjustment is allowed after a Stage A0 stop.

### 10.2 Stage A1: residual-flow attribution

Only after Stage A0 GO or CONDITIONAL GO, run D1, D4, and B4 for the same fixed
six epochs.

Required epoch-6 comparisons:

- each arm versus F0;
- D1 versus C0;
- D4 versus D1;
- B4 versus D4;
- B4 versus F0.

Required reporting:

- Y-PSNR and Y-SSIM;
- per-category, natural mean, and macro weighted;
- 10,000 paired image-bootstrap CI95, seed 42;
- trainable and total parameters;
- time/image, peak memory, and receiver calls;
- all six epoch results, not only the selected maximum.

Decision:

- B4 versus F0 lower CI95 > 0 **and** B4 versus D4 lower CI95 > 0:
  **BFN GO**;
- B4 beats F0 but not D4: residual-refiner gain only, **no BFN claim**;
- D4 and B4 both fail F0: **NO-GO**;
- any significant category loss of at least 0.02 dB: **NO-GO**;
- any CI covering zero is reported as "no clear difference detected", never
  as equivalence.

If the B4-D4 point estimate is below 0.10 dB, at least three independent
training seeds are mandatory before any paper claim. Image bootstrap does not
replace training-seed replication.

## 11. Stage B Lock

Stage B is launch-locked until BFN GO.

If unlocked, both D4 and B4 must:

- start from their unchanged Stage A implementation;
- unfreeze only `backbone.model.output`;
- use SSR output-layer LR `1e-6`;
- use receiver LR in `[2e-5, 5e-5]`, fixed before launch;
- run the same 3-epoch schedule;
- retain all parameter matching and reporting requirements.

Do not unfreeze the complete SSR or any earlier decoder level.

Stage B stops both paired arms if either arm has an Outdoor-Rain delta of at
most -0.02 dB with CI95 upper bound below zero. The decision must be paired;
one arm may not continue alone.

## 12. Required Implementation Artifacts

Implementation is incomplete until all of the following exist:

```text
docs/V27_RESIDUAL_BFN_DESIGN_CONTRACT.md
configs/experiments/_ssr_residual_bfn_v27_common.yaml
configs/experiments/ssr_residual_v27_c0_proxy.yaml
configs/experiments/ssr_residual_v27_d1_proxy.yaml
configs/experiments/ssr_residual_v27_d4_proxy.yaml
configs/experiments/ssr_residual_v27_b4_proxy.yaml
scripts/preflight_v27_residual_bfn.py
scripts/profile_v27_residual_bfn.py
scripts/analyze_v27_proxy.py
tests/test_v27_residual_bfn.py
```

The preflight must emit a machine-readable JSON containing:

- source implementation fingerprint;
- config fingerprints;
- foundation checkpoint fingerprint;
- train/dev manifest fingerprints;
- trainable parameter names, shapes, and counts;
- identity and gradient results;
- optimizer groups;
- registered arm differences.

Each arm's profile JSON must record the matching preflight/config fingerprints,
200-step throughput, peak GPU memory, optimizer geometry, and receiver calls.

Release packaging remains a complete standalone source tar.gz. Checkpoints and
datasets remain external absolute paths or environment-variable references.
No separate `.sha256` upload is required.

## 13. Execution Order

From the standalone v27 source release:

```bash
cd /path/to/SGBformer-v27-residual-bfn
conda activate diffusionad

export SGB_DATA_ROOT=/path/to/allweather
export SSR_REPO=/path/to/SSR
export SSR_CHECKPOINT="$SSR_REPO/Allweather/pretrained_models/net_g_best.pth"
```

Run the common preflight:

```bash
python3 scripts/preflight_v27_residual_bfn.py \
  --config-c0 configs/experiments/ssr_residual_v27_c0_proxy.yaml \
  --config-d1 configs/experiments/ssr_residual_v27_d1_proxy.yaml \
  --config-d4 configs/experiments/ssr_residual_v27_d4_proxy.yaml \
  --config-b4 configs/experiments/ssr_residual_v27_b4_proxy.yaml \
  --output-dir runs/residual_bfn_v27/preflight \
  --device cuda
```

Profile C0:

```bash
python3 scripts/profile_v27_residual_bfn.py \
  --config configs/experiments/ssr_residual_v27_c0_proxy.yaml \
  --preflight runs/residual_bfn_v27/preflight/preflight.json \
  --output runs/residual_bfn_v27/profiles/C0.json \
  --device cuda
```

Only then may C0 be launched:

```bash
python3 train.py \
  --config configs/experiments/ssr_residual_v27_c0_proxy.yaml \
  --device cuda
```

After C0 finishes, evaluate every EMA epoch into one directory:

```bash
mkdir -p runs/residual_bfn_v27/eval/C0
for checkpoint in runs/residual_bfn_v27/c0_proxy/epoch_000[1-6].pt; do
  name="$(basename "$checkpoint" .pt)"
  python3 evaluate.py \
    --checkpoint "$checkpoint" \
    --input-dir "$SGB_DATA_ROOT/input" \
    --target-dir "$SGB_DATA_ROOT/gt" \
    --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
    --device cuda \
    --test-y-channel \
    --crop-border 0 \
    --compare-structural \
    --quiet \
    --json "runs/residual_bfn_v27/eval/C0/${name}.json"
done
```

Produce the Stage A0 decision:

```bash
python3 scripts/analyze_v27_proxy.py \
  --baseline "$DEV_FOUNDATION_JSON" \
  --arm C0=runs/residual_bfn_v27/eval/C0 \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --output-dir runs/residual_bfn_v27/analysis/stage_a0 \
  --bootstrap-samples 10000 \
  --seed 42
```

The D1/D4/B4 training entry points verify this decision file and refuse to run
unless it contains GO or CONDITIONAL_GO. Their profiles must then be generated
with the same profile command and their corresponding config/output names.

## 14. Explicit Non-Goals

The following are not part of v27 Stage A:

- explicit frequency-domain latent targets;
- category labels or category-balanced sampling;
- learned confidence gates;
- internal SSR adapters;
- SSR fine-tuning;
- TTA, checkpoint averaging, LPIPS optimization, or tiling;
- loss-weight sweeps;
- width or depth sweeps;
- official four-set testing;
- claims based on the 903-image dev set alone.

These exclusions preserve one causal question:

> Does a matched Bayesian residual flow add restoration value beyond a matched
> deterministic residual refiner while preserving the original SSR behavior?

## 15. Post-C0 Structural Correction: C0-prime

C0-prime is a separate preregistered exploratory arm created after C0 and
C0-E1 exposed rain-category conflict. It does not alter the original v27
Stage-A evidence or unlock D1/D4/B4.

Its sole behavioral change relative to C0 is:

```text
shared GroupNorm -> SiLU -> Conv3x3 readout
    ->
three private GroupNorm -> SiLU -> Conv3x3 heads
mixed by a GAP -> Linear -> softmax router
```

All head output convolutions and the router weight are zero initialized. The
router biases are equal `log(1/3)`, so routing starts uniformly and the initial
correction is exactly zero. Deterministic `1e-3` private GroupNorm affine
offsets break expert gradient symmetry without changing that identity.

The SSR, residual receiver trunk, shuffle sampler, loss, optimizer, LR,
schedule, epochs, seed, batch, accumulation, augmentation, direct deterministic
mode, and one-step execution remain identical to C0. Router supervision is
disabled and `loss.adapter_weight` remains zero.

For this arm only, the existing detached `adapter_regularization` diagnostic
carries router entropy. Consequently, `train.adapter` in `metrics.jsonl` is
the epoch-average router entropy, not a loss or regularization term.

The implementation and fixed endpoint are:

```text
config:    configs/experiments/ssr_residual_v27_c0prime_routed.yaml
preflight: scripts/preflight_v27_c0prime.py
profile:   scripts/profile_v27_residual_bfn.py
analysis:  scripts/analyze_v27_c0prime.py
endpoint:  epoch_0003.pt only
```

The complete static rationale and acceptance gates are recorded in
`docs/v27_c0prime_static_audit.md`.
