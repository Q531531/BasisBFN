# v27 C0-prime Category-Routed Readout Static Audit

## 0. Status and scope

- Phase: **Step 0 static audit only**
- Selected design: **A, shared trunk + three category-routed readout heads**
- Code implementation: **not started**
- Config implementation: **not started**
- Preflight/profile/training: **not started**
- Stage A0 and C0-E1 conclusions remain frozen and unchanged.

This audit treats C0-prime as a new exploratory structural experiment. It does
not replace the preregistered C0 epoch-6 STOP result and does not reinterpret
C0 epoch 3 or C0-E1 as preregistered evidence.

## 1. Existing implementation boundary

The current deterministic C0 path is:

```text
frozen SSR
  -> y0 and detached decoder_level1 feature
  -> ResidualReceiver shared U-Net trunk
  -> output_norm (GroupNorm, 16 channels)
  -> SiLU
  -> output Conv2d(16, 3, kernel_size=3)
  -> correction
  -> clamp(y0 + 0.2 * correction)
```

Relevant implementation ownership:

```text
sgbformer/models/receiver.py
  ResidualReceiver
    time_embedding
    input_projection
    encoder_stages / middle / decoder_stages
    output_norm
    output

sgbformer/models/model.py
  SGBFormer.receiver_prediction
  SGBFormer.training_outputs
  SGBFormer.restore
```

The actual shared readout is only the final:

```text
GroupNorm(16) -> SiLU -> Conv2d(16, 3, 3x3)
```

It is not the complete 614,455-parameter receiver. Therefore design A does not
require three copies of the full receiver.

## 2. Causal target

The only behavioral variable is:

```text
shared readout
    ->
three private readout heads + one soft router
```

Everything before the readout remains shared and identical to C0:

- frozen SSR and its checkpoint;
- detached SSR output and decoder feature;
- residual receiver U-Net trunk;
- direct deterministic mode;
- one receiver call;
- residual scale 0.2;
- identity-safe output equation;
- shuffle sampling;
- loss, optimizer, LR, scheduler, epochs, seed, batch, accumulation, and
  augmentation.

This experiment tests whether the shared final readout causes incompatible
category corrections. It does not test BFN.

## 3. Candidate A versus B

### 3.1 Candidate A: selected

The selected module boundary is:

```text
ResidualReceiver shared trunk
            |
            v
       feature f [B,16,H,W]
        /       |       \
       /        |        \
head_snow   head_outdoor   head_raindrop
       \        |        /
        \       |       /
      soft router probabilities [B,3]
                 |
                 v
 correction = sum_k p_k * correction_k
```

Each private head copies the current readout block:

```text
GroupNorm(16) -> SiLU -> Conv2d(16, 3, 3x3)
```

Every final `Conv2d(16,3,3x3)` weight and bias is initialized to exactly zero.
The router is:

```text
AdaptiveAvgPool2d(1) -> Linear(16,3) -> softmax
```

The router linear weight is zero initialized. All three biases initialize to
`log(1/3)`, yielding exact uniform routing.

### 3.2 Candidate B: rejected for C0-prime

Candidate B would retain one shared output convolution and insert three routed
FiLM parameter sets before it.

Estimated extra parameters:

```text
three (gamma,beta) vectors: 3 * 2 * 16 = 96
router Linear(16,3):                    51
total extra:                           147
```

It is smaller, but it preserves the shared final projection that the current
causal hypothesis identifies as the conflict point. It also reintroduces a
feature-modulation mechanism immediately after v26 showed that internal
FiLM/spectral modulation did not provide robust restoration value.

Candidate A is therefore preferred. Its parameter increase is far below the
3x fallback threshold.

## 4. Parameter estimate

Current C0 receiver:

```text
614,455 trainable parameters
```

Current readout block:

```text
GroupNorm(16):             16 weights + 16 biases = 32
Conv2d(16,3,3x3):          3*16*3*3 + 3       = 435
shared readout total:                              467
```

Candidate A:

```text
three private readout blocks: 3 * 467 = 1,401
router Linear(16,3):                    51
replacement total:                  1,452
```

Expected C0-prime trainable count:

```text
614,455 - 467 + 1,452 = 615,440
```

Increase versus C0:

```text
+985 parameters
+0.1603%
ratio = 1.001603x
```

This is below both the requested 2x hard ceiling and the 3x A/B fallback
threshold.

The estimate must become an exact test assertion after implementation.

## 5. Symmetry and supervision audit

### 5.1 Constraint conflict

The proposed A design optionally adds router cross-entropy with weight 0.05.
That would change the loss and introduce category-label supervision, violating
the hard single-variable requirement:

```text
do not change loss
```

C0-prime therefore must use:

```text
router_ce_weight = 0
```

No category label may enter model forward or loss computation.

### 5.2 Symmetry failure in the naive implementation

If all three heads have:

- identical architecture;
- identical internal initialization;
- zero final convolutions;
- a uniform router;
- only downstream image loss;

then all head gradients are identical. The heads remain identical and the
router gradient remains zero. The routed model degenerates exactly to a shared
head and cannot test the stated hypothesis.

This must be prevented structurally without adding a loss.

### 5.3 Selected zero-preserving symmetry break

All heads keep the same layer count and channel dimensions. Their private
GroupNorm affine weights receive deterministic, zero-mean, head-specific
perturbations at initialization. Final convolutions remain exactly zero.

Consequences:

- initial correction remains exactly zero;
- initial restored output remains exactly F0;
- final-convolution gradients differ between heads after the first backward;
- once head outputs diverge, downstream image loss supplies router gradients;
- no category label or auxiliary loss is introduced.

The perturbation magnitude is frozen at:

```text
epsilon = 1e-3
```

It must be deterministic under seed 42 and tested for:

- exact zero correction before training;
- non-identical head gradients after the first backward;
- nonzero router gradient after one optimizer step;
- reproducible initialization.

The names `snow`, `outdoor_rain`, and `raindrop` are bookkeeping labels only.
Without router CE, the three experts are permutation-identifiable; no paper
claim may assert that a specific expert semantically represents its namesake
category without post-hoc evidence.

## 6. Router entropy without train.py changes

The requirement to print epoch-average router entropy conflicts with the
prohibition on modifying `train.py`.

The compatible design is:

1. `CategoryRoutedReadout` stores detached `last_router_entropy`.
2. For C0-prime only, `SGBFormer.adapter_regularization()` returns this detached
   entropy diagnostic.
3. `loss.adapter_weight` remains exactly `0.0`.
4. Existing training logs therefore expose router entropy through the existing
   `adapter=` diagnostic and epoch `train.adapter` metric.

This changes diagnostics only. It does not add a loss term or alter gradients.
The audit and README must state:

```text
For C0-prime, train.adapter is router entropy, not a regularization loss.
```

No `train.py`, `sgbformer/training.py`, scheduler, or loss-weight change is
required.

## 7. Proposed module class graph

```text
SGBFormer
├── backbone: ExternalSSRFoundation
│   ├── model: SSR                       [frozen]
│   └── adapters: {}                     [empty]
├── flow: ResidualBayesianFlow           [parameter-free, unused by C0-prime]
└── receiver: ResidualReceiver
    ├── time_embedding                   [shared]
    ├── input_projection                 [shared]
    ├── encoder_stages                   [shared]
    ├── middle                           [shared]
    ├── decoder_stages                   [shared]
    └── readout: CategoryRoutedReadout
        ├── router
        │   ├── AdaptiveAvgPool2d(1)
        │   └── Linear(16,3)              [zero weight, equal log(1/3) bias]
        └── heads
            ├── snow: RoutedResidualHead
            │   ├── GroupNorm(16)
            │   ├── SiLU
            │   └── Conv2d(16,3,3x3)      [zero initialized]
            ├── outdoor_rain: RoutedResidualHead
            └── raindrop: RoutedResidualHead
```

`ResidualReceiver` retains the original shared-readout path for C0 and all
historical checkpoints. Routed behavior is enabled only by an explicit config
block.

## 8. Proposed configuration diff

Target file:

```text
configs/experiments/ssr_residual_v27_c0prime_routed.yaml
```

Proposed diff relative to C0:

```yaml
_base_: ssr_residual_v27_c0_proxy.yaml

protocol:
  name: residual_bfn_v27_exploratory
  arm: C0-prime
  preregistered: true
  primary_endpoint: fixed_epoch_3
  checkpoint_selection: fixed_epoch_3_only

model:
  bfn:
    routed_readout:
      enabled: true
      categories: [snow, outdoor_rain, raindrop]
      router_type: global_average_linear
      router_supervision: none
      router_bias_init: uniform_log_probability
      head_type: groupnorm_silu_conv3x3
      head_symmetry_break_epsilon: 0.001
      zero_init_output: true

training:
  output_dir: runs/residual_bfn_v27_exploratory/c0prime_routed
```

Required inherited values, unchanged:

```text
training.sampling.mode = shuffle
training.epochs = 6
training.stages = [{name: residual, epochs: 6}]
training.learning_rate = 2e-4
training.warmup_epochs = 1
training.minimum_learning_rate_ratio = 0.1
training.batch_size = 16
training.micro_batch_size = 2
seed = 42
loss.final_weight = 1.0
loss.bfn_weight = 0.0
model.bfn.inference_mode = direct
model.bfn.update_mode = deterministic
model.bfn.steps = 1
```

Allowed administrative differences:

- protocol metadata;
- output directory;
- preflight/profile artifact paths.

Allowed behavioral difference:

```text
model.bfn.routed_readout
```

Every other resolved config field must compare equal to C0.

## 9. Planned implementation diff

No files below are modified in Step 0. Step 1 would use:

### New files

```text
configs/experiments/ssr_residual_v27_c0prime_routed.yaml
scripts/preflight_v27_c0prime.py
scripts/analyze_v27_c0prime.py
tests/test_v27_c0prime.py
```

### Modified files

```text
sgbformer/models/receiver.py
sgbformer/models/model.py
sgbformer/v27.py
sgbformer/protocol.py
scripts/profile_v27_residual_bfn.py
README.md
docs/V27_RESIDUAL_BFN_DESIGN_CONTRACT.md
```

### Explicitly untouched

```text
train.py
sgbformer/training.py
sgbformer/losses.py
all D4/B4 configs
all C0/C0-E1 configs
```

## 10. Step 1 acceptance checks

Before any profile or training:

1. Eight fixed dev images reproduce F0 with max absolute difference `<=1e-6`.
2. Zero input produces exactly zero correction.
3. Every routed final convolution is exactly zero initialized.
4. Initial router probabilities equal exactly `1/3` within floating tolerance.
5. Trainable parameters equal exactly 615,440.
6. Trainable parameters are below `2 * 614,455`.
7. SSR parameters have no gradients and remain byte-identical.
8. C0-prime and C0 resolved configs differ only at registered paths.
9. Sampling remains `shuffle`.
10. First backward separates head gradients; after one optimizer step the
    router receives a nonzero gradient.
11. Router entropy is finite and lies in `[0, log(3)]`.
12. `train.adapter` is documented and tested as detached router entropy.

## 11. Fixed future decision

Only `epoch_0003.pt` may be evaluated for the decision.

Required comparisons:

- C0-prime epoch 3 versus F0;
- C0-prime epoch 3 versus original C0 epoch 3.

All acceptance conditions remain exactly:

1. macro delta PSNR > 0 and CI95 lower bound >= -0.002;
2. Snow delta >= 0, or degradation <= 0.005 with CI95 upper bound >= 0;
3. Outdoor-Rain delta >= -0.003 with CI95 upper bound >= 0;
4. RainDrop delta >= -0.005 with CI95 upper bound >= 0;
5. Outdoor-Rain and RainDrop point estimates both improve versus C0 epoch 3.

Any failure produces C0-prime STOP. No checkpoint scan, official-test access,
loss/LR/schedule adjustment, or D4/B4 launch is permitted.

## 12. Step 0 decision

Proceed with candidate A only after explicit confirmation of these two
interpretations:

1. no router CE is used, because loss must remain identical to C0;
2. epoch-average router entropy is carried through the existing detached
   `adapter` diagnostic, because `train.py` must remain unchanged.
