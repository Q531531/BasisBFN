# v27 C0-prime Epoch-3 Route Probe

## Status

- Phase: implementation audit and probe tooling complete
- Runtime measurement: pending server execution
- Allowed checkpoint: `epoch_0003.pt` only
- Dataset: existing 903-image dev split only
- Training: prohibited
- Official test access: prohibited
- D4/B4 launch: prohibited

`scripts/probe_v27_c0prime_routing.py` replaces this document with measured
results after the registered read-only server run.

## Implementation Audit

### Actual structure

C0-prime is:

```text
frozen SSR
  -> shared ResidualReceiver trunk
  -> shared [B,16,H,W] feature
       -> private Snow output head
       -> private Outdoor-Rain output head
       -> private RainDrop output head
       -> image-level GAP + Linear router
  -> soft weighted sum of the three head corrections
```

It is not three complete receivers and it is not routed FiLM.

Source:

- `sgbformer/models/receiver.py:131-240`
- `sgbformer/models/receiver.py:243-366`
- `sgbformer/models/model.py:204-224`

### Parameter delta

The original C0 readout contains:

```text
GroupNorm(16):            32
Conv2d(16,3,3x3):        435
total:                   467
```

C0-prime replaces it with:

```text
three private heads: 3 * 467 = 1,401
router Linear(16,3):             51
replacement total:           1,452
delta:               1,452 - 467 = 985
```

Therefore:

```text
C0:       614,455
C0-prime: 615,440
```

The receiver trunk remains shared.

### Router granularity

The router applies:

```text
AdaptiveAvgPool2d(1) -> Linear(16,3) -> softmax(dim=1)
```

It produces one three-expert probability vector per image, not a spatial
router. See `sgbformer/models/receiver.py:200-206,224-237`.

### Supervision

There is no category label input and no router cross-entropy:

```text
model.bfn.routed_readout.router_supervision = none
loss.adapter_weight = 0
loss.latent_category_weight = 0
```

The expert names are bookkeeping labels and are permutation-identifiable.

### Initialization and symmetry

Every head output convolution starts at exactly zero. The three heads do not
have exactly identical internal initialization: private GroupNorm affine
weights receive deterministic offsets `[-1,0,+1] * 1e-3`. This preserves zero
initial correction while separating first-step head gradients.

Source: `sgbformer/models/receiver.py:150-163`.

### `train.adapter`

For routed receivers, `SGBFormer.adapter_regularization()` returns detached
router entropy:

- generation: `sgbformer/models/model.py:533-539`
- loss diagnostic extraction: `sgbformer/losses.py:155-191`
- epoch accumulation and serialization:
  `train.py:732-742,902-903,987-1012`

Because `loss.adapter_weight=0`, this field is diagnostic only.

## Probe Method

The probe registers read-only forward hooks on the router linear layer and
the three private heads. One normal forward pass per image captures:

- router logits and probabilities;
- all three expert corrections;
- the deployed learned-soft correction.

The weighted head sum must reproduce the model correction within `1e-6`.
Forced experts, six category mappings, and the per-image oracle are then
assembled from captured tensors without mutating the model.

The probe verifies before exit:

- checkpoint SHA-256 unchanged;
- model state SHA-256 unchanged;
- no parameter gradients;
- reconstructed soft correction matches the deployed output;
- F0 and learned-soft per-image metrics reproduce the registered JSONs.

## Attribution Rules

The script emits exactly one main attribution:

- **A Router did not learn:** near-uniform router with useful expert headroom.
- **B Expert symmetry collapse:** nearly identical corrections and forced
  results.
- **C Wrong allocation:** differentiated router/experts with oracle routing
  headroom.
- **D Receiver/objective mismatch:** no material expert headroom, including
  the target-leaking per-image oracle.

All thresholds are serialized into `summary.json`. Oracle results are
mechanism diagnostics and cannot be reported as deployable model performance.
