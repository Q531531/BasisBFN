# v27 C0-accept Identity-Safe Acceptance Gate

## 1. Evidence

The fixed C0-double-prime epoch-3 geometry audit selected:

```text
E2. Category-dependent accept/reject mismatch
```

Measured fixed scale optima were:

```text
Snow          0.75
Outdoor-Rain  0.25
RainDrop     -0.25
```

The current RGB-MSE gradient retained substantial alignment with the formal
Y-metric direction (`0.836105`), while the learned correction was not aligned
with the current loss descent direction (`-0.025746`). The next experiment
therefore tests only whether rejecting or attenuating the fixed correction can
protect rain images.

## 2. Single Behavioral Variable

C0-accept adds one image-level scalar gate after the existing C0-double-prime
soft expert mixture:

```text
frozen shared receiver feature
    -> GAP
    -> Linear(16,1), weight=0, bias=0
    -> hard clamp [0,1]
    -> acceptance a(x)

frozen C0-double-prime correction c(x)
    -> a(x) * c(x)
```

The gate has exactly 17 trainable parameters. The SSR foundation, receiver
trunk, three private bottleneck experts, and existing three-way soft router
are frozen.

The gate cannot invert the correction. RainDrop's negative diagnostic alpha is
interpreted as a rejection signal, so the deployable gate maps it toward zero.

## 3. Warm Start And Identity

Starting the gate and C0-double-prime experts from zero would produce a
zero-gradient deadlock. C0-accept therefore:

1. loads only the fixed C0-double-prime `epoch_0003.pt` EMA weights;
2. copies the complete frozen correction basis into the candidate;
3. initializes only the new gate weight and bias to exactly zero;
4. saves a fresh epoch `-1`, global-step `0` initialization checkpoint with a
   fresh optimizer, scheduler, and gate-only EMA.

At initialization:

```text
acceptance = clamp(0, 0, 1) = 0
restored = F0
```

PyTorch's clamp derivative at the lower boundary is `1`, and the warm-start
correction is nonzero, so the gate receives a first-step gradient while every
frozen parameter remains gradient-free.

## 4. Frozen Contract

Unchanged from C0-double-prime:

- train/dev manifests and shuffle sampling;
- seed 42;
- RGB MSE objective;
- optimizer, learning rate, schedule, batch and accumulation;
- six epochs and fixed epoch 3 endpoint;
- augmentation and EMA settings;
- direct deterministic one-call receiver path;
- no category labels, router CE, or adapter loss;
- no official test access.

For C0-accept only, `train.adapter` records detached mean acceptance rather
than router entropy. Its loss weight remains zero.

## 5. Preflight And Profile Gates

Preflight must verify:

1. source is C0-double-prime EMA `epoch_0003.pt`;
2. source checkpoint hash remains unchanged;
3. every shared state tensor is exactly transplanted;
4. only gate weight and bias are missing from the source state;
5. only 17 gate parameters require gradients;
6. eight seeded dev images exactly reproduce F0;
7. the hidden ungated correction exactly reproduces C0-double-prime;
8. gate output is exactly zero and lies in `[0,1]`;
9. first backward gives a nonzero gate gradient;
10. no frozen parameter receives a gradient;
11. the generated initialization checkpoint resumes at epoch 1 with matching
    config, optimizer, scheduler, and gate-only EMA.

The 200-step CUDA profile must load that exact initialization checkpoint,
verify its SHA against the preflight report, preserve the 2 GiB memory gate,
and require finite mean acceptance strictly inside `(0,1]` so a dead gate
cannot proceed to formal training.

## 6. Fixed Decision

Only C0-accept `epoch_0003.pt` is eligible.

The candidate must satisfy the existing F0 safety gates and improve both rain
categories over C0-double-prime epoch 3. Any failure is `STOP`.

No checkpoint scan, D4/B4 launch, official-test access, or additional gate,
router, expert, sampling, loss, or schedule change is permitted.
