# v27 C0-double-prime Private Bottleneck Static Audit

## 1. Decision Basis

The fixed C0-prime epoch-3 route probe attributed the failure to **B. Expert
symmetry collapse**:

- router entropy: `1.098611267`;
- `ln(3) - entropy`: `0.000001022`;
- forced-expert macro delta range: `0.000086 dB`;
- category-oracle gain over learned routing: `0.000063 dB`;
- per-image target-leaking oracle gain: `0.000134 dB`.

The router had no materially different experts to select. C0-double-prime
therefore changes expert parameterization only. It does not add router labels,
change the residual objective, rebalance categories, or alter training.

## 2. Single Behavioral Variable

Reference: `C0-prime`

Candidate: `C0-double-prime`

The only behavioral config difference is:

```text
model.bfn.routed_readout.head_type
```

Unchanged fields include:

- shared residual receiver trunk;
- image-level `GAP -> Linear(16, 3) -> softmax(dim=1)` router;
- no category labels and no router cross-entropy;
- uniform router initialization;
- zero-initialized final RGB projection;
- shuffle sampling and the original ordered train/dev manifests;
- seed, optimizer, learning rate, schedule, loss, augmentation, batch size,
  accumulation, EMA, and six-epoch budget;
- fixed epoch 3 as the only eligible endpoint;
- frozen SSR foundation and one direct deterministic receiver call.

## 3. Private Expert Head

C0-prime uses this shallow head:

```text
GroupNorm(16)
SiLU
Conv2d(16, 3, 3x3) [zero initialized]
```

C0-double-prime uses three separately constructed private heads:

```text
GroupNorm(16)
Conv2d(16, 16, 3x3) [independent default initialization]
SiLU
GroupNorm(16)
SiLU
Conv2d(16, 3, 3x3) [zero initialized]
```

The final convolution keeps the initial correction exactly zero. The private
bottleneck convolution produces different hidden features before that final
projection, so the three final projections receive materially different
gradients on the first backward pass. Router gradients are expected to be zero
on the first backward pass because all expert outputs are zero, then nonzero
after one optimizer update.

The existing `head_symmetry_break_epsilon=0.001` GroupNorm offset remains
unchanged. It is no longer the primary symmetry-breaking mechanism.

## 4. Exact Parameter Budget

At 16 receiver channels and 3 RGB residual channels:

| Component | Parameters |
| --- | ---: |
| first GroupNorm per head | 32 |
| private `Conv2d(16,16,3x3)` per head | 2,320 |
| second GroupNorm per head | 32 |
| zero RGB `Conv2d(16,3,3x3)` per head | 435 |
| one private head | 2,819 |
| three private heads | 8,457 |
| shared router `Linear(16,3)` | 51 |
| complete routed readout | 8,508 |

The C0 shared readout has 467 parameters. The expected model count is:

```text
614,455 - 467 + 8,508 = 622,496
```

This is `+8,041` parameters versus C0 (`+1.3086%`) and remains below the
existing `2x C0` ceiling.

## 5. Preflight Acceptance

Before any training, the dedicated preflight must prove:

1. resolved C0-double-prime differs from C0-prime only at the registered
   protocol/artifact paths and `routed_readout.head_type`;
2. trainable parameter count is exactly `622,496`;
3. all SSR parameters are frozen and only `receiver.*` is trainable;
4. all three private bottleneck convolutions are independently initialized;
5. every final RGB projection and the router weight are exactly zero;
6. eight seeded dev images and a zero input reproduce F0 to `1e-6`;
7. the initial candidate loss differs from F0 by at most 1%;
8. all final heads receive nonzero first-step gradients;
9. minimum pairwise relative gradient distance is at least `0.01`;
10. first-step router gradient is zero and second-step router gradient is
    finite and greater than `1e-12`;
11. router entropy is finite, detached, and initially equal to `ln(3)`;
12. the SSR state hash is unchanged after the gradient probe.

The 200-step CUDA profile remains mandatory after preflight. It preserves the
2 GiB peak-allocation gate and records throughput and router entropy.

## 6. Fixed Endpoint And Stop Rule

Only `epoch_0003.pt` may be evaluated. The analyzer reuses the preregistered
C0-prime gates against F0 and original C0 epoch 3:

- positive macro point estimate;
- macro CI95 lower bound at least `-0.002 dB`;
- Snow safety gate;
- Outdoor-Rain point estimate at least `-0.003 dB` and CI95 upper bound
  nonnegative;
- RainDrop point estimate at least `-0.005 dB` and CI95 upper bound
  nonnegative;
- both rain categories improve over original C0 epoch 3.

Any failed gate is `STOP`. No checkpoint scan is permitted.

## 7. Current Stop Point

This implementation package ends after local tests and packaging. It does not:

- start C0-double-prime training;
- evaluate any checkpoint;
- start D4 or B4;
- access official test data;
- reinterpret deterministic-arm diagnostics as BFN evidence.
