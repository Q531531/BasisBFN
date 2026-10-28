# BasisBFN

**Bayesian Residual Refinement with Image-Dependent Bases for Adverse Weather Image Restoration**

Official PyTorch implementation of **BasisBFN**, a compact Bayesian residual
refiner attached to a frozen SSR restoration backbone.

BasisBFN constructs four image-dependent RGB residual bases once, recurrently
updates the means and precisions of their spatial coefficients, and applies a
posterior-conditioned acceptance gate before adding the correction to the SSR
base image.

![BasisBFN architecture](docs/assets/basisbfn_architecture.png)

## Highlights

- **Frozen strong backbone.** SSR produces the base restoration and a
  48-channel decoder feature. SSR parameters remain frozen.
- **One-pass residual representation.** A 61-channel conditional U-Net builds
  a shared context and four local RGB residual bases once per image.
- **Coefficient-space Bayesian flow.** A shared observer performs four
  precision-weighted updates over `4 x H x W` coefficient means and
  precisions.
- **Posterior-conditioned acceptance.** A spatial gate uses image context,
  final coefficient means, and log-precisions to scale the proposed
  correction.
- **Identity-safe initialization.** Zero-initialized observer and gate output
  projections preserve the frozen SSR output at initialization.
- **Compact refinement.** The complete refinement branch has 624,217
  trainable parameters.

## Paper Configuration

The main model is **B4-G**:

| Setting | Value |
| --- | --- |
| Residual bases | `K = 4` |
| Coefficient updates | `T = 4` |
| Update | Bayesian precision-weighted fusion |
| Gate | Context + final mean + final log-precision |
| Residual scale | `0.2` |
| Coefficient target | Detached ridge projection, ridge `1e-3` |
| Coefficient BFN loss | Equation (15), weight `4e-5` |
| Optimizer | AdamW, LR `2e-4`, weight decay `1e-4` |
| Training | 6 epochs, batch 16, BF16, EMA `0.999` |
| Paper seeds | `42, 43, 44, 45` |

The matched ablation matrix is:

| Arm | Coefficient update | Acceptance gate |
| --- | --- | --- |
| `D4` | additive | off |
| `D4-G` | additive | on |
| `B4` | Bayesian | off |
| `B4-G` | Bayesian | on |

## Repository Layout

```text
sgbformer/models/coefficient_state.py  # basis, observer, flow, gate
sgbformer/models/receiver.py           # one-pass conditional U-Net
sgbformer/models/foundation.py         # external frozen SSR integration
sgbformer/models/bfn.py                # Gaussian sender and Bayesian update
sgbformer/models/model.py              # BasisBFN end-to-end model
configs/experiments/basisbfn_*.yaml    # B4-G and matched controls
scripts/preflight_basisbfn.py          # identity/gradient/protocol checks
scripts/profile_basisbfn.py            # 200-step CUDA smoke profile
scripts/analyze_basisbfn.py            # fixed endpoint matrix analysis
```

The Python package retains `sgbformer` as an import namespace for checkpoint
compatibility. The public model and loss names are `BasisBFN` and
`BasisBFNLoss`; `SGBFormer` and `SGBFormerLoss` remain aliases for older
checkpoints.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The official SSR source and checkpoint are runtime dependencies and are not
redistributed here. Set:

```bash
export SSR_REPO=/path/to/SSR
export SSR_CHECKPOINT="$SSR_REPO/Allweather/pretrained_models/net_g_best.pth"
export SGB_DATA_ROOT=/path/to/allweather
export ALLWEATHER_TEST_ROOT=/path/to/official/testallweather
```

`SGB_DATA_ROOT` must contain:

```text
input/
gt/
train_pairs.tsv
val_pairs.tsv
```

The paper split contains 17,166 training pairs and 903 held-out development
pairs.

## Generate A Paper-Seed Config

Generate an arm/seed overlay so runs and verification artifacts cannot
overwrite one another:

```bash
python3 scripts/make_basisbfn_config.py \
  --arm B4-G \
  --seed 42 \
  --output configs/generated/basisbfn_b4g_seed42.yaml
```

Valid paper seeds are `42`, `43`, `44`, and `45`.

## Preflight And CUDA Profile

The training entry point requires a matching preflight and 200-step profile.
For B4-G seed 42:

```bash
python3 scripts/preflight_basisbfn.py \
  --config configs/generated/basisbfn_b4g_seed42.yaml \
  --output-dir runs/basisbfn/artifacts/b4-g/seed_42/preflight \
  --device cuda

python3 scripts/profile_basisbfn.py \
  --config configs/generated/basisbfn_b4g_seed42.yaml \
  --preflight runs/basisbfn/artifacts/b4-g/seed_42/preflight/preflight_b4-g.json \
  --output runs/basisbfn/artifacts/b4-g/seed_42/profile.json \
  --device cuda
```

The preflight verifies the exact parameter count, frozen SSR state, initial
identity, DCT initialization, finite coefficient BFN loss, posterior precision
growth, gate bounds, and first/second-step gradient reachability.

## Training

```bash
python3 train.py \
  --config configs/generated/basisbfn_b4g_seed42.yaml \
  --device cuda
```

The paper endpoint is the epoch-6 EMA checkpoint:

```text
runs/basisbfn/b4-g/seed_42/epoch_0006.pt
```

Repeat the same sequence for seeds `43`, `44`, and `45`. Matched D4, D4-G,
and B4 controls use the same generator with a different `--arm`.

## Inference

```bash
python3 infer.py \
  --checkpoint runs/basisbfn/b4-g/seed_42/epoch_0006.pt \
  --input /path/to/degraded/images \
  --output results/basisbfn \
  --device cuda \
  --deterministic
```

## Evaluation

Development evaluation:

```bash
python3 evaluate.py \
  --checkpoint runs/basisbfn/b4-g/seed_42/epoch_0006.pt \
  --input-dir "$SGB_DATA_ROOT/input" \
  --target-dir "$SGB_DATA_ROOT/gt" \
  --pairs-file "$SGB_DATA_ROOT/val_pairs.tsv" \
  --device cuda \
  --test-y-channel \
  --compare-structural \
  --json results/basisbfn_b4g_seed42.json
```

Paper-facing evaluation uses the untouched Snow100K-S, Snow100K-L,
Outdoor-Rain Test1, and RainDrop Test-A subsets. See
[`docs/BASISBFN.md`](docs/BASISBFN.md) and the scripts in `scripts/` for the
fixed matrix analysis.

## Reported Manuscript Results

The manuscript reports four-seed means:

| Model | Average PSNR | Average SSIM |
| --- | ---: | ---: |
| SSR | 34.15 | 0.9442 |
| BasisBFN B4-G | **34.50** | **0.9483** |

Within the matched refinement matrix, the manuscript reports `+0.10 dB` from
Bayesian fusion and `+0.08 dB` from acceptance gating. These are paper results,
not claims produced by the unit tests in this repository.

## Tests

```bash
python3 -m pytest -q
```

## Third-Party Code And Weights

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). SSR source and weights
are not included because the upstream repository did not provide a license
file when this integration was prepared.

## Citation

Citation metadata will be added after the anonymous review process.
