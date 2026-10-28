#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_ROOT="${SSR_THIRD_PARTY_ROOT:-$(dirname "$PROJECT_ROOT")/third_party}"
SSR_REPO="${SSR_REPO:-$THIRD_PARTY_ROOT/SSR}"
SSR_CHECKPOINT="${SSR_CHECKPOINT:-$SSR_REPO/Allweather/pretrained_models/net_g_best.pth}"
SSR_FILE_ID="1adNJ-UiTcM9DUsqpvQknOpU-putgY_3u"
SSR_SHA256="18897da4561959ca122bb2e43aa3e70bb585e5255406305da3f43365f19d7284"

mkdir -p "$THIRD_PARTY_ROOT"
if [[ ! -d "$SSR_REPO/.git" ]]; then
  git clone --depth 1 https://github.com/LIDAYU-DayuLi/SSR.git "$SSR_REPO"
fi

python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  einops ninja packaging "numpy>=2,<2.3"

if ! python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" \
  >/dev/null 2>&1; then
  read -r torch_version cuda_version < <(
    python - <<'PY'
import torch
print(torch.__version__.split("+", 1)[0], torch.version.cuda or "none")
PY
  )
  if [[ "$torch_version" == 2.11.* && "$cuda_version" == 12.8* ]]; then
    python -m pip install \
      --index-url https://wheels.astral.sh/simple/cu128/ \
      --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --only-binary=:all: --no-deps \
      "mamba-ssm==2.3.2.post1+cu.12.8.torch.2.11"
  elif command -v nvcc >/dev/null 2>&1; then
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --no-build-isolation --no-deps mamba-ssm
  else
    echo "No matching Mamba wheel and nvcc is unavailable." >&2
    echo "Install a CUDA toolkit matching PyTorch before retrying." >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "$SSR_CHECKPOINT")"
if [[ ! -s "$SSR_CHECKPOINT" ]]; then
  temporary="${SSR_CHECKPOINT}.part"
  curl --fail --location --retry 5 --connect-timeout 30 \
    --output "$temporary" \
    "https://drive.usercontent.google.com/download?id=${SSR_FILE_ID}&export=download&confirm=t"
  mv "$temporary" "$SSR_CHECKPOINT"
fi

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python - "$SSR_REPO" "$SSR_CHECKPOINT" "$SSR_SHA256" <<'PY'
import hashlib
import sys
from pathlib import Path

from sgbformer.models.foundation import ExternalSSRFoundation

repo, checkpoint = map(lambda value: Path(value).resolve(), sys.argv[1:3])
actual_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if actual_sha256 != sys.argv[3]:
    raise RuntimeError(
        f"SSR checkpoint SHA-256 mismatch: {actual_sha256}"
    )
model = ExternalSSRFoundation(
    repo_path=str(repo),
    checkpoint=str(checkpoint),
    freeze=True,
    checkpoint_compatible_edffn=True,
    architecture={
        "inp_channels": 3,
        "out_channels": 3,
        "dim": 48,
        "num_blocks": [3, 3, 6],
        "ffn_expansion_factor": 3,
        "bias": False,
    },
)
print(f"SSR repository: {repo}")
print(f"SSR checkpoint: {checkpoint}")
print(f"SSR checkpoint SHA-256: {actual_sha256}")
print("SSR architecture and checkpoint load successfully")
PY

printf '\nSet these variables before training:\n'
printf 'export SSR_REPO=%q\n' "$SSR_REPO"
printf 'export SSR_CHECKPOINT=%q\n' "$SSR_CHECKPOINT"
