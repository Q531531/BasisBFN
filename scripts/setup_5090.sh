#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"

cd "${PROJECT_DIR}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Using active conda environment: ${CONDA_DEFAULT_ENV:-${CONDA_PREFIX}}"
else
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
  echo "Using virtual environment: ${VENV_DIR}"
fi

python -m pip install --upgrade pip setuptools wheel
if ! python - <<'PY'
import torch

version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
assert version >= (2, 7)
assert torch.cuda.is_available()
torch.ones(1, device="cuda")
PY
then
  python -m pip install --upgrade \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7" "torchvision>=0.22"
fi
python -m pip install -e ".[all]" gdown

python - <<'PY'
import torch
import open_clip
import lpips

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment")

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("open_clip:", getattr(open_clip, "__version__", "installed"))
print("lpips:", getattr(lpips, "__version__", "installed"))
PY

echo "5090 environment is ready: ${VENV_DIR}"
