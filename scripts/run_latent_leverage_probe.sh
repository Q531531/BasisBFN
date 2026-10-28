#!/usr/bin/env bash
# Run the fixed-dev latent leverage probe from an unpacked source release.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${SGB_DATA_ROOT:?Set SGB_DATA_ROOT to the directory containing input/, gt/, and val_pairs.tsv}"
: "${CHECKPOINT:?Set CHECKPOINT to the selected EMA epoch checkpoint}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
INPUT_DIR="${PROBE_INPUT_DIR:-$SGB_DATA_ROOT/input}"
TARGET_DIR="${PROBE_TARGET_DIR:-$SGB_DATA_ROOT/gt}"
PAIRS_FILE="${PROBE_PAIRS_FILE:-$SGB_DATA_ROOT/val_pairs.tsv}"
CHECKPOINT_BASENAME="$(basename "$CHECKPOINT")"
CHECKPOINT_STEM="${CHECKPOINT_BASENAME%.*}"
OUTPUT_DIR="${PROBE_OUTPUT_DIR:-runs/latent_leverage/$CHECKPOINT_STEM}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
SEED="${SEED:-42}"

[[ -f "$CHECKPOINT" ]] || {
  printf 'Checkpoint does not exist: %s\n' "$CHECKPOINT" >&2
  exit 2
}
[[ -d "$INPUT_DIR" ]] || {
  printf 'Input directory does not exist: %s\n' "$INPUT_DIR" >&2
  exit 2
}
[[ -d "$TARGET_DIR" ]] || {
  printf 'Target directory does not exist: %s\n' "$TARGET_DIR" >&2
  exit 2
}
[[ -f "$PAIRS_FILE" ]] || {
  printf 'Pairs manifest does not exist: %s\n' "$PAIRS_FILE" >&2
  exit 2
}

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

command=(
  "$PYTHON_BIN" scripts/probe_latent_leverage.py
  --checkpoint "$CHECKPOINT"
  --input-dir "$INPUT_DIR"
  --target-dir "$TARGET_DIR"
  --pairs-file "$PAIRS_FILE"
  --device "$DEVICE"
  --output-dir "$OUTPUT_DIR"
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"
  --seed "$SEED"
)

if [[ -n "${PROBE_CONFIG:-}" ]]; then
  [[ -f "$PROBE_CONFIG" ]] || {
    printf 'PROBE_CONFIG does not exist: %s\n' "$PROBE_CONFIG" >&2
    exit 2
  }
  command+=(--config "$PROBE_CONFIG")
fi
if [[ "${PROBE_FUSION_SWEEP:-0}" == "1" ]]; then
  command+=(--fusion-sweep)
fi
if [[ "${PROBE_SINGLE_INJECTION_SWEEP:-0}" == "1" ]]; then
  command+=(--single-injection-sweep)
fi
if [[ "${PROBE_FUSION_ALPHA_MODE:-relative}" != "relative" ]]; then
  command+=(--fusion-alpha-mode "$PROBE_FUSION_ALPHA_MODE")
fi
if [[ -n "${PROBE_FUSION_ALPHAS:-}" ]]; then
  IFS=',' read -r -a fusion_alphas <<<"$PROBE_FUSION_ALPHAS"
  for alpha in "${fusion_alphas[@]}"; do
    command+=(--fusion-alpha "$alpha")
  done
fi
if [[ "${PROBE_ORACLE:-0}" == "1" ]]; then
  command+=(--oracle)
fi
if [[ -n "${PROBE_ORACLE_PRECISION_SCALES:-}" ]]; then
  IFS=',' read -r -a precision_scales <<<"$PROBE_ORACLE_PRECISION_SCALES"
  for scale in "${precision_scales[@]}"; do
    command+=(--oracle-precision-scale "$scale")
  done
fi
if [[ "${PROBE_OVERWRITE:-0}" == "1" ]]; then
  command+=(--overwrite)
fi

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
