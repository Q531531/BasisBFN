#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${ALLWEATHER_TEST_ROOT:?Set ALLWEATHER_TEST_ROOT to testallweather}"
: "${SSR_REPO:?Set SSR_REPO to the official SSR repository}"
: "${SSR_CHECKPOINT:?Set SSR_CHECKPOINT to net_g_best.pth}"

CONFIG="${1:-configs/experiments/basisbfn_b4g.yaml}"
OUTPUT_DIR="${2:-results/ssr_official}"
mkdir -p "$OUTPUT_DIR"

python scripts/check_allweather_test.py --root "$ALLWEATHER_TEST_ROOT"

names=(snow100k_s snow100k_l outdoor_rain raindrop)
inputs=(
  "$ALLWEATHER_TEST_ROOT/snow100k/Snow100K-S/synthetic"
  "$ALLWEATHER_TEST_ROOT/snow100k/Snow100K-L/synthetic"
  "$ALLWEATHER_TEST_ROOT/Test1/input"
  "$ALLWEATHER_TEST_ROOT/raindrop/test_a/data"
)
targets=(
  "$ALLWEATHER_TEST_ROOT/snow100k/Snow100K-S/gt"
  "$ALLWEATHER_TEST_ROOT/snow100k/Snow100K-L/gt"
  "$ALLWEATHER_TEST_ROOT/Test1/gt"
  "$ALLWEATHER_TEST_ROOT/raindrop/test_a/gt"
)

for index in "${!names[@]}"; do
  python evaluate_foundation.py \
    --config "$CONFIG" \
    --input-dir "${inputs[$index]}" \
    --target-dir "${targets[$index]}" \
    --device cuda \
    --test-y-channel --crop-border 0 --lpips \
    --json "$OUTPUT_DIR/${names[$index]}.json" \
    > "$OUTPUT_DIR/${names[$index]}.log"
  grep "Average:" "$OUTPUT_DIR/${names[$index]}.log"
done

python - "$OUTPUT_DIR" "${names[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = sys.argv[2:]
datasets = {
    name: json.loads((root / f"{name}.json").read_text())["average"]
    for name in names
}
metrics = sorted(set.intersection(*(set(row) for row in datasets.values())))
macro_average = {
    metric: sum(row[metric] for row in datasets.values()) / len(datasets)
    for metric in metrics
}
output = {
    "protocol": "Official All-Weather; Y PSNR/SSIM; RGB LPIPS; crop_border=0",
    "datasets": datasets,
    "macro_average": macro_average,
}
(root / "benchmark_summary.json").write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
)
print("Official All-Weather macro average:")
for metric, value in macro_average.items():
    print(f"  {metric}: {value:.6f}")
PY
