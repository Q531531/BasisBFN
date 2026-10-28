#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 CHECKPOINT [OUTPUT_DIR] [--uncertainty]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${ALLWEATHER_TEST_ROOT:?Set ALLWEATHER_TEST_ROOT to testallweather}"
CHECKPOINT="$1"
OUTPUT_DIR="${2:-results/allweather}"
BFN_GAIN="${BFN_GAIN:-1.0}"
extra=()
if [[ $# -eq 3 ]]; then
  if [[ "$3" != "--uncertainty" ]]; then
    echo "Only --uncertainty is supported as the third argument" >&2
    exit 2
  fi
  extra=(--uncertainty)
fi
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
  python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --input-dir "${inputs[$index]}" \
    --target-dir "${targets[$index]}" \
    --device cuda \
    --bfn-gain "$BFN_GAIN" \
    --test-y-channel --crop-border 0 \
    --lpips --compare-structural \
    --json "$OUTPUT_DIR/${names[$index]}.json" \
    "${extra[@]}"
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
average = {
    metric: sum(row[metric] for row in datasets.values()) / len(datasets)
    for metric in metrics
}
output = {"protocol": "All-Weather Y channel, crop_border=0",
          "datasets": datasets, "macro_average": average}
(root / "benchmark_summary.json").write_text(
    json.dumps(output, indent=2) + "\n", encoding="utf-8"
)
print("All-Weather macro average:")
for metric, value in average.items():
    print(f"  {metric}: {value:.6f}")
PY
