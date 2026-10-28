#!/usr/bin/env python3
"""Compare two four-set evaluations using matching image names and protocols."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgbformer.comparison import paired_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    names = ("snow100k_s", "snow100k_l", "outdoor_rain", "raindrop")

    def read(root: Path) -> dict:
        return {
            name: json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
            for name in names
        }

    result = paired_comparison(
        read(args.baseline), read(args.candidate),
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for metric, values in result["macro"].items():
        low, high = values["paired_image_bootstrap_ci95"]
        print(f"{metric}: delta={values['delta']:+.6f}, paired-image CI95=[{low:+.6f}, {high:+.6f}]")
    print(result["limitation"])


if __name__ == "__main__":
    main()
