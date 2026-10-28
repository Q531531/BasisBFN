"""Deterministic sampling helpers for controlled degradation mixtures."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.data import Sampler


def classify_paths(
    paths: Sequence[str | Path],
    patterns: Mapping[str, str],
    expected_counts: Mapping[str, int] | None = None,
) -> list[str]:
    if not patterns:
        raise ValueError("Category patterns must not be empty")
    compiled = {
        name: re.compile(pattern, flags=re.IGNORECASE)
        for name, pattern in patterns.items()
    }
    labels = []
    for path in paths:
        filename = Path(path).name
        matches = [
            name for name, pattern in compiled.items()
            if pattern.fullmatch(filename)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{filename!r} must match exactly one category; matched {matches}"
            )
        labels.append(matches[0])
    counts = Counter(labels)
    if expected_counts is not None:
        expected = {str(name): int(count) for name, count in expected_counts.items()}
        if dict(counts) != expected:
            raise ValueError(
                f"Training category counts changed: actual={dict(counts)}, "
                f"expected={expected}"
            )
    return labels


class CategoryBalancedSampler(Sampler[tuple[int, int]]):
    """Sample a fixed category mixture reproducibly at each epoch.

    The second yielded integer is a draw identifier. Datasets can include it
    in augmentation seeds so repeated minority samples receive different crops.
    """

    def __init__(
        self,
        labels: Sequence[str],
        target_fractions: Mapping[str, float],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        if not labels:
            raise ValueError("Sampler labels must not be empty")
        self.indices: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.indices[str(label)].append(index)
        if set(self.indices) != set(target_fractions):
            raise ValueError(
                "Target fractions must name every observed category exactly"
            )
        fractions = {
            str(name): float(value)
            for name, value in target_fractions.items()
        }
        if any(not math.isfinite(value) or value <= 0 for value in fractions.values()):
            raise ValueError("Target fractions must be finite and positive")
        total = sum(fractions.values())
        self.fractions = {name: value / total for name, value in fractions.items()}
        self.num_samples = int(num_samples if num_samples is not None else len(labels))
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.seed = int(seed)
        self.epoch = 0
        self.category_counts = Counter(labels)
        self.target_counts = self._allocate_counts()

    def _allocate_counts(self) -> dict[str, int]:
        raw = {
            name: self.num_samples * fraction
            for name, fraction in self.fractions.items()
        }
        counts = {name: math.floor(value) for name, value in raw.items()}
        remainder = self.num_samples - sum(counts.values())
        order = sorted(
            raw,
            key=lambda name: (-(raw[name] - counts[name]), name),
        )
        for name in order[:remainder]:
            counts[name] += 1
        return counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _draw(
        indices: Sequence[int],
        count: int,
        generator: torch.Generator,
    ) -> list[int]:
        selected = []
        while len(selected) < count:
            order = torch.randperm(len(indices), generator=generator).tolist()
            remaining = count - len(selected)
            selected.extend(indices[position] for position in order[:remaining])
        return selected

    def __iter__(self) -> Iterator[tuple[int, int]]:
        generator = torch.Generator().manual_seed(
            (self.seed + self.epoch) % (2**63)
        )
        selected = []
        for name in sorted(self.indices):
            selected.extend(
                self._draw(
                    self.indices[name],
                    self.target_counts[name],
                    generator,
                )
            )
        order = torch.randperm(len(selected), generator=generator).tolist()
        return iter(
            (selected[position], draw_id)
            for draw_id, position in enumerate(order)
        )

    def __len__(self) -> int:
        return self.num_samples
