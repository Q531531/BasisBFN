from pathlib import Path
from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from sgbformer.data import (
    CategoryBalancedSampler,
    PairedImageDataset,
    classify_paths,
)


def write_image(path: Path, value: int, size: tuple[int, int] = (12, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_relative_path_pairing_and_crop(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    target_dir = tmp_path / "target"
    write_image(input_dir / "weather" / "sample.jpg", 32)
    write_image(target_dir / "weather" / "sample.png", 64)

    dataset = PairedImageDataset(
        input_dir,
        target_dir,
        patch_size=16,
        augment=False,
    )
    item = dataset[0]
    assert item["name"] == "weather/sample"
    assert item["degraded"].shape == (3, 16, 16)
    assert item["target"].shape == (3, 16, 16)


def test_manifest_pairing(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    target_dir = tmp_path / "target"
    write_image(input_dir / "rain-1.png", 32)
    write_image(target_dir / "clean-1.png", 64)
    manifest = tmp_path / "pairs.tsv"
    manifest.write_text("rain-1.png\tclean-1.png\n", encoding="utf-8")

    dataset = PairedImageDataset(
        input_dir,
        target_dir,
        pairs_file=manifest,
    )
    assert len(dataset) == 1
    assert dataset[0]["name"] == "rain-1"


def test_transweather_one_column_manifest(tmp_path: Path) -> None:
    root = tmp_path / "train"
    write_image(root / "snow" / "input" / "sample.png", 32)
    write_image(root / "snow" / "gt" / "sample.png", 64)
    manifest = root / "allweather.txt"
    manifest.write_text("snow/input/sample.png\n", encoding="utf-8")

    dataset = PairedImageDataset(root, root, pairs_file=manifest)
    assert len(dataset) == 1
    assert dataset[0]["name"] == "snow/input/sample"


def test_seeded_crops_do_not_depend_on_model_rng_or_access_order(tmp_path: Path) -> None:
    input_dir, target_dir = tmp_path / "input", tmp_path / "gt"
    input_dir.mkdir()
    target_dir.mkdir()
    image = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3).astype(np.uint8)
    for index in range(3):
        Image.fromarray(image).save(input_dir / f"{index}.png")
        Image.fromarray(image).save(target_dir / f"{index}.png")
    dataset = PairedImageDataset(
        input_dir, target_dir, patch_size=8, augment=True, sample_seed=42
    )
    dataset.set_epoch(2)
    first = dataset[0]["degraded"]
    torch.randn(1234)
    dataset[2]
    torch.testing.assert_close(dataset[0]["degraded"], first, atol=0, rtol=0)
    dataset.set_epoch(3)
    assert not torch.equal(dataset[0]["degraded"], first)
    resumed = PairedImageDataset(
        input_dir, target_dir, patch_size=8, augment=True, sample_seed=42
    )
    resumed.set_epoch(2)
    torch.testing.assert_close(resumed[0]["degraded"], first, atol=0, rtol=0)
    torch.testing.assert_close(resumed[0]["degraded"], resumed[0]["target"])
    loader = torch.utils.data.DataLoader(
        resumed, batch_size=1, num_workers=1, persistent_workers=True,
        multiprocessing_context="spawn",
    )
    torch.testing.assert_close(list(loader)[0]["degraded"][0], first, atol=0, rtol=0)
    resumed.set_epoch(3)
    torch.testing.assert_close(
        list(loader)[0]["degraded"][0], dataset[0]["degraded"], atol=0, rtol=0
    )


def test_strict_category_classification_and_expected_counts() -> None:
    paths = [
        "winter_house_00001.jpg",
        "im_0001_s80_a04.png",
        "0_rain.png",
    ]
    patterns = {
        "snow": r".*\.jpe?g",
        "outdoor_rain": r"im_\d+_s\d+_a0[456]\.png",
        "raindrop": r"\d+_rain\.png",
    }
    assert classify_paths(
        paths, patterns,
        {"snow": 1, "outdoor_rain": 1, "raindrop": 1},
    ) == ["snow", "outdoor_rain", "raindrop"]
    with pytest.raises(ValueError, match="exactly one"):
        classify_paths(["unknown.png"], patterns)
    with pytest.raises(ValueError, match="counts changed"):
        classify_paths(paths, patterns, {"snow": 3})


def test_category_balanced_sampler_is_exact_and_epoch_deterministic() -> None:
    labels = ["snow"] * 5 + ["outdoor_rain"] * 4 + ["raindrop"]
    sampler = CategoryBalancedSampler(
        labels,
        {"snow": 0.5, "outdoor_rain": 0.25, "raindrop": 0.25},
        num_samples=12,
        seed=42,
    )
    assert sampler.target_counts == {
        "snow": 6, "outdoor_rain": 3, "raindrop": 3
    }
    first = list(sampler)
    assert first == list(sampler)
    assert len(first) == 12
    assert len({draw_id for _, draw_id in first}) == 12
    assert Counter(labels[index] for index, _ in first) == sampler.target_counts
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_repeated_balanced_draws_use_distinct_reproducible_crops(
    tmp_path: Path,
) -> None:
    input_dir, target_dir = tmp_path / "input", tmp_path / "gt"
    input_dir.mkdir()
    target_dir.mkdir()
    image = np.arange(48 * 48 * 3, dtype=np.uint16).reshape(
        48, 48, 3
    ).astype(np.uint8)
    Image.fromarray(image).save(input_dir / "0.png")
    Image.fromarray(image).save(target_dir / "0.png")
    dataset = PairedImageDataset(
        input_dir, target_dir, patch_size=8, augment=True, sample_seed=42
    )
    first = dataset[(0, 0)]["degraded"]
    repeated = dataset[(0, 1)]["degraded"]
    torch.testing.assert_close(dataset[(0, 0)]["degraded"], first)
    assert not torch.equal(first, repeated)
