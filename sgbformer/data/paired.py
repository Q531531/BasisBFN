"""Paired degraded/clean image dataset."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_rgb(path: str | Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _relative_stem(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix(""))


class PairedImageDataset(Dataset[dict[str, Any]]):
    """Pair images by relative path stem or an explicit tab-separated manifest."""

    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path,
        patch_size: int | None = None,
        augment: bool = False,
        pairs_file: str | Path | None = None,
        manifest_target_replace: Sequence[str] = ("input", "gt"),
        strict: bool = True,
        sample_seed: int | None = None,
    ) -> None:
        self.input_dir = Path(input_dir).expanduser()
        self.target_dir = Path(target_dir).expanduser()
        self.patch_size = patch_size
        self.augment = augment
        self.sample_seed = sample_seed
        self._epoch = (
            torch.zeros((), dtype=torch.int64).share_memory_()
            if sample_seed is not None
            else None
        )
        self.manifest_target_replace = tuple(manifest_target_replace)
        if len(self.manifest_target_replace) != 2:
            raise ValueError("manifest_target_replace must contain two strings")
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.target_dir.is_dir():
            raise FileNotFoundError(
                f"Target directory does not exist: {self.target_dir}"
            )

        if pairs_file is not None:
            self.pairs = self._read_manifest(Path(pairs_file).expanduser())
        else:
            input_map = {
                _relative_stem(path, self.input_dir): path
                for path in _image_files(self.input_dir)
            }
            target_map = {
                _relative_stem(path, self.target_dir): path
                for path in _image_files(self.target_dir)
            }
            common = sorted(input_map.keys() & target_map.keys())
            missing_targets = sorted(input_map.keys() - target_map.keys())
            if strict and missing_targets:
                preview = ", ".join(missing_targets[:5])
                raise ValueError(
                    f"{len(missing_targets)} inputs have no matching target: {preview}"
                )
            self.pairs = [
                (input_map[key], target_map[key], key) for key in common
            ]
        if not self.pairs:
            raise ValueError("No paired images were found")

    def _read_manifest(self, manifest: Path) -> list[tuple[Path, Path, str]]:
        pairs = []
        for line_number, raw_line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) == 1:
                source, replacement = self.manifest_target_replace
                if source not in fields[0]:
                    raise ValueError(
                        f"{manifest}:{line_number}: '{source}' is absent from "
                        "the one-column input path"
                    )
                fields.append(fields[0].replace(source, replacement))
            elif len(fields) != 2:
                raise ValueError(
                    f"{manifest}:{line_number}: expected one path or two "
                    "tab-separated paths"
                )
            input_path = self.input_dir / fields[0]
            target_path = self.target_dir / fields[1]
            if not input_path.is_file() or not target_path.is_file():
                raise FileNotFoundError(
                    f"{manifest}:{line_number}: pair does not exist"
                )
            name = str(Path(fields[0]).with_suffix(""))
            pairs.append((input_path, target_path, name))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        if self._epoch is not None:
            self._epoch.fill_(epoch)

    def _crop(
        self,
        degraded: Tensor,
        target: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.patch_size is None:
            return degraded, target
        size = self.patch_size
        height, width = degraded.shape[-2:]
        pad_height = max(0, size - height)
        pad_width = max(0, size - width)
        if pad_height or pad_width:
            mode = (
                "reflect"
                if height > pad_height and width > pad_width
                else "replicate"
            )
            padding = (0, pad_width, 0, pad_height)
            degraded = F.pad(degraded, padding, mode=mode)
            target = F.pad(target, padding, mode=mode)
            height, width = degraded.shape[-2:]
        top = int(torch.randint(0, height - size + 1, (), generator=generator).item())
        left = int(torch.randint(0, width - size + 1, (), generator=generator).item())
        return (
            degraded[:, top : top + size, left : left + size],
            target[:, top : top + size, left : left + size],
        )

    def __getitem__(
        self, index: int | tuple[int, int]
    ) -> dict[str, Any]:
        draw_id = 0
        if isinstance(index, tuple):
            index, draw_id = index
        generator = None
        if self.sample_seed is not None:
            # Independent of model initialization, worker count, and resume epoch.
            generator = torch.Generator().manual_seed(
                (
                    self.sample_seed
                    + int(self._epoch) * len(self)
                    + index
                    + draw_id * 1_000_003
                )
                % (2**63)
            )
        input_path, target_path, name = self.pairs[index]
        degraded = load_rgb(input_path)
        target = load_rgb(target_path)
        if degraded.shape != target.shape:
            raise ValueError(
                f"Pair shape mismatch for {input_path} and {target_path}: "
                f"{tuple(degraded.shape)} vs {tuple(target.shape)}"
            )
        degraded, target = self._crop(degraded, target, generator)
        if self.augment:
            if bool(torch.randint(0, 2, (), generator=generator).item()):
                degraded = degraded.flip(-1)
                target = target.flip(-1)
            if bool(torch.randint(0, 2, (), generator=generator).item()):
                degraded = degraded.flip(-2)
                target = target.flip(-2)
            rotations = int(torch.randint(0, 4, (), generator=generator).item())
            degraded = torch.rot90(degraded, rotations, (-2, -1))
            target = torch.rot90(target, rotations, (-2, -1))
        return {
            "degraded": degraded.contiguous(),
            "target": target.contiguous(),
            "name": name,
            "input_path": str(input_path),
            "target_path": str(target_path),
        }
