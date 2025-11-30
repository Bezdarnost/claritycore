# Copyright (c) Aman Urumbekov and other contributors.
"""Unified dataset for pixel-to-pixel image tasks."""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from claritycore.utils import print_warning

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class NormConfig:
    """
    Configuration for image normalization.

    Args:
        mode: Normalization mode.
            - 'zero_one': Normalize to [0, 1]
            - 'minus_one_one': Normalize to [-1, 1] (default)
            - 'none': No normalization (keep original range)
        max_value: Maximum value for division (default: 255.0 for 8-bit images).
        use_log: Apply log1p transform before normalization.
        mean: Per-channel mean for normalization (optional).
        std: Per-channel std for normalization (optional).
    """

    mode: Literal["zero_one", "minus_one_one", "none"] = "minus_one_one"
    max_value: float = 255.0
    use_log: bool = False
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None


@dataclass
class DatasetConfig:
    """
    Configuration for the unified dataset.

    Args:
        target_dir: Directory containing target (ground truth) images.
        input_dir: Directory containing input images (optional, for paired mode).
        scale: Scale factor between input and target (default: 1).
        input_suffix: Suffix pattern in input filenames (e.g., 'x4' for '0001x4.png').
            If None, will auto-detect common patterns.
        patch_size: Size of random crops for training (None for full images).
        augment: Enable data augmentation (flip, rotation).
        normalize: Normalization configuration.
    """

    target_dir: str
    input_dir: str | None = None
    scale: int = 1
    input_suffix: str | None = None
    patch_size: int | None = 256
    augment: bool = True
    normalize: NormConfig = field(default_factory=NormConfig)


class Pixel2PixelDataset(Dataset):
    """
    Unified dataset for pixel-to-pixel image tasks (SR, denoising, deblurring, etc.).

    Supports:
    - Paired mode: Separate input/target directories
    - Self-supervised mode: Generate input by downscaling target
    - Identity mode (scale=1): 1:1 mapping with warning
    - Flexible normalization ([0,1], [-1,1], log, custom)
    - Auto-detection of filename suffix patterns

    Args:
        config: Dataset configuration.
        mode: 'train' or 'val'.
        transform: Optional additional transform.

    Example:
        # Paired dataset with x4 downscaling
        config = DatasetConfig(
            target_dir="data/HR",
            input_dir="data/LR_x4",
            scale=4,
        )
        dataset = Pixel2PixelDataset(config, mode="train")

        # Self-supervised (generate LR on the fly)
        config = DatasetConfig(
            target_dir="data/HR",
            scale=4,
        )
        dataset = Pixel2PixelDataset(config, mode="train")
    """

    def __init__(
        self,
        config: DatasetConfig,
        mode: Literal["train", "val"] = "train",
        transform: Callable | None = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self.transform = transform

        self.target_root = Path(config.target_dir)
        self.input_root = Path(config.input_dir) if config.input_dir else None
        self.scale = config.scale
        self.patch_size = config.patch_size
        self.augment = config.augment and mode == "train"
        self.norm = config.normalize

        # Find target images first (needed for suffix detection)
        self.target_paths = sorted([p for p in self.target_root.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS])

        if not self.target_paths:
            raise FileNotFoundError(f"No images found in {config.target_dir}")

        # Determine suffix pattern (after target_paths is set)
        self.input_suffix = self._resolve_suffix()

        # Validate pairs if in paired mode
        if self.input_root is not None:
            self._validate_pairs()

        # Warn for identity mapping
        if self.scale == 1 and self.input_root is None:
            print_warning(
                "scale=1 without input_dir: dataset returns identical input/target pairs. "
                "This is only useful for denoising or similar tasks."
            )

    def _resolve_suffix(self) -> str | None:
        """Auto-detect or validate input filename suffix."""
        if self.config.input_suffix is not None:
            return self.config.input_suffix

        if self.input_root is None:
            return None

        # Try to auto-detect common patterns
        first_target = next(iter(self.target_paths), None)
        if first_target is None:
            return None

        # Common suffix patterns to try
        patterns = [
            f"x{self.scale}",  # DIV2K style: 0001x4.png
            f"_x{self.scale}",  # Alternative: 0001_x4.png
            f"_{self.scale}x",  # Another: 0001_4x.png
            "",  # Same name
        ]

        for pattern in patterns:
            test_name = f"{first_target.stem}{pattern}{first_target.suffix}"
            test_path = self.input_root / test_name

            if test_path.exists():
                return pattern if pattern else None

        return None

    def _get_input_path(self, target_path: Path) -> Path:
        """Get corresponding input path for a target image."""
        rel_path = target_path.relative_to(self.target_root)

        if self.input_suffix:
            new_name = f"{rel_path.stem}{self.input_suffix}{rel_path.suffix}"
            return self.input_root / rel_path.parent / new_name

        return self.input_root / rel_path

    def _validate_pairs(self) -> None:
        """Validate that input images exist for all target images."""
        valid_pairs = []
        missing = 0

        for target_path in self.target_paths:
            input_path = self._get_input_path(target_path)
            if input_path.exists():
                valid_pairs.append(target_path)
            else:
                missing += 1

        if missing > 0:
            print_warning(f"Missing {missing} input pairs (skipped)")

        self.target_paths = valid_pairs

    def __len__(self) -> int:
        return len(self.target_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        target_path = self.target_paths[idx]

        # Load target image
        target = Image.open(target_path).convert("RGB")

        # Load or generate input image
        if self.input_root is not None:
            input_path = self._get_input_path(target_path)
            input_img = Image.open(input_path).convert("RGB")
        else:
            # Self-supervised: generate input by downscaling
            input_img = self._generate_input(target)

        # Random crop for training
        if self.patch_size and self.mode == "train":
            target, input_img = self._random_crop(target, input_img)

        # Convert to numpy
        target = np.array(target, dtype=np.float32)
        input_img = np.array(input_img, dtype=np.float32)

        # Apply augmentation
        if self.augment:
            target, input_img = self._augment(target, input_img)

        # Apply custom transform
        if self.transform:
            target, input_img = self.transform(target, input_img)

        # Normalize
        target = self._normalize(target)
        input_img = self._normalize(input_img)

        # Convert to tensors (C, H, W)
        target = torch.from_numpy(target.transpose(2, 0, 1).copy())
        input_img = torch.from_numpy(input_img.transpose(2, 0, 1).copy())

        return {"input": input_img, "target": target}

    def _generate_input(self, target: Image.Image) -> Image.Image:
        """Generate input image from target (self-supervised mode)."""
        if self.scale == 1:
            # Identity mapping
            return target.copy()

        # Downsample
        w, h = target.size
        new_w, new_h = w // self.scale, h // self.scale
        return target.resize((new_w, new_h), Image.BICUBIC)

    def _random_crop(
        self,
        target: Image.Image,
        input_img: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        """Apply synchronized random crop."""
        t_w, t_h = target.size

        # Ensure crop size fits
        crop_h = min(self.patch_size, t_h)
        crop_w = min(self.patch_size, t_w)

        # Random position on target
        top = random.randint(0, t_h - crop_h)
        left = random.randint(0, t_w - crop_w)

        # Crop target
        target = target.crop((left, top, left + crop_w, top + crop_h))

        # Corresponding input crop (accounting for scale)
        i_top = top // self.scale
        i_left = left // self.scale
        i_crop_h = crop_h // self.scale
        i_crop_w = crop_w // self.scale
        input_img = input_img.crop((i_left, i_top, i_left + i_crop_w, i_top + i_crop_h))

        return target, input_img

    def _augment(
        self,
        target: np.ndarray,
        input_img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random augmentation."""
        # Random horizontal flip
        if random.random() > 0.5:
            target = np.flip(target, axis=1).copy()
            input_img = np.flip(input_img, axis=1).copy()

        # Random vertical flip
        if random.random() > 0.5:
            target = np.flip(target, axis=0).copy()
            input_img = np.flip(input_img, axis=0).copy()

        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            target = np.rot90(target, k).copy()
            input_img = np.rot90(input_img, k).copy()

        return target, input_img

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Apply normalization based on config."""
        # Apply log transform if requested
        if self.norm.use_log:
            img = np.log1p(img)

        # Divide by max value
        img = img / self.norm.max_value

        # Apply mean/std normalization if provided
        if self.norm.mean is not None and self.norm.std is not None:
            mean = np.array(self.norm.mean, dtype=np.float32).reshape(1, 1, -1)
            std = np.array(self.norm.std, dtype=np.float32).reshape(1, 1, -1)
            img = (img - mean) / std
        elif self.norm.mode == "minus_one_one":
            # Scale from [0, 1] to [-1, 1]
            img = img * 2.0 - 1.0
        # mode == "zero_one" or "none": already in [0, 1] after division

        return img

    @staticmethod
    def denormalize(
        tensor: torch.Tensor,
        norm: NormConfig,
    ) -> torch.Tensor:
        """
        Reverse normalization for visualization/saving.

        Args:
            tensor: Normalized tensor (C, H, W) or (B, C, H, W).
            norm: Normalization config used during training.

        Returns:
            Denormalized tensor in [0, 1] range.
        """
        if norm.mean is not None and norm.std is not None:
            mean = torch.tensor(norm.mean, device=tensor.device).view(-1, 1, 1)
            std = torch.tensor(norm.std, device=tensor.device).view(-1, 1, 1)
            tensor = tensor * std + mean
        elif norm.mode == "minus_one_one":
            tensor = (tensor + 1.0) / 2.0

        if norm.use_log:
            tensor = torch.expm1(tensor * norm.max_value) / norm.max_value

        return tensor.clamp(0, 1)


__all__ = [
    "Pixel2PixelDataset",
    "DatasetConfig",
    "NormConfig",
]
