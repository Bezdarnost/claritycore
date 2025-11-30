# Copyright (c) Aman Urumbekov and other contributors.
"""Dataset classes for image restoration tasks."""

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from loguru import logger

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    """
    Simple dataset for single images.

    Useful for inference or self-supervised training.

    Args:
        root: Path to image directory.
        transform: Optional transform to apply.
        recursive: Whether to search subdirectories.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        recursive: bool = True,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        pattern = "**/*" if recursive else "*"
        self.image_paths = sorted([
            p for p in self.root.glob(pattern)
            if p.suffix.lower() in VALID_EXTENSIONS
        ])

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root}")

        logger.info(f"Found {len(self.image_paths)} images in {root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0

        if self.transform:
            image = self.transform(image)

        # Convert to tensor (C, H, W)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1))

        return {"image": image, "path": str(path)}


class ImagePairDataset(Dataset):
    """
    Dataset for paired LQ-HQ image restoration.

    Supports three modes:
    1. Paired: Separate LQ and HQ directories with matching filenames
    2. Self-supervised: Generate LQ by downscaling HQ
    3. HQ-only: Return only HQ images (for inference)

    Args:
        hq_root: Path to high-quality images.
        lq_root: Path to low-quality images (None for self-supervised).
        scale: Downscaling factor for self-supervised mode.
        patch_size: Size of random crops (None for full images).
        transform: Optional transform to apply to both images.
        mode: 'train' or 'val'.
        lq_suffix: Suffix in LQ filenames (e.g., 'x4' for '0001x4.png').
    """

    def __init__(
        self,
        hq_root: str | Path,
        lq_root: str | Path | None = None,
        scale: int = 4,
        patch_size: int | None = 256,
        transform: Callable | None = None,
        mode: str = "train",
        lq_suffix: str | None = None,
    ) -> None:
        self.hq_root = Path(hq_root)
        self.lq_root = Path(lq_root) if lq_root else None
        self.scale = scale
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode
        self.lq_suffix = lq_suffix  # e.g., "x4" for DIV2K

        # Find HQ images
        self.hq_paths = sorted([
            p for p in self.hq_root.rglob("*")
            if p.suffix.lower() in VALID_EXTENSIONS
        ])

        if not self.hq_paths:
            raise FileNotFoundError(f"No images found in {hq_root}")

        # Validate paired mode
        if self.lq_root is not None:
            self._validate_pairs()

        logger.info(
            f"Loaded {len(self.hq_paths)} image pairs "
            f"({'paired' if self.lq_root else 'self-supervised'} mode)"
        )

    def _validate_pairs(self) -> None:
        """Validate that LQ images exist for all HQ images."""
        valid_pairs = []
        for hq_path in self.hq_paths:
            lq_path = self._get_lq_path(hq_path)

            if lq_path.exists():
                valid_pairs.append(hq_path)
            else:
                logger.warning(f"No LQ pair for {hq_path}")

        self.hq_paths = valid_pairs

    def _get_lq_path(self, hq_path: Path) -> Path:
        """Get corresponding LQ path for an HQ image."""
        rel_path = hq_path.relative_to(self.hq_root)

        if self.lq_suffix:
            # Handle suffix pattern: 0001.png -> 0001x4.png
            stem = rel_path.stem
            suffix = rel_path.suffix
            new_name = f"{stem}{self.lq_suffix}{suffix}"
            lq_path = self.lq_root / rel_path.parent / new_name
        else:
            lq_path = self.lq_root / rel_path

        return lq_path

    def __len__(self) -> int:
        return len(self.hq_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hq_path = self.hq_paths[idx]

        # Load HQ image
        hq = Image.open(hq_path).convert("RGB")

        # Load or generate LQ image
        if self.lq_root is not None:
            lq_path = self._get_lq_path(hq_path)
            lq = Image.open(lq_path).convert("RGB")
        else:
            # Self-supervised: downsample HQ
            w, h = hq.size
            lq = hq.resize((w // self.scale, h // self.scale), Image.BICUBIC)

        # Random crop for training
        if self.patch_size and self.mode == "train":
            hq, lq = self._random_crop(hq, lq)

        # Convert to numpy arrays
        hq = np.array(hq, dtype=np.float32) / 255.0
        lq = np.array(lq, dtype=np.float32) / 255.0

        # Apply transforms
        if self.transform:
            hq, lq = self.transform(hq, lq)

        # Random horizontal flip for training
        if self.mode == "train" and random.random() > 0.5:
            hq = np.flip(hq, axis=1).copy()
            lq = np.flip(lq, axis=1).copy()

        # Convert to tensors (C, H, W)
        hq = torch.from_numpy(hq.transpose(2, 0, 1).copy())
        lq = torch.from_numpy(lq.transpose(2, 0, 1).copy())

        return {"lq": lq, "gt": hq}

    def _random_crop(
        self,
        hq: Image.Image,
        lq: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        """Apply synchronized random crop."""
        hq_w, hq_h = hq.size
        lq_w, lq_h = lq.size

        # Crop size for HQ
        crop_h = min(self.patch_size, hq_h)
        crop_w = min(self.patch_size, hq_w)

        # Random position
        top = random.randint(0, hq_h - crop_h)
        left = random.randint(0, hq_w - crop_w)

        # Crop HQ
        hq = hq.crop((left, top, left + crop_w, top + crop_h))

        # Corresponding LQ crop
        lq_top = top // self.scale
        lq_left = left // self.scale
        lq_crop_h = crop_h // self.scale
        lq_crop_w = crop_w // self.scale
        lq = lq.crop((lq_left, lq_top, lq_left + lq_crop_w, lq_top + lq_crop_h))

        return hq, lq


__all__ = ["ImageDataset", "ImagePairDataset"]

