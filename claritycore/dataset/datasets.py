# Copyright (c) Aman Urumbekov and other contributors.
import random
from collections.abc import Callable
from pathlib import Path
from typing import Union

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from claritycore.utils.registry import DATASET_REGISTRY

VALID_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
# Use the modern Resampling enum for Pillow
BICUBIC = Image.Resampling.BICUBIC


@DATASET_REGISTRY.register("pixel2pixel")
class Pixel2PixelDataset(Dataset):
    """
    A flexible dataset for pixel-to-pixel vision tasks (e.g., super-resolution, denoising).

    This dataset can operate in three primary modes:

    1.  Single Image Mode (Target-only):
        - Returns a single image (the "target").
        - Activated by providing only `target_root`.

    2.  Self-Generated Input Mode:
        - Returns an (input, target) pair.
        - The input is created on-the-fly by downsampling the target image.
        - Activated by providing `target_root` and `scale_factor`.

    3.  Paired Image Mode:
        - Returns an (input, target) pair from separate directories, matching filenames via suffixes.
        - Activated by providing both `input_root` and `target_root`.

    Args:
        target_root (str): Path to the root directory for target images.
        input_root (str | None): Path to the root directory for input images. If provided, activates paired mode.
        glob (str): A glob pattern to find image files within `target_root`. Defaults to `**/*`.
        target_suffix (str): In paired mode, the suffix to identify and strip from target filenames. Default: "".
        input_suffix (str): In paired mode, the suffix to add to the base name to find the input filename. Default: "".
        target_channels (int): Channels for the target output image (e.g., 1 for grayscale, 3 for RGB).
        input_channels (int | None): Channels for the input image. If None, defaults to `target_channels`.
        scale_factor (int | None): The scale factor between input and target. Required for self-generated mode.
            For paired mode, it defaults to 1 if not provided.
        target_size (int | None): Target patch size for images. If an image is smaller, it's skipped.
            If larger, a random crop is taken. Input images are cropped accordingly. Default: 256.
        transform (Callable | None): A transform to apply to the image(s).
        verify_scale (bool): In paired mode, verify that the target is `scale_factor` times larger than the input.
        pre_filter_size (bool): If True, filters out images smaller than `target_size` during initialization.
            Can be slow on large datasets. Set to False if you know your data is clean. Default: True.
    """

    def __init__(
        self,
        target_root: str,
        input_root: str | None = None,
        glob: str = "**/*",
        target_suffix: str = "",
        input_suffix: str = "",
        target_channels: int = 3,
        input_channels: int | None = None,
        scale_factor: int | None = None,
        target_size: int | None = 256,
        transform: Callable | None = None,
        verify_scale: bool = True,
        pre_filter_size: bool = True,
    ):
        super().__init__()
        self.transform = transform
        self.target_channels = target_channels
        self.input_channels = input_channels if input_channels is not None else target_channels
        self.target_size = target_size
        self.image_paths: list[Path] | list[dict[str, Path]] = []
        self.scale_tolerance: int = 0

        # --- Determine dataset mode and validate configuration ---
        if input_root is not None:
            self._mode = "paired"
            self.scale_factor = scale_factor if scale_factor is not None else 1
            self.verify_scale = verify_scale
            self.target_root = Path(target_root)
            self.input_root = Path(input_root)
            self._find_and_validate_pairs(glob, target_suffix, input_suffix)
        elif scale_factor is not None:
            self._mode = "self_supervised"
            self.scale_factor = scale_factor
            self.target_root = Path(target_root)
            self.image_paths = self._scan_files(self.target_root, glob)
        else:
            self._mode = "single"
            self.target_root = Path(target_root)
            self.image_paths = self._scan_files(self.target_root, glob)

        if hasattr(self, "scale_factor"):
            self.scale_tolerance = max(0, self.scale_factor - 1)

        if not self.image_paths:
            raise FileNotFoundError(
                f"No valid image files or pairs found in the specified directories with glob '{glob}'."
            )

        if self.target_size is not None and pre_filter_size:
            self._pre_filter_images_by_size()

        if not self.image_paths:
            raise FileNotFoundError(f"No images remain after filtering by target_size ({self.target_size}px).")

        logger.info(f"Prepared {len(self.image_paths)} samples for mode '{self._mode}'.")

    def _pre_filter_images_by_size(self):
        """Filters the image list to remove any images smaller than the target patch size."""
        original_count = len(self.image_paths)
        if self.target_size is None:
            return

        logger.info(f"Pre-filtering images smaller than {self.target_size}px. This may take a while...")
        image_list = self.image_paths
        if self._mode in ("single", "self_supervised"):
            filtered_paths = []
            for p in image_list:
                try:
                    w, h = Image.open(p).size
                    if h >= self.target_size and w >= self.target_size:
                        filtered_paths.append(p)
                except Exception:
                    logger.exception(f"Failed to open image '{p}' for size check; skipping.")
            self.image_paths = filtered_paths
        elif self._mode == "paired":
            filtered_pairs = []
            for pair in image_list:
                try:
                    w, h = Image.open(pair["target"]).size
                    if h >= self.target_size and w >= self.target_size:
                        filtered_pairs.append(pair)
                except Exception:
                    logger.exception(
                        f"Failed to open target image '{pair.get('target')}' for size check; skipping pair."
                    )
            self.image_paths = filtered_pairs

        filtered_count = original_count - len(self.image_paths)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} images smaller than {self.target_size}px.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor], None]:
        try:
            target_img, input_img = self._load_images(index)
            if self.target_size is not None:
                target_img, input_img = self._apply_cropping(target_img, input_img)

            target_tensor = self._process_image(np.array(target_img), self.target_channels)
            if self._mode == "single":
                if self.transform:
                    target_tensor = self.transform(target_tensor)
                return target_tensor

            input_tensor = self._process_image(np.array(input_img), self.input_channels)
            output = (input_tensor, target_tensor)
            if self.transform:
                output = self.transform(output)
            return output
        except Exception:
            path_info = self.image_paths[index]
            logger.exception(f"Error loading image at index {index} with path info: {path_info}")
            return None

    def _load_images(self, index: int) -> tuple[Image.Image, Image.Image | None]:
        """Loads target and optionally input images based on the dataset mode."""
        if self._mode == "single":
            target_path = self.image_paths[index]
            return Image.open(target_path), None
        if self._mode == "self_supervised":
            target_path = self.image_paths[index]
            target_img = Image.open(target_path)
            w, h = target_img.size
            input_w, input_h = w // self.scale_factor, h // self.scale_factor
            input_img = target_img.resize((input_w, input_h), BICUBIC)
            return target_img, input_img
        if self._mode == "paired":
            pair = self.image_paths[index]
            target_path, input_path = pair["target"], pair["input"]
            target_img = Image.open(target_path)
            input_img = Image.open(input_path)
            if self.verify_scale:
                target_w, target_h = target_img.size
                input_w, input_h = input_img.size
                expected_h = input_h * self.scale_factor
                expected_w = input_w * self.scale_factor
                if (
                    abs(target_h - expected_h) > self.scale_tolerance
                    or abs(target_w - expected_w) > self.scale_tolerance
                ):
                    logger.warning(
                        f"Scale mismatch for {target_path.name} (tolerance={self.scale_tolerance}px). "
                        f"Target: ({target_w}, {target_h}), Input: ({input_w}, {input_h}), "
                        f"Expected Target: ({expected_w}, {expected_h})"
                    )
            return target_img, input_img

    def _apply_cropping(
        self, target_img: Image.Image, input_img: Image.Image | None
    ) -> tuple[Image.Image, Image.Image | None]:
        """Applies synchronized random cropping to target and input images."""
        w, h = target_img.size
        # This check is a safeguard; pre-filtering should prevent this from being triggered.
        if h < self.target_size or w < self.target_size:
            raise ValueError(f"Image ({w}x{h}) smaller than target_size ({self.target_size}) after pre-filtering.")
        top = random.randint(0, h - self.target_size)
        left = random.randint(0, w - self.target_size)
        target_img_cropped = target_img.crop((left, top, left + self.target_size, top + self.target_size))
        input_img_cropped = None
        if input_img is not None:
            input_size = self.target_size // self.scale_factor
            input_top = top // self.scale_factor
            input_left = left // self.scale_factor
            input_img_cropped = input_img.crop((input_left, input_top, input_left + input_size, input_top + input_size))
        return target_img_cropped, input_img_cropped

    def _find_and_validate_pairs(self, glob: str, target_suffix: str, input_suffix: str):
        """Scans for target images and finds corresponding input images using suffixes."""
        target_paths = self._scan_files(self.target_root, glob)
        validated_pairs = []
        for target_path in target_paths:
            target_name = target_path.name
            if target_suffix and not target_name.endswith(target_suffix):
                continue

            # Use modern `removesuffix` for cleaner code (requires Python 3.9+)
            base_name = target_name.removesuffix(target_suffix)
            input_name = f"{base_name}{input_suffix}"

            relative_dir = target_path.relative_to(self.target_root).parent
            input_path = self.input_root / relative_dir / input_name

            if input_path.exists():
                validated_pairs.append({"input": input_path, "target": target_path})
            else:
                logger.warning(
                    f"Could not find matching input for target '{target_path}' at expected path '{input_path}'"
                )
        self.image_paths = validated_pairs

    @staticmethod
    def _process_image(img_np: np.ndarray, target_channels: int) -> torch.Tensor:
        """Converts a NumPy image to a PyTorch tensor with correct channels and range."""
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        elif img_np.dtype == np.uint16:
            img_np = img_np.astype(np.float32) / 65535.0
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=2)
        current_channels = img_np.shape[2]
        if current_channels != target_channels:
            if current_channels == 4 and target_channels == 3:
                img_np = img_np[:, :, :3]
            elif current_channels == 1 and target_channels == 3:
                img_np = np.concatenate([img_np] * 3, axis=2)
            elif target_channels == 1:
                if current_channels >= 3:
                    rgb_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32).reshape(1, 1, 3)
                    img_np = np.sum(img_np[:, :, :3] * rgb_weights, axis=2, keepdims=True)
                else:
                    img_np = img_np[:, :, :1]
            else:
                raise ValueError(f"Cannot convert image with {current_channels} to {target_channels} channels.")
        return torch.from_numpy(img_np.transpose(2, 0, 1)).contiguous()

    @staticmethod
    def _scan_files(directory: Path, glob_pattern: str) -> list[Path]:
        """Scans a directory for valid image files using a glob pattern."""
        return sorted([p for p in directory.glob(glob_pattern) if p.suffix.lower() in VALID_IMG_EXTENSIONS])


__all__ = ["Pixel2PixelDataset"]
