# Copyright (c) Aman Urumbekov and other contributors.
import random
from collections.abc import Callable
from pathlib import Path

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
    A flexible dataset for low-level vision tasks (pixel-to-pixel).

    This dataset can operate in three primary modes based on the provided arguments:

    1.  Single Image Mode:
        - Returns a single image from the `root` directory.
        - Activated when `scale_factor` and `lr_template` are `None`.

    2.  Self-Supervised Super-Resolution Mode:
        - Returns an (LR, HR) pair created on-the-fly by downsampling the HR image.
        - Activated when `scale_factor` is provided, but `lr_template` is `None`.

    3.  Paired Image Mode:
        - Returns an (LR, HR) pair from separate files, located via a flexible template.
        - Activated when `scale_factor` and `lr_template` are provided.

    Args:
        root (str): Path to the root directory. For paired mode, this is where HR images are located.
        hr_channels (int): Number of channels for the high-res output image (e.g., 1 for grayscale, 3 for RGB).
        lr_channels (Optional[int]): Channels for the low-res output. If None, defaults to `hr_channels`.
        scale_factor (Optional[int]): Super-resolution scale factor. Required for modes 2 and 3.
        hr_size (Optional[int]): Target patch size for HR images. If an image is smaller, it's skipped.
            If larger, a random crop is taken. LR images are cropped accordingly. Set to None to disable. Default: 256.
        lr_template (Optional[str]): A template to find LR images relative to HR names.
            Placeholders: {basename}, {basename_nosuffix}, {stem}.
            Example: If HR is `img_HR.png` and LR is `img.png`,
                use `hr_suffix="_HR"` and `lr_template="{basename_nosuffix}.png"`.
        hr_glob (str): A glob pattern to find HR images within `root`. Defaults to all files.
        hr_suffix (Optional[str]): An optional suffix to remove from HR filenames before applying the LR template.
        transform (Optional[Callable]): A transform to apply to the image(s).
        verify_dims (bool): In paired mode, verify that HR is `scale_factor` times larger than LR.
    """

    def __init__(
        self,
        root: str,
        hr_channels: int,
        lr_channels: int | None = None,
        scale_factor: int | None = None,
        hr_size: int | None = 256,
        lr_template: str | None = None,
        hr_glob: str = "*",
        hr_suffix: str | None = None,
        transform: Callable | None = None,
        verify_dims: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.hr_channels = hr_channels
        self.lr_channels = lr_channels if lr_channels is not None else hr_channels
        self.hr_size = hr_size
        self.image_paths: list[Path] | list[dict[str, Path]] = []

        # --- Determine dataset mode and validate configuration ---
        is_paired = scale_factor is not None and lr_template is not None
        is_self_supervised = scale_factor is not None and lr_template is None
        is_single = scale_factor is None and lr_template is None

        if is_single:
            self._mode = "single"
            self.image_paths = self._scan_files(self.root, hr_glob)
        elif is_self_supervised:
            self._mode = "self_supervised"
            self.scale_factor = scale_factor
            self.image_paths = self._scan_files(self.root, hr_glob)
        elif is_paired:
            self._mode = "paired"
            self.scale_factor = scale_factor
            self.verify_dims = verify_dims
            self._find_and_validate_pairs(lr_template, hr_glob, hr_suffix)
        else:
            raise ValueError("Invalid combination of arguments. Please check docstring for valid modes.")

        if not self.image_paths:
            raise FileNotFoundError(f"No valid image files or pairs found in '{self.root}' with glob '{hr_glob}'.")

        # Pre-filter too-small images if hr_size is set, so __getitem__ never returns None.
        if self.hr_size is not None:
            if self._mode in ("single", "self_supervised"):
                filtered: list[Path] = []
                for p in self.image_paths:  # type: ignore[assignment]
                    try:
                        w, h = Image.open(p).size
                        if h >= self.hr_size and w >= self.hr_size:
                            filtered.append(p)
                    except Exception:
                        logger.exception(f"Failed to open image '{p}' for size check; skipping.")
                self.image_paths = filtered
            elif self._mode == "paired":
                filtered_pairs: list[dict[str, Path]] = []
                for pair in self.image_paths:  # type: ignore[assignment]
                    try:
                        w, h = Image.open(pair["hr"]).size
                        if h >= self.hr_size and w >= self.hr_size:
                            filtered_pairs.append(pair)
                    except Exception:
                        logger.exception(f"Failed to open HR image '{pair.get('hr')}' for size check; skipping pair.")
                self.image_paths = filtered_pairs

        if not self.image_paths:
            raise FileNotFoundError("No images remain after filtering by hr_size.")

        logger.info(f"Prepared {len(self.image_paths)} samples for mode '{self._mode}'.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        try:
            # Step 1: Load image(s) into PIL objects
            hr_img, lr_img = self._load_images(index)

            # Step 2: Apply synchronized cropping if required
            if self.hr_size is not None:
                hr_img, lr_img = self._apply_cropping(hr_img, lr_img)

            # Step 3: Convert to NumPy, then to Tensors
            hr_tensor = self._process_image(np.array(hr_img), self.hr_channels)

            if self._mode == "single":
                if self.transform:
                    hr_tensor = self.transform(hr_tensor)
                return hr_tensor

            # For paired modes
            lr_tensor = self._process_image(np.array(lr_img), self.lr_channels)
            output = (lr_tensor, hr_tensor)
            if self.transform:
                output = self.transform(output)
            return output

        except Exception:
            path_info = self.image_paths[index]
            logger.exception(f"Error loading image at index {index} with path info: {path_info}")
            return None

    def _load_images(self, index: int) -> tuple[Image.Image, Image.Image | None]:
        """Loads HR and optionally LR images based on the dataset mode."""
        if self._mode == "single":
            hr_path = self.image_paths[index]
            return Image.open(hr_path), None

        if self._mode == "self_supervised":
            hr_path = self.image_paths[index]
            hr_img = Image.open(hr_path)
            w, h = hr_img.size
            lr_w, lr_h = w // self.scale_factor, h // self.scale_factor
            lr_img = hr_img.resize((lr_w, lr_h), BICUBIC)
            return hr_img, lr_img

        if self._mode == "paired":
            pair = self.image_paths[index]
            hr_path, lr_path = pair["hr"], pair["lr"]
            hr_img = Image.open(hr_path)
            lr_img = Image.open(lr_path)

            if self.verify_dims:
                hr_w, hr_h = hr_img.size
                lr_w, lr_h = lr_img.size
                if hr_h != lr_h * self.scale_factor or hr_w != lr_w * self.scale_factor:
                    logger.warning(
                        f"Dimension mismatch for {hr_path.name}. "
                        f"HR: ({hr_h}, {hr_w}), LR: ({lr_h}, {lr_w}), Scale: {self.scale_factor}"
                    )
            return hr_img, lr_img

    def _apply_cropping(
        self, hr_img: Image.Image, lr_img: Image.Image | None
    ) -> tuple[Image.Image | None, Image.Image | None]:
        """Applies synchronized random cropping to HR and LR images."""
        w, h = hr_img.size
        if h < self.hr_size or w < self.hr_size:
            # Should not happen after pre-filter; guard anyway.
            raise ValueError("Image smaller than hr_size after pre-filtering.")

        # Get random crop coordinates for HR
        top = random.randint(0, h - self.hr_size)
        left = random.randint(0, w - self.hr_size)
        hr_img_cropped = hr_img.crop((left, top, left + self.hr_size, top + self.hr_size))

        lr_img_cropped = None
        if lr_img is not None:
            lr_size = self.hr_size // self.scale_factor
            lr_top = top // self.scale_factor
            lr_left = left // self.scale_factor
            lr_img_cropped = lr_img.crop((lr_left, lr_top, lr_left + lr_size, lr_top + lr_size))

        return hr_img_cropped, lr_img_cropped

    def _find_and_validate_pairs(self, lr_template: str, hr_glob: str, hr_suffix: str | None):
        """Scans for HR images and validates that a corresponding LR image exists."""
        hr_paths = self._scan_files(self.root, hr_glob)
        validated_pairs = []
        for hr_path in hr_paths:
            basename = hr_path.stem
            basename_nosuffix = basename
            if hr_suffix:
                if not basename.endswith(hr_suffix):
                    continue
                basename_nosuffix = basename[: -len(hr_suffix)]

            template_vars = {"basename": basename, "basename_nosuffix": basename_nosuffix, "stem": hr_path.name}
            lr_path = (hr_path.parent / lr_template.format_map(template_vars)).resolve()

            if lr_path.exists():
                validated_pairs.append({"lr": lr_path, "hr": hr_path})
            else:
                logger.warning(f"Could not find matching LR image for HR '{hr_path.name}' at expected path '{lr_path}'")
        self.image_paths = validated_pairs

    @staticmethod
    def _process_image(img_np: np.ndarray, target_channels: int) -> torch.Tensor:
        """Converts a NumPy image to a PyTorch tensor with correct channels and range."""
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        elif img_np.dtype == np.uint16:
            img_np = img_np.astype(np.float32) / 65535.0

        # Add channel dimension if not present
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=2)

        # Convert to target channels
        current_channels = img_np.shape[2]
        if current_channels != target_channels:
            if current_channels == 4 and target_channels == 3:  # RGBA -> RGB
                img_np = img_np[:, :, :3]
            elif current_channels == 1 and target_channels == 3:  # Gray -> RGB
                img_np = np.concatenate([img_np] * 3, axis=2)
            elif target_channels == 1:  # Any -> Gray
                if current_channels >= 3:  # RGB/RGBA -> Gray
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
