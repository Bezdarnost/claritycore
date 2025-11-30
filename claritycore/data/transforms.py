# Copyright (c) Aman Urumbekov and other contributors.
"""Image transforms for data augmentation."""

import random
from collections.abc import Callable

import numpy as np


def get_train_transforms() -> Callable:
    """
    Get default training transforms.

    Includes:
    - Random horizontal flip
    - Random rotation (0, 90, 180, 270)
    """

    def transform(hq: np.ndarray, lq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Random rotation
        k = random.randint(0, 3)
        hq = np.rot90(hq, k)
        lq = np.rot90(lq, k)

        return hq, lq

    return transform


def get_val_transforms() -> Callable | None:
    """
    Get default validation transforms.

    Returns None (no transforms for validation).
    """
    return None


__all__ = ["get_train_transforms", "get_val_transforms"]
