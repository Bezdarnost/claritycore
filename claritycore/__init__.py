"""
ClarityCore: Next-generation toolkit for image & video restoration.

Engineered for state-of-the-art performance in Super-Resolution,
Denoising, Deblurring, and more.

Example usage:
    from claritycore.models import AutoModel, AutoConfig
    from claritycore.data import Pixel2PixelDataset, DatasetConfig

    # Create a model
    config = AutoConfig.from_name("rrdbnet", scale=4)
    model = AutoModel.from_config(config)

    # Training
    from claritycore.training import Trainer, TrainingConfig

    data_config = DatasetConfig(target_dir="data/HR", scale=4)
    dataset = Pixel2PixelDataset(data_config)
    trainer = Trainer(model, train_loader, optimizer, config)
    trainer.train()

Copyright 2025 Aman Urumbekov. Apache License 2.0.
"""

from importlib.metadata import version

__version__ = version("claritycore")

# Convenience imports
from claritycore.data import DatasetConfig, NormConfig, Pixel2PixelDataset
from claritycore.metrics import psnr, ssim
from claritycore.models import AutoConfig, AutoModel
from claritycore.training import Trainer, TrainingConfig

__all__ = [
    "__version__",
    "AutoConfig",
    "AutoModel",
    "Trainer",
    "TrainingConfig",
    "Pixel2PixelDataset",
    "DatasetConfig",
    "NormConfig",
    "psnr",
    "ssim",
]
