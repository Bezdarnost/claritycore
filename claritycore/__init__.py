"""
ClarityCore: Next-generation toolkit for image & video restoration.

Engineered for state-of-the-art performance in Super-Resolution,
Denoising, Deblurring, and more.

Example usage:
    from claritycore.models import AutoModel, AutoConfig

    # Create a model
    config = AutoConfig.from_name("rrdbnet", scale=4)
    model = AutoModel.from_config(config)

    # Training
    from claritycore.training import Trainer, TrainingConfig
    from claritycore.data import ImagePairDataset

    dataset = ImagePairDataset(hq_root="data/train/HQ", scale=4)
    trainer = Trainer(model, train_loader, optimizer, config)
    trainer.train()

Copyright 2025 Aman Urumbekov. Apache License 2.0.
"""

from importlib.metadata import version

__version__ = version("claritycore")

# Convenience imports
from claritycore.models import AutoConfig, AutoModel
from claritycore.training import Trainer, TrainingConfig
from claritycore.data import ImagePairDataset
from claritycore.metrics import psnr, ssim

__all__ = [
    "__version__",
    "AutoConfig",
    "AutoModel",
    "Trainer",
    "TrainingConfig",
    "ImagePairDataset",
    "psnr",
    "ssim",
]
