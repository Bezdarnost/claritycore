# Copyright (c) Aman Urumbekov and other contributors.
"""Configuration for RRDBNet model."""

from dataclasses import dataclass, field
from typing import Literal

from claritycore.models.auto import register_config
from claritycore.models.base import BaseConfig


@dataclass
class RRDBNetTrainingPreset:
    """
    Training preset for RRDBNet.

    Defines recommended hyperparameters for different scale factors and variants.
    """

    # Training
    total_steps: int = 400000
    batch_size: int = 16
    patch_size: int = 256
    lr: float = 2e-4
    loss: str = "l1"

    # Logging
    log_freq: int = 100
    val_freq: int = 5000
    save_freq: int = 5000

    # Features
    use_amp: bool = True
    use_ema: bool = False

    # Data
    norm: Literal["zero_one", "minus_one_one", "none"] = "minus_one_one"
    dataset: str = "div2k"
    dataset_paths: list[str] = field(default_factory=lambda: ["datasets/DIV2K", "data/DIV2K"])


# Training presets for different RRDBNet configurations
RRDBNET_PRESETS: dict[str, tuple["RRDBNetConfig", RRDBNetTrainingPreset]] = {}


def _register_preset(name: str, config: "RRDBNetConfig", preset: RRDBNetTrainingPreset):
    """Register a training preset."""
    RRDBNET_PRESETS[name] = (config, preset)


@register_config("rrdbnet")
@dataclass
class RRDBNetConfig(BaseConfig):
    """
    Configuration for RRDBNet (ESRGAN generator).

    RRDBNet is a powerful CNN architecture for super-resolution based on
    Residual-in-Residual Dense Blocks (RRDB). It achieves excellent perceptual
    quality when trained with GAN losses.

    Architecture Details:
        - Uses Residual-in-Residual Dense Blocks (RRDB)
        - No batch normalization for stable training
        - Leaky ReLU activations
        - Pixel shuffle upsampling

    Variants:
        - Full (64 feat, 23 blocks): ~16.7M params - Best quality
        - Lite (32 feat, 6 blocks): ~1M params - Fast inference

    Reference:
        ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
        Wang et al., ECCV 2018 Workshop
        https://arxiv.org/abs/1809.00219

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output image channels.
        scale: Upscaling factor (2, 3, 4, or 8).
        num_feat: Number of intermediate feature channels (32 for lite, 64 for full).
        num_block: Number of RRDB blocks (6 for lite, 23 for full).
        num_grow_ch: Growth channels in dense blocks.

    Example:
        >>> config = RRDBNetConfig(scale=4, num_feat=64, num_block=23)
        >>> model = AutoModel.from_config(config)
    """

    model_type: str = "rrdbnet"

    # Architecture
    in_channels: int = 3
    out_channels: int = 3
    scale: int = 4
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32

    @classmethod
    def get_preset(cls, name: str) -> tuple["RRDBNetConfig", RRDBNetTrainingPreset] | None:
        """Get a named preset configuration."""
        return RRDBNET_PRESETS.get(name.lower())

    @classmethod
    def list_presets(cls) -> list[str]:
        """List available preset names."""
        return list(RRDBNET_PRESETS.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Standard RRDBNet Presets (64 features, 23 blocks, ~16.7M params)
# ═══════════════════════════════════════════════════════════════════════════════

_register_preset(
    "rrdbnetx2",
    RRDBNetConfig(scale=2, num_feat=64, num_block=23),
    RRDBNetTrainingPreset(
        patch_size=128,
        total_steps=400000,
        batch_size=16,
    ),
)

_register_preset(
    "rrdbnetx3",
    RRDBNetConfig(scale=3, num_feat=64, num_block=23),
    RRDBNetTrainingPreset(
        patch_size=192,
        total_steps=400000,
        batch_size=16,
    ),
)

_register_preset(
    "rrdbnetx4",
    RRDBNetConfig(scale=4, num_feat=64, num_block=23),
    RRDBNetTrainingPreset(
        patch_size=256,
        total_steps=400000,
        batch_size=16,
    ),
)

_register_preset(
    "rrdbnetx8",
    RRDBNetConfig(scale=8, num_feat=64, num_block=23),
    RRDBNetTrainingPreset(
        patch_size=512,
        total_steps=500000,
        batch_size=8,  # Larger patches need smaller batch
    ),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Lightweight RRDBNet Presets (32 features, 6 blocks, ~1M params)
# ═══════════════════════════════════════════════════════════════════════════════

_register_preset(
    "rrdbnet-litex2",
    RRDBNetConfig(scale=2, num_feat=32, num_block=6),
    RRDBNetTrainingPreset(
        patch_size=128,
        total_steps=200000,
        batch_size=32,  # Smaller model = larger batch
    ),
)

_register_preset(
    "rrdbnet-litex3",
    RRDBNetConfig(scale=3, num_feat=32, num_block=6),
    RRDBNetTrainingPreset(
        patch_size=192,
        total_steps=200000,
        batch_size=32,
    ),
)

_register_preset(
    "rrdbnet-litex4",
    RRDBNetConfig(scale=4, num_feat=32, num_block=6),
    RRDBNetTrainingPreset(
        patch_size=256,
        total_steps=200000,
        batch_size=32,
    ),
)

_register_preset(
    "rrdbnet-litex8",
    RRDBNetConfig(scale=8, num_feat=32, num_block=6),
    RRDBNetTrainingPreset(
        patch_size=512,
        total_steps=250000,
        batch_size=16,
    ),
)


__all__ = ["RRDBNetConfig", "RRDBNetTrainingPreset", "RRDBNET_PRESETS"]
