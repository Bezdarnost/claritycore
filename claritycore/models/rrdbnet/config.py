# Copyright (c) Aman Urumbekov and other contributors.
"""Configuration for RRDBNet model."""

from dataclasses import dataclass

from claritycore.models.base import BaseConfig
from claritycore.models.auto import register_config


@register_config("rrdbnet")
@dataclass
class RRDBNetConfig(BaseConfig):
    """
    Configuration for RRDBNet (ESRGAN generator).

    RRDBNet is a powerful CNN architecture for super-resolution based on
    Residual-in-Residual Dense Blocks (RRDB).

    Reference:
        ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
        https://arxiv.org/abs/1809.00219

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output image channels.
        scale: Upscaling factor (1, 2, or 4).
        num_feat: Number of intermediate feature channels.
        num_block: Number of RRDB blocks in the network.
        num_grow_ch: Growth channels in dense blocks.
    """

    model_type: str = "rrdbnet"

    # Architecture
    in_channels: int = 3
    out_channels: int = 3
    scale: int = 4
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32


__all__ = ["RRDBNetConfig"]

