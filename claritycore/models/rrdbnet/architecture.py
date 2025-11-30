# Copyright (c) Aman Urumbekov and other contributors.
"""
RRDBNet architecture for ESRGAN-style super-resolution.

Reference:
    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
    https://arxiv.org/abs/1809.00219

Based on implementation from:
    https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
"""

import torch
import torch.nn as nn

from claritycore.models.common.layers import Upsample
from claritycore.models.common.init import default_init_weights


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block.

    Features dense connections within a residual block, allowing the network
    to learn complex hierarchical features.

    Args:
        num_feat: Number of input/output feature channels.
        num_grow_ch: Number of channels to grow at each conv layer.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialize weights
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], scale=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))

        # Residual scaling (0.2 empirically works well)
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block.

    Consists of 3 RDB blocks with residual connection.

    Args:
        num_feat: Number of feature channels.
        num_grow_ch: Growth channels in dense blocks.
    """

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()

        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        # Residual scaling
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDBNet: Networks consisting of Residual in Residual Dense Blocks.

    The generator architecture used in ESRGAN for high-quality super-resolution.

    Architecture:
        1. Shallow feature extraction (single conv)
        2. Deep feature extraction (RRDB blocks)
        3. Upsampling module
        4. Reconstruction (conv layers)

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        scale: Upscaling factor (1, 2, or 4).
        num_feat: Number of intermediate feature channels.
        num_block: Number of RRDB blocks.
        num_grow_ch: Growth channels in dense blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()

        self.scale = scale

        # Handle scale 1 and 2 with pixel unshuffle
        num_in_ch = in_channels
        if scale == 2:
            num_in_ch = in_channels * 4
        elif scale == 1:
            num_in_ch = in_channels * 16

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # Deep feature extraction (RRDB blocks)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling
        self.upsample = Upsample(
            scale=scale,
            num_feat=num_feat,
            mode="lightweight",
            one_step=False,
        )

        # Reconstruction
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different scales with pixel unshuffle
        if self.scale == 2:
            feat = nn.functional.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = nn.functional.pixel_unshuffle(x, 4)
        else:
            feat = x

        # Shallow features
        feat = self.conv_first(feat)

        # Deep features with global residual
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsample and reconstruct
        feat = self.lrelu(self.upsample(feat))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        return out


__all__ = ["RRDBNet", "RRDB", "ResidualDenseBlock"]

