# Copyright (c) Aman Urumbekov and other contributors.
"""RRDBNet model wrapper with training support."""

import torch
import torch.nn as nn

from claritycore.models.base import BaseModel
from claritycore.models.auto import register_model
from claritycore.models.rrdbnet.config import RRDBNetConfig
from claritycore.models.rrdbnet.architecture import RRDBNet


@register_model("rrdbnet")
class RRDBNetModel(BaseModel):
    """
    RRDBNet model for super-resolution.

    This class wraps the RRDBNet architecture with training functionality:
    - Loss computation
    - EMA support
    - Tiled inference for large images

    Usage:
        config = RRDBNetConfig(scale=4, num_block=23)
        model = RRDBNetModel(config)

        # Training
        model.add_loss("l1", nn.L1Loss())
        output = model(lr_image)
        loss, loss_dict = model.compute_loss(output, hr_image)

        # Inference
        sr_image = model.inference(lr_image)
    """

    config_class = RRDBNetConfig

    def __init__(self, config: RRDBNetConfig) -> None:
        super().__init__(config)

    def _build_network(self) -> None:
        """Build the RRDBNet architecture."""
        cfg = self.config
        self.net = RRDBNet(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            scale=cfg.scale,
            num_feat=cfg.num_feat,
            num_block=cfg.num_block,
            num_grow_ch=cfg.num_grow_ch,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Low-resolution input (B, C, H, W).

        Returns:
            Super-resolved output (B, C, H*scale, W*scale).
        """
        return self.net(x)

    @torch.no_grad()
    def inference(self, x: torch.Tensor, use_ema: bool = True) -> torch.Tensor:
        """
        Run inference.

        Args:
            x: Low-resolution input.
            use_ema: Whether to use EMA weights if available.

        Returns:
            Super-resolved output.
        """
        was_training = self.training
        self.eval()

        if use_ema and self.net_ema is not None:
            output = self.net_ema(x)
        else:
            output = self.net(x)

        if was_training:
            self.train()

        return output

    @torch.no_grad()
    def inference_tiled(
        self,
        x: torch.Tensor,
        tile_size: int = 256,
        tile_overlap: int = 32,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Tiled inference for large images that don't fit in memory.

        Splits input into overlapping tiles, processes each, and blends results.

        Args:
            x: Low-resolution input (B, C, H, W).
            tile_size: Size of each tile.
            tile_overlap: Overlap between tiles.
            use_ema: Whether to use EMA weights.

        Returns:
            Super-resolved output.
        """
        was_training = self.training
        self.eval()

        net = self.net_ema if (use_ema and self.net_ema is not None) else self.net
        scale = self.config.scale

        b, c, h, w = x.shape
        tile_size = min(tile_size, h, w)
        stride = tile_size - tile_overlap

        # Output dimensions
        h_out, w_out = h * scale, w * scale
        output = torch.zeros((b, c, h_out, w_out), device=x.device, dtype=x.dtype)
        count = torch.zeros((b, 1, h_out, w_out), device=x.device, dtype=x.dtype)

        # Process tiles
        for y in range(0, h, stride):
            for x_pos in range(0, w, stride):
                # Extract tile with boundary handling
                y_end = min(y + tile_size, h)
                x_end = min(x_pos + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                tile = x[:, :, y_start:y_end, x_start:x_end]
                tile_out = net(tile)

                # Place in output
                y_out_s, y_out_e = y_start * scale, y_end * scale
                x_out_s, x_out_e = x_start * scale, x_end * scale

                output[:, :, y_out_s:y_out_e, x_out_s:x_out_e] += tile_out
                count[:, :, y_out_s:y_out_e, x_out_s:x_out_e] += 1

        # Average overlapping regions
        output = output / count.clamp(min=1)

        if was_training:
            self.train()

        return output


__all__ = ["RRDBNetModel"]

