# Copyright (c) Aman Urumbekov and other contributors.
from typing import Any, List, Optional, Type

import torch.nn as nn


class MLP(nn.Module):
    """
    MLP module.

    Args:
        dim (int): Number of input features.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        act_layer (nn.Module): Activation layer.
        drop (float): Dropout rate.

    Returns:
        torch.Tensor: Output tensor.
    """

    def __init__(self, dim, mlp_ratio=4.0, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Upsample(nn.Sequential):
    """
    A comprehensive and architecturally consistent upsampling module.

    This module provides two core upsampling strategies ('classic' and 'lightweight').
    The structure of the operation for both modes is controlled by the `one_step`
    flag, which toggles between a single-block and a multi-block chained operation.

    Args:
        scale (int): The desired upsampling scale factor.
        num_feat (int): The number of channels in the input feature maps.
        mode (str): The upsampling strategy. Supported modes: 'classic', 'lightweight'.
        one_step (bool): Controls the operation's structure. Defaults to `False`.
            - `False`: Decomposes `scale` into its prime factors and creates a
              chain of upsampling blocks.
            - `True`: Performs a single, direct upsampling block for the
              entire `scale`.
        act_layer (Optional[Type[nn.Module]]): Activation layer to use between
            the blocks in multi-step (`one_step=False`) modes. For example,
            `nn.ReLU` or `nn.GELU`. Defaults to `None` (no activation).
        **kwargs (Any): Additional keyword arguments passed to the underlying
            `nn.Upsample` layers in 'lightweight' mode.
    """

    def __init__(
        self,
        scale: int,
        num_feat: int,
        mode: str = "classic",
        one_step: bool = False,
        act_layer: Optional[Type[nn.Module]] = None,
        **kwargs: Any,
    ):
        layers: List[nn.Module] = []

        # --- Mode: 'classic' (PixelShuffle based) ---
        if mode == "classic":
            if one_step:
                # Single-step: A single Conv + PixelShuffle block.
                layers.extend([nn.Conv2d(num_feat, num_feat * (scale**2), 3, 1, 1), nn.PixelShuffle(scale)])
            else:
                # Multi-step: Chain of Conv + PixelShuffle blocks from prime factors.
                factors = self._get_prime_factors(scale)
                for i, factor in enumerate(factors):
                    layers.extend([nn.Conv2d(num_feat, num_feat * (factor**2), 3, 1, 1), nn.PixelShuffle(factor)])
                    if act_layer is not None and i < len(factors) - 1:
                        layers.append(act_layer())

        # --- Mode: 'lightweight' (Interpolation based) ---
        elif mode == "lightweight":
            kwargs.setdefault("mode", "nearest")
            if one_step:
                # Single-step: A single Upsample + Conv block.
                layers.extend([nn.Upsample(scale_factor=scale, **kwargs), nn.Conv2d(num_feat, num_feat, 3, 1, 1)])
            else:
                # Multi-step: Chain of Upsample + Conv blocks from prime factors.
                factors = self._get_prime_factors(scale)
                for i, factor in enumerate(factors):
                    layers.extend([nn.Upsample(scale_factor=factor, **kwargs), nn.Conv2d(num_feat, num_feat, 3, 1, 1)])
                    if act_layer is not None and i < len(factors) - 1:
                        layers.append(act_layer())

        else:
            raise NotImplementedError(
                f"Upsample mode '{mode}' is not supported. Available modes: 'classic', 'lightweight'."
            )

        super().__init__(*layers)

    @staticmethod
    def _get_prime_factors(n: int) -> List[int]:
        """
        Decomposes an integer into its sorted prime factors.
        Example: _get_prime_factors(12) -> [2, 2, 3]
        """
        factors = []
        d = 2
        temp_n = n
        while d * d <= temp_n:
            while (temp_n % d) == 0:
                factors.append(d)
                temp_n //= d
            d += 1
        if temp_n > 1:
            factors.append(temp_n)
        return sorted(factors)
