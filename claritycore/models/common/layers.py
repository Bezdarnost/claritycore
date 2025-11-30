# Copyright (c) Aman Urumbekov and other contributors.
"""Common neural network layers for ClarityCore models."""

from typing import Any

import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron block.

    A simple feed-forward network with one hidden layer, commonly used
    in Transformer-style architectures.

    Args:
        dim: Number of input/output features.
        mlp_ratio: Ratio of hidden dim to input dim.
        act_layer: Activation function class.
        drop: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
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
    Flexible upsampling module supporting multiple strategies.

    Modes:
        - 'classic': PixelShuffle-based (ESPCN style)
        - 'lightweight': Interpolation + Conv (efficient)

    Args:
        scale: Upsampling scale factor.
        num_feat: Number of feature channels.
        mode: Upsampling strategy ('classic' or 'lightweight').
        one_step: If True, upsample in one step. If False, decompose into
            prime factors for multi-step upsampling.
        act_layer: Optional activation between steps.
        **kwargs: Additional args for nn.Upsample in lightweight mode.
    """

    def __init__(
        self,
        scale: int,
        num_feat: int,
        mode: str = "classic",
        one_step: bool = False,
        act_layer: type[nn.Module] | None = None,
        **kwargs: Any,
    ) -> None:
        layers: list[nn.Module] = []

        if mode == "classic":
            # PixelShuffle-based upsampling
            if one_step:
                layers.extend([
                    nn.Conv2d(num_feat, num_feat * (scale**2), 3, 1, 1),
                    nn.PixelShuffle(scale),
                ])
            else:
                factors = self._prime_factors(scale)
                for i, factor in enumerate(factors):
                    layers.extend([
                        nn.Conv2d(num_feat, num_feat * (factor**2), 3, 1, 1),
                        nn.PixelShuffle(factor),
                    ])
                    if act_layer is not None and i < len(factors) - 1:
                        layers.append(act_layer())

        elif mode == "lightweight":
            # Interpolation-based upsampling
            kwargs.setdefault("mode", "nearest")
            if one_step:
                layers.extend([
                    nn.Upsample(scale_factor=scale, **kwargs),
                    nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                ])
            else:
                factors = self._prime_factors(scale)
                for i, factor in enumerate(factors):
                    layers.extend([
                        nn.Upsample(scale_factor=factor, **kwargs),
                        nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                    ])
                    if act_layer is not None and i < len(factors) - 1:
                        layers.append(act_layer())
        else:
            raise ValueError(f"Unknown upsample mode: {mode}. Use 'classic' or 'lightweight'.")

        super().__init__(*layers)

    @staticmethod
    def _prime_factors(n: int) -> list[int]:
        """Decompose n into sorted prime factors."""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return sorted(factors)


__all__ = ["MLP", "Upsample"]

