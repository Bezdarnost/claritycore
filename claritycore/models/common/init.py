# Copyright (c) Aman Urumbekov and other contributors.
"""Weight initialization utilities for ClarityCore models."""

import math
import warnings

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """
    Fill tensor with values from a truncated normal distribution.

    Values are drawn from N(mean, std²) with values outside [a, b] redrawn.

    Args:
        tensor: Tensor to fill.
        mean: Mean of the normal distribution.
        std: Standard deviation.
        a: Minimum cutoff.
        b: Maximum cutoff.

    Returns:
        The filled tensor.
    """

    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in trunc_normal_. The distribution may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)

        return tensor


@torch.no_grad()
def default_init_weights(
    module_list: nn.Module | list[nn.Module],
    scale: float = 1.0,
    bias_fill: float = 0.0,
    **kwargs,
) -> None:
    """
    Initialize network weights with Kaiming normal.

    Commonly used for CNN architectures like ESRGAN.

    Args:
        module_list: Module(s) to initialize.
        scale: Scale factor for weights (useful for residual blocks).
        bias_fill: Value to fill biases with.
        **kwargs: Additional args for kaiming_normal_.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]

    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


__all__ = ["trunc_normal_", "default_init_weights"]
