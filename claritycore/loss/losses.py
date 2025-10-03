# Copyright (c) Aman Urumbekov and other contributors.
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from claritycore.utils.registry import LOSS_REGISTRY

Reduction = Literal["none", "mean", "sum"]


class LossBase(nn.Module):
    """
    Base class for all custom losses. Inherits from nn.Module.
    Handles the reduction boilerplate.
    """

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"'{reduction}' is not a valid reduction. Choose from 'none', 'mean', 'sum'.")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")

    @staticmethod
    def _reduce_loss(loss: torch.Tensor, reduction: Reduction) -> torch.Tensor:
        """Static helper to apply reduction to a raw loss tensor."""
        if reduction == "none":
            return loss
        if reduction == "sum":
            return torch.sum(loss)
        return torch.mean(loss)


@LOSS_REGISTRY.register("l1")
class L1Loss(LossBase):
    """Wrapper around torch.nn.functional.l1_loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target, reduction=self.reduction)


@LOSS_REGISTRY.register("mse")
class MSELoss(LossBase):
    """Wrapper around torch.nn.functional.mse_loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


@LOSS_REGISTRY.register("smooth_l1")
class SmoothL1Loss(LossBase):
    """
    Wrapper around torch.nn.functional.smooth_l1_loss.

    Args:
        beta (float): The threshold at which to switch from L2 to L1 loss. This value must be positive. Default: 1.0.
        reduction (str): The reduction method ('none', 'mean', 'sum').
    """

    def __init__(self, beta: float = 1.0, reduction: Reduction = "mean") -> None:
        super().__init__(reduction)
        if beta <= 0:
            raise ValueError("Beta must be positive.")
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.beta, reduction=self.reduction)


@LOSS_REGISTRY.register("huber")
class HuberLoss(LossBase):
    """
    Wrapper around torch.nn.functional.huber_loss.

    Args:
        delta (float): The threshold at which to change between delta-scaled L1 and L2 loss.
                       Default: 1.0. When delta equals 1, this loss is equivalent to SmoothL1Loss.
        reduction (str): The reduction method ('none', 'mean', 'sum').
    """

    def __init__(self, delta: float = 1.0, reduction: Reduction = "mean") -> None:
        super().__init__(reduction)
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta, reduction=self.reduction)


@LOSS_REGISTRY.register("charbonnier")
class CharbonnierLoss(LossBase):
    """
    Charbonnier loss, a differentiable L1 variant: sqrt(x^2 + eps).

    Args:
        eps (float): Small value to ensure stability and differentiability.
        reduction (str): The reduction method ('none', 'mean', 'sum').
    """

    def __init__(self, eps: float = 1e-8, reduction: Reduction = "mean") -> None:
        super().__init__(reduction)
        if eps <= 0:
            raise ValueError("Epsilon (eps) must be positive.")
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return self._reduce_loss(loss, self.reduction)


# Defines the public API of this module.
__all__ = [
    "L1Loss",
    "MSELoss",
    "SmoothL1Loss",
    "HuberLoss",
    "CharbonnierLoss",
]
