# Copyright (c) Aman Urumbekov and other contributors.
"""Loss functions for ClarityCore models."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Reduction = Literal["none", "mean", "sum"]


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss."""

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (Mean Squared Error) loss."""

    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (differentiable L1 variant).

    L = sqrt((pred - target)² + ε)

    More robust than L1 for small errors.

    Args:
        eps: Small constant for numerical stability.
        reduction: Reduction method.
    """

    def __init__(self, eps: float = 1e-6, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) loss.

    Combines L1 and L2 loss - L2 for small errors, L1 for large.

    Args:
        beta: Threshold for switching from L2 to L1.
        reduction: Reduction method.
    """

    def __init__(self, beta: float = 1.0, reduction: Reduction = "mean") -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.beta, reduction=self.reduction)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.

    Compares high-level features from a pretrained VGG network.

    Args:
        layer_weights: Dict mapping layer name to weight.
        use_input_norm: Normalize inputs to ImageNet stats.
        reduction: Reduction method.
    """

    def __init__(
        self,
        layer_weights: dict[str, float] | None = None,
        use_input_norm: bool = True,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()

        self.layer_weights = layer_weights or {
            "conv1_2": 0.1,
            "conv2_2": 0.1,
            "conv3_4": 1.0,
            "conv4_4": 1.0,
            "conv5_4": 1.0,
        }
        self.use_input_norm = use_input_norm
        self.reduction = reduction

        # Load VGG lazily to avoid import overhead
        self._vgg = None

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _load_vgg(self, device: torch.device) -> nn.Module:
        """Lazily load VGG model."""
        if self._vgg is None:
            from torchvision.models import vgg19, VGG19_Weights

            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad_(False)

            self._vgg = vgg.to(device)

        return self._vgg

    def _extract_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract VGG features at specified layers."""
        vgg = self._load_vgg(x.device)

        if self.use_input_norm:
            x = (x - self.mean) / self.std

        features = {}
        layer_name_mapping = {
            3: "conv1_2",
            8: "conv2_2",
            17: "conv3_4",
            26: "conv4_4",
            35: "conv5_4",
        }

        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in layer_name_mapping:
                name = layer_name_mapping[i]
                if name in self.layer_weights:
                    features[name] = x

        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        loss = torch.tensor(0.0, device=pred.device)

        for name, weight in self.layer_weights.items():
            if name in pred_features:
                layer_loss = F.l1_loss(
                    pred_features[name],
                    target_features[name],
                    reduction=self.reduction,
                )
                loss = loss + weight * layer_loss

        return loss


class GANLoss(nn.Module):
    """
    GAN loss for discriminator training.

    Supports multiple GAN variants.

    Args:
        gan_type: Type of GAN loss ('vanilla', 'lsgan', 'wgan', 'hinge').
        real_label: Target value for real samples.
        fake_label: Target value for fake samples.
    """

    def __init__(
        self,
        gan_type: Literal["vanilla", "lsgan", "wgan", "hinge"] = "vanilla",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()

        self.gan_type = gan_type
        self.real_label = real_label
        self.fake_label = fake_label

        if gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_type in ("wgan", "hinge"):
            self.loss = None
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")

    def forward(
        self,
        pred: torch.Tensor,
        target_is_real: bool,
        is_discriminator: bool = False,
    ) -> torch.Tensor:
        """
        Compute GAN loss.

        Args:
            pred: Discriminator output.
            target_is_real: Whether target is real or fake.
            is_discriminator: Whether computing for discriminator or generator.
        """
        if self.gan_type in ("vanilla", "lsgan"):
            target = pred.new_full(pred.shape, self.real_label if target_is_real else self.fake_label)
            return self.loss(pred, target)

        elif self.gan_type == "wgan":
            return -pred.mean() if target_is_real else pred.mean()

        elif self.gan_type == "hinge":
            if is_discriminator:
                if target_is_real:
                    return F.relu(1 - pred).mean()
                else:
                    return F.relu(1 + pred).mean()
            else:
                return -pred.mean()


# Convenience function to create losses
def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Get a loss function by name.

    Args:
        name: Loss name ('l1', 'mse', 'charbonnier', 'perceptual', 'gan').
        **kwargs: Loss-specific arguments.

    Returns:
        Loss module.
    """
    losses = {
        "l1": L1Loss,
        "mse": MSELoss,
        "charbonnier": CharbonnierLoss,
        "smooth_l1": SmoothL1Loss,
        "perceptual": PerceptualLoss,
        "gan": GANLoss,
    }

    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")

    return losses[name](**kwargs)


__all__ = [
    "L1Loss",
    "MSELoss",
    "CharbonnierLoss",
    "SmoothL1Loss",
    "PerceptualLoss",
    "GANLoss",
    "get_loss",
]

