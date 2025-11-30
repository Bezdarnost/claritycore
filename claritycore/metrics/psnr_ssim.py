# Copyright (c) Aman Urumbekov and other contributors.
"""PSNR and SSIM metrics for image quality assessment."""

import torch
import torch.nn.functional as F


def _rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to Y channel (luminance).

    Uses ITU-R BT.601 conversion coefficients.

    Args:
        img: RGB image tensor (B, 3, H, W) in range [0, 1].

    Returns:
        Y channel tensor (B, 1, H, W).
    """
    weight = torch.tensor(
        [[65.481], [128.553], [24.966]],
        device=img.device,
        dtype=img.dtype,
    )
    y = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    return y / 255.0


def _create_gaussian_kernel(
    size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.outer(g)


@torch.no_grad()
def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_border: int = 0,
    convert_to_y: bool = False,
) -> torch.Tensor:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).

    Args:
        pred: Predicted image (B, C, H, W) in range [0, 1].
        target: Ground truth image (B, C, H, W) in range [0, 1].
        crop_border: Pixels to crop from each border.
        convert_to_y: Calculate on Y channel only.

    Returns:
        PSNR value for each image in batch.
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

    # Crop borders
    if crop_border > 0:
        pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
        target = target[..., crop_border:-crop_border, crop_border:-crop_border]

    # Convert to Y channel
    if convert_to_y and pred.shape[1] == 3:
        pred = _rgb_to_y(pred)
        target = _rgb_to_y(target)

    # Calculate MSE and PSNR
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr_val = 10.0 * torch.log10(1.0 / (mse + 1e-8))

    return psnr_val


@torch.no_grad()
def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_border: int = 0,
    convert_to_y: bool = False,
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Calculate SSIM (Structural Similarity Index).

    Args:
        pred: Predicted image (B, C, H, W) in range [0, 1].
        target: Ground truth image (B, C, H, W) in range [0, 1].
        crop_border: Pixels to crop from each border.
        convert_to_y: Calculate on Y channel only.
        kernel_size: Size of Gaussian kernel.
        sigma: Standard deviation of Gaussian kernel.

    Returns:
        SSIM value for each image in batch.
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

    # Crop borders
    if crop_border > 0:
        pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
        target = target[..., crop_border:-crop_border, crop_border:-crop_border]

    # Convert to Y channel
    if convert_to_y and pred.shape[1] == 3:
        pred = _rgb_to_y(pred)
        target = _rgb_to_y(target)

    # Scale to 0-255 range (standard SSIM calculation)
    pred = pred * 255.0
    target = target * 255.0

    # SSIM constants
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # Create Gaussian kernel
    kernel = _create_gaussian_kernel(kernel_size, sigma, pred.device, pred.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.expand(pred.shape[1], 1, kernel_size, kernel_size)

    # Compute SSIM
    mu1 = F.conv2d(pred, kernel, groups=pred.shape[1], padding=0)
    mu2 = F.conv2d(target, kernel, groups=target.shape[1], padding=0)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, groups=pred.shape[1], padding=0) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, groups=target.shape[1], padding=0) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, groups=pred.shape[1], padding=0) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim_map.mean(dim=(1, 2, 3))


__all__ = ["psnr", "ssim"]
