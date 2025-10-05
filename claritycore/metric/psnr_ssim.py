# Copyright (c) Aman Urumbekov and other contributors.
import torch
import torch.nn.functional as F

from claritycore.utils.color_util import rgb2ycbcr
from claritycore.utils.registry import METRIC_REGISTRY
from claritycore.utils.preprocess_util import preprocess_psnr_ssim


def _create_gaussian_window(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel using only PyTorch.

    Args:
        kernel_size (int): The size of the kernel.
        sigma (float): The sigma of the kernel.
        device (torch.device): The device of the kernel.
        dtype (torch.dtype): The dtype of the kernel.

    Returns:
        torch.Tensor: The Gaussian kernel.
    """
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords -= (kernel_size - 1) / 2.0

    g = coords**2
    g = (-g / (2 * sigma**2)).exp()

    g /= g.sum()
    return g.outer(g)


def _ssim(pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
    """
    Calculate SSIM.

    Reference implementation can be found in BasicSR:
    https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/psnr_ssim.py

    Args:
        pred_img (torch.Tensor): The predicted image.
        target_img (torch.Tensor): The target image.

    Returns:
        torch.Tensor: The SSIM map.
    """
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # Create a Gaussian kernel and expand it to match the number of channels
    window = _create_gaussian_window(11, 1.5, pred_img.device, pred_img.dtype)
    window = window.view(1, 1, 11, 11).expand(pred_img.size(1), 1, 11, 11)

    # Use valid convolution with groups for per-channel calculation
    mu1 = F.conv2d(pred_img, window, stride=1, padding=0, groups=pred_img.shape[1])
    mu2 = F.conv2d(target_img, window, stride=1, padding=0, groups=target_img.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(pred_img * pred_img, window, stride=1, padding=0, groups=pred_img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target_img * target_img, window, stride=1, padding=0, groups=target_img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred_img * target_img, window, stride=1, padding=0, groups=target_img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


@METRIC_REGISTRY.register("ssim")
def ssim(
    pred_img: torch.Tensor,
    target_img: torch.Tensor,
    crop_border: int = 0,
    convert_to_y_channel: bool = False,
) -> torch.Tensor:
    """Calculate SSIM (Structural Similarity).

    Reference implementation can be found in BasicSR:
    https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/psnr_ssim.py

    Args:
        pred_img (torch.Tensor): Predicted image (range [0, 1], shape BCHW).
        target_img (torch.Tensor): Ground-truth image (range [0, 1], shape BCHW).
        crop_border (int): Pixels to crop from each border.
        convert_to_y_channel (bool): Whether to calculate on the Y channel of YCbCr.

    Returns:
        torch.Tensor: The SSIM result per image in the batch.
    """
    assert pred_img.shape == target_img.shape, f"Image shapes are different: {pred_img.shape}, {target_img.shape}."

    pred_img = preprocess_psnr_ssim(pred_img, crop_border, convert_to_y_channel)
    target_img = preprocess_psnr_ssim(target_img, crop_border, convert_to_y_channel)

    # The original SSIM calculation is designed for images with a range of [0, 255]
    return _ssim(pred_img * 255.0, target_img * 255.0)


@METRIC_REGISTRY.register("psnr")
def psnr(
    pred_img: torch.Tensor,
    target_img: torch.Tensor,
    crop_border: int = 0,
    convert_to_y_channel: bool = False,
) -> torch.Tensor:
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference implementation can be found in BasicSR:
    https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/psnr_ssim.py

    Args:
        pred_img (torch.Tensor): Predicted image (range [0, 1], shape BCHW).
        target_img (torch.Tensor): Ground-truth image (range [0, 1], shape BCHW).
        crop_border (int): Pixels to crop from each border.
        convert_to_y_channel (bool): Whether to calculate on the Y channel of YCbCr.

    Returns:
        torch.Tensor: The PSNR result per image in the batch.
    """
    assert pred_img.shape == target_img.shape, f"Image shapes are different: {pred_img.shape}, {target_img.shape}."

    pred_img = preprocess_psnr_ssim(pred_img, crop_border, convert_to_y_channel)
    target_img = preprocess_psnr_ssim(target_img, crop_border, convert_to_y_channel)

    mse = torch.mean((pred_img - target_img) ** 2, dim=(1, 2, 3))
    # Add a small epsilon to avoid log(0)
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))


__all__ = ["psnr", "ssim"]
