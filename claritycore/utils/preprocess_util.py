# Copyright (c) Aman Urumbekov and other contributors.
import torch

from claritycore.utils.color_util import rgb2ycbcr


def preprocess_psnr_ssim(img: torch.Tensor, crop_border: int = 0, convert_to_y_channel: bool = False) -> torch.Tensor:
    """
    Unified preprocessing for PSNR and SSIM metrics.

    Args:
        img (torch.Tensor): The input image.
        crop_border (int): The border to crop from the image. Default: 0.
        convert_to_y_channel (bool): Whether to convert the image to Y channel. Default: False.

    Returns:
        torch.Tensor: The preprocessed image with dtype torch.float64.
    """
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]

    if convert_to_y_channel:
        img = rgb2ycbcr(img, y_only=True)

    return img.to(torch.float64)
