# Copyright (c) Aman Urumbekov and other contributors.
import torch


def rgb2ycbcr(img: torch.Tensor, y_only: bool = False) -> torch.Tensor:
    """
    Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Reference implementation can be found in BasicSR:
    https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/color_util.py

    Args:
        img (torch.Tensor): The input RGB image.
        y_only (bool): Whether to return only the Y channel.

    Returns:
        torch.Tensor: The converted YCbCr image.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]], device=img.device, dtype=img.dtype)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor(
            [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]],
            device=img.device,
            dtype=img.dtype,
        )
        bias = torch.tensor([16, 128, 128], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    return out_img / 255.0
