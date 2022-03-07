# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
from math import sqrt
import warnings

import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "MsSSIM",
    "Ssim",
    "ms_ssim",
    "ssim",
    "psnr",
    "PSNR"
]

_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]


def _fspecial_gauss_1d(size, sigma, device=None):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, device=device).float()
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(x, win):
    r""" Blur x with 1-D kernel
    Args:
        x (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blurred tensors
    """
    assert all(ws == 1 for ws in win.shape[1:-1]), win.shape
    if len(x.shape) == 4:
        conv = F.conv2d
    elif len(x.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(x.shape)

    C = x.shape[1]
    out = x
    for i, s in enumerate(x.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {x.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, win)
    mu2 = _gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (_gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (_gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (_gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(X, Y, win, data_range=255, sizeAverage=True, K=(0.01, 0.03), nonnegative_ssim=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        sizeAverage (bool, optional): if sizeAverage=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    assert len(X.shape) in (4, 5), f"Input images should be 4-d or 5-d tensors, but got {X.shape}"

    assert X.type() == Y.type(), "Input images should have the same dtype."

    win_size = win.shape[-1]

    assert win_size % 2 == 1, "Window size should be odd."

    ssim_per_channel, _ = _ssim(X, Y, data_range=data_range, win=win, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if sizeAverage:
        return ssim_per_channel.mean()
    return ssim_per_channel.mean(1)


def ms_ssim(X, Y, win, weights, poolMethod, data_range=255, sizeAverage=True, K=(0.01, 0.03)):

    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        sizeAverage (bool, optional): if sizeAverage=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (Tensor, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    assert X.shape == Y.shape, "Input images should have the same dimensions."

    # for d in range(len(X.shape) - 1, 1, -1):
    #     X = X.squeeze(dim=d)
    #     Y = Y.squeeze(dim=d)

    win_size = win.shape[-1]
    assert win_size % 2 == 1, "Window size should be odd."

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    levels = weights.shape[0]
    mcs = []
    ssim_per_channel = None
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = poolMethod(X, kernel_size=2, padding=padding)
            Y = poolMethod(Y, kernel_size=2, padding=padding)

    if ssim_per_channel is None:
        raise ValueError()

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=1)  # (batch, level, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(1, -1, 1), dim=1)

    if sizeAverage:
        return ms_ssim_val.mean()
    return ms_ssim_val.mean(1)


class Ssim(nn.Module):
    def __init__(self, data_range=255, sizeAverage=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2, K=(0.01, 0.03), nonnegative_ssim=False):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            sizeAverage (bool, optional): if sizeAverage=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super().__init__()
        self.register_buffer("win", _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims))
        self.sizeAverage = sizeAverage
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return 1.0 - ssim(X, Y, self.win, data_range=self.data_range, sizeAverage=self.sizeAverage, K=self.K, nonnegative_ssim=self.nonnegative_ssim)


class MsSSIM(nn.Module):
    def __init__(self, shape=4, data_range=255, sizeAverage=True, win_size=11, win_sigma=1.5, channel=3, spatial_dims=2, weights=None, K=(0.01, 0.03)):
        r""" class for ms-ssim
        Args:
            shape (int): 4 for NCHW, 5 for NCTHW
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            sizeAverage (bool, optional): if sizeAverage=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super().__init__()

        if shape == 4:
            self._avg_pool = F.avg_pool2d
        elif shape == 5:
            self._avg_pool = F.avg_pool3d
        else:
            raise ValueError(f"Input shape should be 4-d or 5-d tensors, but got {shape}")

        self.register_buffer("win", _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims), persistent=False)
        self.sizeAverage = sizeAverage
        self.data_range = data_range
        if weights is None:
            weights = torch.tensor(_WEIGHTS)
        else:
            weights = torch.tensor(weights)
        self.register_buffer("weights", weights, persistent=False)
        self.K = K

    def forward(self, X, Y):
        return 1.0 - ms_ssim(X, Y, self.win, self.weights, self._avg_pool, data_range=self.data_range, sizeAverage=self.sizeAverage, K=self.K)


def psnr(x: torch.Tensor, y: torch.Tensor, sizeAverage: bool = False, upperBound: float = 255.0):
    mse = ((x.float() - y.float()) ** 2).mean(dim=(1, 2, 3))
    res = 10 * (upperBound ** 2 / mse).log10()
    return res.mean() if sizeAverage else res


class PSNR(nn.Module):
    def __init__(self, sizeAverage: bool = False, upperBound: float = 255.0):
        super().__init__()
        self.register_buffer("_upperBound", torch.tensor(upperBound ** 2))
        self._average = sizeAverage

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mse = ((x.float() - y.float()) ** 2).mean(dim=(1, 2, 3))
        res = 10 * (self._upperBound / mse).log10()
        return res.mean() if self._average else res
