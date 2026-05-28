"""SSIM family: per-voxel SSIM map, mean SSIM, mask-aware SSIM.

Adapted from https://github.com/bryanlimy/clinical-super-mri (originally 2D
only via `F.conv2d`). Dispatches to `F.conv2d` or `F.conv3d` based on input
rank so the same API works on 2D images and 3D MRI volumes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel_1d(
    size: int, sigma: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g


def _separable_gaussian_filter(
    inputs: torch.Tensor,
    kernel_1d: torch.Tensor,
    spatial_dims: int,
    same_size: bool = False,
) -> torch.Tensor:
    """Apply a separable 1D Gaussian along each spatial axis.

    Args:
        inputs: (N, C, *spatial) — `spatial_dims` trailing spatial axes.
        kernel_1d: 1D kernel of shape (k,).
        spatial_dims: 2 for images, 3 for volumes.
        same_size: replicate-pad so output spatial size equals input.
    """
    assert spatial_dims in (2, 3), f"only 2D/3D supported, got {spatial_dims}"
    conv = F.conv2d if spatial_dims == 2 else F.conv3d
    channel = inputs.shape[1]
    k = kernel_1d.shape[0]

    out = inputs
    if same_size:
        pad_each = k // 2
        pad = [pad_each] * (2 * spatial_dims)
        out = F.pad(out, pad, mode="replicate")

    for axis in range(spatial_dims):
        kshape = [1] * spatial_dims
        kshape[axis] = k
        w = kernel_1d.view(*([1, 1] + kshape)).expand(channel, 1, *kshape).contiguous()
        if out.shape[2 + axis] >= k:
            out = conv(out, weight=w, stride=1, padding=0, groups=channel)
    return out


def ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    max_value: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    same_size: bool = False,
) -> torch.Tensor:
    """Per-voxel SSIM map.

    Args:
        same_size: if True, output spatial size equals input (replicate
            padding). Required when multiplying by a mask on the original grid.

    Returns:
        Tensor of shape (N, C, *spatial). When `same_size=False`, spatial
        dims are reduced by `win_size - 1`.
    """
    assert x.shape == y.shape, f"shape mismatch: {x.shape} vs {y.shape}"
    assert x.ndim in (4, 5), f"expected 4D or 5D tensor, got {x.ndim}D"
    assert win_size % 2 == 1, f"win_size must be odd, got {win_size}"

    spatial_dims = x.ndim - 2
    smallest = min(x.shape[2:])
    assert smallest >= win_size, (
        f"win_size={win_size} larger than smallest spatial dim {smallest}"
    )

    kernel = _gaussian_kernel_1d(win_size, win_sigma, dtype=x.dtype, device=x.device)
    C1 = (K1 * max_value) ** 2
    C2 = (K2 * max_value) ** 2

    g = lambda t: _separable_gaussian_filter(t, kernel, spatial_dims, same_size=same_size)
    mu1 = g(x)
    mu2 = g(y)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = g(x * x) - mu1_sq
    sigma2_sq = g(y * y) - mu2_sq
    sigma12 = g(x * y) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    return ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    max_value: float = 1.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """Structural Similarity Index for 2D (N,C,H,W) or 3D (N,C,D,H,W).

    Returns:
        Scalar mean (size_average=True) or per-batch (N,) tensor.
    """
    smap = ssim_map(x, y, max_value, win_size, win_sigma, K1, K2, same_size=False)
    per_channel = smap.flatten(start_dim=2).mean(dim=-1)
    return per_channel.mean() if size_average else per_channel.mean(dim=1)


def masked_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    max_value: float = 1.0,
    win_size: int = 11,
    win_sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """SSIM averaged over masked voxels only.

    The SSIM map is aligned to the original grid via replicate padding.
    Note: voxels within `win_size//2` of the mask boundary are still
    influenced by neighboring out-of-mask values through the Gaussian.

    Args:
        x, y: (N, C, *spatial).
        mask: bool/float, broadcastable to (N, C, *spatial). Empty masks in
            a sample contribute 0 (skipped from per-sample average) when
            `reduction == 'mean'`.
        reduction: 'mean' (over batch+channel) or 'none' → returns (N, C).
    """
    smap = ssim_map(x, y, max_value, win_size, win_sigma, K1, K2, same_size=True)
    if mask.dtype != x.dtype:
        mask = mask.to(dtype=x.dtype)
    while mask.ndim < smap.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(smap)

    weighted = smap * mask
    spatial_axes = tuple(range(2, smap.ndim))
    num = weighted.sum(dim=spatial_axes)
    den = mask.sum(dim=spatial_axes).clamp(min=eps)
    per_nc = num / den

    if reduction == "none":
        return per_nc
    valid = mask.sum(dim=spatial_axes) > 0
    if valid.any():
        return per_nc[valid].mean()
    return torch.zeros((), device=x.device, dtype=x.dtype)
